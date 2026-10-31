"""Spatio-Temporal 3D-CNN with Bidirectional Mamba (Bi-Mamba) architecture.

Combines a 4-block volumetric 3D-CNN spatial encoder, a 2-layer Bidirectional Mamba
sequence backbone with pure PyTorch selective scan fallback, a Gated Temporal Attention
Pooling layer with boolean mask support, and a classification head.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as cuda_selective_scan_fn
    HAS_CUDA_MAMBA = True
except (ImportError, Exception):
    HAS_CUDA_MAMBA = False
    cuda_selective_scan_fn = None

__all__ = [
    "Spatial3DCNNEncoder",
    "selective_scan_fn",
    "MambaBlock",
    "BiMambaSequenceModel",
    "GatedTemporalAttentionPooling",
    "ClassificationHead",
    "SpatioTemporalBiMamba",
]


class Spatial3DCNNEncoder(nn.Module):
    """4-block volumetric 3D-CNN spatial encoder for 3D BOLD volumes.

    Extracts a 128-dimensional latent representation z_t per 3D BOLD volume using 4
    convolutional blocks with channel progression 32 -> 64 -> 128 -> 256, Batch Normalization,
    LeakyReLU activations, and max pooling downsampling, followed by 3D adaptive average pooling
    and a linear projection.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 128,
        leaky_relu_slope: float = 0.01,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the 4-block 3D-CNN spatial encoder.

        Args:
            in_channels: Number of input volumetric channels (default: 1).
            out_features: Dimensionality of output spatial embedding vector (default: 128).
            leaky_relu_slope: Negative slope for LeakyReLU activations (default: 0.01).
            dropout: Dropout probability before output projection (default: 0.1).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features

        # 4 volumetric convolutional stages (channels 32 -> 64 -> 128 -> 256)
        self.block1 = self._make_block(in_channels, 32, leaky_relu_slope)
        self.block2 = self._make_block(32, 64, leaky_relu_slope)
        self.block3 = self._make_block(64, 128, leaky_relu_slope)
        self.block4 = self._make_block(128, 256, leaky_relu_slope)

        # Global spatial aggregation and projection
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(256, out_features)

    @staticmethod
    def _make_block(in_c: int, out_c: int, slope: float) -> nn.Sequential:
        """Create a 3D convolutional stage with Batch Normalization, LeakyReLU, and MaxPool3d."""
        return nn.Sequential(
            nn.Conv3d(
                in_c,
                out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial embeddings from 5D or 6D neuroimaging volume inputs.

        Args:
            x: Input tensor of shape (N, C, H, W, D) for 5D single-volume batches,
               or (B, T, C, H, W, D) for 6D temporal volume sequences.

        Returns:
            Extracted spatial feature tensor of shape (N, out_features) for 5D input,
            or (B, T, out_features) for 6D input.

        Raises:
            ValueError: If input tensor dimensionality is neither 5D nor 6D.
        """
        if x.dim() == 5:
            # Batch of 3D volumes: (N, C, H, W, D)
            is_sequence = False
            batch_size, time_steps = x.size(0), 1
            x_flat = x
        elif x.dim() == 6:
            # Sequence of 3D volumes: (B, T, C, H, W, D)
            is_sequence = True
            batch_size, time_steps, c, h, w, d = x.shape
            x_flat = x.reshape(batch_size * time_steps, c, h, w, d)
        else:
            raise ValueError(
                f"Expected 5D (N, C, H, W, D) or 6D (B, T, C, H, W, D) input tensor, "
                f"got shape {tuple(x.shape)}"
            )

        # Pass through 4 convolutional stages
        h1 = self.block1(x_flat)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)

        # Pool spatially to (N_total, 256, 1, 1, 1) and flatten
        pooled = self.global_pool(h4).flatten(1)
        dropped = self.dropout(pooled)
        out = self.fc(dropped)

        if is_sequence:
            return out.reshape(batch_size, time_steps, self.out_features)
        return out


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """Pure PyTorch selective scan reference implementation.

    Computes the selective state space sequence recurrence:
        h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
        y_t = sum_N(h_t * C_t) + D * u_t
        y_t = y_t * silu(z_t) (if z is provided)

    Args:
        u: Input signal tensor of shape (B, D, L).
        delta: Discretization step tensor of shape (B, D, L).
        A: State transition matrix of shape (D, N), strictly negative real values.
        B: Input projection matrix of shape (B, N, L).
        C: Output projection matrix of shape (B, N, L).
        D: Skip connection parameter of shape (D,) or None.
        z: Gating branch tensor of shape (B, D, L) or None.
        delta_bias: Optional bias tensor for delta of shape (D,).
        delta_softplus: Whether to apply softplus activation to delta.

    Returns:
        Output tensor y of shape (B, D, L).
    """
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    batch_size, dim, seq_len = u.shape
    d_state = A.shape[1]

    # Initialize continuous state representation: (B, D, N)
    state = torch.zeros(batch_size, dim, d_state, device=u.device, dtype=u.dtype)

    ys = []
    for t in range(seq_len):
        u_t = u[:, :, t]  # (B, D)
        delta_t = delta[:, :, t]  # (B, D)
        b_t = B[:, :, t]  # (B, N)
        c_t = C[:, :, t]  # (B, N)

        # Discretize continuous matrices:
        # A_bar = exp(delta_t * A) -> (B, D, N)
        # B_bar = (delta_t * B_t) -> (B, D, N)
        a_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
        b_bar = delta_t.unsqueeze(-1) * b_t.unsqueeze(1)

        # State update: h_t = A_bar * h_{t-1} + B_bar * u_t
        state = a_bar * state + b_bar * u_t.unsqueeze(-1)

        # Output observation: y_t = sum_n(state * c_t) -> (B, D)
        y_t = torch.sum(state * c_t.unsqueeze(1), dim=-1)

        if D is not None:
            y_t = y_t + u_t * D.unsqueeze(0)

        ys.append(y_t)

    # Stack along sequence length dimension: (B, D, L)
    y = torch.stack(ys, dim=-1)

    if z is not None:
        y = y * F.silu(z)

    return y


class MambaBlock(nn.Module):
    """Single selective state space Mamba block with dual-mode CUDA/PyTorch fallback.

    Implements the selective state space architecture (Gu & Dao, 2023):
    1. Input projection to 2 * d_inner (signal branch and gate branch).
    2. 1D depthwise causal convolution and SiLU activation.
    3. Input-dependent SSM parameter projections (delta, B, C).
    4. Hardware-accelerated or pure PyTorch selective scan recurrence.
    5. Multiplicative gating and linear output projection.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        force_pure_pytorch: bool = False,
    ) -> None:
        """Initialize Mamba block.

        Args:
            d_model: Model input and output dimensionality (default: 128).
            d_state: State space expansion dimension (default: 16).
            d_conv: 1D convolution kernel size along sequence length (default: 4).
            expand: Inner dimension expansion factor (default: 2 -> d_inner = 256).
            dt_rank: Dimensionality of delta projection (default: ceil(d_model / 16)).
            force_pure_pytorch: Force execution of pure PyTorch selective scan fallback.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.force_pure_pytorch = force_pure_pytorch

        # Input projection expanding to signal and gate branches: d_model -> 2 * d_inner
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise 1D convolution along time dimension
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Input-dependent projection to (delta, B, C)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )

        # Projection from dt_rank to inner dimension
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A parameter with log-space negative values
        a_log = torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(a_log)

        # Skip parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output linear projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence tensor through selective state space block.

        Args:
            x: Input sequence tensor of shape (B, L, d_model).

        Returns:
            Processed output tensor of shape (B, L, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: In-projection and split into signal and gate
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Step 2: 1D Depthwise convolution along sequence length
        # Permute (B, L, D) -> (B, D, L) for Conv1d
        x_conv = self.conv1d(x_branch.transpose(1, 2))[..., :seq_len]
        x_act = F.silu(x_conv)  # (B, d_inner, L)

        # Step 3: Project to delta, B, C
        x_act_t = x_act.transpose(1, 2)  # (B, L, d_inner)
        ssm_params = self.x_proj(x_act_t)
        delta, b_proj, c_proj = torch.split(
            ssm_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        delta = self.dt_proj(delta).transpose(1, 2)  # (B, d_inner, L)
        b_proj = b_proj.transpose(1, 2)  # (B, d_state, L)
        c_proj = c_proj.transpose(1, 2)  # (B, d_state, L)
        z_t = z.transpose(1, 2)  # (B, d_inner, L)

        # A is parameterized as -exp(A_log)
        A = -torch.exp(self.A_log)

        # Step 4: Selective scan (CUDA kernel if available on GPU, else pure PyTorch)
        can_use_cuda = (
            HAS_CUDA_MAMBA
            and not self.force_pure_pytorch
            and x.is_cuda
            and cuda_selective_scan_fn is not None
        )

        if can_use_cuda:
            # Compiled CUDA kernel execution
            y = cuda_selective_scan_fn(
                x_act,
                delta,
                A,
                b_proj,
                c_proj,
                self.D,
                z=z_t,
                delta_bias=None,
                delta_softplus=True,
            )
        else:
            # Pure PyTorch fallback execution
            y = selective_scan_fn(
                u=x_act,
                delta=delta,
                A=A,
                B=b_proj,
                C=c_proj,
                D=self.D,
                z=z_t,
                delta_bias=None,
                delta_softplus=True,
            )

        # Step 5: Output projection back to d_model
        y_t = y.transpose(1, 2)  # (B, L, d_inner)
        out = self.out_proj(y_t)
        return out


class BiMambaSequenceModel(nn.Module):
    """2-layer Bidirectional Mamba sequence model with dual-mode selective scan fallback.

    Scans temporal sequence representations in both forward and reverse chronological directions:
        Y_fwd = Mamba_fwd(LayerNorm(X))
        Y_rev = flip(Mamba_rev(LayerNorm(flip(X))))
        X_{l+1} = X_l + Y_fwd + Y_rev

    Features linear O(T) computational and memory complexity, capturing bidirectional
    hemodynamic dependencies across time frames.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        force_pure_pytorch: bool = False,
    ) -> None:
        """Initialize the 2-layer Bidirectional Mamba sequence backbone.

        Args:
            d_model: Dimensionality of temporal feature representations (default: 128).
            d_state: State space expansion dimension (default: 16).
            d_conv: Temporal convolution kernel size (default: 4).
            expand: Inner channel expansion factor (default: 2).
            num_layers: Number of bidirectional Mamba layers (default: 2).
            dropout: Dropout probability applied between layers (default: 0.1).
            force_pure_pytorch: Whether to force pure PyTorch selective scan fallback.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers

        self.fwd_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                force_pure_pytorch=force_pure_pytorch,
            )
            for _ in range(num_layers)
        ])

        self.rev_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                force_pure_pytorch=force_pure_pytorch,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _flip_sequence(
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flip sequence along time dimension respecting valid sequence lengths.

        When mask is None, applies uniform temporal flip along dim 1.
        When boolean mask (B, T) is provided, flips each batch item's valid prefix
        and preserves trailing zeros, preventing padded frames from contaminating
        the recurrent state space of reverse scan models.
        """
        if mask is None:
            return torch.flip(tensor, dims=[1])

        batch_size, seq_len, _ = tensor.shape
        flipped = torch.zeros_like(tensor)
        valid_lengths = mask.sum(dim=1).long()

        for b in range(batch_size):
            length = valid_lengths[b].item()
            if length > 0:
                flipped[b, :length] = torch.flip(tensor[b, :length], dims=[0])

        return flipped

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process sequence tensor bidirectionally across time frames.

        Args:
            x: Input sequence tensor of shape (B, T, d_model).
            mask: Optional boolean temporal mask of shape (B, T), where True indicates
                  a valid biological frame and False indicates a padded frame.

        Returns:
            Processed sequence tensor of shape (B, T, d_model).
        """
        out = x
        for fwd_block, rev_block, norm, dropout in zip(
            self.fwd_layers, self.rev_layers, self.norms, self.dropouts
        ):
            # Pre-layer normalization
            normed = norm(out)
            if mask is not None:
                normed = normed.masked_fill(~mask.unsqueeze(-1), 0.0)

            # Forward chronological scan
            fwd_out = fwd_block(normed)
            if mask is not None:
                fwd_out = fwd_out.masked_fill(~mask.unsqueeze(-1), 0.0)

            # Reverse chronological scan: flip valid prefix
            normed_rev = self._flip_sequence(normed, mask)
            rev_out_flipped = rev_block(normed_rev)
            rev_out = self._flip_sequence(rev_out_flipped, mask)
            if mask is not None:
                rev_out = rev_out.masked_fill(~mask.unsqueeze(-1), 0.0)

            # Residual aggregation with dropout
            layer_out = dropout(fwd_out + rev_out)
            out = out + layer_out

            # Apply mask to suppress padded frame leakage
            if mask is not None:
                out = out.masked_fill(~mask.unsqueeze(-1), 0.0)

        out = self.final_norm(out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)

        return out


class GatedTemporalAttentionPooling(nn.Module):
    """Gated Temporal Attention Pooling module for sequential neuroimaging representations.

    Computes dynamic attention weights alpha_t in [0, 1] over time frames:
        alpha_t = softmax(w^T * (tanh(W_a * h_t) * sigmoid(W_b * h_t)))
        z = sum_t alpha_t * h_t

    Applies a -inf pre-softmax mask over padded time frames to guarantee that non-biological
    padding frames receive strictly zero attention weight.
    """

    def __init__(
        self,
        in_features: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """Initialize Gated Temporal Attention Pooling.

        Args:
            in_features: Input feature dimension per frame (default: 128).
            latent_dim: Attention projection latent dimension (default: 64).
            dropout: Dropout probability applied to gated attention features (default: 0.0).
        """
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.linear_a = nn.Linear(in_features, latent_dim)
        self.linear_b = nn.Linear(in_features, latent_dim)
        self.linear_w = nn.Linear(latent_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute temporal attention weights and aggregated subject representation.

        Args:
            h: Feature sequence tensor of shape (B, T, D) or unbatched (T, D).
            mask: Optional boolean temporal mask of shape (B, T) or unbatched (T,),
                  where True indicates a valid biological frame and False indicates padding.

        Returns:
            Tuple of (pooled, attention_weights):
            - pooled: Aggregated subject representation of shape (B, D) or (D,).
            - attention_weights: Normalized attention weights of shape (B, T) or (T,)
              satisfying probability simplex constraints.

        Raises:
            ValueError: If input tensor dimensionality is neither 2D nor 3D.
        """
        if h.dim() == 2:
            unbatched = True
            h = h.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        elif h.dim() == 3:
            unbatched = False
        else:
            raise ValueError(
                f"Expected 2D (T, D) or 3D (B, T, D) input tensor, got shape {tuple(h.shape)}"
            )

        # Gated attention projection: tanh(W_a * h) * sigmoid(W_b * h)
        a_out = torch.tanh(self.linear_a(h))
        b_out = torch.sigmoid(self.linear_b(h))
        gated = self.dropout(a_out * b_out)

        # Scalar attention scores: (B, T)
        scores = self.linear_w(gated).squeeze(-1)

        # Apply -inf pre-softmax mask for padded frames
        if mask is not None:
            bool_mask = mask.bool()
            scores = scores.masked_fill(~bool_mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        # Handle potential all-masked edge cases safely
        if mask is not None:
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Weighted combination: sum_t alpha_t * h_t -> (B, D)
        pooled = torch.sum(attention_weights.unsqueeze(-1) * h, dim=1)

        if unbatched:
            return pooled.squeeze(0), attention_weights.squeeze(0)
        return pooled, attention_weights


class ClassificationHead(nn.Module):
    """Classification head mapping subject embeddings to diagnostic class logits.

    Consists of Layer Normalization, Dropout (default: 0.4), and a Linear projection
    to binary diagnostic logits (ASD vs. TC).
    """

    def __init__(
        self,
        in_features: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        """Initialize classification head.

        Args:
            in_features: Dimensionality of aggregated subject embedding (default: 128).
            num_classes: Number of diagnostic classes (default: 2 for ASD vs. TC).
            dropout: Dropout probability before linear projection (default: 0.4).
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map subject embedding to class logits.

        Args:
            x: Subject embedding tensor of shape (B, in_features) or (in_features,).

        Returns:
            Logits tensor of shape (B, num_classes) or (num_classes,).
        """
        out = self.norm(x)
        out = self.dropout(out)
        return self.fc(out)


class SpatioTemporalBiMamba(nn.Module):
    """Integrated Spatio-Temporal 3D-CNN with Bidirectional Mamba architecture.

    Processes 5D neuroimaging sequences (B, T, C, H, W, D) with accompanying boolean
    temporal masks (B, T):
    1. Spatial3DCNNEncoder: Maps each 3D BOLD volume to a 128-dimensional embedding z_t.
    2. BiMambaSequenceModel: Models chronological dynamics bidirectionally with linear O(T) complexity.
    3. GatedTemporalAttentionPooling: Dynamically weights informative time frames while zeroing out padding frames.
    4. ClassificationHead: LayerNorm(128) -> Dropout(0.4) -> Linear(128, 2) outputting ASD vs. TC logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        num_classes: int = 2,
        spatial_dropout: float = 0.1,
        temporal_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        head_dropout: float = 0.4,
        force_pure_pytorch: bool = False,
    ) -> None:
        """Initialize the Spatio-Temporal 3D-CNN + Bi-Mamba model.

        Args:
            in_channels: Number of input volumetric channels (default: 1).
            d_model: Intermediate spatial and temporal embedding dimension (default: 128).
            d_state: State space expansion dimension for Mamba (default: 16).
            d_conv: 1D convolution kernel size for Mamba (default: 4).
            expand: Expansion factor for inner Mamba dimension (default: 2 -> 256).
            num_layers: Number of bidirectional Mamba layers (default: 2).
            num_classes: Number of target diagnostic classes (default: 2 for ASD vs. TC).
            spatial_dropout: Dropout probability in 3D-CNN spatial encoder (default: 0.1).
            temporal_dropout: Dropout probability in Bi-Mamba backbone (default: 0.1).
            attention_dropout: Dropout probability in temporal attention pooling (default: 0.0).
            head_dropout: Dropout probability in classification head (default: 0.4).
            force_pure_pytorch: Force pure PyTorch fallback for selective scan.
        """
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_classes = num_classes

        # Component 1: Volumetric 3D-CNN spatial encoder
        self.spatial_encoder = Spatial3DCNNEncoder(
            in_channels=in_channels,
            out_features=d_model,
            dropout=spatial_dropout,
        )

        # Component 2: 2-layer Bidirectional Mamba sequence model
        self.sequence_model = BiMambaSequenceModel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            dropout=temporal_dropout,
            force_pure_pytorch=force_pure_pytorch,
        )

        # Component 3: Gated Temporal Attention Pooling
        self.attention_pooling = GatedTemporalAttentionPooling(
            in_features=d_model,
            latent_dim=d_model // 2 if d_model >= 2 else d_model,
            dropout=attention_dropout,
        )

        # Component 4: Classification head
        self.classification_head = ClassificationHead(
            in_features=d_model,
            num_classes=num_classes,
            dropout=head_dropout,
        )

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial embeddings from 5D neuroimaging sequence batches.

        Args:
            x: Input tensor of shape (B, T, C, H, W, D).

        Returns:
            Spatial embeddings tensor of shape (B, T, d_model).
        """
        return self.spatial_encoder(x)

    def forward_from_spatial_features(
        self,
        spatial_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run temporal sequence modeling and classification from pre-extracted spatial features.

        Args:
            spatial_features: Latent spatial representations of shape (B, T, d_model).
            mask: Optional boolean temporal mask of shape (B, T).

        Returns:
            Tuple of (logits, attention_weights):
            - logits: Diagnostic class logits of shape (B, num_classes).
            - attention_weights: Temporal attention distribution of shape (B, T).
        """
        # Temporal bidirectional sequence scanning
        h_seq = self.sequence_model(spatial_features, mask=mask)

        # Dynamic temporal attention pooling
        pooled, attention_weights = self.attention_pooling(h_seq, mask=mask)

        # Diagnostic classification head
        logits = self.classification_head(pooled)

        return logits, attention_weights

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass mapping 5D neuroimaging batches to diagnostic logits.

        Args:
            x: Input 5D neuroimaging sequence batch of shape (B, T, C, H, W, D).
            mask: Optional boolean temporal mask of shape (B, T).

        Returns:
            Tuple of (logits, attention_weights):
            - logits: Diagnostic class logits of shape (B, num_classes).
            - attention_weights: Temporal attention distribution of shape (B, T).
        """
        spatial_features = self.extract_spatial_features(x)
        return self.forward_from_spatial_features(spatial_features, mask=mask)


