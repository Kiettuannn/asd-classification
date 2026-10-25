"""Pretrained 2.5D ResNet-18 with Gated Attention Multiple Instance Learning (MIL).

Implements the transfer learning architecture for multi-view fMRI volumetric
ASD vs. TC classification:
1. ImageNet-pretrained ResNet-18 backbone acting as a 2.5D tri-slice feature extractor.
2. Gated Attention MIL pooling mechanism (Ilse et al., ICML 2018) aggregating K=60
   slice representations into a single 512-dimensional patient embedding.
3. Two-layer MLP classification head predicting binary diagnostic logits.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

__all__ = ["GatedAttention", "ResNet18MIL"]


class GatedAttention(nn.Module):
    """Gated Attention Multiple Instance Learning (MIL) pooling module.

    Computes instance attention weights over a bag of feature representations using
    the gated attention mechanism proposed by Ilse et al. (ICML 2018):
        a_k = softmax(w^T * (tanh(V * h_k) * sigmoid(U * h_k)))
        z = sum_{k=1}^K a_k * h_k

    This non-linear gating dynamically weighs discriminative regions while suppressing
    uninformative background slices (e.g. skull, air).
    """

    def __init__(
        self,
        in_features: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        """Initialize GatedAttention pooling module.

        Args:
            in_features: Dimensionality D of input slice representations (default: 512).
            latent_dim: Dimensionality L of the hidden attention projection space (default: 128).
            dropout: Dropout probability applied to the gated attention representations (default: 0.0).
        """
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        # Non-linear projection branches for gated attention
        self.linear_v = nn.Linear(in_features, latent_dim)
        self.linear_u = nn.Linear(in_features, latent_dim)
        self.linear_w = nn.Linear(latent_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute slice attention weights and aggregated bag embedding.

        Args:
            h: Slice feature representations of shape (B, K, D) or unbatched (K, D).

        Returns:
            Tuple of (pooled, attention_weights):
            - pooled: Bag-level representation of shape (B, D) or (D,).
            - attention_weights: Normalized attention weights of shape (B, K) or (K,)
              satisfying probability simplex constraints (sum == 1.0, non-negative).

        Raises:
            ValueError: If input tensor dimensionality is not 2D or 3D.
        """
        if h.dim() == 2:
            unbatched = True
            h = h.unsqueeze(0)
        elif h.dim() == 3:
            unbatched = False
        else:
            raise ValueError(
                f"Expected 2D (K, D) or 3D (B, K, D) input tensor, got shape {tuple(h.shape)}"
            )

        # Compute gated attention scores: tanh(V * h) * sigmoid(U * h)
        # Shape: (B, K, L)
        v_out = torch.tanh(self.linear_v(h))
        u_out = torch.sigmoid(self.linear_u(h))
        gated = self.dropout(v_out * u_out)

        # Project to scalar logits per slice and squeeze channel dimension: (B, K)
        scores = self.linear_w(gated).squeeze(-1)

        # Normalize attention weights along slice dimension (simplex constraint)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted combination: sum_{k=1}^K a_k * h_k -> (B, D)
        pooled = torch.sum(attention_weights.unsqueeze(-1) * h, dim=1)

        if unbatched:
            return pooled.squeeze(0), attention_weights.squeeze(0)
        return pooled, attention_weights


class ResNet18MIL(nn.Module):
    """Pretrained 2.5D ResNet-18 with Gated Attention Multiple Instance Learning (MIL).

    Combines an ImageNet-pretrained ResNet-18 slice feature extractor with a Gated
    Attention MIL pooling module and an MLP classification head for ASD vs. TC diagnosis.

    Architecture:
    1. Backbone: ResNet-18 (early layers conv1, bn1, layer1, layer2 frozen by default;
       deep layers layer3, layer4 trainable) extracting 512-dim slice feature vectors.
    2. Attention Pooling: Gated Attention mechanism (Ilse et al., 2018) mapping K=60
       slice representations into a single 512-dim patient representation.
    3. Classifier: 2-layer MLP (Linear(512, 128) -> ReLU -> Dropout(0.4) -> Linear(128, 2))
       predicting diagnostic logits.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights: Optional[Union[ResNet18_Weights, str]] = "DEFAULT",
        num_classes: int = 2,
        in_features: int = 512,
        attention_dim: int = 128,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.4,
        freeze_early: bool = True,
    ) -> None:
        """Initialize ResNet18MIL network.

        Args:
            pretrained: Whether to load ImageNet pretrained weights (default: True).
            weights: Specific torchvision ResNet18_Weights or string alias (default: 'DEFAULT').
            num_classes: Number of target classification classes (default: 2 for ASD vs. TC).
            in_features: Output feature dimension of the ResNet-18 backbone (default: 512).
            attention_dim: Latent dimension for gated attention projections (default: 128).
            classifier_hidden_dim: Hidden layer size for the classification MLP (default: 128).
            dropout: Dropout rate applied within the classifier head (default: 0.4).
            freeze_early: Whether to freeze early backbone stages (conv1, bn1, layer1, layer2).
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.attention_dim = attention_dim
        self.classifier_hidden_dim = classifier_hidden_dim
        self.dropout_rate = dropout
        self.freeze_early = freeze_early

        # Resolve pretrained weights specification
        weights_obj: Optional[ResNet18_Weights]
        if not pretrained or weights is None:
            weights_obj = None
        elif isinstance(weights, str):
            if weights.upper() == "DEFAULT":
                weights_obj = ResNet18_Weights.DEFAULT
            else:
                weights_obj = getattr(ResNet18_Weights, weights)
        else:
            weights_obj = weights

        # Standardize pretrained weight selection or initialize random weights for testing
        self.backbone = resnet18(weights=weights_obj)
        # Replace the final 1000-class classification head with an identity pass-through
        # so self.backbone(x) directly yields the 512-dimensional pooled slice embedding
        self.backbone.fc = nn.Identity()

        # Learnable gated attention weights dynamically suppress non-informative slices
        self.attention_pooling = GatedAttention(
            in_features=in_features,
            latent_dim=attention_dim,
            dropout=0.0,
        )

        # Map patient-level pooled embedding to diagnostic binary class logits
        self.classifier = nn.Sequential(
            nn.Linear(in_features, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

        # Freeze early layers to prevent catastrophic forgetting of ImageNet representations
        if self.freeze_early:
            self.freeze_early_layers()

    @property
    def conv1(self) -> nn.Module:
        """First convolutional layer of the ResNet backbone."""
        return self.backbone.conv1

    @property
    def bn1(self) -> nn.Module:
        """First batch normalization layer of the ResNet backbone."""
        return self.backbone.bn1

    @property
    def layer1(self) -> nn.Module:
        """Stage 1 residual blocks of the ResNet backbone."""
        return self.backbone.layer1

    @property
    def layer2(self) -> nn.Module:
        """Stage 2 residual blocks of the ResNet backbone."""
        return self.backbone.layer2

    @property
    def layer3(self) -> nn.Module:
        """Stage 3 residual blocks of the ResNet backbone."""
        return self.backbone.layer3

    @property
    def layer4(self) -> nn.Module:
        """Stage 4 residual blocks of the ResNet backbone."""
        return self.backbone.layer4

    def freeze_early_layers(self) -> None:
        """Freeze early ResNet-18 layers (conv1, bn1, layer1, layer2).

        Sets requires_grad=False for all parameters in early stages to retain general
        visual feature representations and reduce parameter footprint during transfer learning.
        """
        for module in [self.backbone.conv1, self.backbone.bn1, self.backbone.layer1, self.backbone.layer2]:
            for param in module.parameters():
                param.requires_grad = False
        self.freeze_early = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters across backbone, attention, and classifier."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_early = False

    def train(self, mode: bool = True) -> "ResNet18MIL":
        """Set module and submodules into training or evaluation mode.

        When freeze_early is True, early backbone stages (conv1, bn1, layer1, layer2)
        remain in evaluation mode (eval) even during model training so that BatchNorm
        running statistics do not drift on small-batch neuroimaging inputs.

        Args:
            mode: Boolean flag indicating whether to set training mode (True) or eval mode (False).

        Returns:
            Self instance.
        """
        super().train(mode)
        if mode and self.freeze_early:
            self.backbone.conv1.eval()
            self.backbone.bn1.eval()
            self.backbone.layer1.eval()
            self.backbone.layer2.eval()
        return self

    def extract_slice_features(self, x_slices: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional visual embeddings for a batch of 2.5D slices.

        Args:
            x_slices: 2.5D tri-slice tensor of shape (N, 3, H, W).

        Returns:
            Feature representations of shape (N, 512).
        """
        return self.backbone(x_slices)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward bag of 2.5D tri-slices to produce diagnostic class logits and attention weights.

        Args:
            x: Input bag tensor of shape (B, K, 3, H, W) or unbatched (K, 3, H, W),
               where K is the number of sampled tri-slices (e.g. 60) and 3 channels
               represent adjacent orthogonal depths [z-1, z, z+1].

        Returns:
            Tuple of (logits, attention_weights):
            - logits: Diagnostic prediction logits of shape (B, num_classes) or (num_classes,).
            - attention_weights: Normalized attention weights of shape (B, K) or (K,)
              satisfying probability simplex constraints.

        Raises:
            ValueError: If input tensor rank is neither 4 nor 5, or if channel dimension != 3.
        """
        if x.dim() == 4:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() == 5:
            unbatched = False
        else:
            raise ValueError(
                f"Expected 4D (K, C, H, W) or 5D (B, K, C, H, W) tensor, got shape {tuple(x.shape)}"
            )

        B, K, C, H, W = x.shape
        if C != 3:
            raise ValueError(
                f"Expected 3 channels representing [z-1, z, z+1] tri-slices, got {C} channels."
            )

        # Flatten bags into a single batch of 2D slices for ResNet-18: (B * K, 3, H, W)
        # Using reshape ensures compatibility even when slices are non-contiguous in memory
        x_flat = x.reshape(B * K, C, H, W)

        # Extract 512-dimensional slice feature representations: (B * K, 512)
        features_flat = self.extract_slice_features(x_flat)

        # Reshape back to bag representations: (B, K, 512)
        features = features_flat.reshape(B, K, self.in_features)

        # Gated Attention MIL pooling: (B, 512) and (B, K)
        pooled, attention_weights = self.attention_pooling(features)

        # Classification head: (B, num_classes)
        logits = self.classifier(pooled)

        if unbatched:
            return logits.squeeze(0), attention_weights.squeeze(0)
        return logits, attention_weights

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        backbone_weight_decay: float = 1e-4,
        head_weight_decay: float = 1e-3,
    ) -> List[Dict[str, Any]]:
        """Configure parameter groups for differential learning rate optimization.

        Assigns a low learning rate to fine-tune deep backbone stages (layer3 and layer4)
        and a higher learning rate to the attention pooling and classifier heads.
        Frozen early layers (conv1, bn1, layer1, layer2) are excluded.

        Args:
            backbone_lr: Learning rate for backbone parameters (default: 1e-5).
            head_lr: Learning rate for attention and classifier heads (default: 1e-4).
            backbone_weight_decay: Weight decay for backbone parameters (default: 1e-4).
            head_weight_decay: Weight decay for attention and classifier (default: 1e-3).

        Returns:
            List of parameter group dictionaries compatible with torch.optim.Optimizer.
        """
        backbone_params = [
            param for name, param in self.backbone.named_parameters()
            if param.requires_grad
        ]

        head_params = [
            param
            for module in [self.attention_pooling, self.classifier]
            for param in module.parameters()
            if param.requires_grad
        ]

        return [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": backbone_weight_decay,
                "name": "backbone_deep",
            },
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": head_weight_decay,
                "name": "attention_and_classifier",
            },
        ]
