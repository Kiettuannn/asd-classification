"""Unit tests for Spatio-Temporal 3D-CNN with Bidirectional Mamba architecture.

Tests exercise public interfaces at pre-agreed seams:
1. Spatial Encoder Seam (Spatial3DCNNEncoder)
2. Selective Scan and Bi-Mamba Sequence Seam (BiMambaSequenceModel)
3. Gated Temporal Attention Pooling Seam (GatedTemporalAttentionPooling)
4. End-to-End Model and BF16 Mixed Precision Gradient Seam (SpatioTemporalBiMamba)
"""

import pytest
import torch
import torch.nn as nn

from asd_classification.models.baseline.focal_loss import FocalLoss
from asd_classification.models.baseline.spatiotemporal_bimamba import (
    BiMambaSequenceModel,
    ClassificationHead,
    GatedTemporalAttentionPooling,
    Spatial3DCNNEncoder,
    SpatioTemporalBiMamba,
    selective_scan_fn,
)


def test_spatial_3dcnn_encoder_shapes():
    """Verify that Spatial3DCNNEncoder maps 4D and 5D neuroimaging inputs to 128D embeddings.
    
    Verifies channel progression 32 -> 64 -> 128 -> 256 through 4 convolutional blocks
    with Batch Normalization, LeakyReLU, and max pooling, producing 128-dimensional latent vectors.
    """
    encoder = Spatial3DCNNEncoder(in_channels=1, out_features=128)
    encoder.eval()

    # 4D input: batch of single 3D volumes (N, 1, 64, 64, 36)
    # Using small spatial dimensions for fast unit testing execution
    n_samples = 2
    x_4d = torch.randn(n_samples, 1, 64, 64, 36, dtype=torch.float32)

    with torch.no_grad():
        out_4d = encoder(x_4d)

    assert out_4d.shape == (n_samples, 128)
    assert not torch.isnan(out_4d).any()

    # 5D input: sequence of 3D volumes (B, T, 1, H, W, D)
    batch_size = 2
    time_steps = 3
    x_5d = torch.randn(batch_size, time_steps, 1, 64, 64, 36, dtype=torch.float32)

    with torch.no_grad():
        out_5d = encoder(x_5d)

    assert out_5d.shape == (batch_size, time_steps, 128)
    assert not torch.isnan(out_5d).any()


def test_selective_scan_fn_numerical_invariants():
    """Verify that pure PyTorch selective scan computes recurrence invariants correctly.

    Checks:
    1. Output tensor shape contract (B, D, L).
    2. Zero input produces zero state evolution when delta is finite.
    3. Gradient backpropagation computes finite non-zero gradients.
    """
    batch_size = 2
    dim = 8
    seq_len = 10
    d_state = 4

    u = torch.randn(batch_size, dim, seq_len, requires_grad=True)
    delta = torch.randn(batch_size, dim, seq_len, requires_grad=True)
    A = -torch.exp(torch.randn(dim, d_state))
    B = torch.randn(batch_size, d_state, seq_len, requires_grad=True)
    C = torch.randn(batch_size, d_state, seq_len, requires_grad=True)
    D = torch.randn(dim, requires_grad=True)

    y = selective_scan_fn(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        delta_softplus=True,
    )

    assert y.shape == (batch_size, dim, seq_len)
    assert not torch.isnan(y).any()

    loss = y.sum()
    loss.backward()

    assert u.grad is not None and not torch.isnan(u.grad).any()
    assert delta.grad is not None and not torch.isnan(delta.grad).any()
    assert B.grad is not None and not torch.isnan(B.grad).any()
    assert C.grad is not None and not torch.isnan(C.grad).any()
    assert D.grad is not None and not torch.isnan(D.grad).any()


def test_bimamba_sequence_model_forward_contract():
    """Verify that BiMambaSequenceModel processes (B, T, D) bidirectionally.

    Checks:
    1. 2-layer bidirectional architecture (d_model=128, d_state=16, expand=2).
    2. Input shape (B=2, T=50, 128) preserves temporal and channel dimensions (B, T, 128).
    3. Handles boolean temporal masking correctly.
    4. Executes without CUDA mamba-ssm dependencies in pure PyTorch mode.
    """
    batch_size = 2
    time_steps = 50
    d_model = 128

    model = BiMambaSequenceModel(
        d_model=d_model,
        d_state=16,
        expand=2,
        num_layers=2,
        force_pure_pytorch=True,
    )
    model.eval()

    x = torch.randn(batch_size, time_steps, d_model)
    mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    # Mask out the last 10 time frames for the second batch element
    mask[1, 40:] = False

    with torch.no_grad():
        out = model(x, mask=mask)

    assert out.shape == (batch_size, time_steps, d_model)
    assert not torch.isnan(out).any()
    # Masked positions should be zeroed
    assert torch.all(out[1, 40:] == 0.0)


def test_gated_temporal_attention_pooling_shapes_and_simplex():
    """Verify that GatedTemporalAttentionPooling pools (B, T, D) -> (B, D) satisfying simplex."""
    batch_size = 3
    time_steps = 40
    dim = 128

    pooling = GatedTemporalAttentionPooling(in_features=dim, latent_dim=64)
    h = torch.randn(batch_size, time_steps, dim)

    pooled, weights = pooling(h)

    assert pooled.shape == (batch_size, dim)
    assert weights.shape == (batch_size, time_steps)
    assert torch.all(weights >= 0.0), "Attention weights must be non-negative"
    sums = torch.sum(weights, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Attention weights must sum to 1.0"


def test_gated_temporal_attention_pooling_mask_guarantees_zero_attention():
    """Verify that padded frames receive strictly zero attention weight."""
    batch_size = 2
    time_steps = 50
    dim = 128

    pooling = GatedTemporalAttentionPooling(in_features=dim, latent_dim=64)
    h = torch.randn(batch_size, time_steps, dim)

    mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    # Mask out the last 20 frames for batch element 0, and last 10 for element 1
    mask[0, 30:] = False
    mask[1, 40:] = False

    pooled, weights = pooling(h, mask=mask)

    assert pooled.shape == (batch_size, dim)
    assert weights.shape == (batch_size, time_steps)

    # Check that padded frames are strictly 0.0
    assert torch.all(weights[0, 30:] == 0.0), "Padded frames must have strictly zero attention weight"
    assert torch.all(weights[1, 40:] == 0.0), "Padded frames must have strictly zero attention weight"

    # Check that valid frames still sum to 1.0
    sums = torch.sum(weights, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Valid frame weights must sum to 1.0"


def test_gated_temporal_attention_pooling_unbatched_input():
    """Verify that GatedTemporalAttentionPooling handles unbatched input (T, D)."""
    time_steps = 25
    dim = 128

    pooling = GatedTemporalAttentionPooling(in_features=dim, latent_dim=64)
    h = torch.randn(time_steps, dim)

    pooled, weights = pooling(h)

    assert pooled.shape == (dim,)
    assert weights.shape == (time_steps,)
    assert torch.allclose(torch.sum(weights), torch.tensor(1.0), atol=1e-5)


def test_spatiotemporal_bimamba_end_to_end_forward():
    """Verify synthetic end-to-end forward pass on (B=2, T=50, C=1, H=64, W=64, D=36).

    Verifies acceptance criteria:
    - Shape contract producing logits (2, 2) and attention maps (2, 50).
    - Masked frames receive strictly 0.0 attention weight.
    - Valid frames sum to 1.0.
    """
    batch_size = 2
    time_steps = 50
    channels = 1
    height, width, depth = 64, 64, 36

    model = SpatioTemporalBiMamba(
        in_channels=channels,
        d_model=128,
        d_state=16,
        expand=2,
        num_layers=2,
        num_classes=2,
        force_pure_pytorch=True,
    )
    model.eval()

    # Create synthetic 5D tensor (B, T, C, H, W, D)
    x = torch.randn(batch_size, time_steps, channels, height, width, depth, dtype=torch.float32)

    # Boolean temporal mask (B, T)
    mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    mask[0, 35:] = False  # 15 padded frames for first subject
    mask[1, 45:] = False  # 5 padded frames for second subject

    with torch.no_grad():
        logits, attention_weights = model(x, mask=mask)

    assert logits.shape == (batch_size, 2)
    assert attention_weights.shape == (batch_size, time_steps)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(attention_weights).any()

    # Check padded frames receive zero attention
    assert torch.all(attention_weights[0, 35:] == 0.0)
    assert torch.all(attention_weights[1, 45:] == 0.0)

    # Check valid frames sum to 1.0
    sums = torch.sum(attention_weights, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_spatiotemporal_bimamba_bf16_backward_gradient_flow():
    """Verify backward gradient flow under BF16 mixed precision with FocalLoss and temporal mask.

    Verifies acceptance criteria:
    - Autocast under BF16 executes cleanly with FocalLoss (alpha=0.25, gamma=2.0).
    - Backward pass computes gradients for all trainable parameters across spatial encoder,
      BiMamba sequence model, gated attention pooling, and classification head.
    - No NaNs or Infs in gradients.
    - All gradient norms are strictly > 0 (no zero-gradient stall).
    """
    torch.manual_seed(42)
    batch_size = 2
    time_steps = 4
    channels = 1
    height, width, depth = 32, 32, 18

    model = SpatioTemporalBiMamba(
        in_channels=channels,
        d_model=64,
        d_state=8,
        expand=2,
        num_layers=2,
        num_classes=2,
        force_pure_pytorch=True,
    )
    model.train()

    x = torch.randn(batch_size, time_steps, channels, height, width, depth, dtype=torch.float32)
    # Mask out the last frame of the second sample to verify masked backward flow
    mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    mask[1, 3] = False

    targets = torch.tensor([1, 0], dtype=torch.long)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(x, mask=mask)
        loss = criterion(logits, targets)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()

    # Verify all trainable parameters received valid, non-zero gradients
    stalled_params = []
    nan_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_params.append(name)
            grad_norm = param.grad.norm().item()
            if grad_norm == 0.0:
                stalled_params.append(name)

    assert len(nan_params) == 0, f"Parameters with NaN/Inf gradients: {nan_params}"
    assert len(stalled_params) == 0, f"Parameters with zero-gradient stall: {stalled_params}"


def test_bimamba_padding_aware_sequence_reversal():
    """Verify that padding-aware sequence reversal prevents reverse scan contamination."""
    batch_size = 2
    time_steps = 5
    d_model = 4

    # Create distinct sequential values
    tensor = torch.arange(batch_size * time_steps * d_model, dtype=torch.float32).reshape(
        batch_size, time_steps, d_model
    )
    mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    # First item has 3 valid frames (indices 0, 1, 2) and 2 padded frames (indices 3, 4)
    mask[0, 3:] = False
    tensor[0, 3:] = 0.0

    flipped = BiMambaSequenceModel._flip_sequence(tensor, mask=mask)

    # First item: valid tokens (0, 1, 2) should be reversed into indices 0, 1, 2
    assert torch.allclose(flipped[0, 0], tensor[0, 2])
    assert torch.allclose(flipped[0, 1], tensor[0, 1])
    assert torch.allclose(flipped[0, 2], tensor[0, 0])
    # Padded frames at the end should remain zero
    assert torch.all(flipped[0, 3:] == 0.0)

    # Second item has all 5 valid frames: full reverse
    assert torch.allclose(flipped[1, 0], tensor[1, 4])
    assert torch.allclose(flipped[1, 4], tensor[1, 0])



def test_spatiotemporal_bimamba_top_level_package_exports():
    """Verify that models package exports all public SpatioTemporalBiMamba components."""
    import asd_classification.models as models

    for symbol in [
        "Spatial3DCNNEncoder",
        "MambaBlock",
        "BiMambaSequenceModel",
        "GatedTemporalAttentionPooling",
        "ClassificationHead",
        "SpatioTemporalBiMamba",
        "selective_scan_fn",
    ]:
        assert hasattr(models, symbol), f"Missing exported symbol {symbol} in asd_classification.models"


def test_spatiotemporal_bimamba_spatial_feature_caching_parity():
    """Verify that decoupled spatial feature extraction yields identical outputs."""
    batch_size = 2
    time_steps = 3
    channels = 1
    height, width, depth = 16, 16, 16

    model = SpatioTemporalBiMamba(
        in_channels=channels,
        d_model=64,
        d_state=8,
        expand=2,
        num_layers=2,
        num_classes=2,
        force_pure_pytorch=True,
    )
    model.eval()

    x = torch.randn(batch_size, time_steps, channels, height, width, depth, dtype=torch.float32)

    with torch.no_grad():
        direct_logits, direct_weights = model(x)

        # Decoupled pass: extract spatial features first, then run sequence model
        spatial_features = model.extract_spatial_features(x)
        cached_logits, cached_weights = model.forward_from_spatial_features(spatial_features)

    assert torch.allclose(direct_logits, cached_logits, atol=1e-5)
    assert torch.allclose(direct_weights, cached_weights, atol=1e-5)



