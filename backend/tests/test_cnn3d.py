"""Unit tests for Baseline 3D-CNN architecture, Focal Loss, and Volume-Consistent Augmentation.

Verifies acceptance criteria for Ticket #7:
1. Baseline3DCNN channel progression (32 -> 64 -> 128 -> 128 -> 256 -> 256) and Dropout 0.4.
2. Forward pass output shape (2, 2) on batch (2, 1, 150, 224, 224).
3. Forward and backward propagation under Automatic Mixed Precision (torch.amp.autocast).
4. FocalLoss finite values, down-weighting of easy samples, and Cross-Entropy toggle.
5. VolumeConsistentAugmenter slice invariance, parameter bounds, and evaluation bypass.
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from asd_classification.models.baseline.cnn3d import (
    Baseline3DCNN,
    SEBlock3D,
    SEResBlock3D,
    SEResNet3D,
)
from asd_classification.models.baseline.focal_loss import FocalLoss
from asd_classification.models.baseline.augmentation import VolumeConsistentAugmenter


# ---------------------------------------------------------------------------
# 1. Baseline3DCNN Architectural & Forward/Backward Tests
# ---------------------------------------------------------------------------

def test_baseline_3dcnn_channel_progression_and_dropout():
    """Verify channel progression (32 -> 64 -> 128 -> 128 -> 256 -> 256) and Dropout 0.4."""
    model = Baseline3DCNN(in_channels=1, num_classes=2, dropout=0.4)

    assert model.channel_progression == [32, 64, 128, 128, 256, 256]
    assert math.isclose(model.dropout_rate, 0.4, rel_tol=1e-5)

    # Check that convolutional layers reflect the specification
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv3d)]
    assert len(conv_layers) == 6
    expected_channels = [
        (1, 32),
        (32, 64),
        (64, 128),
        (128, 128),
        (128, 256),
        (256, 256),
    ]
    for conv, (expected_in, expected_out) in zip(conv_layers, expected_channels):
        assert conv.in_channels == expected_in
        assert conv.out_channels == expected_out
        assert conv.kernel_size == (3, 3, 3)
        assert conv.stride == (1, 1, 1)
        assert conv.padding == (1, 1, 1)

    # Verify BatchNorm3d and MaxPool3d placement
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm3d)]
    assert len(bn_layers) == 6

    pool_layers = [m for m in model.modules() if isinstance(m, nn.MaxPool3d)]
    assert len(pool_layers) == 4  # Blocks 1 to 4 have MaxPool3d(2), blocks 5 and 6 do not

    # Check classifier head
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 1
    assert linear_layers[0].in_features == 256
    assert linear_layers[0].out_features == 2


def test_baseline_3dcnn_forward_pass_shape():
    """Verify forward pass on synthetic batch (2, 1, 150, 224, 224) outputs logits of shape (2, 2)."""
    model = Baseline3DCNN(in_channels=1, num_classes=2, dropout=0.4)
    model.eval()

    # Synthetic batch of 2 subjects with depth 150 and spatial size 224x224
    inputs = torch.randn(2, 1, 150, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_baseline_3dcnn_amp_forward_backward():
    """Verify forward and backward propagation under Automatic Mixed Precision updates all gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    model = Baseline3DCNN(in_channels=1, num_classes=2, dropout=0.4).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)

    # Use a reduced spatial dimension for fast CI execution while verifying gradient flow
    inputs = torch.randn(2, 1, 16, 32, 32, dtype=torch.float32, device=device)
    targets = torch.tensor([0, 1], dtype=torch.long, device=device)

    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=amp_dtype):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

    assert torch.isfinite(loss).item()
    loss.backward()

    # Verify that every trainable parameter has a valid, finite gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients."
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients."
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients."

    optimizer.step()


def test_baseline_3dcnn_predict_and_proba():
    """Verify predict and predict_proba return expected numpy arrays and probabilities sum to 1.0."""
    model = Baseline3DCNN(in_channels=1, num_classes=2, dropout=0.4)
    model.eval()

    inputs = torch.randn(3, 1, 16, 32, 32, dtype=torch.float32)
    preds = model.predict(inputs)
    probas = model.predict_proba(inputs)

    assert isinstance(preds, np.ndarray)
    assert isinstance(probas, np.ndarray)
    assert preds.shape == (3,)
    assert probas.shape == (3, 2)
    np.testing.assert_allclose(probas.sum(axis=-1), np.ones(3), atol=1e-5)


def test_baseline_3dcnn_save_and_load(tmp_path: Path):
    """Verify model weights can be saved to and loaded from disk."""
    model = Baseline3DCNN(in_channels=1, num_classes=2, dropout=0.4)
    save_file = tmp_path / "baseline_3dcnn.pt"

    model.save(save_file)
    assert save_file.is_file()

    restored = Baseline3DCNN(in_channels=1, num_classes=2, dropout=0.4)
    restored.load(save_file)

    for p1, p2 in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# 2. FocalLoss Mathematical & Functional Tests
# ---------------------------------------------------------------------------

def test_focal_loss_finite_and_reduction_modes():
    """Verify FocalLoss computes finite values across all reduction modes."""
    criterion_mean = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    criterion_sum = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")
    criterion_none = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5], [0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 0], dtype=torch.long)

    loss_mean = criterion_mean(logits, targets)
    loss_sum = criterion_sum(logits, targets)
    loss_none = criterion_none(logits, targets)

    assert torch.isfinite(loss_mean)
    assert loss_mean.ndim == 0
    assert torch.isfinite(loss_sum)
    assert loss_sum.ndim == 0
    assert torch.all(torch.isfinite(loss_none))
    assert loss_none.shape == (3,)
    assert torch.all(loss_none > 0)
    assert math.isclose(loss_mean.item(), loss_sum.item() / 3.0, rel_tol=1e-5)


def test_focal_loss_places_higher_weight_on_difficult_samples():
    """Verify FocalLoss suppresses easy samples and places higher relative weight on hard samples."""
    criterion = FocalLoss(alpha=None, gamma=2.0, reduction="none")
    ce_criterion = nn.CrossEntropyLoss(reduction="none")

    # Easy sample: model is confident and correct (logits: [5.0, -5.0], target: 0)
    # Hard sample: model is confident but wrong (logits: [-5.0, 5.0], target: 0)
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]], dtype=torch.float32)
    targets = torch.tensor([0, 0], dtype=torch.long)

    focal_losses = criterion(logits, targets)
    ce_losses = ce_criterion(logits, targets)

    easy_focal = focal_losses[0].item()
    hard_focal = focal_losses[1].item()
    easy_ce = ce_losses[0].item()
    hard_ce = ce_losses[1].item()

    # The modulating factor (1 - p_t)^gamma heavily attenuates the easy sample
    focal_ratio = hard_focal / (easy_focal + 1e-12)
    ce_ratio = hard_ce / (easy_ce + 1e-12)

    assert easy_focal < easy_ce
    assert focal_ratio > ce_ratio


def test_focal_loss_cross_entropy_toggle():
    """Verify FocalLoss matches CrossEntropyLoss when use_cross_entropy=True or gamma=0 with alpha=None."""
    criterion_toggle = FocalLoss(use_cross_entropy=True, reduction="mean")
    criterion_gamma_zero = FocalLoss(alpha=None, gamma=0.0, reduction="mean")
    standard_ce = nn.CrossEntropyLoss(reduction="mean")

    logits = torch.randn(10, 2, dtype=torch.float32)
    targets = torch.randint(0, 2, (10,), dtype=torch.long)

    expected_loss = standard_ce(logits, targets)
    toggle_loss = criterion_toggle(logits, targets)
    gamma_zero_loss = criterion_gamma_zero(logits, targets)

    torch.testing.assert_close(toggle_loss, expected_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gamma_zero_loss, expected_loss, atol=1e-5, rtol=1e-5)


def test_focal_loss_alpha_weighting():
    """Verify alpha balancing weights positive (ASD=1) and negative (TC=0) classes."""
    alpha = 0.25
    criterion = FocalLoss(alpha=alpha, gamma=0.0, reduction="none")

    # Symmetric logits where class 0 and class 1 have identical probability of 0.5
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    losses = criterion(logits, targets)
    loss_class_0 = losses[0].item()
    loss_class_1 = losses[1].item()

    # For class 0: weight is 1 - alpha = 0.75; for class 1: weight is alpha = 0.25
    expected_ratio = (1.0 - alpha) / alpha
    actual_ratio = loss_class_0 / loss_class_1
    assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-4)


def test_focal_loss_invalid_reduction():
    """Verify invalid reduction raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported reduction"):
        FocalLoss(reduction="invalid_mode")


# ---------------------------------------------------------------------------
# 3. VolumeConsistentAugmenter Invariance & Contract Tests
# ---------------------------------------------------------------------------

def test_volume_consistent_augmenter_eval_mode_bypass():
    """Verify augmenter behaves as an identity function when training=False."""
    augmenter = VolumeConsistentAugmenter()
    augmenter.eval()

    tensor = torch.randn(1, 150, 224, 224, dtype=torch.float32)
    augmented = augmenter(tensor)

    torch.testing.assert_close(augmented, tensor)


def test_volume_consistent_augmenter_slice_invariance():
    """Verify exact same rotation, flips, and scaling are applied across all 150 slices."""
    augmenter = VolumeConsistentAugmenter(
        max_rotation_angle=10.0,
        p_flip_horizontal=1.0,
        p_flip_vertical=1.0,
        intensity_scale_range=(0.9, 1.1),
    )
    augmenter.train()

    # Construct volume where slice 0 (axial) and slice 149 (sagittal) have identical content
    volume = torch.zeros(1, 150, 224, 224, dtype=torch.float32)
    pattern = torch.randn(224, 224, dtype=torch.float32)
    volume[0, 0] = pattern
    volume[0, 149] = pattern

    # Apply deterministic transform parameters
    params = {
        "rotation_angle": 7.5,
        "flip_horizontal": True,
        "flip_vertical": True,
        "intensity_scale": 1.08,
    }
    augmented = augmenter(volume, params=params)

    # Slice 0 and slice 149 must remain identical
    slice_diff = (augmented[0, 0] - augmented[0, 149]).abs().max().item()
    assert slice_diff < 1e-6, f"Slice invariance violated: max diff {slice_diff}"


def test_volume_consistent_augmenter_shapes():
    """Verify augmenter supports single volume, batched volume, and 3D tensors."""
    augmenter = VolumeConsistentAugmenter()
    augmenter.train()

    # Single volume: (1, 150, 224, 224)
    x1 = torch.randn(1, 150, 224, 224, dtype=torch.float32)
    out1 = augmenter(x1)
    assert out1.shape == (1, 150, 224, 224)

    # Batched volume: (2, 1, 150, 224, 224)
    x2 = torch.randn(2, 1, 150, 224, 224, dtype=torch.float32)
    out2 = augmenter(x2)
    assert out2.shape == (2, 1, 150, 224, 224)

    # 3D tensor: (150, 224, 224)
    x3 = torch.randn(150, 224, 224, dtype=torch.float32)
    out3 = augmenter(x3)
    assert out3.shape == (150, 224, 224)


def test_volume_consistent_augmenter_parameter_bounds():
    """Verify sampled random parameters fall strictly within specified bounds."""
    augmenter = VolumeConsistentAugmenter(
        max_rotation_angle=10.0,
        intensity_scale_range=(0.9, 1.1),
    )

    for _ in range(50):
        params = augmenter.sample_parameters()
        assert -10.0 <= params["rotation_angle"] <= 10.0
        assert 0.9 <= params["intensity_scale"] <= 1.1
        assert isinstance(params["flip_horizontal"], bool)
        assert isinstance(params["flip_vertical"], bool)


# ---------------------------------------------------------------------------
# 4. SEResNet3D, SEResBlock3D, and SEBlock3D Architecture Tests
# ---------------------------------------------------------------------------

def test_se_block_3d_forward_and_scaling():
    """Verify SEBlock3D produces identical output shapes and channel recalibration scales."""
    se = SEBlock3D(channels=64, reduction=16)
    inputs = torch.randn(2, 64, 8, 14, 14, dtype=torch.float32)

    outputs = se(inputs)
    assert outputs.shape == inputs.shape
    assert not torch.isnan(outputs).any()
    assert not torch.isinf(outputs).any()

    # Verify floor on bottleneck channel reduction for small channel counts
    se_small = SEBlock3D(channels=32, reduction=16)
    # 32 // 16 = 2, but max(2, 8) = 8
    linear_layer = [m for m in se_small.fc if isinstance(m, nn.Linear)][0]
    assert linear_layer.out_features == 8


def test_seres_block_3d_identity_and_downsample():
    """Verify SEResBlock3D identity shortcut for stride=1 and downsampling shortcut for stride=2."""
    # Identity shortcut when stride=1 and in_channels == out_channels
    identity_block = SEResBlock3D(in_channels=64, out_channels=64, stride=1)
    assert isinstance(identity_block.shortcut, nn.Identity)

    x_same = torch.randn(2, 64, 8, 14, 14, dtype=torch.float32)
    out_same = identity_block(x_same)
    assert out_same.shape == (2, 64, 8, 14, 14)

    # Downsampling shortcut when stride=2 and channels change
    downsample_block = SEResBlock3D(in_channels=32, out_channels=64, stride=2)
    assert isinstance(downsample_block.shortcut, nn.Sequential)

    x_down = torch.randn(2, 32, 16, 28, 28, dtype=torch.float32)
    out_down = downsample_block(x_down)
    assert out_down.shape == (2, 64, 8, 14, 14)


def test_seresnet_3d_architecture_and_parameter_count():
    """Verify SEResNet3D stage progression, dropout rate, and ~3.6M parameter capacity."""
    model = SEResNet3D(in_channels=1, num_classes=2, dropout=0.4)

    assert model.channel_progression == [32, 64, 128, 256]
    assert math.isclose(model.dropout_rate, 0.4, rel_tol=1e-5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # SEResNet3D has ~3.598M parameters, well matched with Baseline3DCNN's 3.38M
    assert 3_500_000 <= total_params <= 3_700_000, f"Unexpected parameter count: {total_params}"


def test_seresnet_3d_forward_pass_shape():
    """Verify forward pass on synthetic batch (2, 1, 150, 224, 224) outputs logits of shape (2, 2)."""
    model = SEResNet3D(in_channels=1, num_classes=2, dropout=0.4)
    model.eval()

    inputs = torch.randn(2, 1, 150, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_seresnet_3d_amp_forward_backward():
    """Verify forward and backward propagation under Automatic Mixed Precision updates all gradients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    model = SEResNet3D(in_channels=1, num_classes=2, dropout=0.4).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)

    inputs = torch.randn(2, 1, 16, 32, 32, dtype=torch.float32, device=device)
    targets = torch.tensor([0, 1], dtype=torch.long, device=device)

    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=amp_dtype):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

    assert torch.isfinite(loss).item()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients."
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients."
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients."

    optimizer.step()


def test_seresnet_3d_predict_and_proba():
    """Verify predict and predict_proba return expected numpy arrays and probabilities sum to 1.0."""
    model = SEResNet3D(in_channels=1, num_classes=2, dropout=0.4)
    model.eval()

    inputs = torch.randn(3, 1, 16, 32, 32, dtype=torch.float32)
    preds = model.predict(inputs)
    probas = model.predict_proba(inputs)

    assert isinstance(preds, np.ndarray)
    assert isinstance(probas, np.ndarray)
    assert preds.shape == (3,)
    assert probas.shape == (3, 2)
    np.testing.assert_allclose(probas.sum(axis=-1), np.ones(3), atol=1e-5)


def test_seresnet_3d_save_and_load(tmp_path: Path):
    """Verify SEResNet3D model weights can be saved to and loaded from disk."""
    model = SEResNet3D(in_channels=1, num_classes=2, dropout=0.4)
    save_file = tmp_path / "seresnet3d.pt"

    model.save(save_file)
    assert save_file.is_file()

    restored = SEResNet3D(in_channels=1, num_classes=2, dropout=0.4)
    restored.load(save_file)

    for p1, p2 in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p1, p2)

