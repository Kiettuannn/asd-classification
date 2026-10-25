"""Unit tests for Pretrained 2.5D ResNet-18 with Gated Attention MIL architecture.

Verifies:
1. Gated Attention MIL pooling forward contract and probability simplex constraints.
2. ResNet18MIL forward pass shape contract for batched and unbatched tri-slice bags.
3. Parameter freezing for early ResNet-18 stages (conv1, bn1, layer1, layer2).
4. Gradient flow to deep layers (layer3, layer4), attention pooling, and classification head.
5. Differential optimizer parameter group configuration.
"""

from typing import Tuple
import pytest
import torch
import torch.nn as nn

from asd_classification.models.baseline.resnet_mil import GatedAttention, ResNet18MIL


def _assert_probability_simplex(weights: torch.Tensor) -> None:
    """Helper to verify that attention weights satisfy probability simplex constraints."""
    assert torch.all(weights >= 0.0), "Attention weights must be non-negative"
    sums = torch.sum(weights, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Attention weights must sum to 1.0"


def test_gated_attention_output_shapes_and_simplex_constraints():
    """Verify that GatedAttention pools (B, K, D) to (B, D) and attention weights satisfy simplex."""
    batch_size = 4
    num_slices = 60
    in_features = 512
    latent_dim = 128

    attention = GatedAttention(in_features=in_features, latent_dim=latent_dim)
    h = torch.randn(batch_size, num_slices, in_features, dtype=torch.float32)

    pooled, weights = attention(h)

    assert pooled.shape == (batch_size, in_features)
    assert weights.shape == (batch_size, num_slices)
    _assert_probability_simplex(weights)


def test_gated_attention_unbatched_input():
    """Verify that GatedAttention correctly handles single unbatched bags of shape (K, D)."""
    num_slices = 60
    in_features = 512

    attention = GatedAttention(in_features=in_features, latent_dim=128)
    h = torch.randn(num_slices, in_features, dtype=torch.float32)

    pooled, weights = attention(h)

    assert pooled.shape == (in_features,)
    assert weights.shape == (num_slices,)
    _assert_probability_simplex(weights)


def test_resnet18_mil_forward_shape_contract():
    """Verify that ResNet18MIL accepts (B, K, 3, 224, 224) bags and outputs (B, 2) and (B, K)."""
    batch_size = 2
    num_slices = 60
    model = ResNet18MIL(pretrained=False, num_classes=2)
    model.eval()

    # Synthetic input bag (B, K, 3, 224, 224)
    x = torch.randn(batch_size, num_slices, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        logits, attention_weights = model(x)

    assert logits.shape == (batch_size, 2)
    assert attention_weights.shape == (batch_size, num_slices)
    _assert_probability_simplex(attention_weights)


def test_resnet18_mil_unbatched_input():
    """Verify that ResNet18MIL handles unbatched bag input (K, 3, H, W)."""
    num_slices = 10
    model = ResNet18MIL(pretrained=False, num_classes=2)
    model.eval()

    # Unbatched bag of shape (K, 3, 64, 64)
    x = torch.randn(num_slices, 3, 64, 64, dtype=torch.float32)

    with torch.no_grad():
        logits, attention_weights = model(x)

    assert logits.shape == (2,)
    assert attention_weights.shape == (num_slices,)
    _assert_probability_simplex(attention_weights)


def test_resnet18_mil_non_contiguous_input():
    """Verify that ResNet18MIL reliably handles non-contiguous bag tensors."""
    model = ResNet18MIL(pretrained=False, num_classes=2)
    model.eval()

    # Create a non-contiguous tensor via slicing and transpose
    base = torch.randn(2, 60, 64, 64, 3, dtype=torch.float32)
    x_non_contiguous = base.permute(0, 1, 4, 2, 3)
    assert not x_non_contiguous.is_contiguous()

    with torch.no_grad():
        logits, attention_weights = model(x_non_contiguous)

    assert logits.shape == (2, 2)
    assert attention_weights.shape == (2, 60)
    _assert_probability_simplex(attention_weights)


def test_resnet18_mil_invalid_dimensions():
    """Verify that ResNet18MIL raises ValueError when tensor rank is neither 4 nor 5."""
    model = ResNet18MIL(pretrained=False)

    with pytest.raises(ValueError, match="Expected 4D.*or 5D"):
        model(torch.randn(3, 224, 224))

    with pytest.raises(ValueError, match="Expected 3 channels"):
        model(torch.randn(1, 10, 1, 224, 224))


def test_early_layer_freezing_and_gradient_flow():
    """Verify that early stages are frozen without gradients, and deep stages receive gradients."""
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    model.train()

    # 1. Check requires_grad flag
    for module in [model.conv1, model.bn1, model.layer1, model.layer2]:
        for param in module.parameters():
            assert not param.requires_grad, "Early stage parameters must have requires_grad=False"

    for module in [model.layer3, model.layer4, model.attention_pooling, model.classifier]:
        for param in module.parameters():
            assert param.requires_grad, "Deep stages, attention, and head must have requires_grad=True"

    # 2. Check backpropagation and gradient flow
    x = torch.randn(2, 10, 3, 64, 64, dtype=torch.float32)
    logits, attention_weights = model(x)
    loss = logits.sum() + attention_weights.sum()
    loss.backward()

    # Frozen early stages must have received NO gradients (grad is None)
    for stage_name, module in [
        ("conv1", model.conv1),
        ("bn1", model.bn1),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
    ]:
        for name, param in module.named_parameters():
            assert (
                param.grad is None
            ), f"Frozen parameter {stage_name}.{name} received gradient during backward pass"

    # Trainable stages must have received gradients
    for stage_name, module in [
        ("layer3", model.layer3),
        ("layer4", model.layer4),
        ("attention_pooling", model.attention_pooling),
        ("classifier", model.classifier),
    ]:
        for name, param in module.named_parameters():
            assert (
                param.grad is not None
            ), f"Trainable parameter {stage_name}.{name} did not receive gradient"
            assert (
                torch.count_nonzero(param.grad) > 0
            ), f"Trainable parameter {stage_name}.{name} has all-zero gradient"


def test_unfreeze_and_refreeze():
    """Verify unfreeze_all and freeze_early_layers toggle parameter requires_grad state."""
    model = ResNet18MIL(pretrained=False, freeze_early=True)

    # Initial state: early frozen
    assert not any(p.requires_grad for p in model.conv1.parameters())

    # Unfreeze all
    model.unfreeze_all()
    assert all(p.requires_grad for p in model.parameters())
    model.train()
    assert model.conv1.training
    assert model.layer1.training

    # Re-freeze early layers
    model.freeze_early_layers()
    assert not any(p.requires_grad for p in model.conv1.parameters())
    assert not any(p.requires_grad for p in model.bn1.parameters())
    assert not any(p.requires_grad for p in model.layer1.parameters())
    assert not any(p.requires_grad for p in model.layer2.parameters())
    assert all(p.requires_grad for p in model.layer3.parameters())
    assert all(p.requires_grad for p in model.layer4.parameters())
    model.train()
    assert not model.conv1.training
    assert not model.layer1.training


def test_train_mode_keeps_frozen_stages_in_eval():
    """Verify that train() preserves eval mode on frozen stages to avoid BatchNorm drift."""
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    model.train()

    assert not model.conv1.training
    assert not model.bn1.training
    assert not model.layer1.training
    assert not model.layer2.training

    assert model.layer3.training
    assert model.layer4.training
    assert model.classifier.training


def test_differential_parameter_groups_and_optimizer_step():
    """Verify that get_parameter_groups correctly splits parameters and interfaces with Adam."""
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    groups = model.get_parameter_groups(
        backbone_lr=1e-5,
        head_lr=1e-4,
        backbone_weight_decay=1e-4,
        head_weight_decay=1e-3,
    )

    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[0]["name"] == "backbone_deep"

    assert groups[1]["lr"] == 1e-4
    assert groups[1]["weight_decay"] == 1e-3
    assert groups[1]["name"] == "attention_and_classifier"

    # All group parameters must require grad
    for group in groups:
        for p in group["params"]:
            assert p.requires_grad

    # Total parameters across groups must match total trainable parameters
    total_group_params = sum(len(g["params"]) for g in groups)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_group_params == total_trainable

    # Verify compatibility with torch.optim.Adam and execute a mock optimization step
    optimizer = torch.optim.Adam(groups)
    model.train()
    x = torch.randn(2, 5, 3, 32, 32)
    logits, _ = model(x)
    loss = logits.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_resnet18_mil_pretrained_initialization():
    """Verify that ResNet18MIL initializes with ImageNet pretrained weights."""
    model = ResNet18MIL(pretrained=True, num_classes=2)
    assert model.backbone is not None
    assert isinstance(model.attention_pooling, GatedAttention)
    assert not any(p.requires_grad for p in model.conv1.parameters())
    assert any(p.requires_grad for p in model.layer3.parameters())



