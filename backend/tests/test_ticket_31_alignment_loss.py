"""
Unit tests for PhenotypicTextEncoder, InfoNCELoss, and DualModalLoss (Seam 2).
"""

import pytest
import torch
import torch.nn as nn

from asd_classification.models.losses import DualModalLoss, InfoNCELoss
from asd_classification.models.text_encoder import PhenotypicTextEncoder


def test_phenotypic_text_encoder_projection_and_normalization():
    """Verify text encoder projects prompts to d=128 with unit L2 norm."""
    embed_dim = 128
    # Use deterministic lightweight backbone for fast unit testing
    encoder = PhenotypicTextEncoder(
        embed_dim=embed_dim,
        use_mock_backbone=True,
        hidden_dim=256,
    )

    prompts = [
        "A 14.5-year-old male, Full-scale IQ 105, Verbal IQ 98, evaluated on ADOS.",
        "A 22.0-year-old female, Full-scale IQ 112, Verbal IQ 115, evaluated on ADOS.",
        "A 9.2-year-old male, Full-scale IQ not documented, Verbal IQ not documented.",
    ]

    z_text = encoder(prompts)
    assert z_text.shape == (len(prompts), embed_dim)

    # Verify L2 normalization: norm of each vector must be 1.0
    norms = torch.norm(z_text, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_phenotypic_text_encoder_layer_freezing():
    """Verify early transformer layers are frozen while the projection head and late layers are trainable."""
    encoder = PhenotypicTextEncoder(
        embed_dim=128,
        use_mock_backbone=True,
        hidden_dim=256,
        freeze_early_layers=True,
    )

    # Projection head must be trainable
    for param in encoder.proj.parameters():
        assert param.requires_grad is True

    # Check that frozen layers have requires_grad = False
    frozen_params = [p for p in encoder.backbone_early.parameters()]
    assert len(frozen_params) > 0
    for param in frozen_params:
        assert param.requires_grad is False


def test_infonce_loss_symmetric_computation():
    """Verify symmetric temperature-scaled InfoNCE loss computes cross-entropy over similarities."""
    batch_size = 4
    embed_dim = 128

    criterion = InfoNCELoss(temperature=0.07, learnable_temp=True)

    # Perfectly aligned embeddings
    z_brain = torch.randn(batch_size, embed_dim)
    z_brain = torch.nn.functional.normalize(z_brain, p=2, dim=-1)
    z_text = z_brain.clone()  # Identical embeddings -> low loss

    loss_aligned = criterion(z_brain, z_text)
    assert loss_aligned > 0.0
    assert not torch.isnan(loss_aligned)

    # Perturbed embeddings -> higher loss
    z_perturbed = torch.randn(batch_size, embed_dim)
    z_perturbed = torch.nn.functional.normalize(z_perturbed, p=2, dim=-1)
    loss_perturbed = criterion(z_brain, z_perturbed)

    assert loss_perturbed > loss_aligned


def test_dual_modal_loss_combination_and_gradients():
    """Verify DualModalLoss combines label-smoothed CE and InfoNCE with non-zero gradient backprop."""
    batch_size = 4
    num_classes = 2
    embed_dim = 128

    loss_fn = DualModalLoss(lambda_weight=0.3, label_smoothing=0.1, temperature=0.07)

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    z_brain = torch.randn(batch_size, embed_dim, requires_grad=True)
    z_text = torch.randn(batch_size, embed_dim, requires_grad=True)

    loss_dict = loss_fn(logits, labels, z_brain, z_text)
    assert "loss" in loss_dict
    assert "loss_ce" in loss_dict
    assert "loss_infonce" in loss_dict

    total_loss = loss_dict["loss"]
    expected_loss = loss_dict["loss_ce"] + 0.3 * loss_dict["loss_infonce"]
    assert torch.allclose(total_loss, expected_loss, atol=1e-5)

    # Verify backpropagation produces non-zero gradients to both modalities and logits
    total_loss.backward()
    assert logits.grad is not None and torch.any(logits.grad != 0.0)
    assert z_brain.grad is not None and torch.any(z_brain.grad != 0.0)
    assert z_text.grad is not None and torch.any(z_text.grad != 0.0)
