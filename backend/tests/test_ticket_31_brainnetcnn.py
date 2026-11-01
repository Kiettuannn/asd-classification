"""
Unit tests for BrainNetCNN architecture and layers (Seam 1).
"""

import pytest
import torch

from asd_classification.models.brainnetcnn import (
    BrainNetCNN,
    E2EBlock,
    E2NBlock,
    N2GBlock,
)


def test_e2e_block_forward_dimensions():
    """Verify E2EBlock cross convolutions transform edge tensor dimensions correctly."""
    batch_size = 4
    num_rois = 200
    in_channels = 1
    out_channels = 32

    block = E2EBlock(
        num_rois=num_rois,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=0.1,
    )
    x = torch.randn(batch_size, in_channels, num_rois, num_rois)
    out = block(x)

    assert out.shape == (batch_size, out_channels, num_rois, num_rois)
    assert not torch.isnan(out).any()


def test_e2n_block_forward_dimensions():
    """Verify E2NBlock edge-to-node convolution pools edge features into node representations."""
    batch_size = 4
    num_rois = 200
    in_channels = 32
    out_channels = 64

    block = E2NBlock(
        num_rois=num_rois,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=0.1,
    )
    x = torch.randn(batch_size, in_channels, num_rois, num_rois)
    out = block(x)

    assert out.shape == (batch_size, out_channels, num_rois)
    assert not torch.isnan(out).any()


def test_n2g_block_forward_dimensions():
    """Verify N2GBlock node-to-graph convolution aggregates node features into graph vectors."""
    batch_size = 4
    num_rois = 200
    in_channels = 64
    out_channels = 128

    block = N2GBlock(
        num_rois=num_rois,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    x = torch.randn(batch_size, in_channels, num_rois)
    out = block(x)

    # Acceptance criterion: maps (B, 64, 200) -> (B, 128, 1)
    assert out.shape == (batch_size, out_channels, 1)
    assert not torch.isnan(out).any()


def test_brainnetcnn_forward_and_embeddings():
    """Verify BrainNetCNN produces valid logits and L2-normalized latent vectors."""
    batch_size = 4
    num_rois = 200

    model = BrainNetCNN(
        num_rois=num_rois,
        in_channels=1,
        e2e_channels=32,
        e2n_channels=64,
        n2g_channels=128,
        embed_dim=128,
        num_classes=2,
    )

    # 4D input: (B, 1, 200, 200)
    x_4d = torch.randn(batch_size, 1, num_rois, num_rois)
    logits = model(x_4d)
    assert logits.shape == (batch_size, 2)

    # Calling with return_embedding=True
    logits, z_brain = model(x_4d, return_embedding=True)
    assert logits.shape == (batch_size, 2)
    assert z_brain.shape == (batch_size, 128)

    # Verify L2 normalization: norm of each embedding vector must be approx 1.0
    norms = torch.norm(z_brain, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # 3D input: (B, 200, 200) auto-unsqueeze channel dimension
    x_3d = torch.randn(batch_size, num_rois, num_rois)
    logits_3d = model(x_3d)
    assert logits_3d.shape == (batch_size, 2)


def test_brainnetcnn_gradient_flow():
    """Verify non-zero gradients flow through all components during backward pass."""
    batch_size = 2
    num_rois = 200

    model = BrainNetCNN(num_rois=num_rois, in_channels=1, embed_dim=128, num_classes=2)
    x = torch.randn(batch_size, 1, num_rois, num_rois, requires_grad=True)
    logits, z_brain = model(x, return_embedding=True)

    loss = logits.sum() + z_brain.sum()
    loss.backward()

    # Check gradients on input
    assert x.grad is not None
    assert torch.any(x.grad != 0.0)

    # Check gradients on model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients."
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients."
