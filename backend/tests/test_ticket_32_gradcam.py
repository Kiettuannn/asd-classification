"""
Unit tests for Ticket #32: BrainNetCNN Grad-CAM ROI Attribution on E2N Layer.

Verifies:
- Grad-CAM hooks correctly into E2N layer of BrainNetCNN and DualModalBrainNetModel.
- Computes alpha_k channel weights and ReLU-gated ROI saliency S_i = ReLU(sum_k alpha_k A_{k, i}).
- Output saliency weights are strictly non-negative (S_i >= 0).
- Output shapes are preserved: (200,) for single input and (B, 200) for batch inputs.
- Clean hook registration and unregistration to prevent memory leaks.
"""

import numpy as np
import pytest
import torch

from asd_classification.models.brainnetcnn import BrainNetCNN
from asd_classification.models.dual_modal_engine import DualModalBrainNetModel
from asd_classification.models.gradcam import BrainNetGradCAM


@pytest.fixture
def mock_brainnet_model():
    """Create a deterministic BrainNetCNN model for Grad-CAM tests."""
    torch.manual_seed(42)
    model = BrainNetCNN(num_rois=200, in_channels=1, num_classes=2)
    model.eval()
    return model


@pytest.fixture
def mock_dual_modal_model():
    """Create a deterministic DualModalBrainNetModel for Grad-CAM tests."""
    torch.manual_seed(42)
    model = DualModalBrainNetModel(
        num_rois=200,
        in_channels=1,
        num_classes=2,
        use_mock_text_backbone=True,
        epochs=1,
    )
    return model


def test_gradcam_e2n_hook_and_shapes(mock_brainnet_model):
    """Verify Grad-CAM extracts non-negative saliency weights of shape (200,) for a single FC matrix."""
    explainer = BrainNetGradCAM(mock_brainnet_model, target_layer="e2n")

    fc_matrix = torch.randn(1, 1, 200, 200)

    # Attribute for class ASD (class index 1)
    saliency = explainer.attribute(fc_matrix, target_class=1)

    assert isinstance(saliency, np.ndarray)
    assert saliency.shape == (200,)
    # Acceptance criterion: Saliency weights must be non-negative
    assert np.all(saliency >= 0.0)

    # Clean up hooks
    explainer.remove_hooks()


def test_gradcam_batch_attribution(mock_brainnet_model):
    """Verify Grad-CAM correctly computes batch attributions of shape (B, 200)."""
    with BrainNetGradCAM(mock_brainnet_model, target_layer="e2n") as explainer:
        batch_fc = torch.randn(4, 1, 200, 200)
        batch_saliency = explainer.attribute(batch_fc, target_class=1)

        assert batch_saliency.shape == (4, 200)
        assert np.all(batch_saliency >= 0.0)


def test_gradcam_numpy_and_2d_input(mock_brainnet_model):
    """Verify Grad-CAM handles 2D numpy arrays (200, 200) directly."""
    np.random.seed(42)
    fc_np = np.random.randn(200, 200).astype(np.float32)
    # Symmetrize
    fc_np = (fc_np + fc_np.T) / 2.0

    with BrainNetGradCAM(mock_brainnet_model) as explainer:
        saliency = explainer.attribute(fc_np, target_class=0)
        assert saliency.shape == (200,)
        assert np.all(saliency >= 0.0)


def test_gradcam_dual_modal_engine_support(mock_dual_modal_model):
    """Verify Grad-CAM operates directly on DualModalBrainNetModel wrappers."""
    with BrainNetGradCAM(mock_dual_modal_model) as explainer:
        fc_np = np.random.randn(1, 200, 200).astype(np.float32)
        saliency = explainer.attribute(fc_np, target_class=1)
        assert saliency.shape == (200,)
        assert np.all(saliency >= 0.0)
