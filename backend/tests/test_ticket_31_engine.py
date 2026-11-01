"""
Unit tests for DualModalBrainNetModel engine and unimodal inference (Seam 3).
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from asd_classification.models.dual_modal_engine import DualModalBrainNetModel


class SyntheticDualModalDataset(torch.utils.data.Dataset):
    """Synthetic dataset generating paired connectome matrices, clinical texts, and labels."""

    def __init__(self, n_samples: int = 8, num_rois: int = 200) -> None:
        self.n_samples = n_samples
        self.fc_matrices = torch.randn(n_samples, 1, num_rois, num_rois)
        self.prompts = [
            f"A {10 + i}.0-year-old male, Full-scale IQ {100 + i}, evaluated on ADOS."
            for i in range(n_samples)
        ]
        self.labels = torch.tensor([i % 2 for i in range(n_samples)], dtype=torch.long)
        self.site_codes = torch.zeros(n_samples, dtype=torch.long)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.fc_matrices[idx], self.prompts[idx], self.labels[idx], self.site_codes[idx]


def test_dual_modal_engine_training_convergence():
    """Verify AdamW + CosineAnnealingLR multi-task training loop converges and reduces loss."""
    dataset = SyntheticDualModalDataset(n_samples=8, num_rois=200)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DualModalBrainNetModel(
        num_rois=200,
        use_mock_text_backbone=True,
        epochs=5,
        lr_brain=1e-2,
        lr_text=1e-3,
        use_amp=False,  # CPU execution for unit tests
    )

    history = model.fit(train_loader)

    assert "loss" in history
    assert len(history["loss"]) == 5
    # Loss at end should be lower than initial loss
    assert history["loss"][-1] < history["loss"][0]


def test_strict_unimodal_inference_parity():
    """Verify evaluation and prediction functions require only fMRI matrix and classify without text."""
    n_samples = 6
    num_rois = 200
    model = DualModalBrainNetModel(
        num_rois=num_rois,
        use_mock_text_backbone=True,
        epochs=1,
    )

    # Input as 3D numpy array (N, 200, 200)
    x_numpy_3d = np.random.randn(n_samples, num_rois, num_rois).astype(np.float32)

    # 1. Test predict_proba
    probs = model.predict_proba(x_numpy_3d)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (n_samples, 2)
    # Probabilities must sum to 1.0 along class dimension
    assert np.allclose(probs.sum(axis=1), np.ones(n_samples), atol=1e-5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # 2. Test predict
    preds = model.predict(x_numpy_3d)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (n_samples,)
    assert set(np.unique(preds)).issubset({0, 1})

    # 3. Input as 4D torch Tensor (N, 1, 200, 200)
    x_torch_4d = torch.randn(n_samples, 1, num_rois, num_rois)
    preds_torch = model.predict(x_torch_4d)
    assert preds_torch.shape == (n_samples,)

    # 4. Test unimodal evaluate(X, y)
    y_test = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    eval_metrics = model.evaluate(x_numpy_3d, y_test)
    assert "loss" in eval_metrics and "accuracy" in eval_metrics
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0
    assert eval_metrics["loss"] >= 0.0


def test_model_persistence_and_weight_restoration(tmp_path: Path):
    """Verify save and load preserve model weights and inference outputs identically."""
    num_rois = 200
    model = DualModalBrainNetModel(
        num_rois=num_rois,
        use_mock_text_backbone=True,
    )

    save_path = tmp_path / "brainnet_dual_modal.pt"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = DualModalBrainNetModel(
        num_rois=num_rois,
        use_mock_text_backbone=True,
    )
    loaded_model.load(save_path)

    test_input = np.random.randn(4, 1, num_rois, num_rois).astype(np.float32)
    probs_orig = model.predict_proba(test_input)
    probs_loaded = loaded_model.predict_proba(test_input)

    assert np.allclose(probs_orig, probs_loaded, atol=1e-5)
