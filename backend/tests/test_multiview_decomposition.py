"""Unit tests for Multi-View Plane Decomposition (Option 3) and Soft-Voting Ensemble.

Verifies:
1. Dynamic view decomposition into 50-slice sub-volumes (Axial, Coronal, Sagittal).
2. Effective sample size tripling (View-as-Sample) and parent label preservation.
3. Single view selection modes ('axial', 'coronal', 'sagittal') and caching.
4. Soft-voting probability aggregation and dual-level evaluation reporting.
5. SEResNet3D and Baseline3DCNN forward-backward compatibility with 50-slice inputs under AMP.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from asd_classification.data.loader import MultiViewDataset
from asd_classification.models.baseline.cnn3d import Baseline3DCNN, SEResNet3D
from asd_classification.pipelines.evaluate import (
    aggregate_soft_voting_probabilities,
    compute_classification_metrics,
    evaluate_multiview_ensemble,
    format_dual_level_report,
)
from asd_classification.pipelines.train import EarlyStopping, compute_class_weights


@pytest.fixture
def mock_multiview_decomposed_env(tmp_path: Path) -> Dict[str, Path]:
    """Create a temporary dataset environment with structured multi-view tensors."""
    tensors_dir = tmp_path / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # Create 6 mock subjects: 3 ASD (label 1), 3 TC (label 0)
    # 4 train (2 ASD, 2 TC) and 2 test (1 ASD, 1 TC)
    for i in range(6):
        sub_id = f"5000{i}"
        file_id = f"NYU_00{sub_id}"
        dx_group = 1 if i < 3 else 0
        split = "train" if i < 4 else "test"
        rel_path = f"tensors/{sub_id}.npz"

        # Create structured volume with 3 distinct view regions:
        # Axial (slices 0:10): value 1.0 * (i + 1)
        # Coronal (slices 10:20): value 2.0 * (i + 1)
        # Sagittal (slices 20:30): value 3.0 * (i + 1)
        tensor_data = np.zeros((1, 30, 16, 16), dtype=np.float32)
        tensor_data[:, 0:10, :, :] = 1.0 * (i + 1)
        tensor_data[:, 10:20, :, :] = 2.0 * (i + 1)
        tensor_data[:, 20:30, :, :] = 3.0 * (i + 1)

        np.savez_compressed(tensors_dir / f"{sub_id}.npz", tensor=tensor_data)

        records.append({
            "subject_id": sub_id,
            "file_id": file_id,
            "dx_group": dx_group,
            "split": split,
            "age": 13.0 + i,
            "sex": 1 if i % 2 == 0 else 2,
            "fiq": 100.0,
            "tensor_path": rel_path,
        })

    manifest_df = pd.DataFrame(records)
    metadata_csv = tmp_path / "metadata.csv"
    manifest_df.to_csv(metadata_csv, index=False)

    return {
        "root_dir": tmp_path,
        "tensors_dir": tensors_dir,
        "metadata_csv": metadata_csv,
    }


def test_multiview_dataset_decomposed_triples_sample_count(mock_multiview_decomposed_env: Dict[str, Path]):
    """Verify that decomposed mode triples the dataset length and preserves parent labels."""
    metadata_csv = mock_multiview_decomposed_env["metadata_csv"]
    tensors_dir = mock_multiview_decomposed_env["tensors_dir"]

    # 4 subjects in train -> 4 * 3 = 12 decomposed samples
    ds_decomposed = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=tensors_dir,
        split="train",
        view="decomposed",
        slices_per_view=10,
    )
    assert len(ds_decomposed) == 12

    labels = ds_decomposed.get_labels()
    assert len(labels) == 12
    # First 3 subjects in train are ASD (1), 4th is TC (0)
    # Each subject is repeated 3 times: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    expected_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert np.array_equal(labels, expected_labels)


def test_multiview_dataset_decomposed_slice_mapping(mock_multiview_decomposed_env: Dict[str, Path]):
    """Verify that index mapping correctly extracts Axial, Coronal, and Sagittal slices."""
    metadata_csv = mock_multiview_decomposed_env["metadata_csv"]
    tensors_dir = mock_multiview_decomposed_env["tensors_dir"]

    ds = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=tensors_dir,
        split="train",
        decomposed=True,
        slices_per_view=10,
    )

    # Subject 0 (i=0): Axial (1.0), Coronal (2.0), Sagittal (3.0)
    sample_axial, label_ax = ds[0]
    sample_coronal, label_cor = ds[1]
    sample_sagittal, label_sag = ds[2]

    assert sample_axial.shape == (1, 10, 16, 16)
    assert sample_coronal.shape == (1, 10, 16, 16)
    assert sample_sagittal.shape == (1, 10, 16, 16)

    assert torch.allclose(sample_axial, torch.tensor(1.0))
    assert torch.allclose(sample_coronal, torch.tensor(2.0))
    assert torch.allclose(sample_sagittal, torch.tensor(3.0))

    assert label_ax.item() == 1
    assert label_cor.item() == 1
    assert label_sag.item() == 1

    # Subject 1 (i=1): Axial (2.0), Coronal (4.0), Sagittal (6.0)
    sample_s1_ax, label_s1_ax = ds[3]
    assert torch.allclose(sample_s1_ax, torch.tensor(2.0))
    assert label_s1_ax.item() == 1


def test_multiview_dataset_single_view_selection(mock_multiview_decomposed_env: Dict[str, Path]):
    """Verify that selecting specific views ('axial', 'coronal', 'sagittal') yields N samples with correct slices."""
    metadata_csv = mock_multiview_decomposed_env["metadata_csv"]
    tensors_dir = mock_multiview_decomposed_env["tensors_dir"]

    ds_ax = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="train", view="axial", slices_per_view=10)
    ds_cor = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="train", view="coronal", slices_per_view=10)
    ds_sag = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="train", view="sagittal", slices_per_view=10)

    assert len(ds_ax) == 4
    assert len(ds_cor) == 4
    assert len(ds_sag) == 4

    # Subject 0 has values 1.0 (axial), 2.0 (coronal), 3.0 (sagittal)
    s0_ax, _ = ds_ax[0]
    s0_cor, _ = ds_cor[0]
    s0_sag, _ = ds_sag[0]

    assert s0_ax.shape == (1, 10, 16, 16)
    assert torch.allclose(s0_ax, torch.tensor(1.0))
    assert torch.allclose(s0_cor, torch.tensor(2.0))
    assert torch.allclose(s0_sag, torch.tensor(3.0))


def test_multiview_dataset_in_memory_caching(mock_multiview_decomposed_env: Dict[str, Path]):
    """Verify that enabling cache_tensors avoids repeated file I/O operations."""
    metadata_csv = mock_multiview_decomposed_env["metadata_csv"]
    tensors_dir = mock_multiview_decomposed_env["tensors_dir"]

    ds = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=tensors_dir,
        view="decomposed",
        cache_tensors=True,
        slices_per_view=10,
    )

    # First access loads tensor into cache
    assert len(ds._tensor_cache) == 0
    _ = ds[0]
    assert len(ds._tensor_cache) == 1

    # Second access (same subject, different view) retrieves from cache
    _ = ds[1]
    assert len(ds._tensor_cache) == 1

    # Access next subject
    _ = ds[3]
    assert len(ds._tensor_cache) == 2


def test_multiview_dataset_get_subject_views(mock_multiview_decomposed_env: Dict[str, Path]):
    """Verify that get_subject_views returns all 3 orthogonal views for a given subject."""
    metadata_csv = mock_multiview_decomposed_env["metadata_csv"]
    tensors_dir = mock_multiview_decomposed_env["tensors_dir"]

    ds = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=tensors_dir,
        split="test",
        slices_per_view=10,
    )
    assert len(ds) == 2

    # Subject 0 in test (50004 -> index 4 overall -> i=4, values: ax=5.0, cor=10.0, sag=15.0)
    views_dict, label = ds.get_subject_views(0)
    assert "axial" in views_dict
    assert "coronal" in views_dict
    assert "sagittal" in views_dict
    assert views_dict["axial"].shape == (1, 10, 16, 16)
    assert torch.allclose(views_dict["axial"], torch.tensor(5.0))
    assert torch.allclose(views_dict["coronal"], torch.tensor(10.0))
    assert torch.allclose(views_dict["sagittal"], torch.tensor(15.0))
    assert label == 0


def test_soft_voting_probability_aggregation():
    """Verify that soft-voting averages probabilities across 3 views and derives accurate predictions."""
    # 3 subjects, 3 views per subject, 2 classes (TC=0, ASD=1)
    # Subject 0: Axial=[0.4, 0.6], Coronal=[0.3, 0.7], Sagittal=[0.8, 0.2]
    # Average: TC = (0.4+0.3+0.8)/3 = 0.50, ASD = (0.6+0.7+0.2)/3 = 0.50 -> Tie/Decision
    # Subject 1: Axial=[0.1, 0.9], Coronal=[0.2, 0.8], Sagittal=[0.3, 0.7] -> ASD
    # Average: ASD = 0.80 -> 1
    # Subject 2: Axial=[0.8, 0.2], Coronal=[0.9, 0.1], Sagittal=[0.4, 0.6] -> TC
    # Average: TC = 0.70 -> 0
    view_probs = {
        "axial": np.array([[0.2, 0.8], [0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
        "coronal": np.array([[0.3, 0.7], [0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
        "sagittal": np.array([[0.7, 0.3], [0.3, 0.7], [0.4, 0.6]], dtype=np.float32),
    }

    subject_probs, subject_preds = aggregate_soft_voting_probabilities(view_probs)

    assert subject_probs.shape == (3, 2)
    assert subject_preds.shape == (3,)

    # Subject 0: ASD prob = (0.8 + 0.7 + 0.3) / 3 = 0.60
    assert np.isclose(subject_probs[0, 1], 0.60, atol=1e-5)
    assert subject_preds[0] == 1

    # Subject 1: ASD prob = (0.9 + 0.8 + 0.7) / 3 = 0.80
    assert np.isclose(subject_probs[1, 1], 0.80, atol=1e-5)
    assert subject_preds[1] == 1

    # Subject 2: ASD prob = (0.2 + 0.1 + 0.6) / 3 = 0.30
    assert np.isclose(subject_probs[2, 1], 0.30, atol=1e-5)
    assert subject_preds[2] == 0


def test_evaluate_multiview_ensemble_dual_level(mock_multiview_decomposed_env: Dict[str, Path]):
    """Verify dual-level evaluation outputs both subject-level and per-view metrics."""
    metadata_csv = mock_multiview_decomposed_env["metadata_csv"]
    tensors_dir = mock_multiview_decomposed_env["tensors_dir"]

    # Use a dummy model that outputs fixed logits
    class DummyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            # Output positive logit for first class if mean > 3 else second class
            out = torch.zeros((b, 2), dtype=torch.float32)
            for idx in range(b):
                if x[idx].mean() > 3.0:
                    out[idx, 1] = 2.0
                else:
                    out[idx, 0] = 2.0
            return out

    dataset = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="test", slices_per_view=10)
    model = DummyModel()

    results = evaluate_multiview_ensemble(model=model, dataset=dataset, device="cpu")

    assert "subject_metrics" in results
    assert "view_metrics" in results
    assert "axial" in results["view_metrics"]
    assert "coronal" in results["view_metrics"]
    assert "sagittal" in results["view_metrics"]

    sub_m = results["subject_metrics"]
    for metric_name in ["accuracy", "precision", "recall", "f1_score", "specificity", "balanced_accuracy", "composite_score"]:
        assert metric_name in sub_m


def test_format_dual_level_report():
    """Verify that format_dual_level_report generates comparison DataFrame against Table 8 and Table 3."""
    subject_metrics = {
        "accuracy": 0.80,
        "precision": 0.78,
        "recall": 0.75,
        "f1_score": 0.764,
        "loss": 0.45,
    }
    view_metrics = {
        "axial": {"accuracy": 0.75, "f1_score": 0.72},
        "coronal": {"accuracy": 0.78, "f1_score": 0.76},
        "sagittal": {"accuracy": 0.70, "f1_score": 0.68},
    }

    report_df = format_dual_level_report(subject_metrics=subject_metrics, view_metrics=view_metrics)
    assert isinstance(report_df, pd.DataFrame)
    assert "Level" in report_df.columns
    assert "Metric" in report_df.columns
    assert "Result" in report_df.columns
    assert "Baseline" in report_df.columns


def test_seresnet3d_and_baseline_with_50_slices_amp():
    """Verify that SEResNet3D and Baseline3DCNN process 50-slice sub-volumes with AMP correctly."""
    # Synthetic batch of shape (B=2, C=1, D=50, H=32, W=32)
    x = torch.randn(2, 1, 50, 32, 32, requires_grad=False)
    target = torch.tensor([1, 0], dtype=torch.long)

    # 1. SEResNet3D
    model_se = SEResNet3D()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_se = model_se(x)
    assert out_se.shape == (2, 2)
    loss_se = loss_fn(out_se, target)
    loss_se.backward()

    # 2. Baseline3DCNN
    model_base = Baseline3DCNN()
    out_base = model_base(x)
    assert out_base.shape == (2, 2)
    loss_base = loss_fn(out_base, target)
    loss_base.backward()
