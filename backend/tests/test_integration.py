"""End-to-end integration smoke tests for ABIDE I NYU 3D-CNN fMRI classification.

Verifies the integration contract between Notebook 1's preprocessing artifacts
and Notebook 2's dataset reader, executes mock/smoke training for 1 epoch on
CPU and GPU without numerical errors or memory leaks, validates Table 8
benchmark comparison utilities, and verifies notebook interface consistency.
"""

import gc
import json
from pathlib import Path
from typing import Dict, Generator, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asd_classification.data.loader import MultiViewDataset
from asd_classification.models.baseline.augmentation import VolumeConsistentAugmenter
from asd_classification.models.baseline.cnn3d import Baseline3DCNN
from asd_classification.models.baseline.focal_loss import FocalLoss
from asd_classification.pipelines.evaluate import (
    PUBLISHED_TABLE_8_BENCHMARKS,
    compute_classification_metrics,
    format_benchmark_comparison,
    plot_roc_curve,
)
from asd_classification.pipelines.train import (
    EarlyStopping,
    evaluate_epoch,
    train_3dcnn_model,
    train_epoch,
)
from asd_classification.preprocessing.multiview import (
    assemble_multiview_tensor,
    compute_temporal_mean,
    export_dataset_manifest,
    extract_orthogonal_slices,
    load_multiview_tensor,
    save_multiview_tensor,
    trim_temporal_frames,
    z_score_normalize,
)


@pytest.fixture
def mock_kaggle_environment(tmp_path: Path) -> Generator[Dict[str, Path], None, None]:
    """Create a temporary directory structure mimicking Kaggle dataset input/output paths.

    Produces:
    - metadata.csv with 8 training subjects (4 ASD, 4 TC) and 4 test subjects (2 ASD, 2 TC)
    - dataset-metadata.json formatted for Kaggle Dataset publishing
    - tensors/ directory with compressed .npz multi-view tensors of shape (1, 16, 16, 16)
      for fast, lightweight test execution across all convolutional and pooling layers.
    """
    dataset_dir = tmp_path / "abide-i-nyu-multiview-fmri"
    tensors_dir = dataset_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # 12 mock subjects: 6 ASD (dx_group=1), 6 TC (dx_group=0)
    # Train: 8 subjects (4 ASD, 4 TC); Test: 4 subjects (2 ASD, 2 TC)
    for idx in range(12):
        sub_id = f"500{idx:02d}"
        file_id = f"NYU_00{sub_id}"
        dx_group = 1 if idx % 2 == 0 else 0
        split = "train" if idx < 8 else "test"
        rel_tensor_path = f"tensors/{sub_id}.npz"

        # Generate synthetic multi-view tensor
        # Compact shape (1, 16, 16, 16) allows 4 successive MaxPool3d(2) operations
        tensor_data = np.random.randn(1, 16, 16, 16).astype(np.float32)
        save_multiview_tensor(tensor_data, tensors_dir / f"{sub_id}.npz")

        records.append({
            "subject_id": sub_id,
            "file_id": file_id,
            "dx_group": dx_group,
            "split": split,
            "age": 10.0 + idx * 0.5,
            "sex": 1 if idx % 3 == 0 else 2,
            "fiq": 100.0 + idx * 2.0,
            "tensor_path": rel_tensor_path,
        })

    manifest_df = pd.DataFrame(records)
    csv_path, json_path = export_dataset_manifest(
        metadata_df=manifest_df,
        output_dir=dataset_dir,
        dataset_title="ABIDE I NYU Multi-View fMRI 3D-CNN Dataset",
        dataset_slug="abide-i-nyu-multiview-fmri",
    )

    env = {
        "dataset_dir": dataset_dir,
        "tensors_dir": tensors_dir,
        "metadata_csv": csv_path,
        "metadata_json": json_path,
        "working_dir": tmp_path / "working",
    }
    env["working_dir"].mkdir(parents=True, exist_ok=True)

    yield env

    # Teardown: ensure matplotlib figures and memory are cleaned up
    plt.close("all")
    gc.collect()


# ---------------------------------------------------------------------------
# 1. Dataset Reader Contract Tests (Notebook 1 -> Notebook 2)
# ---------------------------------------------------------------------------

def test_notebook1_to_notebook2_artifact_contract(mock_kaggle_environment: Dict[str, Path]):
    """Verify that Notebook 1's exported metadata.csv and tensors match Notebook 2's dataset reader."""
    metadata_csv = mock_kaggle_environment["metadata_csv"]
    tensors_dir = mock_kaggle_environment["tensors_dir"]
    metadata_json = mock_kaggle_environment["metadata_json"]

    # 1. Verify JSON metadata descriptor adheres to Kaggle specification
    assert metadata_json.is_file()
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == "antigravity/abide-i-nyu-multiview-fmri"
    assert "ABIDE I NYU Multi-View" in meta["title"]

    # 2. Verify metadata.csv contains required columns and correct split assignments
    df = pd.read_csv(metadata_csv)
    expected_cols = {"subject_id", "file_id", "dx_group", "split", "age", "sex", "fiq", "tensor_path"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 12
    assert (df["split"] == "train").sum() == 8
    assert (df["split"] == "test").sum() == 4

    # 3. Instantiate Notebook 2's MultiViewDataset directly on the exported directory
    # Test loading full dataset
    dataset_all = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split=None)
    assert len(dataset_all) == 12

    # Test loading train split
    dataset_train = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="train")
    assert len(dataset_train) == 8

    # Test loading test split
    dataset_test = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="test")
    assert len(dataset_test) == 4

    # 4. Verify item extraction contract
    tensor, label = dataset_train[0]
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert tensor.ndim == 4  # (C, D, H, W)
    assert tensor.shape == (1, 16, 16, 16)
    assert tensor.dtype == torch.float32
    assert label.dtype == torch.long
    assert label.item() in (0, 1)

    # 5. Verify relative path resolution without explicit tensors_dir
    # MultiViewDataset should resolve tensor_path relative to metadata_csv parent directory
    dataset_auto = MultiViewDataset(manifest=metadata_csv, split="train")
    auto_tensor, _ = dataset_auto[0]
    assert auto_tensor.shape == (1, 16, 16, 16)


def test_multiview_standard_resolution_contract(tmp_path: Path):
    """Verify standard publication resolution (1, 150, 224, 224) loads and indexes properly."""
    sub_id = "50001"
    tensor_path = tmp_path / "tensors" / f"{sub_id}.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)

    # Standard paper resolution: 1 channel, 150 depth slices (50x3 views), 224x224 pixels
    standard_shape = (1, 150, 224, 224)
    standard_data = np.zeros(standard_shape, dtype=np.float32)
    save_multiview_tensor(standard_data, tensor_path)

    manifest_df = pd.DataFrame([{
        "subject_id": sub_id,
        "file_id": f"NYU_00{sub_id}",
        "dx_group": 1,
        "split": "train",
        "tensor_path": f"tensors/{sub_id}.npz",
    }])
    csv_path = tmp_path / "metadata.csv"
    manifest_df.to_csv(csv_path, index=False)

    dataset = MultiViewDataset(manifest=csv_path, tensors_dir=tmp_path / "tensors")
    loaded_tensor, loaded_label = dataset[0]

    assert loaded_tensor.shape == standard_shape
    assert loaded_tensor.dtype == torch.float32
    assert loaded_label.item() == 1


# ---------------------------------------------------------------------------
# 2. Mock / Smoke Training & Evaluation Execution Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_smoke_training_epoch_and_evaluation(mock_kaggle_environment: Dict[str, Path], target_device: str):
    """Execute 1 full training epoch and evaluation step on CPU/GPU verifying gradients and memory."""
    metadata_csv = mock_kaggle_environment["metadata_csv"]
    tensors_dir = mock_kaggle_environment["tensors_dir"]

    # 1. Initialize data loaders matching Table 4 parameters
    # Physical batch size 2, gradient accumulation 4 steps -> effective batch size 8
    augmenter = VolumeConsistentAugmenter(
        max_rotation_angle=10.0,
        p_flip_horizontal=0.5,
        p_flip_vertical=0.5,
        intensity_scale_range=(0.9, 1.1),
    )

    train_ds = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="train", transform=augmenter)
    test_ds = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="test", transform=None)

    # 8 train samples / batch size 2 = 4 batches (exactly 1 accumulation cycle)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    # 2. Instantiate model, loss, optimizer
    model = Baseline3DCNN(in_channels=1, num_classes=2).to(target_device)
    initial_weights = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda") if (target_device == "cuda") else None

    # 3. Execute 1 full training epoch
    train_loss = train_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        accumulation_steps=4,
        device=target_device,
        augmenter=augmenter,
    )

    # Verify training loss invariants
    assert isinstance(train_loss, float)
    assert not np.isnan(train_loss), "Training loss returned NaN"
    assert not np.isinf(train_loss), "Training loss returned Inf"
    assert train_loss > 0.0, "Focal loss must be strictly positive"

    # Verify weight updates occurred
    updated_weights = [p.detach() for p in model.parameters() if p.requires_grad]
    weight_changes = [
        torch.norm(u - i).item() for u, i in zip(updated_weights, initial_weights)
    ]
    assert any(change > 0.0 for change in weight_changes), "Model weights did not update after training epoch"

    # 4. Execute 1 evaluation epoch
    val_loss, metrics, y_true, y_pred, y_prob = evaluate_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=target_device,
    )

    # Verify evaluation metrics
    assert not np.isnan(val_loss)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert len(y_true) == 4
    assert len(y_pred) == 4
    assert y_prob.shape == (4, 2)


def test_training_memory_safety_and_no_leak(mock_kaggle_environment: Dict[str, Path]):
    """Verify that multiple successive training steps do not accumulate residual memory or graph references."""
    metadata_csv = mock_kaggle_environment["metadata_csv"]
    tensors_dir = mock_kaggle_environment["tensors_dir"]

    train_ds = MultiViewDataset(manifest=metadata_csv, tensors_dir=tensors_dir, split="train")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    model = Baseline3DCNN(in_channels=1, num_classes=2)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Warm-up run for 1 step to initialize Adam optimizer internal momentum state tensors
    train_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=None,
        accumulation_steps=4,
        device="cpu",
    )

    gc.collect()
    initial_tensor_count = len([
        obj for obj in gc.get_objects()
        if type(obj) is torch.Tensor
    ])

    # Execute 3 subsequent epochs
    for _ in range(3):
        loss_val = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=None,
            accumulation_steps=4,
            device="cpu",
        )
        assert not np.isnan(loss_val)

    gc.collect()
    final_tensor_count = len([
        obj for obj in gc.get_objects()
        if type(obj) is torch.Tensor
    ])

    # After warm-up, persistent tensor count should remain steady with zero leakage
    tensor_diff = final_tensor_count - initial_tensor_count
    assert tensor_diff <= 2, f"Potential tensor memory leak detected: {tensor_diff} new tensors retained"


# ---------------------------------------------------------------------------
# 3. End-to-End Pipeline Smoke Test (Preprocessing -> Model -> Benchmark)
# ---------------------------------------------------------------------------

def test_full_pipeline_end_to_end_smoke(tmp_path: Path):
    """Smoke test executing full workflow from synthetic 4D scan to evaluation report and ROC curve."""
    from asd_classification.preprocessing.multiview import extract_multiview_tensor

    # Step A: Synthesize a raw 4D fMRI volume (X, Y, Z, T)
    raw_shape = (25, 25, 25, 30)  # 30 temporal timepoints
    raw_4d = np.random.uniform(50.0, 150.0, size=raw_shape).astype(np.float32)

    # Step B through F: Extract multi-view tensor using integrated extraction function
    # n_slices=6 extracts 6 slices along 3 planes = 18 depth slices (>= 16 for 4 MaxPool3d layers)
    multiview_tensor = extract_multiview_tensor(
        fmri_input=raw_4d,
        n_slices=6,
        target_shape=(16, 16),
        initial_frames=2,
        terminal_frames=2,
    )
    assert multiview_tensor.shape == (1, 18, 16, 16)

    # Step G: Save tensors and export manifest
    tensors_dir = tmp_path / "dataset" / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(4):
        sid = f"5010{i}"
        tensor_file = tensors_dir / f"{sid}.npz"
        # Scale slightly to provide variation
        sub_tensor = multiview_tensor * (1.0 + i * 0.1)
        save_multiview_tensor(sub_tensor, tensor_file)
        records.append({
            "subject_id": sid,
            "file_id": f"NYU_00{sid}",
            "dx_group": 1 if i % 2 == 0 else 0,
            "split": "train" if i < 2 else "test",
            "age": 12.0,
            "sex": 1,
            "fiq": 110.0,
            "tensor_path": f"tensors/{sid}.npz",
        })

    manifest_df = pd.DataFrame(records)
    csv_path, json_path = export_dataset_manifest(
        metadata_df=manifest_df,
        output_dir=tmp_path / "dataset",
        dataset_title="ABIDE I NYU Multi-View Smoke Dataset",
        dataset_slug="abide-nyu-smoke",
    )

    # Step H: Load into Dataset and run 1 epoch training with EarlyStopping
    train_ds = MultiViewDataset(manifest=csv_path, split="train")
    test_ds = MultiViewDataset(manifest=csv_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    model = Baseline3DCNN(in_channels=1, num_classes=2)
    checkpoint_file = tmp_path / "working" / "best_model.pt"

    train_results = train_3dcnn_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=1,
        accumulation_steps=1,
        early_stopping_patience=2,
        checkpoint_path=checkpoint_file,
        device="cpu",
    )

    assert "model" in train_results
    assert len(train_results["history"]["train_loss"]) == 1

    # Step I: Evaluate and export ROC curve
    _, eval_metrics, y_true, y_pred, y_prob = evaluate_epoch(
        model=model,
        dataloader=test_loader,
        criterion=FocalLoss(),
        device="cpu",
    )

    roc_plot_file = tmp_path / "working" / "roc_curve.png"
    auc_score, saved_plot = plot_roc_curve(y_true, y_prob, output_path=roc_plot_file)
    assert saved_plot.is_file()
    assert roc_plot_file.stat().st_size > 0

    # Step J: Format benchmark comparison table against Table 8
    comparison_df = format_benchmark_comparison(eval_metrics)
    assert isinstance(comparison_df, pd.DataFrame)
    assert "Published Baseline (Table 8)" in comparison_df.columns
    assert "Delta" in comparison_df.columns


# ---------------------------------------------------------------------------
# 4. Table 8 Benchmark Verification Checklist Contract Tests
# ---------------------------------------------------------------------------

def test_published_table_8_metrics_benchmark_verification():
    """Verify published Table 8 metrics dictionary and comparison formatting adhere to paper."""
    # Check baseline constants from Hammash & Younis (2026), Table 8
    assert PUBLISHED_TABLE_8_BENCHMARKS["Accuracy"] == 0.8124
    assert PUBLISHED_TABLE_8_BENCHMARKS["Precision"] == 0.8023
    assert PUBLISHED_TABLE_8_BENCHMARKS["Recall"] == 0.7905
    assert PUBLISHED_TABLE_8_BENCHMARKS["F1-Score"] == 0.7963
    assert PUBLISHED_TABLE_8_BENCHMARKS["Focal Loss"] == 0.4932

    # Verify formatting comparison logic
    mock_metrics = {
        "accuracy": 0.8124,
        "precision": 0.8023,
        "recall": 0.7905,
        "f1_score": 0.7963,
        "loss": 0.4932,
    }
    comparison_df = format_benchmark_comparison(mock_metrics)

    for _, row in comparison_df.iterrows():
        metric_name = row["Metric"]
        current = row["Current Result"]
        baseline = row["Published Baseline (Table 8)"]
        delta = row["Delta"]

        assert current == baseline
        if metric_name == "Focal Loss":
            assert delta == "+0.0000"
        else:
            assert delta == "+0.00%"


def test_notebooks_interface_and_slug_consistency():
    """Verify that Notebook 1 and Notebook 2 maintain mutually consistent paths and configurations."""
    nb1_path = Path("notebooks/01_fmri_3dcnn_preprocessing.ipynb")
    nb2_path = Path("notebooks/02_fmri_3dcnn_training.ipynb")

    if not nb1_path.is_file():
        nb1_path = Path("../notebooks/01_fmri_3dcnn_preprocessing.ipynb")
        nb2_path = Path("../notebooks/02_fmri_3dcnn_training.ipynb")

    with open(nb1_path, "r", encoding="utf-8") as f:
        nb1_data = json.load(f)
    with open(nb2_path, "r", encoding="utf-8") as f:
        nb2_data = json.load(f)

    nb1_text = " ".join("".join(c.get("source", [])) for c in nb1_data["cells"])
    nb2_text = " ".join("".join(c.get("source", [])) for c in nb2_data["cells"])

    # Verify shared Kaggle dataset slug
    shared_slug = "abide-i-nyu-multiview-fmri"
    assert shared_slug in nb1_text
    assert shared_slug in nb2_text

    # Verify shared input path
    dataset_input_path = f"/kaggle/input/{shared_slug}"
    assert dataset_input_path in nb1_text or shared_slug in nb1_text
    assert dataset_input_path in nb2_text

    # Verify identical split sizes
    assert "64" in nb1_text and "84" in nb1_text  # 148 train
    assert "15" in nb1_text and "21" in nb1_text  # 36 test
    assert "15" in nb2_text and "21" in nb2_text

    # Verify identical tensor shape specification (1, 150, 224, 224)
    assert "(1, 150, 224, 224)" in nb1_text
    assert "(1, 150, 224, 224)" in nb2_text
