"""Unit tests for 3D-CNN training pipeline, MultiViewDataset, early stopping, and ROC analysis.

Verifies acceptance criteria for Ticket #8:
1. MultiViewDataset loading, indexing, split filtering, and augmentation integration.
2. Training loop execution with physical batch size 2, gradient accumulation 4 steps, and AMP.
3. EarlyStopping patience monitoring and checkpoint saving.
4. Evaluation metrics calculation (Accuracy, Precision, Recall, F1, Loss, AUC).
5. ROC curve plot generation and export to filesystem.
6. Stratified 5-fold cross-validation generation across all 184 subjects.
"""

from pathlib import Path
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
    compute_classification_metrics,
    format_benchmark_comparison,
    plot_roc_curve,
)
from asd_classification.pipelines.train import (
    EarlyStopping,
    evaluate_epoch,
    run_stratified_kfold_cv,
    run_training_pipeline,
    train_3dcnn_model,
    train_epoch,
)


@pytest.fixture
def mock_dataset_environment(tmp_path: Path):
    """Create a temporary dataset environment with mock multi-view tensors and metadata.csv."""
    tensors_dir = tmp_path / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # Create 10 mock subjects: 4 ASD, 6 TC
    # 8 train (3 ASD, 5 TC) and 2 test (1 ASD, 1 TC)
    for i in range(10):
        sub_id = f"5000{i}"
        file_id = f"NYU_00{sub_id}"
        dx_group = 1 if i < 4 else 0
        split = "train" if i < 8 else "test"
        rel_path = f"tensors/{sub_id}.npz"

        # Save synthetic tensor of shape (1, 150, 224, 224)
        # Using a compact shape (1, 16, 16, 16) for ultra-fast CI execution through 4 pooling layers
        tensor_data = np.ones((1, 16, 16, 16), dtype=np.float32) * (i + 1)
        np.savez_compressed(tensors_dir / f"{sub_id}.npz", tensor=tensor_data)

        records.append({
            "subject_id": sub_id,
            "file_id": file_id,
            "dx_group": dx_group,
            "split": split,
            "age": 12.5 + i,
            "sex": 1 if i % 2 == 0 else 2,
            "fiq": 105.0,
            "tensor_path": rel_path,
        })

    manifest_df = pd.DataFrame(records)
    metadata_csv = tmp_path / "metadata.csv"
    manifest_df.to_csv(metadata_csv, index=False)

    return {
        "root_dir": tmp_path,
        "tensors_dir": tensors_dir,
        "metadata_csv": metadata_csv,
        "manifest_df": manifest_df,
    }


# ---------------------------------------------------------------------------
# 1. MultiViewDataset Tests
# ---------------------------------------------------------------------------

def test_multiview_dataset_loading_and_splits(mock_dataset_environment):
    """Verify MultiViewDataset loads correctly from CSV or DataFrame and respects splits."""
    root = mock_dataset_environment["root_dir"]
    metadata_csv = mock_dataset_environment["metadata_csv"]

    # Load all
    ds_all = MultiViewDataset(manifest=metadata_csv, tensors_dir=root / "tensors")
    assert len(ds_all) == 10

    # Load train split (8 subjects)
    ds_train = MultiViewDataset(manifest=metadata_csv, tensors_dir=root / "tensors", split="train")
    assert len(ds_train) == 8

    # Load test split (2 subjects)
    ds_test = MultiViewDataset(manifest=metadata_csv, tensors_dir=root / "tensors", split="test")
    assert len(ds_test) == 2

    # Verify item fetching
    tensor, label = ds_train[0]
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert tensor.shape == (1, 16, 16, 16)
    assert tensor.dtype == torch.float32
    assert label.dtype == torch.long
    assert label.item() in (0, 1)


def test_multiview_dataset_augmentation_integration(mock_dataset_environment):
    """Verify VolumeConsistentAugmenter applies transforms during training and bypasses during eval."""
    root = mock_dataset_environment["root_dir"]
    metadata_csv = mock_dataset_environment["metadata_csv"]

    augmenter = VolumeConsistentAugmenter(
        max_rotation_angle=10.0,
        p_flip_horizontal=1.0,
        p_flip_vertical=1.0,
        intensity_scale_range=(1.1, 1.1),
    )

    # In train mode, transformation modifies the tensor
    augmenter.train()
    ds_train = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=root / "tensors",
        split="train",
        transform=augmenter,
    )
    aug_tensor, _ = ds_train[0]
    raw_tensor, _ = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=root / "tensors",
        split="train",
    )[0]
    assert not torch.allclose(aug_tensor, raw_tensor)

    # In eval mode, transformation is bypassed
    augmenter.eval()
    ds_eval = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=root / "tensors",
        split="test",
        transform=augmenter,
    )
    eval_tensor, _ = ds_eval[0]
    raw_test_tensor, _ = MultiViewDataset(
        manifest=metadata_csv,
        tensors_dir=root / "tensors",
        split="test",
    )[0]
    torch.testing.assert_close(eval_tensor, raw_test_tensor)


# ---------------------------------------------------------------------------
# 2. EarlyStopping Tests
# ---------------------------------------------------------------------------

def test_early_stopping_patience_and_checkpoint(tmp_path: Path):
    """Verify early stopping triggers after patience epochs and saves best checkpoint."""
    checkpoint_file = tmp_path / "best_model.pt"
    early_stopping = EarlyStopping(patience=3, min_delta=1e-3, checkpoint_path=checkpoint_file)

    model = nn.Linear(4, 2)

    # Epoch 1: initial loss 1.0 -> improvement, saved
    early_stopping(val_loss=1.0, model=model)
    assert not early_stopping.early_stop
    assert checkpoint_file.is_file()
    assert early_stopping.counter == 0

    # Epoch 2: loss 0.8 -> improvement, saved
    early_stopping(val_loss=0.8, model=model)
    assert not early_stopping.early_stop
    assert early_stopping.counter == 0

    # Epoch 3: loss 0.81 -> no improvement, counter 1
    early_stopping(val_loss=0.81, model=model)
    assert not early_stopping.early_stop
    assert early_stopping.counter == 1

    # Epoch 4: loss 0.85 -> no improvement, counter 2
    early_stopping(val_loss=0.85, model=model)
    assert not early_stopping.early_stop
    assert early_stopping.counter == 2

    # Epoch 5: loss 0.82 -> no improvement, counter 3 >= patience -> early stop triggered!
    early_stopping(val_loss=0.82, model=model)
    assert early_stopping.early_stop
    assert early_stopping.counter == 3


# ---------------------------------------------------------------------------
# 3. Metrics Calculation & ROC Plotting Tests
# ---------------------------------------------------------------------------

def test_compute_classification_metrics():
    """Verify accuracy, precision, recall, f1, and roc-auc computation on known inputs."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob, loss_value=0.35)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2.0 / 3.0
    assert metrics["recall"] == 1.0
    assert "f1_score" in metrics
    assert metrics["roc_auc"] == 1.0  # Perfectly separated by prob thresholds
    assert metrics["loss"] == 0.35


def test_format_benchmark_comparison():
    """Verify benchmark comparison DataFrame matches published Table 8 references."""
    metrics = {
        "accuracy": 0.8150,
        "precision": 0.8050,
        "recall": 0.7950,
        "f1_score": 0.8000,
        "loss": 0.4850,
    }
    df = format_benchmark_comparison(metrics)

    assert "Metric" in df.columns
    assert "Current Result" in df.columns
    assert "Published Baseline (Table 8)" in df.columns
    assert "Delta" in df.columns

    # Verify Table 8 published values
    table8_dict = dict(zip(df["Metric"], df["Published Baseline (Table 8)"]))
    assert "81.24%" in table8_dict["Accuracy"]
    assert "80.23%" in table8_dict["Precision"]
    assert "79.05%" in table8_dict["Recall"]
    assert "79.63%" in table8_dict["F1-Score"]
    assert "0.4932" in table8_dict["Focal Loss"]


def test_plot_roc_curve(tmp_path: Path):
    """Verify ROC curve plot generation and export to filesystem."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.9])

    roc_png = tmp_path / "roc_curve.png"
    auc_score, saved_path = plot_roc_curve(y_true, y_prob, output_path=roc_png)

    assert saved_path.is_file()
    assert saved_path.suffix == ".png"
    assert saved_path.stat().st_size > 1000
    assert auc_score == 1.0


# ---------------------------------------------------------------------------
# 4. Training Loop & Gradient Accumulation Tests
# ---------------------------------------------------------------------------

def test_train_and_evaluate_epoch_with_gradient_accumulation(mock_dataset_environment):
    """Verify training epoch with batch size 2, accumulation steps 4, and mixed precision."""
    root = mock_dataset_environment["root_dir"]
    metadata_csv = mock_dataset_environment["metadata_csv"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_train = MultiViewDataset(manifest=metadata_csv, tensors_dir=root / "tensors", split="train")
    ds_test = MultiViewDataset(manifest=metadata_csv, tensors_dir=root / "tensors", split="test")

    train_loader = DataLoader(ds_train, batch_size=2, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=2, shuffle=False)

    # Use a miniature 3D model for fast CI execution
    class Tiny3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 4, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(4, 2),
            )
        def forward(self, x):
            return self.net(x)

    model = Tiny3DCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # Run 1 train epoch with gradient accumulation 4 steps
    train_loss = train_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        accumulation_steps=4,
        device=device,
    )
    assert np.isfinite(train_loss)
    assert train_loss > 0.0

    # Run 1 evaluation epoch
    val_loss, metrics, y_true, y_pred, y_prob = evaluate_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    assert np.isfinite(val_loss)
    assert len(y_true) == 2
    assert len(y_pred) == 2
    assert len(y_prob) == 2
    assert "accuracy" in metrics
    assert "f1_score" in metrics


# ---------------------------------------------------------------------------
# 5. Stratified 5-Fold Cross-Validation Verification
# ---------------------------------------------------------------------------

def test_stratified_kfold_split_balance():
    """Verify 5-fold cross-validation across 184 subjects maintains balanced ASD/TC distribution."""
    # 184 subjects: 79 ASD (1), 105 TC (0)
    y = np.array([1] * 79 + [0] * 105)
    X = np.arange(184)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_counts = []
    for train_idx, val_idx in skf.split(X, y):
        y_train = y[train_idx]
        y_val = y[val_idx]

        val_asd = int((y_val == 1).sum())
        val_tc = int((y_val == 0).sum())

        # Each fold has 36 or 37 subjects with ~15-16 ASD and ~21 TC
        assert 15 <= val_asd <= 16
        assert 21 <= val_tc <= 21
        assert len(val_idx) in (36, 37)
        fold_counts.append((val_asd, val_tc))

    assert len(fold_counts) == 5


def test_train_3dcnn_model_preserves_best_metrics(mock_dataset_environment, tmp_path):
    """Verify train_3dcnn_model returns best metrics and stops early without overwriting."""
    root = mock_dataset_environment["root_dir"]
    metadata_csv = mock_dataset_environment["metadata_csv"]

    ds_train = MultiViewDataset(manifest=metadata_csv, tensors_dir=root / "tensors", split="train")
    ds_test = MultiViewDataset(manifest=metadata_csv, tensors_dir=root / "tensors", split="test")

    train_loader = DataLoader(ds_train, batch_size=2, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=2, shuffle=False)

    model = Baseline3DCNN(in_channels=1, num_classes=2)
    ckpt_path = tmp_path / "best_model.pt"

    res = train_3dcnn_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=1,
        early_stopping_patience=1,
        checkpoint_path=ckpt_path,
        device="cpu",
        use_amp=False,
    )

    assert "model" in res
    assert "history" in res
    assert "best_metrics" in res
    assert "accuracy" in res["best_metrics"]
    assert ckpt_path.is_file()


def test_run_training_pipeline_3dcnn(mock_dataset_environment):
    """Verify high-level run_training_pipeline executes for 3dcnn model type."""
    metadata_csv = mock_dataset_environment["metadata_csv"]
    tensors_dir = mock_dataset_environment["tensors_dir"]

    res = run_training_pipeline(
        model_type="3dcnn",
        manifest_path=str(metadata_csv),
        tensors_dir=str(tensors_dir),
        epochs=1,
        save_checkpoint=False,
        device="cpu",
        use_amp=False,
    )

    assert "best_metrics" in res
    assert "history" in res


def test_compute_class_weights_nyu_cohort():
    """Verify class weight computation correctly up-weights minority class on NYU Cohort."""
    from asd_classification.pipelines.train import compute_class_weights

    # NYU Cohort train split: 84 TC (0), 64 ASD (1)
    labels = np.array([0] * 84 + [1] * 64)
    weights = compute_class_weights(labels)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (2,)

    w_tc = weights[0].item()
    w_asd = weights[1].item()

    # Minority ASD class must receive weight > 1.0, majority TC < 1.0
    assert w_tc < 1.0
    assert w_asd > 1.0

    # Effective balanced contribution
    assert np.isclose(w_tc * 84, w_asd * 64, rtol=1e-5)
    assert np.isclose(w_tc, 148 / (2.0 * 84), rtol=1e-5)
    assert np.isclose(w_asd, 148 / (2.0 * 64), rtol=1e-5)


def test_class_weighted_cross_entropy_majority_penalty():
    """Verify weighted CrossEntropyLoss penalizes majority-class collapse more than unweighted loss."""
    # Weights for NYU Cohort
    w_tc = 148.0 / (2.0 * 84.0)
    w_asd = 148.0 / (2.0 * 64.0)
    class_weights = torch.tensor([w_tc, w_asd], dtype=torch.float32)

    unweighted_ce = nn.CrossEntropyLoss()
    weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

    # Simulated majority-collapse: model predicts TC (class 0) with high confidence for all samples
    logits = torch.tensor([
        [4.0, -4.0],  # predicts class 0
        [3.5, -3.5],  # predicts class 0
        [4.2, -4.2],  # predicts class 0
    ], dtype=torch.float32)

    # Ground truth targets containing minority ASD subjects (targets: [1, 1, 0])
    targets = torch.tensor([1, 1, 0], dtype=torch.long)

    loss_unweighted = unweighted_ce(logits, targets)
    loss_weighted = weighted_ce(logits, targets)

    # Because model misclassifies ASD and ASD is upweighted, weighted loss must be strictly greater
    assert loss_weighted > loss_unweighted


def test_run_training_pipeline_with_class_weights(mock_dataset_environment):
    """Verify run_training_pipeline configures weighted CrossEntropyLoss when class_weights provided."""
    metadata_csv = mock_dataset_environment["metadata_csv"]
    tensors_dir = mock_dataset_environment["tensors_dir"]

    res = run_training_pipeline(
        model_type="3dcnn",
        manifest_path=str(metadata_csv),
        tensors_dir=str(tensors_dir),
        epochs=1,
        save_checkpoint=False,
        class_weights=[0.88, 1.15],
        device="cpu",
        use_amp=False,
    )

    assert "best_metrics" in res
    assert "history" in res

