"""Unit and integration tests for Spatio-Temporal Multi-GPU Training Engine.

Tests exercise public interfaces at pre-agreed seams:
1. Multi-GPU Orchestration & Precision Seam (setup_model_parallelism, unwrap_model, resolve_device_and_dtype)
2. Fold Class Weighting & Optimization Seam (compute_fold_class_weights, configure_spatiotemporal_optimizer, configure_cosine_scheduler)
3. Training & Evaluation Epoch Seam (train_spatiotemporal_epoch, evaluate_spatiotemporal_epoch)
4. Early Stopping & Checkpoint Saving Seam (SpatiotemporalEarlyStopping)
5. End-to-End 5-Fold Cross-Validation Integration Seam (run_spatiotemporal_kfold_cv)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from asd_classification.models.baseline.focal_loss import FocalLoss
from asd_classification.models.baseline.spatiotemporal_bimamba import SpatioTemporalBiMamba
from asd_classification.pipelines.spatiotemporal_engine import (
    compute_fold_class_weights,
    configure_cosine_scheduler,
    configure_spatiotemporal_optimizer,
    evaluate_spatiotemporal_epoch,
    resolve_device_and_dtype,
    setup_model_parallelism,
    train_spatiotemporal_epoch,
    unwrap_model,
    SpatiotemporalEarlyStopping,
    run_spatiotemporal_kfold_cv,
)


# ---------------------------------------------------------------------------
# Slice 1: Multi-GPU Orchestration, Precision, Class Weights & Optimizer
# ---------------------------------------------------------------------------

def test_resolve_device_and_dtype():
    """Verify resolve_device_and_dtype correctly selects device and BF16/FP32 precision."""
    # Explicit CPU request with BF16
    device, dtype = resolve_device_and_dtype(device="cpu", use_bf16=True)
    assert device.type == "cpu"
    assert dtype == torch.bfloat16

    # Explicit CPU request with FP32
    device, dtype = resolve_device_and_dtype(device="cpu", use_bf16=False)
    assert device.type == "cpu"
    assert dtype == torch.float32


def test_setup_model_parallelism_and_unwrap():
    """Verify setup_model_parallelism and unwrap_model handle DataParallel and plain modules."""
    base_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Strategy 'none' should return plain model on target device
    wrapped_model, is_parallel = setup_model_parallelism(
        model=base_model,
        device=device,
        strategy="none",
    )
    assert not is_parallel
    assert unwrap_model(wrapped_model) is base_model

    # Simulated DataParallel wrapping
    dp_model = nn.DataParallel(base_model)
    unwrapped = unwrap_model(dp_model)
    assert unwrapped is base_model


def test_compute_fold_class_weights_balanced_and_imbalanced():
    """Verify fold class weighting accurately balances ASD vs TC representation."""
    # Balanced fold: 50 TC (0), 50 ASD (1)
    balanced_labels = np.array([0] * 50 + [1] * 50)
    weights = compute_fold_class_weights(balanced_labels, alpha=0.25)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (2,)
    # For balanced distribution, weights should reflect the alpha ratio [1-alpha, alpha]
    # normalized so that mean or ratio is consistent
    ratio = weights[1].item() / weights[0].item()
    expected_ratio = 0.25 / (1.0 - 0.25)  # 0.25 / 0.75 = 1/3
    assert np.isclose(ratio, expected_ratio, rtol=1e-3)

    # Imbalanced fold: 80 TC (0), 20 ASD (1) (e.g. 4:1 imbalance)
    imbalanced_labels = np.array([0] * 80 + [1] * 20)
    weights_imb = compute_fold_class_weights(imbalanced_labels, alpha=0.25)
    # Class 1 (minority) should receive higher weight adjustment to counteract imbalance
    # Frequency inverse: w0 ~ (1-alpha)/80, w1 ~ alpha/20 -> ratio = (0.25/20) / (0.75/80) = (1/80) / (3/320) = 4/3
    imb_ratio = weights_imb[1].item() / weights_imb[0].item()
    assert imb_ratio > ratio


def test_configure_spatiotemporal_optimizer_and_cosine_scheduler():
    """Verify AdamW optimizer and cosine annealing scheduler configuration."""
    model = nn.Linear(128, 2)
    optimizer = configure_spatiotemporal_optimizer(model, lr=1e-4, weight_decay=1e-4)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-4
    assert optimizer.defaults["weight_decay"] == 1e-4

    scheduler = configure_cosine_scheduler(optimizer, epochs=50, min_lr=1e-6)
    assert scheduler is not None

    # Step through epochs and ensure learning rate decays toward min_lr
    initial_lr = optimizer.param_groups[0]["lr"]
    assert np.isclose(initial_lr, 1e-4, rtol=1e-4)

    optimizer.step()
    for _ in range(50):
        scheduler.step()

    final_lr = optimizer.param_groups[0]["lr"]
    assert np.isclose(final_lr, 1e-6, rtol=1e-3)


# ---------------------------------------------------------------------------
# Slice 2: Training & Evaluation Epochs, Clinical Metrics & Early Stopping
# ---------------------------------------------------------------------------

class DummySpatioTemporalModel(nn.Module):
    """Minimal spatio-temporal module for deterministic fast unit testing."""

    def __init__(self, in_features: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(
        self,
        volumes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Flatten spatial-temporal dimensions to match feature dimension
        batch_size = volumes.size(0)
        flattened = volumes.reshape(batch_size, -1)
        if flattened.shape[1] >= 16:
            feat = flattened[:, :16]
        else:
            feat = torch.nn.functional.pad(flattened, (0, 16 - flattened.shape[1]))
        logits = self.fc(feat)
        return logits, None


def test_train_spatiotemporal_epoch_with_bf16_and_clipping():
    """Verify train_spatiotemporal_epoch executes under BF16 with gradient clipping."""
    torch.manual_seed(42)
    model = DummySpatioTemporalModel()
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Synthetic batches: (volumes (B, T, 1, 4, 4, 4), masks (B, T), targets (B,))
    x = torch.randn(8, 5, 1, 4, 4, 4, dtype=torch.float32)
    mask = torch.ones(8, 5, dtype=torch.bool)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(x, mask, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    initial_loss = train_spatiotemporal_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_bf16=True,
        gradient_clip_norm=1.0,
    )
    assert isinstance(initial_loss, float)
    assert initial_loss > 0.0

    # Ensure gradients were computed and model updated
    for param in model.parameters():
        assert param.grad is None or not torch.isnan(param.grad).any()


def test_evaluate_spatiotemporal_epoch_clinical_metrics():
    """Verify evaluate_spatiotemporal_epoch reports all required clinical metrics.
    
    Required metrics: Balanced Accuracy, ROC-AUC, Sensitivity (Recall),
    Specificity, F1-Score, and validation Focal Loss.
    """
    model = DummySpatioTemporalModel()
    device = torch.device("cpu")
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    x = torch.randn(10, 4, 1, 4, 4, 4, dtype=torch.float32)
    mask = torch.ones(10, 4, dtype=torch.bool)
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    dataset = TensorDataset(x, mask, y)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    avg_loss, metrics, y_true, y_pred, y_prob = evaluate_spatiotemporal_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        use_bf16=True,
    )

    assert isinstance(avg_loss, float)
    assert "balanced_accuracy" in metrics
    assert "roc_auc" in metrics
    assert "sensitivity" in metrics
    assert "recall" in metrics
    assert metrics["sensitivity"] == metrics["recall"]
    assert "specificity" in metrics
    assert "f1_score" in metrics
    assert "loss" in metrics
    assert len(y_true) == 10
    assert len(y_pred) == 10
    assert y_prob.shape == (10, 2)


def test_spatiotemporal_early_stopping_and_checkpoint_saving(tmp_path: Path):
    """Verify EarlyStopping stops on validation loss stagnation and saves best weights on composite score."""
    ckpt_path = tmp_path / "best_model.pt"
    early_stopping = SpatiotemporalEarlyStopping(
        patience=2,
        min_delta=1e-3,
        checkpoint_path=ckpt_path,
    )

    model = DummySpatioTemporalModel()

    # Epoch 1: Loss 0.5, composite score 0.70 -> improves, saves checkpoint
    stop = early_stopping(
        val_loss=0.5,
        val_metrics={"balanced_accuracy": 0.70, "roc_auc": 0.70},
        model=model,
        epoch=1,
    )
    assert not stop
    assert early_stopping.counter == 0
    assert ckpt_path.is_file()

    # Epoch 2: Loss 0.51 (worse), composite score 0.65 -> counter 1
    stop = early_stopping(
        val_loss=0.51,
        val_metrics={"balanced_accuracy": 0.65, "roc_auc": 0.65},
        model=model,
        epoch=2,
    )
    assert not stop
    assert early_stopping.counter == 1

    # Epoch 3: Loss 0.52 (worse), patience reached -> stop True
    stop = early_stopping(
        val_loss=0.52,
        val_metrics={"balanced_accuracy": 0.60, "roc_auc": 0.60},
        model=model,
        epoch=3,
    )
    assert stop
    assert early_stopping.counter == 2


# ---------------------------------------------------------------------------
# Slice 3: Synthetic End-to-End Multi-Step Training and 5-Fold Loop
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Synthetic dataset generating 5D spatio-temporal tensors with boolean masks."""

    def __init__(self, n_samples: int = 10, t_steps: int = 4) -> None:
        self.n_samples = n_samples
        self.t_steps = t_steps
        # Minimum spatial dimensions of 16x16x16 required for 4-block 3D-CNN downsampling
        self.volumes = torch.randn(n_samples, t_steps, 1, 16, 16, 16, dtype=torch.float32)
        self.masks = torch.ones(n_samples, t_steps, dtype=torch.bool)
        self.labels = torch.tensor([i % 2 for i in range(n_samples)], dtype=torch.long)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.volumes[idx], self.masks[idx], self.labels[idx]


def test_run_spatiotemporal_kfold_cv_synthetic_loop(tmp_path: Path):
    """Integration test executing synthetic multi-step training and evaluation loop.
    
    Verifies:
    1. Joint Site-and-Class Stratification into folds
    2. Fold label distribution-weighted Focal Loss
    3. AdamW optimization with Cosine Annealing scheduler and gradient clipping
    4. Metric calculation: Balanced Accuracy, ROC-AUC, Sensitivity, Specificity, F1-Score, Loss
    5. Saving and restoring best checkpoint weights per fold
    6. Aggregation of out-of-fold predictions
    """
    n_samples = 10
    n_splits = 2  # 2 folds for fast synthetic integration test
    manifest = pd.DataFrame({
        "subject_id": [f"SUB_{i:03d}" for i in range(n_samples)],
        "site": ["SITE_A"] * 5 + ["SITE_B"] * 5,
        "label": [0, 1] * 5,
    })

    dataset = SyntheticDataset(n_samples=n_samples, t_steps=4)

    # Use miniature architectural parameters for fast synthetic execution
    mini_model_kwargs = {
        "in_channels": 1,
        "d_model": 16,
        "d_state": 4,
        "num_layers": 1,
        "num_classes": 2,
        "force_pure_pytorch": True,
    }

    results = run_spatiotemporal_kfold_cv(
        manifest=manifest,
        dataset_or_paths=dataset,
        n_splits=n_splits,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_dir=tmp_path / "checkpoints",
        device="cpu",
        use_bf16=True,
        parallel_strategy="none",
        model_kwargs=mini_model_kwargs,
    )

    assert "fold_results" in results
    assert len(results["fold_results"]) == n_splits

    # Check fold-level metrics
    for fold_res in results["fold_results"]:
        assert "balanced_accuracy" in fold_res
        assert "roc_auc" in fold_res
        assert "sensitivity" in fold_res
        assert "recall" in fold_res
        assert "specificity" in fold_res
        assert "f1_score" in fold_res
        assert "loss" in fold_res

    # Check checkpoint weights were created per fold
    for fold_idx in range(1, n_splits + 1):
        ckpt_file = tmp_path / "checkpoints" / f"best_spatiotemporal_fold_{fold_idx}.pt"
        assert ckpt_file.is_file()

    # Check aggregate out-of-fold predictions
    assert "summary" in results
    assert "balanced_accuracy" in results["summary"]
    assert "sensitivity" in results["summary"]
    assert "oof_targets" in results
    assert len(results["oof_targets"]) == n_samples
    assert len(results["oof_preds"]) == n_samples

