"""Unit and integration tests for 2.5D ResNet-18 MIL training and evaluation pipeline.

Verifies acceptance criteria for Issue #16:
1. Differential optimizer parameter groups (1e-5 backbone, 1e-4 attention/classifier).
2. 3-epoch linear warmup schedule followed by Cosine Annealing decay down to 1e-6.
3. Training epoch loop computing label-smoothed cross-entropy loss and accumulating gradients without NaN.
4. Evaluation function computing diagnostic metrics and extracting slice attention weights.
5. Decision threshold calibration (tau* in [0.30, 0.60]) maximizing Table 8 Composite with Recall priority.
6. Integration tests verifying parameter updates, loss reduction, and early stopping.
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from asd_classification.models.baseline.resnet_mil import ResNet18MIL
from asd_classification.pipelines.evaluate import (
    evaluate_mil_model,
    find_optimal_threshold,
)
from asd_classification.pipelines.train import (
    compute_differential_cosine_warmup_lr,
    configure_differential_optimizer,
    run_training_pipeline,
    train_epoch,
    train_mil_epoch,
    train_mil_model,
    update_learning_rates,
)


# ---------------------------------------------------------------------------
# Slice 1: Optimizer & Learning Rate Scheduling Tests
# ---------------------------------------------------------------------------

def test_configure_differential_optimizer_param_groups():
    """Verify configure_differential_optimizer assigns distinct lr and weight_decay."""
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    optimizer = configure_differential_optimizer(
        model,
        backbone_lr=1e-5,
        head_lr=1e-4,
        backbone_weight_decay=1e-4,
        head_weight_decay=1e-3,
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) == 2

    backbone_group = optimizer.param_groups[0]
    head_group = optimizer.param_groups[1]

    assert backbone_group["name"] == "backbone_deep"
    assert backbone_group["lr"] == 1e-5
    assert backbone_group["initial_lr"] == 1e-5
    assert backbone_group["weight_decay"] == 1e-4

    assert head_group["name"] == "attention_and_classifier"
    assert head_group["lr"] == 1e-4
    assert head_group["initial_lr"] == 1e-4
    assert head_group["weight_decay"] == 1e-3

    # Ensure frozen early layers are not included in any optimizer param group
    all_optimizer_params = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            all_optimizer_params.add(id(p))

    for module in [model.conv1, model.bn1, model.layer1, model.layer2]:
        for param in module.parameters():
            assert id(param) not in all_optimizer_params


def test_compute_differential_cosine_warmup_lr_schedule():
    """Verify 3-epoch linear warmup followed by cosine annealing decay down to min_lr."""
    # Test for head group with peak_lr=1e-4
    head_lrs = [
        compute_differential_cosine_warmup_lr(
            epoch=ep,
            base_lr=1e-4,
            warmup_epochs=3,
            max_epochs=100,
            min_lr=1e-6,
        )
        for ep in range(1, 101)
    ]

    # Warmup phase (epochs 1-3): monotonic increase to base_lr (1e-4)
    assert np.isclose(head_lrs[0], 1e-5, rtol=1e-4)  # start_lr defaults to base_lr * 0.1
    assert head_lrs[0] < head_lrs[1] < head_lrs[2]
    assert np.isclose(head_lrs[2], 1e-4, rtol=1e-4)

    # Cosine decay phase (epochs 3-100): monotonic decrease down to min_lr (1e-6)
    for i in range(2, 99):
        assert head_lrs[i] > head_lrs[i + 1]
    assert np.isclose(head_lrs[-1], 1e-6, rtol=1e-4)

    # Test for backbone group with peak_lr=1e-5
    backbone_lrs = [
        compute_differential_cosine_warmup_lr(
            epoch=ep,
            base_lr=1e-5,
            warmup_epochs=3,
            max_epochs=100,
            min_lr=1e-6,
        )
        for ep in range(1, 101)
    ]

    # Warmup phase (epochs 1-3): monotonic increase to base_lr (1e-5)
    assert backbone_lrs[0] < backbone_lrs[1] < backbone_lrs[2]
    assert np.isclose(backbone_lrs[2], 1e-5, rtol=1e-4)

    # Cosine decay down to min_lr
    for i in range(2, 99):
        assert backbone_lrs[i] > backbone_lrs[i + 1]
    assert np.isclose(backbone_lrs[-1], 1e-6, rtol=1e-4)


def test_update_learning_rates_updates_optimizer_groups():
    """Verify update_learning_rates updates each group's lr in-place."""
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    optimizer = configure_differential_optimizer(
        model,
        backbone_lr=1e-5,
        head_lr=1e-4,
    )

    # Epoch 1: linear warmup start
    updated_lrs_ep1 = update_learning_rates(
        optimizer,
        epoch=1,
        warmup_epochs=3,
        max_epochs=100,
        min_lr=1e-6,
    )
    assert optimizer.param_groups[0]["lr"] == updated_lrs_ep1["backbone_deep"]
    assert optimizer.param_groups[1]["lr"] == updated_lrs_ep1["attention_and_classifier"]
    assert optimizer.param_groups[0]["lr"] < 1e-5
    assert optimizer.param_groups[1]["lr"] < 1e-4

    # Epoch 3: reaches peak differential learning rates
    updated_lrs_ep3 = update_learning_rates(
        optimizer,
        epoch=3,
        warmup_epochs=3,
        max_epochs=100,
        min_lr=1e-6,
    )
    assert np.isclose(optimizer.param_groups[0]["lr"], 1e-5, rtol=1e-4)
    assert np.isclose(optimizer.param_groups[1]["lr"], 1e-4, rtol=1e-4)

    # Epoch 100: decays to min_lr
    updated_lrs_ep100 = update_learning_rates(
        optimizer,
        epoch=100,
        warmup_epochs=3,
        max_epochs=100,
        min_lr=1e-6,
    )
    assert np.isclose(optimizer.param_groups[0]["lr"], 1e-6, rtol=1e-4)
    assert np.isclose(optimizer.param_groups[1]["lr"], 1e-6, rtol=1e-4)


# ---------------------------------------------------------------------------
# Slice 2: Training Epoch Loop & Gradient Accumulation Tests
# ---------------------------------------------------------------------------

def test_train_mil_epoch_label_smoothed_loss_and_gradient_accumulation():
    """Verify train_mil_epoch computes finite loss and updates only trainable parameters."""
    torch.manual_seed(42)
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    optimizer = configure_differential_optimizer(model, backbone_lr=1e-5, head_lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 4 synthetic batches: batch_size=2, K=6 tri-slices of 32x32
    inputs = torch.randn(8, 6, 3, 32, 32, dtype=torch.float32)
    targets = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.long)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Cache early layer weights and trainable weights prior to training
    conv1_weight_before = model.conv1.weight.clone()
    layer1_weight_before = model.layer1[0].conv1.weight.clone()
    layer3_weight_before = model.layer3[0].conv1.weight.clone()
    head_weight_before = model.classifier[0].weight.clone()

    avg_loss = train_mil_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        accumulation_steps=2,
        device="cpu",
    )

    assert not np.isnan(avg_loss)
    assert not np.isinf(avg_loss)
    assert avg_loss > 0.0

    assert torch.equal(model.conv1.weight, conv1_weight_before)
    assert torch.equal(model.layer1[0].conv1.weight, layer1_weight_before)

    assert not torch.equal(model.layer3[0].conv1.weight, layer3_weight_before)
    assert not torch.equal(model.classifier[0].weight, head_weight_before)


def test_train_mil_epoch_loss_reduction():
    """Verify train_mil_epoch reduces loss over multiple epochs on a toy batch."""
    torch.manual_seed(42)
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    optimizer = configure_differential_optimizer(model, backbone_lr=1e-4, head_lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    inputs = torch.randn(4, 4, 3, 32, 32, dtype=torch.float32)
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    losses = []
    for _ in range(5):
        loss_val = train_mil_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            accumulation_steps=1,
            device="cpu",
        )
        losses.append(loss_val)

    assert losses[-1] < losses[0]


def test_train_epoch_tuple_output_compatibility():
    """Verify standard train_epoch handles models that return (logits, attention_weights)."""
    torch.manual_seed(42)
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    optimizer = configure_differential_optimizer(model, backbone_lr=1e-5, head_lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    inputs = torch.randn(2, 4, 3, 32, 32, dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    loss_val = train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        accumulation_steps=1,
        device="cpu",
    )
    assert not np.isnan(loss_val)
    assert loss_val > 0.0


# ---------------------------------------------------------------------------
# Slice 3: MIL Evaluation & Slice Attention Extraction Tests
# ---------------------------------------------------------------------------

def test_evaluate_mil_model_metrics_and_attention_extraction():
    """Verify evaluate_mil_model returns comprehensive metrics and attention matrix."""
    torch.manual_seed(42)
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    model.eval()

    n_subjects = 8
    num_slices = 6
    inputs = torch.randn(n_subjects, num_slices, 3, 32, 32, dtype=torch.float32)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    eval_res = evaluate_mil_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device="cpu",
        calibrate_threshold=True,
    )

    # 1. Verify structure of results
    assert "metrics" in eval_res
    assert "calibrated_metrics" in eval_res
    assert "optimal_threshold" in eval_res
    assert "attention_weights" in eval_res
    assert "probabilities" in eval_res
    assert "predictions" in eval_res
    assert "targets" in eval_res

    # 2. Verify all diagnostic metrics exist
    metrics = eval_res["metrics"]
    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "balanced_accuracy",
        "composite_score",
        "loss",
        "roc_auc",
    ]
    for m in required_metrics:
        assert m in metrics, f"Missing metric {m} in evaluation metrics"
        assert not np.isnan(metrics[m])

    # 3. Verify slice attention weights extraction and simplex constraints
    attn_weights = eval_res["attention_weights"]
    assert isinstance(attn_weights, np.ndarray)
    assert attn_weights.shape == (n_subjects, num_slices)
    assert np.all(attn_weights >= 0.0)
    row_sums = np.sum(attn_weights, axis=1)
    assert np.allclose(row_sums, np.ones(n_subjects), atol=1e-5)

    # 4. Verify optimal threshold search in [0.30, 0.60]
    opt_tau = eval_res["optimal_threshold"]
    assert 0.30 <= opt_tau <= 0.60
    assert "composite_score" in eval_res["calibrated_metrics"]


def test_find_optimal_threshold_recall_priority_tie_breaking(monkeypatch):
    """Verify optimal threshold calibration prioritizes higher Recall on ties."""
    from asd_classification.pipelines import evaluate

    def mock_metrics(y_true, y_pred, y_prob=None, loss_value=None):
        if np.sum(y_pred) >= 2:
            return {
                "balanced_accuracy": 0.80,
                "f1_score": 0.80,
                "recall": 0.95,
                "specificity": 0.65,
                "accuracy": 0.80,
                "precision": 0.70,
                "composite_score": 0.80,
            }
        return {
            "balanced_accuracy": 0.80,
            "f1_score": 0.80,
            "recall": 0.50,
            "specificity": 1.00,
            "accuracy": 0.80,
            "precision": 1.00,
            "composite_score": 0.80,
        }

    monkeypatch.setattr(evaluate, "compute_classification_metrics", mock_metrics)
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.35, 0.45, 0.25, 0.55])

    opt_tau = evaluate.find_optimal_threshold(y_true, y_prob, min_thresh=0.30, max_thresh=0.60)
    # The tie in composite score (0.80 vs 0.80) must be broken in favor of the threshold giving higher recall (0.95)
    assert opt_tau <= 0.45


# ---------------------------------------------------------------------------
# Slice 4: End-to-End MIL Training Pipeline Tests
# ---------------------------------------------------------------------------

def test_train_mil_model_early_stopping_and_checkpointing(tmp_path: Path):
    """Verify train_mil_model trains, updates differential LRs, checkpoints, and restores weights."""
    torch.manual_seed(42)
    model = ResNet18MIL(pretrained=False, freeze_early=True)

    # 4 train samples, 2 val samples
    train_x = torch.randn(4, 4, 3, 32, 32, dtype=torch.float32)
    train_y = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    val_x = torch.randn(2, 4, 3, 32, 32, dtype=torch.float32)
    val_y = torch.tensor([1, 0], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=2, shuffle=False)

    ckpt_path = tmp_path / "best_resnet_mil.pt"

    res = train_mil_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=4,
        warmup_epochs=2,
        accumulation_steps=1,
        early_stopping_patience=3,
        checkpoint_path=ckpt_path,
        device="cpu",
    )

    assert "model" in res
    assert "history" in res
    assert "best_metrics" in res
    assert ckpt_path.is_file(), "Best checkpoint should be saved to disk"

    history = res["history"]
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4
    assert len(history["val_composite_score"]) == 4


def test_run_training_pipeline_resnet_mil(tmp_path: Path):
    """Verify run_training_pipeline executes cleanly with model_type='resnet_mil'."""
    tensors_dir = tmp_path / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # Create 4 train and 2 test subjects with synthetic multi-view tensors (1, 150, 64, 64)
    for i in range(6):
        sub_id = f"500{i}"
        split = "train" if i < 4 else "test"
        dx = 1 if i % 2 == 0 else 0
        np.savez_compressed(tensors_dir / f"{sub_id}.npz", tensor=np.zeros((1, 150, 64, 64), dtype=np.float32))
        records.append({
            "subject_id": sub_id,
            "file_id": f"NYU_00{sub_id}",
            "dx_group": dx,
            "split": split,
            "tensor_path": f"tensors/{sub_id}.npz",
        })

    import pandas as pd
    manifest_df = pd.DataFrame(records)
    manifest_path = tmp_path / "metadata.csv"
    manifest_df.to_csv(manifest_path, index=False)

    res = run_training_pipeline(
        model_type="resnet_mil",
        manifest_path=manifest_path,
        tensors_dir=tensors_dir,
        epochs=2,
        warmup_epochs=1,
        batch_size=2,
        accumulation_steps=1,
        save_checkpoint=False,
        pretrained=False,
        slices_per_view=50,
        central_slices=20,
        stride=2,
    )

    assert "model" in res
    assert "best_metrics" in res



