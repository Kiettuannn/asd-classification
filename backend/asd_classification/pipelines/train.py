"""Training workflows, mixed-precision optimization, and cross-validation pipelines.

Implements the mixed-precision training pipeline for Baseline 3D-CNN from
Hammash & Younis (MDPI Journal of Imaging, 2026, 12, 109) with:
- Adam optimizer (learning rate 1e-5, L2 weight decay 0.0005)
- Physical batch size 2, gradient accumulation 4 steps (effective batch size 8)
- Automatic Mixed Precision (torch.amp.autocast)
- Early stopping monitored on validation loss (patience 15 epochs, max 120 epochs)
- Stratified 5-fold cross-validation switch across all 184 subjects
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from asd_classification.data.loader import MultiViewDataset, TriSliceDataset
from asd_classification.data.split import get_kfold_splits
from asd_classification.models.baseline.augmentation import (
    BagConsistentAugmenter,
    VolumeConsistentAugmenter,
)
from asd_classification.models.baseline.cnn3d import Baseline3DCNN, SEResNet3D
from asd_classification.models.baseline.focal_loss import FocalLoss
from asd_classification.models.baseline.resnet_mil import ResNet18MIL
from asd_classification.pipelines.evaluate import (
    compute_classification_metrics,
    evaluate_mil_model,
    evaluate_multiview_ensemble,
)


def compute_cosine_warmup_lr(
    epoch: int,
    warmup_epochs: int = 5,
    max_epochs: int = 100,
    peak_lr: float = 1e-4,
    min_lr: float = 1e-6,
    start_lr: float = 1e-5,
) -> float:
    """Compute learning rate for a given epoch using linear warmup and cosine decay.

    Args:
        epoch: Current 1-based epoch index (1 <= epoch <= max_epochs).
        warmup_epochs: Number of linear warmup epochs (default: 5).
        max_epochs: Total optimization epochs (default: 100).
        peak_lr: Maximum learning rate reached after warmup (default: 1e-4).
        min_lr: Minimum learning rate at terminal epoch (default: 1e-6).
        start_lr: Initial learning rate at epoch 1 (default: 1e-5).

    Returns:
        Scheduled learning rate float value.
    """
    if epoch <= warmup_epochs:
        if warmup_epochs <= 1:
            return float(peak_lr)
        alpha = (epoch - 1) / float(warmup_epochs - 1)
        return float(start_lr + alpha * (peak_lr - start_lr))

    # Cosine annealing phase
    progress = (epoch - warmup_epochs) / float(max(max_epochs - warmup_epochs, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (peak_lr - min_lr) * cosine_decay)


def compute_differential_cosine_warmup_lr(
    epoch: int,
    base_lr: float,
    warmup_epochs: int = 3,
    max_epochs: int = 100,
    min_lr: float = 1e-6,
    start_lr: Optional[float] = None,
) -> float:
    """Compute scheduled learning rate for a specific parameter group across epochs.

    Implements a linear warmup over `warmup_epochs` followed by Cosine Annealing
    decay down to `min_lr` at `max_epochs`.

    Args:
        epoch: Current 1-based training epoch (1 <= epoch <= max_epochs).
        base_lr: Peak target learning rate reached at the end of the warmup phase.
        warmup_epochs: Number of linear warmup epochs (default: 3).
        max_epochs: Total optimization epochs (default: 100).
        min_lr: Minimum learning rate floor at terminal epoch (default: 1e-6).
        start_lr: Initial learning rate at epoch 1. If None, defaults to base_lr * 0.1.

    Returns:
        Scheduled learning rate float value.
    """
    if start_lr is None:
        start_lr = base_lr * 0.1

    if epoch <= warmup_epochs:
        if warmup_epochs <= 1:
            return float(base_lr)
        alpha = (epoch - 1) / float(warmup_epochs - 1)
        return float(start_lr + alpha * (base_lr - start_lr))

    progress = (epoch - warmup_epochs) / float(max(max_epochs - warmup_epochs, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (base_lr - min_lr) * cosine_decay)


def configure_differential_optimizer(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    backbone_weight_decay: float = 1e-4,
    head_weight_decay: float = 1e-3,
) -> torch.optim.Adam:
    """Configure Adam optimizer with differential learning rates for ResNet-18 MIL.

    Assigns a low learning rate (1e-5) to deep backbone stages (layer3, layer4)
    and a higher rate (1e-4) to the Gated Attention module and classification head.
    Excludes frozen early stages (conv1, bn1, layer1, layer2).

    Args:
        model: ResNet18MIL network instance.
        backbone_lr: Fine-tuning learning rate for ResNet backbone (default: 1e-5).
        head_lr: Learning rate for attention pooling and classification head (default: 1e-4).
        backbone_weight_decay: L2 regularization factor for backbone (default: 1e-4).
        head_weight_decay: L2 regularization factor for heads (default: 1e-3).

    Returns:
        Configured torch.optim.Adam optimizer instance.
    """
    param_fn = getattr(model, "get_parameter_groups", None)
    if callable(param_fn):
        param_groups = param_fn(
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            backbone_weight_decay=backbone_weight_decay,
            head_weight_decay=head_weight_decay,
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        param_groups = [
            {
                "params": trainable,
                "lr": head_lr,
                "weight_decay": head_weight_decay,
                "name": "all_trainable",
            }
        ]

    for group in param_groups:
        group["initial_lr"] = float(group["lr"])
        group["base_lr"] = float(group["lr"])

    return torch.optim.Adam(param_groups)


def update_learning_rates(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int = 3,
    max_epochs: int = 100,
    min_lr: float = 1e-6,
) -> Dict[str, float]:
    """Update learning rates for all parameter groups in the optimizer for the given epoch.

    Args:
        optimizer: PyTorch optimizer with one or more parameter groups.
        epoch: Current 1-based training epoch.
        warmup_epochs: Number of linear warmup epochs (default: 3).
        max_epochs: Total optimization epochs (default: 100).
        min_lr: Minimum learning rate floor (default: 1e-6).

    Returns:
        Dictionary mapping parameter group name (or index string) to updated learning rate.
    """
    updated: Dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        base_lr = float(group.get("base_lr", group.get("initial_lr", group["lr"])))
        group_min_lr = min(min_lr, base_lr)
        group_lr = compute_differential_cosine_warmup_lr(
            epoch=epoch,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            min_lr=group_min_lr,
        )
        group["lr"] = group_lr
        group_name = str(group.get("name", f"group_{idx}"))
        updated[group_name] = group_lr
    return updated


class EarlyStopping:
    """Early stopping monitor for deep neural network training.

    Terminates optimization when validation metric fails to improve after
    a specified patience window of consecutive epochs, avoiding overfitting.
    Supports warmup epoch suppression to avoid prematurely saving unlearned initial weights.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
        checkpoint_path: Optional[Union[str, Path]] = None,
        warmup_epochs: int = 0,
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs with no improvement before stopping (default: 15).
            min_delta: Minimum change in monitored quantity to qualify as improvement (default: 1e-4).
            mode: Direction of optimization, 'min' for loss or 'max' for score (default: 'min').
            checkpoint_path: Optional path to save best model weights.
            warmup_epochs: Initial epochs during which checkpointing and patience are suppressed (default: 0).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.warmup_epochs = warmup_epochs

        self.counter = 0
        self.best_loss = float("inf") if mode == "min" else float("-inf")
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.early_stop = False

    def __call__(
        self,
        val_score: Optional[float] = None,
        model: Optional[nn.Module] = None,
        epoch: int = 1,
        val_loss: Optional[float] = None,
    ) -> bool:
        """Evaluate validation metric and update early stopping state.

        Args:
            val_score: Validation metric value (loss for 'min', composite score/accuracy for 'max').
            model: Optional PyTorch model instance whose weights will be saved on improvement.
            epoch: Current 1-based training epoch.
            val_loss: Legacy alias for val_score.

        Returns:
            True if early stopping criterion has been triggered, False otherwise.
        """
        if val_score is None:
            val_score = val_loss if val_loss is not None else 0.0

        current = float(val_score)

        # Suppress checkpoint saving and patience count during initial warmup epochs
        if epoch <= self.warmup_epochs:
            return False

        target_best = self.best_score if self.mode == "max" else self.best_loss

        if self.mode == "min":
            improved = current < (target_best - self.min_delta)
        else:
            improved = current > (target_best + self.min_delta)

        if improved:
            self.best_loss = current
            self.best_score = current
            self.best_epoch = epoch
            self.counter = 0
            if self.checkpoint_path is not None and model is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                save_fn = getattr(model, "save", None)
                if callable(save_fn):
                    save_fn(self.checkpoint_path)
                else:
                    torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[torch.amp.GradScaler] = None,
    accumulation_steps: int = 4,
    device: str = "cpu",
    augmenter: Optional[nn.Module] = None,
) -> float:
    """Execute one training epoch with gradient accumulation and Automatic Mixed Precision.

    Args:
        model: PyTorch neural network module.
        dataloader: Training DataLoader delivering batches of (inputs, targets).
        optimizer: PyTorch optimizer (e.g. Adam).
        criterion: Loss module (e.g. FocalLoss).
        scaler: Optional GradScaler for mixed precision (torch.amp).
        accumulation_steps: Number of batches over which gradients are accumulated (default: 4).
        device: Execution device ('cuda' or 'cpu').
        augmenter: Optional data augmentation module.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    if augmenter is not None:
        augmenter.train()

    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()

    amp_enabled = (torch.device(device).type == "cuda") and (scaler is not None)

    for step_idx, (inputs, targets) in enumerate(dataloader):
        if augmenter is not None:
            inputs = augmenter(inputs)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)

        # Forward pass with AMP when GPU is available
        if amp_enabled and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(logits, targets)
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, targets)
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Gradient accumulation step
        is_accumulation_step = ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == len(dataloader))
        if is_accumulation_step:
            if amp_enabled and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(total_samples, 1)


def train_mil_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[torch.amp.GradScaler] = None,
    accumulation_steps: int = 4,
    device: str = "cpu",
    max_norm: float = 1.0,
    augmenter: Optional[nn.Module] = None,
) -> float:
    """Execute one training epoch for ResNet-18 MIL with gradient accumulation and clipping.

    Computes label-smoothed cross-entropy loss over patient bags of 2.5D tri-slices,
    accumulates gradients over `accumulation_steps`, clips gradients to `max_norm` to
    prevent gradient explosion, and updates parameters.

    Args:
        model: PyTorch MIL model (e.g. ResNet18MIL).
        dataloader: Training DataLoader yielding (bag_tensor, target_tensor).
        optimizer: Differential optimizer (e.g. Adam with parameter groups).
        criterion: Loss module (e.g. nn.CrossEntropyLoss(label_smoothing=0.1)).
        scaler: Optional GradScaler for Automatic Mixed Precision.
        accumulation_steps: Number of batches over which gradients are accumulated (default: 4).
        device: Target execution device ('cuda' or 'cpu').
        max_norm: Maximum norm for gradient clipping (default: 1.0).
        augmenter: Optional data augmentation transform module.

    Returns:
        Average training loss over the epoch.

    Raises:
        FloatingPointError: If loss evaluates to NaN or Inf during training.
    """
    model.train()
    if augmenter is not None:
        augmenter.train()

    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()

    amp_enabled = (torch.device(device).type == "cuda") and (scaler is not None)

    for step_idx, (inputs, targets) in enumerate(dataloader):
        if augmenter is not None:
            inputs = augmenter(inputs)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)

        if amp_enabled and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(logits, targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise FloatingPointError("Loss exploded to NaN or Inf during MIL training step.")
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError("Loss exploded to NaN or Inf during MIL training step.")
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        is_accumulation_step = ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == len(dataloader))
        if is_accumulation_step:
            if amp_enabled and scaler is not None:
                scaler.unscale_(optimizer)
                if max_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(total_samples, 1)


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Execute one evaluation epoch on validation or test dataset without augmentation.

    Args:
        model: PyTorch neural network module.
        dataloader: Evaluation DataLoader.
        criterion: Loss module.
        device: Execution device ('cuda' or 'cpu').

    Returns:
        Tuple of (average_loss, metrics_dictionary, all_y_true, all_y_pred, all_y_prob).
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, targets)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=int)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    y_prob = np.concatenate(all_probs) if all_probs else np.empty((0, 2), dtype=float)

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob, loss_value=avg_loss)

    return avg_loss, metrics, y_true, y_pred, y_prob


def train_3dcnn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 120,
    accumulation_steps: int = 4,
    early_stopping_patience: int = 15,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    use_amp: bool = True,
    val_dataset: Optional[Any] = None,
    early_stopping_metric: str = "composite_score",
    early_stopping_mode: str = "max",
    warmup_epochs: int = 0,
) -> Dict[str, Any]:
    """Train 3D-CNN architecture with early stopping and mixed precision.

    Args:
        model: PyTorch 3D-CNN model instance (SEResNet3D or Baseline3DCNN).
        train_loader: Training DataLoader.
        val_loader: Validation or test DataLoader.
        criterion: Loss function (defaults to FocalLoss(alpha=0.25, gamma=2.0)).
        optimizer: Optimizer (defaults to Adam(lr=1e-5, weight_decay=0.0005)).
        epochs: Maximum training epochs (default: 120).
        accumulation_steps: Gradient accumulation steps (default: 4).
        early_stopping_patience: Early stopping patience (default: 15).
        checkpoint_path: Destination path for saving best model weights.
        device: Target execution device ('cuda' or 'cpu').
        use_amp: Enable Automatic Mixed Precision if GPU is available.
        val_dataset: Optional validation MultiViewDataset for 3-view soft voting evaluation.
        early_stopping_metric: Metric to monitor for early stopping (default: 'composite_score').
        early_stopping_mode: Optimization mode ('max' or 'min').
        warmup_epochs: Suppression window epochs for early stopping (default: 0).

    Returns:
        Dictionary containing training history and best validation metrics.
    """
    exec_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(exec_device)

    opt = optimizer or torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
    loss_fn = criterion or FocalLoss(alpha=0.25, gamma=2.0)
    scaler = torch.amp.GradScaler("cuda") if (torch.device(exec_device).type == "cuda" and use_amp) else None
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-4,
        mode=early_stopping_mode,
        checkpoint_path=checkpoint_path,
        warmup_epochs=warmup_epochs,
    )

    history: Dict[str, List[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_composite_score": [],
    }

    best_metrics: Dict[str, float] = {}

    val_ds = val_dataset

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=opt,
            criterion=loss_fn,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            device=exec_device,
        )

        if val_ds is not None and hasattr(val_ds, "get_subject_views"):
            ensemble_res = evaluate_multiview_ensemble(
                model=model,
                dataset=val_ds,
                device=exec_device,
                criterion=loss_fn,
            )
            v_metrics = ensemble_res["subject_metrics"]
            v_loss = float(v_metrics.get("loss", 0.0))
        else:
            v_loss, v_metrics, _, _, _ = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=loss_fn,
                device=exec_device,
            )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_metrics["accuracy"])
        history["val_f1"].append(v_metrics["f1_score"])
        history["val_composite_score"].append(v_metrics.get("composite_score", v_metrics["f1_score"]))

        if early_stopping_mode == "max":
            monitored_val = v_metrics.get(early_stopping_metric, v_metrics.get("composite_score", v_metrics["accuracy"]))
        else:
            monitored_val = v_loss

        if early_stopping(val_score=monitored_val, model=model, epoch=epoch):
            break
        elif early_stopping.counter == 0:
            best_metrics = v_metrics

    # Restore best checkpoint if saved
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        load_fn = getattr(model, "load", None)
        if callable(load_fn):
            load_fn(checkpoint_path)
        else:
            ckpt_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt_dict, dict) and "state_dict" in ckpt_dict:
                model.load_state_dict(ckpt_dict["state_dict"])
            elif isinstance(ckpt_dict, dict):
                model.load_state_dict(ckpt_dict)
        model.to(exec_device)

    return {
        "model": model,
        "history": history,
        "best_metrics": best_metrics,
        "stopped_epoch": len(history["train_loss"]),
    }


def train_mil_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 100,
    accumulation_steps: int = 4,
    early_stopping_patience: int = 15,
    early_stopping_metric: str = "composite_score",
    early_stopping_mode: str = "max",
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    use_amp: bool = True,
    augmenter: Optional[nn.Module] = None,
    max_norm: float = 1.0,
) -> Dict[str, Any]:
    """Train 2.5D ResNet-18 MIL network with differential LR, cosine warmup, and early stopping.

    Args:
        model: ResNet18MIL network instance.
        train_loader: Training DataLoader delivering bags of tri-slices.
        val_loader: Validation DataLoader delivering bags of tri-slices.
        criterion: Loss function (defaults to CrossEntropyLoss with label_smoothing=0.1).
        optimizer: Optimizer with parameter groups (defaults to configure_differential_optimizer).
        epochs: Maximum training epochs (default: 100).
        accumulation_steps: Gradient accumulation steps (default: 4).
        early_stopping_patience: Number of epochs without improvement before stopping (default: 15).
        early_stopping_metric: Validation metric to monitor (default: 'composite_score').
        early_stopping_mode: Optimization direction ('max' or 'min').
        warmup_epochs: Linear warmup epochs (default: 3).
        min_lr: Floor learning rate at final epoch (default: 1e-6).
        checkpoint_path: Optional destination path for saving best model weights.
        device: Target execution device ('cuda' or 'cpu').
        use_amp: Enable Automatic Mixed Precision if GPU is available.
        augmenter: Optional bag-level data augmentation module.
        max_norm: Maximum gradient norm clipping threshold (default: 1.0).

    Returns:
        Dictionary containing:
        - 'model': Model instance with best weights restored.
        - 'history': Training and validation metric trajectories across epochs.
        - 'best_metrics': Highest validation metrics achieved at best epoch.
        - 'optimal_threshold': Optimal decision threshold calibrated at best epoch.
        - 'calibrated_metrics': Calibrated metrics at optimal threshold at best epoch.
        - 'stopped_epoch': Terminal epoch index reached during training.
    """
    exec_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(exec_device)

    opt = optimizer or configure_differential_optimizer(
        model=model,
        backbone_lr=1e-5,
        head_lr=1e-4,
        backbone_weight_decay=1e-4,
        head_weight_decay=1e-3,
    )
    loss_fn = criterion or nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") if (torch.device(exec_device).type == "cuda" and use_amp) else None
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-4,
        mode=early_stopping_mode,
        checkpoint_path=checkpoint_path,
        warmup_epochs=warmup_epochs,
    )

    history: Dict[str, List[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_composite_score": [],
        "learning_rates": [],
    }

    best_metrics: Dict[str, float] = {}
    best_eval_res: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        current_lrs = update_learning_rates(
            optimizer=opt,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
            min_lr=min_lr,
        )

        train_loss_val = train_mil_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=opt,
            criterion=loss_fn,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            device=exec_device,
            max_norm=max_norm,
            augmenter=augmenter,
        )

        eval_res = evaluate_mil_model(
            model=model,
            dataloader=val_loader,
            criterion=loss_fn,
            device=exec_device,
            calibrate_threshold=True,
        )
        val_metrics = eval_res["metrics"]
        calibrated_metrics = eval_res.get("calibrated_metrics", val_metrics)
        val_loss_val = float(val_metrics.get("loss", 0.0))

        history["train_loss"].append(train_loss_val)
        history["val_loss"].append(val_loss_val)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])
        history["val_composite_score"].append(val_metrics.get("composite_score", val_metrics["f1_score"]))
        history["learning_rates"].append(current_lrs)

        if early_stopping_mode == "max":
            monitored_val = calibrated_metrics.get(
                early_stopping_metric,
                val_metrics.get(
                    early_stopping_metric,
                    val_metrics.get("composite_score", val_metrics["accuracy"]),
                ),
            )
        else:
            monitored_val = val_loss_val

        if early_stopping(val_score=monitored_val, model=model, epoch=epoch):
            break
        elif early_stopping.counter == 0:
            best_metrics = val_metrics
            best_eval_res = eval_res

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        load_fn = getattr(model, "load", None)
        if callable(load_fn):
            load_fn(checkpoint_path)
        else:
            ckpt_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt_dict, dict) and "state_dict" in ckpt_dict:
                model.load_state_dict(ckpt_dict["state_dict"])
            elif isinstance(ckpt_dict, dict):
                model.load_state_dict(ckpt_dict)
        model.to(exec_device)

    return {
        "model": model,
        "history": history,
        "best_metrics": best_metrics,
        "optimal_threshold": best_eval_res.get("optimal_threshold", 0.50),
        "calibrated_metrics": best_eval_res.get("calibrated_metrics", best_metrics),
        "stopped_epoch": len(history["train_loss"]),
    }


def run_stratified_kfold_cv(
    manifest: pd.DataFrame,
    tensors_dir: Union[str, Path],
    n_splits: int = 5,
    epochs: int = 120,
    accumulation_steps: int = 2,
    early_stopping_patience: int = 15,
    device: Optional[str] = None,
    random_state: int = 42,
    use_amp: bool = True,
    model_arch: str = "se_resnet3d",
    decomposed: bool = True,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.001,
    label_smoothing: float = 0.1,
    warmup_epochs: int = 15,
) -> Dict[str, Any]:
    """Execute Stratified 5-Fold Cross-Validation across all cohort subjects.

    Supports Multi-View Plane Decomposition (Option 3) with dynamic 3-view training
    and patient-level soft-voting ensemble validation within each fold.

    Args:
        manifest: Full DataFrame of all cohort subjects with binary 'dx_group'.
        tensors_dir: Directory containing preprocessed .npz Multi-View Tensors.
        n_splits: Number of cross-validation folds (default: 5).
        epochs: Max epochs per fold (default: 120).
        accumulation_steps: Gradient accumulation steps (default: 2).
        early_stopping_patience: Early stopping patience (default: 15).
        device: Device ('cuda' or 'cpu').
        random_state: Seed for reproducible fold generation (default: 42).
        use_amp: Enable Automatic Mixed Precision on GPU.
        model_arch: Architecture to train ('se_resnet3d' or 'baseline_3dcnn').
        decomposed: Enable dynamic multi-view decomposition (triples samples).
        batch_size: Physical batch size (default: 4).
        learning_rate: Optimizer learning rate (default: 1e-5).
        weight_decay: L2 weight decay regularization factor (default: 0.001).
        label_smoothing: Label smoothing regularization factor (default: 0.1).
        warmup_epochs: Early stopping warmup suppression window (default: 15).

    Returns:
        Dictionary containing fold metrics and aggregated summary statistics.
    """
    exec_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    y = manifest["dx_group"].to_numpy()
    X = np.arange(len(manifest))

    augmenter = VolumeConsistentAugmenter()
    fold_results: List[Dict[str, float]] = []

    for fold_idx, (train_indices, val_indices) in enumerate(
        get_kfold_splits(X, y, n_splits=n_splits, random_state=random_state), 1
    ):
        train_sub_df = manifest.iloc[train_indices].reset_index(drop=True)
        val_sub_df = manifest.iloc[val_indices].reset_index(drop=True)

        if decomposed:
            ds_train = MultiViewDataset(
                manifest=train_sub_df,
                tensors_dir=tensors_dir,
                split=None,
                view="decomposed",
                transform=augmenter,
            )
        else:
            ds_train = MultiViewDataset(
                manifest=train_sub_df,
                tensors_dir=tensors_dir,
                split=None,
                transform=augmenter,
            )

        ds_val = MultiViewDataset(manifest=val_sub_df, tensors_dir=tensors_dir, split=None, transform=None)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        if model_arch.lower() in ("se_resnet3d", "seresnet3d"):
            model: nn.Module = SEResNet3D()
        else:
            model = Baseline3DCNN()

        train_labels = ds_train.get_labels()
        class_weights = compute_class_weights(train_labels).to(exec_device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        result = train_3dcnn_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=ds_val if decomposed else None,
            criterion=loss_fn,
            optimizer=opt,
            epochs=epochs,
            accumulation_steps=accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric="composite_score",
            early_stopping_mode="max",
            warmup_epochs=warmup_epochs,
            device=exec_device,
            use_amp=use_amp,
        )

        fold_results.append(result["best_metrics"])

    # Aggregate cross-fold performance
    metric_keys = [
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
    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    return {
        "fold_results": fold_results,
        "summary": summary,
    }


def compute_class_weights(labels: Union[np.ndarray, torch.Tensor, List[int]]) -> torch.Tensor:
    """Compute balanced class weights using inverse-frequency weighting.

    Formula: weight_c = total_samples / (num_classes * count_c)

    Args:
        labels: 1D array-like of binary or multi-class labels.

    Returns:
        1D float32 tensor of class weights.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    classes, counts = np.unique(labels_arr, return_counts=True)
    total_samples = len(labels_arr)
    num_classes = len(classes)

    weights = np.zeros(num_classes, dtype=np.float32)
    for cls, count in zip(classes, counts):
        weights[int(cls)] = total_samples / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


def run_training_pipeline(
    model_type: str = "se_resnet3d",
    data_dir: Optional[Path] = None,
    save_checkpoint: bool = True,
    class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    decomposed: Optional[bool] = None,
    batch_size: int = 4,
    accumulation_steps: int = 2,
    weight_decay: float = 0.001,
    label_smoothing: float = 0.1,
    learning_rate: float = 1e-5,
    early_stopping_patience: int = 15,
    warmup_epochs: int = 15,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute model training workflow based on specified architecture.

    Args:
        model_type: Model architecture ('se_resnet3d', 'baseline_3dcnn', 'svm', 'gcn', 'gat').
        data_dir: Data directory containing manifests or raw acquisitions.
        save_checkpoint: Whether to persist best model checkpoint.
        class_weights: Optional class weights for weighted CrossEntropyLoss.
        decomposed: Enable dynamic multi-view decomposition (Option 3). Defaults to True for SEResNet3D.
        batch_size: Physical batch size (default: 4).
        accumulation_steps: Gradient accumulation steps (default: 2).
        weight_decay: L2 regularization factor (default: 0.001).
        label_smoothing: Label smoothing regularization factor (default: 0.1).
        learning_rate: Initial learning rate (default: 1e-5).
        early_stopping_patience: Patience for early stopping (default: 15).
        warmup_epochs: Suppression window epochs for early stopping (default: 15).
        **kwargs: Additional training arguments.

    Returns:
        Summary dictionary with training results.
    """
    if model_type.lower() in ("3dcnn", "baseline_3dcnn", "se_resnet3d", "seresnet3d"):
        manifest_path = kwargs.pop("manifest_path", "output/abide_nyu_multiview/metadata.csv")
        tensors_dir = kwargs.pop("tensors_dir", "output/abide_nyu_multiview/tensors")
        manifest_df = pd.read_csv(manifest_path)
        augmenter = kwargs.pop("transform", VolumeConsistentAugmenter())

        is_decomposed = (
            decomposed
            if decomposed is not None
            else (model_type.lower() in ("se_resnet3d", "seresnet3d"))
        )

        train_ds = MultiViewDataset(
            manifest=manifest_df,
            tensors_dir=tensors_dir,
            split="train",
            view="decomposed" if is_decomposed else "all",
            transform=augmenter,
        )
        test_ds = MultiViewDataset(
            manifest=manifest_df,
            tensors_dir=tensors_dir,
            split="test",
            transform=None,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        if model_type.lower() in ("se_resnet3d", "seresnet3d"):
            model: nn.Module = SEResNet3D()
            ckpt_name = "best_se_resnet3d.pt"
        else:
            model = Baseline3DCNN()
            ckpt_name = "best_baseline_3dcnn.pt"

        ckpt = Path(f"artifacts/models/{ckpt_name}") if save_checkpoint else None

        criterion = kwargs.pop("criterion", None)
        if criterion is None:
            if class_weights is not None:
                weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            else:
                weight_tensor = compute_class_weights(train_ds.get_labels())
            criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

        optimizer = kwargs.pop("optimizer", None)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        return train_3dcnn_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            val_dataset=test_ds if is_decomposed else None,
            criterion=criterion,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric="composite_score",
            early_stopping_mode="max",
            warmup_epochs=warmup_epochs,
            checkpoint_path=ckpt,
            **kwargs,
        )

    if model_type.lower() in ("resnet_mil", "resnet18_mil", "mil", "resnet18"):
        manifest_path = kwargs.pop("manifest_path", "output/abide_nyu_multiview/metadata.csv")
        tensors_dir = kwargs.pop("tensors_dir", "output/abide_nyu_multiview/tensors")
        slices_per_view = kwargs.pop("slices_per_view", 50)
        central_slices = kwargs.pop("central_slices", 40)
        stride = kwargs.pop("stride", 2)
        pretrained = kwargs.pop("pretrained", True)
        freeze_early = kwargs.pop("freeze_early", True)
        augmenter = kwargs.pop("transform", BagConsistentAugmenter())
        epochs = kwargs.pop("epochs", 100)
        min_lr = kwargs.pop("min_lr", 1e-6)

        mil_train_ds = TriSliceDataset(
            manifest=manifest_path,
            tensors_dir=tensors_dir,
            split="train",
            transform=augmenter,
            slices_per_view=slices_per_view,
            central_slices=central_slices,
            stride=stride,
        )
        mil_test_ds = TriSliceDataset(
            manifest=manifest_path,
            tensors_dir=tensors_dir,
            split="test",
            transform=None,
            slices_per_view=slices_per_view,
            central_slices=central_slices,
            stride=stride,
        )

        train_loader_mil = DataLoader(mil_train_ds, batch_size=batch_size, shuffle=True)
        test_loader_mil = DataLoader(mil_test_ds, batch_size=batch_size, shuffle=False)

        mil_model = ResNet18MIL(pretrained=pretrained, freeze_early=freeze_early)
        ckpt_name = "best_resnet18_mil.pt"
        ckpt = Path(f"artifacts/models/{ckpt_name}") if save_checkpoint else None

        criterion = kwargs.pop("criterion", nn.CrossEntropyLoss(label_smoothing=label_smoothing))
        optimizer = kwargs.pop("optimizer", None)

        return train_mil_model(
            model=mil_model,
            train_loader=train_loader_mil,
            val_loader=test_loader_mil,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            accumulation_steps=accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric="composite_score",
            early_stopping_mode="max",
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            checkpoint_path=ckpt,
            augmenter=augmenter,
            **kwargs,
        )

    if model_type.lower() in (
        "spatiotemporal",
        "spatiotemporal_bimamba",
        "bimamba",
        "3dcnn_bimamba",
    ):
        from asd_classification.pipelines.spatiotemporal_engine import run_spatiotemporal_kfold_cv

        manifest = kwargs.pop("manifest", None)
        if manifest is None and "manifest_path" in kwargs:
            manifest = pd.read_csv(kwargs.pop("manifest_path"))
        dataset = kwargs.pop("dataset", None)
        if dataset is None:
            dataset = kwargs.pop("dataset_or_paths", None)

        return run_spatiotemporal_kfold_cv(
            manifest=manifest,
            dataset_or_paths=dataset,
            **kwargs,
        )

    raise NotImplementedError(f"Training pipeline for '{model_type}' is not yet implemented.")
