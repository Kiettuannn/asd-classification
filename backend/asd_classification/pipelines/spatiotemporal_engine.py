"""Multi-GPU Distributed Training Engine with BF16 Mixed Precision, Focal Loss, and Clinical Metrics.

Coordinates training and evaluation for the Spatio-Temporal 3D-CNN + Bidirectional Mamba architecture:
- Multi-GPU orchestration supporting 4x NVIDIA L4 GPUs via DataParallel (DP) or DistributedDataParallel (DDP)
- Native BF16 mixed precision execution using torch.amp.autocast(dtype=torch.bfloat16)
- Focal Loss (alpha=0.25, gamma=2.0) with fold label distribution class weighting
- AdamW optimizer with cosine annealing learning rate scheduler and configurable gradient clipping
- Early stopping monitoring validation loss with configurable epoch patience
- Comprehensive fold-level clinical metrics: Balanced Accuracy, ROC-AUC, Sensitivity, Specificity, F1-Score, Loss
- Best model checkpoint persistence per fold based on validation Balanced Accuracy and ROC-AUC
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from asd_classification.data.spatiotemporal import (
    SpatioTemporalFMRIDataset,
    get_site_and_class_stratified_folds,
    spatiotemporal_collate_fn,
)
from asd_classification.models.baseline.focal_loss import FocalLoss
from asd_classification.models.baseline.spatiotemporal_bimamba import SpatioTemporalBiMamba
from asd_classification.pipelines.evaluate import compute_classification_metrics


def resolve_device_and_dtype(
    device: Optional[Union[str, torch.device]] = None,
    use_bf16: bool = True,
) -> Tuple[torch.device, torch.dtype]:
    """Resolve target execution hardware device and mixed-precision data type.

    Args:
        device: Target execution device. If None, automatically selects available CUDA GPU
            or defaults to CPU.
        use_bf16: When True, selects torch.bfloat16 for native BF16 mixed precision
            supported on NVIDIA L4 GPUs; otherwise defaults to torch.float32.

    Returns:
        Tuple of (resolved_device, target_dtype).
    """
    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    target_dtype = torch.bfloat16 if use_bf16 else torch.float32
    return target_device, target_dtype


def unwrap_model(model: nn.Module) -> nn.Module:
    """Retrieve the core underlying nn.Module from parallel wrappers (DataParallel or DDP).

    Args:
        model: PyTorch model, potentially wrapped in DataParallel or DistributedDataParallel.

    Returns:
        Underlying unwrapped nn.Module instance.
    """
    if hasattr(model, "module"):
        return cast(nn.Module, getattr(model, "module"))
    return model


def setup_model_parallelism(
    model: nn.Module,
    device: torch.device,
    strategy: str = "auto",
) -> Tuple[nn.Module, bool]:
    """Configure multi-GPU execution parallelism across visible accelerators.

    Supports:
    - 'auto': Checks if distributed process group is initialized (DDP), else wraps in
      DataParallel if multiple CUDA devices are available, else runs on single device.
    - 'ddp': Enforces PyTorch DistributedDataParallel wrapping.
    - 'dp': Enforces PyTorch DataParallel wrapping across all available GPUs.
    - 'none': Single-device execution without parallel wrapping.

    Args:
        model: Base neural network module.
        device: Primary hardware execution device.
        strategy: Parallel execution strategy ('auto' | 'ddp' | 'dp' | 'none').

    Returns:
        Tuple of (parallel_model, is_parallel_boolean).
    """
    model = model.to(device)

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if strategy == "none":
        return model, False

    if strategy == "ddp" or (strategy == "auto" and is_distributed):
        local_rank = device.index if device.index is not None else 0
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        return ddp_model, True

    if strategy == "dp" or (strategy == "auto" and not is_distributed and device.type == "cuda" and cuda_devices > 1):
        dp_model = nn.DataParallel(model)
        return dp_model, True

    return model, False


def compute_fold_class_weights(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Calculate fold class weights from fold label distributions for Focal Loss.

    Computes inverse-frequency class weights:
        w_c = N_total / (2 * N_c)
    Optionally modulated by focal alpha balancing factor if alpha is provided.

    Args:
        labels: 1D sequence or array of binary ground-truth labels (0=TC, 1=ASD).
        alpha: Optional focal loss alpha balancing factor. If None, computes
            pure inverse class frequency weights.

    Returns:
        1D float32 torch.Tensor of shape (2,) containing per-class weights.
    """
    if isinstance(labels, torch.Tensor):
        arr = labels.detach().cpu().numpy().ravel()
    else:
        arr = np.asarray(labels).ravel()

    n_samples = len(arr)
    if n_samples == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    count_0 = int(np.sum(arr == 0))
    count_1 = int(np.sum(arr == 1))

    # Guard against folds with zero samples of a class in extreme edge cases
    effective_c0 = max(count_0, 1)
    effective_c1 = max(count_1, 1)

    if alpha is not None:
        raw_w0 = (1.0 - alpha) / float(effective_c0)
        raw_w1 = alpha / float(effective_c1)
        total_w = raw_w0 + raw_w1
        w0_norm = 2.0 * raw_w0 / total_w
        w1_norm = 2.0 * raw_w1 / total_w
        return torch.tensor([w0_norm, w1_norm], dtype=torch.float32)

    # Standard balanced inverse frequency: N_total / (num_classes * N_c)
    w0 = float(n_samples) / (2.0 * float(effective_c0))
    w1 = float(n_samples) / (2.0 * float(effective_c1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def configure_spatiotemporal_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    """Configure AdamW optimizer for Spatio-Temporal 3D-CNN + Bi-Mamba model.

    Args:
        model: PyTorch model instance.
        lr: Base initial learning rate (default: 1e-4).
        weight_decay: L2 weight decay regularization parameter (default: 1e-4).

    Returns:
        Configured torch.optim.AdamW optimizer.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def configure_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Configure Cosine Annealing learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        epochs: Total training epochs (T_max).
        min_lr: Minimum learning rate floor (default: 1e-6).

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR instance.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=min_lr,
    )


class SpatiotemporalEarlyStopping:
    """Early stopping monitor evaluating validation loss and saving top-performing checkpoints.

    Terminates training if validation loss fails to improve for `patience` consecutive epochs.
    Checkpoints the best model weights based on composite clinical validation score
    (0.5 * Balanced Accuracy + 0.5 * ROC-AUC).
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize SpatiotemporalEarlyStopping.

        Args:
            patience: Maximum consecutive non-improving epochs before stopping (default: 15).
            min_delta: Minimum loss decrease required to count as improvement (default: 1e-4).
            checkpoint_path: Optional file path to persist best model weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

        self.counter = 0
        self.best_loss = float("inf")
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.early_stop = False

    def __call__(
        self,
        val_loss: float,
        val_metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        epoch: int = 1,
    ) -> bool:
        """Evaluate validation loss and metrics to update stopping counter and checkpoint.

        Args:
            val_loss: Validation loss for current epoch.
            val_metrics: Dictionary containing validation evaluation metrics.
            model: Model to checkpoint upon metric improvement.
            epoch: Current 1-based training epoch.

        Returns:
            True if early stopping has triggered, False otherwise.
        """
        loss_val = float(val_loss)
        bal_acc = float(val_metrics.get("balanced_accuracy", 0.0))
        roc_auc = float(val_metrics.get("roc_auc", 0.0))
        if math.isnan(roc_auc):
            roc_auc = bal_acc
        composite_score = 0.5 * (bal_acc + roc_auc)

        # Check for loss improvement
        if loss_val < (self.best_loss - self.min_delta):
            self.best_loss = loss_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # Check for composite clinical metric improvement to update checkpoint
        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_epoch = epoch
            if self.checkpoint_path is not None and model is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                core_model = unwrap_model(model)
                save_fn = getattr(core_model, "save", None)
                if callable(save_fn):
                    save_fn(self.checkpoint_path)
                else:
                    torch.save(core_model.state_dict(), self.checkpoint_path)

        return self.early_stop


def train_spatiotemporal_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_bf16: bool = True,
    gradient_clip_norm: float = 1.0,
    accumulation_steps: int = 1,
) -> float:
    """Execute one training epoch for Spatio-Temporal 3D-CNN + Bi-Mamba model.

    Coordinates:
    - Native BF16 mixed precision via torch.amp.autocast(dtype=torch.bfloat16)
    - Gradient accumulation across accumulation_steps
    - Configurable gradient clipping to avoid gradient explosion

    Args:
        model: PyTorch model module (potentially wrapped in DataParallel / DDP).
        dataloader: Training DataLoader delivering batches of (volumes, masks, targets) or (volumes, targets).
        optimizer: PyTorch optimizer instance (e.g. AdamW).
        criterion: Loss criterion (e.g. FocalLoss).
        device: Target execution device.
        use_bf16: Enable native BF16 mixed precision autocasting (default: True).
        gradient_clip_norm: Maximum norm for gradient clipping (default: 1.0).
        accumulation_steps: Gradient accumulation step count (default: 1).

    Returns:
        Average training loss across the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()

    device_type = "cuda" if device.type == "cuda" else "cpu"

    for step_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            volumes = batch["volume"]
            masks = batch.get("mask", None)
            targets = batch["label"]
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            volumes, masks, targets = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            volumes, targets = batch
            masks = None
        else:
            raise ValueError(f"Unsupported batch format from DataLoader: {type(batch)}")

        volumes = volumes.to(device, non_blocking=True)
        if masks is not None:
            masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = targets.size(0)

        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(volumes, mask=masks)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = criterion(logits.float(), targets)
        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError("Loss evaluated to NaN or Inf during training step.")
        scaled_loss = loss / max(accumulation_steps, 1)

        scaled_loss.backward()

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        # Guard against gradient NaN/Inf corruption under mixed precision
        valid_grads = True
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                valid_grads = False
                break

        is_acc_step = ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == len(dataloader))
        if is_acc_step:
            if valid_grads:
                if gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(total_samples, 1)


def evaluate_spatiotemporal_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_bf16: bool = True,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Execute validation or evaluation epoch calculating clinical diagnostic metrics.

    Args:
        model: PyTorch model module.
        dataloader: Evaluation DataLoader yielding batches.
        criterion: Loss criterion (e.g. FocalLoss).
        device: Target execution device.
        use_bf16: Enable native BF16 mixed precision autocasting (default: True).

    Returns:
        Tuple of (average_loss, metrics_dictionary, y_true, y_pred, y_prob).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    device_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                volumes = batch["volume"]
                masks = batch.get("mask", None)
                targets = batch["label"]
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                volumes, masks, targets = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                volumes, targets = batch
                masks = None
            else:
                raise ValueError(f"Unsupported batch format from DataLoader: {type(batch)}")

            volumes = volumes.to(device, non_blocking=True)
            if masks is not None:
                masks = masks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = targets.size(0)

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(volumes, mask=masks)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits.float(), targets)

            probs = torch.softmax(logits.float(), dim=-1)
            preds = torch.argmax(logits, dim=-1)

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=int)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    y_prob = np.concatenate(all_probs) if all_probs else np.empty((0, 2), dtype=float)

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob, loss_value=avg_loss)
    # Ensure Sensitivity is explicitly reported alongside recall per clinical convention
    metrics["sensitivity"] = metrics.get("recall", 0.0)

    return avg_loss, metrics, y_true, y_pred, y_prob


def run_spatiotemporal_kfold_cv(
    manifest: pd.DataFrame,
    dataset_or_paths: Union[Dataset, Sequence[Union[str, Path]]],
    n_splits: int = 5,
    epochs: int = 30,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    early_stopping_patience: int = 15,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    use_bf16: bool = True,
    parallel_strategy: str = "auto",
    random_state: int = 42,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute Site-and-Class Stratified 5-Fold Cross-Validation for Spatio-Temporal Bi-Mamba.

    Orchestrates:
    - Joint Site-and-Class Stratified 5-Fold dataset partitioning
    - Weighted Focal Loss per fold based on training fold label distribution
    - Multi-GPU model wrapping and BF16 execution
    - AdamW optimization with Cosine Annealing scheduler and gradient clipping
    - Early stopping monitoring validation loss
    - Best model checkpoint persistence per fold based on validation Balanced Accuracy and ROC-AUC
    - Out-of-fold predictions aggregation and cross-fold metric summary

    Args:
        manifest: Phenotypic manifest DataFrame containing 'subject_id', 'label', and 'site'.
        dataset_or_paths: SpatioTemporalFMRIDataset or sequence of data paths.
        n_splits: Number of cross-validation folds (default: 5).
        epochs: Maximum training epochs per fold (default: 30).
        batch_size: DataLoader physical batch size (default: 2).
        learning_rate: Initial AdamW learning rate (default: 1e-4).
        weight_decay: L2 weight decay regularization factor (default: 1e-4).
        gradient_clip_norm: Gradient clipping threshold (default: 1.0).
        early_stopping_patience: Early stopping patience window (default: 15).
        checkpoint_dir: Directory where best fold weights are saved (default: 'artifacts/models/spatiotemporal').
        device: Hardware device. If None, auto-selects GPU/CPU.
        use_bf16: Enable native BF16 mixed precision (default: True).
        parallel_strategy: Strategy for multi-GPU acceleration ('auto', 'dp', 'ddp', 'none').
        random_state: Random seed for reproducible stratification (default: 42).
        model_kwargs: Optional architectural overrides for SpatioTemporalBiMamba.

    Returns:
        Dictionary containing fold results, aggregated summary statistics, and out-of-fold predictions.
    """
    exec_device, _ = resolve_device_and_dtype(device=device, use_bf16=use_bf16)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("artifacts/models/spatiotemporal")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Detect site and label columns dynamically
    site_col = "site_id" if "site_id" in manifest.columns else "site"
    label_col = "label" if "label" in manifest.columns else "dx_group"

    folds = get_site_and_class_stratified_folds(
        df=manifest,
        n_splits=n_splits,
        random_state=random_state,
        site_col=site_col,
        label_col=label_col,
    )

    fold_metrics_list: List[Dict[str, float]] = []
    out_of_fold_preds: List[int] = []
    out_of_fold_targets: List[int] = []
    out_of_fold_probs: List[List[float]] = []

    m_kwargs = model_kwargs or {}

    for fold_idx, (train_indices, val_indices) in enumerate(folds, 1):
        train_labels = manifest.iloc[train_indices][label_col].to_numpy()
        class_weights = compute_fold_class_weights(train_labels, alpha=0.25).to(exec_device)

        criterion = FocalLoss(alpha=class_weights, gamma=2.0)

        # Instantiate fresh architecture instance per fold to eliminate cross-fold weight leakage
        model = SpatioTemporalBiMamba(**m_kwargs)
        parallel_model, _ = setup_model_parallelism(model, device=exec_device, strategy=parallel_strategy)

        optimizer = configure_spatiotemporal_optimizer(parallel_model, lr=learning_rate, weight_decay=weight_decay)
        scheduler = configure_cosine_scheduler(optimizer, epochs=epochs, min_lr=1e-6)

        fold_ckpt_path = ckpt_dir / f"best_spatiotemporal_fold_{fold_idx}.pt"
        early_stopping = SpatiotemporalEarlyStopping(
            patience=early_stopping_patience,
            checkpoint_path=fold_ckpt_path,
        )

        # Create disjoint train and validation subsets for the current fold partition
        if isinstance(dataset_or_paths, Dataset):
            train_subset = torch.utils.data.Subset(dataset_or_paths, train_indices.tolist())
            val_subset = torch.utils.data.Subset(dataset_or_paths, val_indices.tolist())
        else:
            raise ValueError("dataset_or_paths must be an instantiated Dataset instance.")

        collate = getattr(dataset_or_paths, "collate_fn", None)
        if collate is None:
            try:
                sample_item = dataset_or_paths[0]
                if isinstance(sample_item, dict):
                    collate = spatiotemporal_collate_fn
            except Exception:
                collate = None

        loader_kwargs: Dict[str, Any] = {"batch_size": batch_size}
        if collate is not None:
            loader_kwargs["collate_fn"] = collate

        # Support DistributedSampler when running in distributed multi-GPU mode
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        train_sampler: Optional[torch.utils.data.distributed.DistributedSampler] = (
            torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=True)
            if is_distributed
            else None
        )

        drop_last = len(train_subset) > batch_size and (len(train_subset) % batch_size == 1)
        train_loader = DataLoader(
            train_subset,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=drop_last,
            **loader_kwargs,
        )
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_spatiotemporal_epoch(
                model=parallel_model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=exec_device,
                use_bf16=use_bf16,
                gradient_clip_norm=gradient_clip_norm,
            )

            val_loss, val_metrics, _, _, _ = evaluate_spatiotemporal_epoch(
                model=parallel_model,
                dataloader=val_loader,
                criterion=criterion,
                device=exec_device,
                use_bf16=use_bf16,
            )

            scheduler.step()

            if early_stopping(val_loss=val_loss, val_metrics=val_metrics, model=parallel_model, epoch=epoch):
                break

        # Restore top-performing weights achieved on composite validation metrics
        if fold_ckpt_path.exists():
            core_model = unwrap_model(parallel_model)
            core_model.load_state_dict(torch.load(fold_ckpt_path, map_location="cpu", weights_only=False))
            core_model.to(exec_device)

        _, final_metrics, y_true_fold, y_pred_fold, y_prob_fold = evaluate_spatiotemporal_epoch(
            model=parallel_model,
            dataloader=val_loader,
            criterion=criterion,
            device=exec_device,
            use_bf16=use_bf16,
        )

        fold_metrics_list.append(final_metrics)
        out_of_fold_targets.extend(y_true_fold.tolist())
        out_of_fold_preds.extend(y_pred_fold.tolist())
        out_of_fold_probs.extend(y_prob_fold.tolist())

    # Aggregate metric summary across all folds
    metric_keys = [
        "balanced_accuracy",
        "roc_auc",
        "sensitivity",
        "recall",
        "specificity",
        "f1_score",
        "loss",
        "accuracy",
        "precision",
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        vals = [f[key] for f in fold_metrics_list if key in f and not math.isnan(f[key])]
        if vals:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    oof_targets_arr = np.array(out_of_fold_targets)
    oof_preds_arr = np.array(out_of_fold_preds)
    oof_probs_arr = np.array(out_of_fold_probs)
    overall_oof_metrics = compute_classification_metrics(
        oof_targets_arr,
        oof_preds_arr,
        y_prob=oof_probs_arr,
    )

    return {
        "fold_results": fold_metrics_list,
        "summary": summary,
        "overall_oof_metrics": overall_oof_metrics,
        "oof_targets": oof_targets_arr,
        "oof_preds": oof_preds_arr,
        "oof_probs": oof_probs_arr,
    }
