"""Model evaluation, ROC curve generation, and benchmark comparison routines.

Implements clinical evaluation metrics (Accuracy, Precision, Recall, F1-Score,
Focal Loss, ROC-AUC) and comparative analysis against published benchmarks from
Hammash & Younis (MDPI Journal of Imaging, 2026, 12, 109).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn.functional as F


# Published benchmarks for rs-fMRI multi-view 50 slices (Hammash & Younis 2026, Table 8)
PUBLISHED_TABLE_8_BENCHMARKS = {
    "Accuracy": 0.8124,
    "Precision": 0.8023,
    "Recall": 0.7905,
    "F1-Score": 0.7963,
    "Focal Loss": 0.4932,
}


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    loss_value: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate diagnostic evaluation metrics for binary ASD classification.

    Computes:
    - Accuracy: Overall classification correctness.
    - Precision: Positive predictive value for ASD detection.
    - Recall (Sensitivity): Proportion of true ASD cases identified.
    - F1-Score: Harmonic mean of precision and recall.
    - ROC-AUC: Area under Receiver Operating Characteristic curve (if probabilities provided).
    - Loss: Mean evaluated loss value (if provided).

    Args:
        y_true: Ground-truth binary labels (1=ASD, 0=TC).
        y_pred: Predicted discrete binary labels (1=ASD, 0=TC).
        y_prob: Optional predicted class probabilities (1D positive class or 2D distribution).
        loss_value: Optional average loss value evaluated over the sample set.

    Returns:
        Dictionary mapping metric names to computed floating point values.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true_np = np.asarray(y_true).astype(int).ravel()
    y_pred_np = np.asarray(y_pred).astype(int).ravel()

    acc = float(accuracy_score(y_true_np, y_pred_np))
    prec = float(precision_score(y_true_np, y_pred_np, zero_division=0))
    rec = float(recall_score(y_true_np, y_pred_np, zero_division=0))
    f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))

    tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))
    fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float(0.5 * (rec + specificity))
    g_mean = float(np.sqrt(max(0.0, rec * specificity)))
    composite = float(0.5 * balanced_acc + 0.5 * g_mean)
    table8_composite = float(0.5 * balanced_acc + 0.5 * f1)

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "g_mean": g_mean,
        "composite_score": composite,
        "table8_composite": table8_composite,
    }

    if loss_value is not None:
        metrics["loss"] = float(loss_value)

    if y_prob is not None:
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.detach().cpu().numpy()
        y_prob_np = np.asarray(y_prob)

        # Extract positive class probabilities if 2D
        if y_prob_np.ndim == 2 and y_prob_np.shape[1] >= 2:
            scores = y_prob_np[:, 1]
        else:
            scores = y_prob_np.ravel()

        # ROC-AUC is defined only when both positive and negative classes are present
        if len(np.unique(y_true_np)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_np, scores))
            except ValueError:
                metrics["roc_auc"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")

    return metrics


def format_benchmark_comparison(metrics: Dict[str, float]) -> pd.DataFrame:
    """Format evaluated metrics alongside published Table 8 benchmark metrics.

    Compares the current model run against published results from Hammash & Younis (2026):
    - Accuracy: 81.24%
    - Precision: 80.23%
    - Recall: 79.05%
    - F1-Score: 79.63%
    - Focal Loss: 0.4932

    Args:
        metrics: Dictionary containing computed model performance metrics.

    Returns:
        Formatted pandas DataFrame comparing current results against published baselines.
    """
    rows = []

    # Map internal metric keys to display names and percentage formatting flags
    metric_mappings = [
        ("Accuracy", "accuracy", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1-Score", "f1_score", True),
        ("Focal Loss", "loss", False),
    ]

    for display_name, key, is_percentage in metric_mappings:
        baseline_val = PUBLISHED_TABLE_8_BENCHMARKS.get(display_name)
        current_val = metrics.get(key)

        if current_val is not None and baseline_val is not None:
            delta = current_val - baseline_val
            if is_percentage:
                curr_str = f"{current_val * 100:.2f}%"
                base_str = f"{baseline_val * 100:.2f}%"
                delta_sign = "+" if delta >= 0 else ""
                delta_str = f"{delta_sign}{delta * 100:.2f}%"
            else:
                curr_str = f"{current_val:.4f}"
                base_str = f"{baseline_val:.4f}"
                delta_sign = "+" if delta >= 0 else ""
                delta_str = f"{delta_sign}{delta:.4f}"
        else:
            curr_str = f"{current_val:.4f}" if current_val is not None else "N/A"
            base_str = f"{baseline_val:.4f}" if baseline_val is not None else "N/A"
            delta_str = "N/A"

        rows.append({
            "Metric": display_name,
            "Current Result": curr_str,
            "Published Baseline (Table 8)": base_str,
            "Delta": delta_str,
        })

    return pd.DataFrame(rows)


def plot_roc_curve(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path] = "/kaggle/working/roc_curve.png",
    title: str = "ROC Curve - Baseline 3D-CNN Multi-View fMRI",
) -> Tuple[float, Path]:
    """Generate and save Receiver Operating Characteristic (ROC) curve plot with AUC.

    Args:
        y_true: Ground-truth binary labels (1=ASD, 0=TC).
        y_prob: Predicted class probabilities for positive class (ASD=1).
        output_path: Filesystem destination path for the saved PNG figure.
        title: Title string for the generated plot.

    Returns:
        Tuple of (calculated AUC score, Path to saved figure).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true_np = np.asarray(y_true).astype(int).ravel()
    y_prob_np = np.asarray(y_prob)
    if y_prob_np.ndim == 2 and y_prob_np.shape[1] >= 2:
        scores = y_prob_np[:, 1]
    else:
        scores = y_prob_np.ravel()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Compute ROC curve and AUC
    if len(np.unique(y_true_np)) > 1:
        fpr, tpr, _ = roc_curve(y_true_np, scores)
        auc_score = float(roc_auc_score(y_true_np, scores))
    else:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc_score = 0.5

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2.5,
        label=f"Baseline 3D-CNN (AUC = {auc_score:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")

    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.02, 1.05))
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    return auc_score, out_file


def find_optimal_threshold(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    min_thresh: float = 0.30,
    max_thresh: float = 0.60,
    num_steps: int = 61,
) -> float:
    """Find the optimal binary decision threshold maximizing the Table 8 Composite score.

    Calibrates decision threshold on predicted probabilities to balance Sensitivity (Recall)
    and Specificity while directly targeting published Table 8 benchmark metrics:
        threshold^* = argmax_{tau in [min_thresh, max_thresh]} Table8Composite(tau)
    where Table8Composite = 0.5 * BalancedAccuracy + 0.5 * F1Score.

    Ties are broken in favor of higher Recall to minimize clinically missed ASD diagnoses.

    Args:
        y_true: Ground-truth binary targets (1=ASD, 0=TC).
        y_prob: Predicted class probabilities (either (N, 2) or (N,)).
        min_thresh: Lower search bound for candidate threshold (default: 0.30).
        max_thresh: Upper search bound for candidate threshold (default: 0.60).
        num_steps: Number of uniformly spaced candidate threshold steps (default: 61, step 0.005).

    Returns:
        Optimal decision threshold float. Falls back to 0.5 if inputs are degenerate.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true_np = np.asarray(y_true).astype(int).ravel()
    y_prob_np = np.asarray(y_prob)
    if y_prob_np.ndim == 2 and y_prob_np.shape[1] >= 2:
        scores = y_prob_np[:, 1]
    else:
        scores = y_prob_np.ravel()

    if len(y_true_np) == 0 or len(np.unique(y_true_np)) < 2:
        return 0.5

    thresholds = np.linspace(min_thresh, max_thresh, num_steps)
    best_thresh = 0.5
    best_score = -1.0
    best_recall = -1.0
    min_dist_to_half = float("inf")

    for tau in thresholds:
        preds = (scores >= tau).astype(int)
        metrics = compute_classification_metrics(y_true_np, preds, scores)
        # Table 8 Composite objective: 0.5 * Balanced Accuracy + 0.5 * F1-Score
        score = float(0.5 * metrics["balanced_accuracy"] + 0.5 * metrics["f1_score"])
        rec = metrics["recall"]
        dist = abs(tau - 0.5)

        if score > best_score + 1e-5:
            best_score = score
            best_thresh = float(tau)
            best_recall = rec
            min_dist_to_half = dist
        elif abs(score - best_score) <= 1e-5:
            # Tie-breaking rule 1: Prefer higher Recall (clinical safety)
            if rec > best_recall + 1e-5:
                best_score = score
                best_thresh = float(tau)
                best_recall = rec
                min_dist_to_half = dist
            elif abs(rec - best_recall) <= 1e-5 and dist < min_dist_to_half:
                # Tie-breaking rule 2: Closer to 0.50
                best_thresh = float(tau)
                min_dist_to_half = dist

    return round(best_thresh, 4)


def aggregate_soft_voting_probabilities(
    view_probs: Union[Dict[str, np.ndarray], np.ndarray],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate multi-view probabilities via soft-voting ensemble.

    Computes patient diagnostic probabilities by taking the arithmetic mean of
    predicted class probabilities across the three orthogonal anatomical planes
    (Axial, Coronal, Sagittal):
        P_subject(ASD) = (P_axial + P_coronal + P_sagittal) / 3

    This cancels uncorrelated per-view prediction variance and provides stable
    subject-level diagnostic decisions.

    Args:
        view_probs: Either a dictionary with keys ('axial', 'coronal', 'sagittal')
            mapping to arrays of shape (N, 2), or a 3D numpy array of shape (N, 3, 2).
        threshold: Decision threshold for positive class (ASD=1) classification (default: 0.5).

    Returns:
        Tuple of (subject_probabilities of shape (N, 2), subject_predictions of shape (N,)).
    """
    if isinstance(view_probs, dict):
        prob_stack = np.stack(
            [view_probs["axial"], view_probs["coronal"], view_probs["sagittal"]],
            axis=1,
        )
    else:
        prob_stack = np.asarray(view_probs)

    # Average along view dimension (axis 1)
    subject_probs = np.mean(prob_stack, axis=1)
    if abs(threshold - 0.5) < 1e-6 and subject_probs.shape[1] == 2:
        subject_preds = np.argmax(subject_probs, axis=1)
    else:
        asd_prob = subject_probs[:, 1] if subject_probs.shape[1] >= 2 else subject_probs.ravel()
        subject_preds = (asd_prob >= threshold).astype(int)

    return subject_probs, subject_preds


def evaluate_multiview_ensemble(
    model: torch.nn.Module,
    dataset: Any,
    device: Optional[str] = None,
    criterion: Optional[torch.nn.Module] = None,
    threshold: Optional[float] = None,
    calibrate_threshold: bool = True,
) -> Dict[str, Any]:
    """Execute dual-level evaluation of 3D-CNN using 3-view soft-voting ensemble.

    For each subject in the dataset:
    1. Evaluates Axial, Coronal, and Sagittal 50-slice sub-volumes independently.
    2. Computes per-view softmax diagnostic probabilities.
    3. Aggregates per-view probabilities via soft-voting average to derive subject-level diagnosis.
    4. Computes comprehensive diagnostic metrics at both Subject Level (at standard and calibrated thresholds) and View Level.

    Args:
        model: Trained 3D-CNN model (SEResNet3D or Baseline3DCNN).
        dataset: MultiViewDataset containing subject tensors and labels.
        device: Target execution device ('cuda' or 'cpu').
        criterion: Optional loss function for computing evaluated loss.
        threshold: Optional custom decision threshold (default: 0.5).
        calibrate_threshold: Whether to search for and evaluate the optimal threshold tau*.

    Returns:
        Dictionary containing:
        - 'subject_metrics': Subject-level diagnostic metrics at default threshold.
        - 'calibrated_metrics': Diagnostic metrics at optimal threshold tau* (if calibrated).
        - 'optimal_threshold': Calculated optimal decision threshold float.
        - 'view_metrics': Per-view metrics dictionary ('axial', 'coronal', 'sagittal').
        - 'subject_probabilities': Array of shape (N, 2) with ensembled probabilities.
        - 'subject_predictions': Array of shape (N,) with discrete class predictions.
        - 'subject_targets': Array of shape (N,) with ground-truth binary labels.
        - 'view_probabilities': Dictionary of per-view probability arrays.
    """
    exec_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(exec_device)
    model.eval()

    n_subjects = len(dataset.manifest_df) if hasattr(dataset, "manifest_df") else len(dataset)
    axial_probs_list = []
    coronal_probs_list = []
    sagittal_probs_list = []
    labels_list = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(n_subjects):
            views_dict, label = dataset.get_subject_views(i)
            labels_list.append(label)

            # Stack 3 views for efficient batch inference of shape (3, 1, D_view, H, W)
            batch_views = torch.stack(
                [views_dict["axial"], views_dict["coronal"], views_dict["sagittal"]],
                dim=0,
            ).to(exec_device)

            logits = model(batch_views)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

            axial_probs_list.append(probs[0])
            coronal_probs_list.append(probs[1])
            sagittal_probs_list.append(probs[2])

            if criterion is not None:
                target_tensor = torch.tensor([label, label, label], device=exec_device, dtype=torch.long)
                loss_val = criterion(logits, target_tensor)
                total_loss += float(loss_val.item()) * 3
                total_samples += 3

    view_probs = {
        "axial": np.array(axial_probs_list, dtype=np.float32),
        "coronal": np.array(coronal_probs_list, dtype=np.float32),
        "sagittal": np.array(sagittal_probs_list, dtype=np.float32),
    }
    y_true = np.array(labels_list, dtype=np.int64)

    # 1. Subject-level soft voting aggregation (default threshold)
    default_thresh = threshold if threshold is not None else 0.5
    subject_probs, subject_preds = aggregate_soft_voting_probabilities(view_probs, threshold=default_thresh)
    mean_loss = (total_loss / total_samples) if total_samples > 0 else None
    subject_metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=subject_preds,
        y_prob=subject_probs,
        loss_value=mean_loss,
    )

    # 2. Optimal threshold calibration (if requested)
    calibrated_metrics = None
    opt_thresh = default_thresh
    if calibrate_threshold:
        opt_thresh = find_optimal_threshold(y_true, subject_probs)
        _, calib_preds = aggregate_soft_voting_probabilities(view_probs, threshold=opt_thresh)
        calibrated_metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=calib_preds,
            y_prob=subject_probs,
            loss_value=mean_loss,
        )
        calibrated_metrics["threshold"] = opt_thresh

    # 3. View-level independent metrics
    view_metrics = {}
    for v_name in ["axial", "coronal", "sagittal"]:
        v_prob = view_probs[v_name]
        v_pred = np.argmax(v_prob, axis=1)
        view_metrics[v_name] = compute_classification_metrics(
            y_true=y_true,
            y_pred=v_pred,
            y_prob=v_prob,
        )

    res = {
        "subject_metrics": subject_metrics,
        "view_metrics": view_metrics,
        "subject_probabilities": subject_probs,
        "subject_predictions": subject_preds,
        "subject_targets": y_true,
        "view_probabilities": view_probs,
        "optimal_threshold": opt_thresh,
    }
    if calibrated_metrics is not None:
        res["calibrated_metrics"] = calibrated_metrics
    return res


def evaluate_mil_model(
    model: torch.nn.Module,
    dataloader: Any,
    criterion: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
    threshold: Optional[float] = 0.50,
    calibrate_threshold: bool = True,
) -> Dict[str, Any]:
    """Execute evaluation for 2.5D ResNet-18 MIL model.

    Evaluates a trained MIL network across patient bags:
    1. Computes predictions, class probabilities, and diagnostic loss.
    2. Extracts slice-level attention weights for anatomical interpretability.
    3. Evaluates diagnostic metrics (accuracy, precision, recall, f1, specificity,
       balanced accuracy, composite score, loss, roc_auc) at standard threshold (tau=0.50).
    4. Calibrates optimal decision threshold tau* in [0.30, 0.60] maximizing Table 8
       composite score with Recall-priority tie-breaking.

    Args:
        model: Trained MIL model (e.g. ResNet18MIL).
        dataloader: Evaluation DataLoader or Dataset yielding (bag_tensors, target_tensors).
        criterion: Optional loss module (e.g. nn.CrossEntropyLoss).
        device: Target execution device ('cuda' or 'cpu').
        threshold: Standard decision threshold (default: 0.50).
        calibrate_threshold: Boolean flag whether to search for optimal threshold tau*.

    Returns:
        Dictionary containing:
        - 'metrics': Diagnostic metrics dictionary at standard decision threshold.
        - 'calibrated_metrics': Diagnostic metrics at calibrated optimal threshold tau*.
        - 'optimal_threshold': Optimal decision threshold float.
        - 'attention_weights': Numpy array of shape (N, K) with slice attention weights.
        - 'probabilities': Numpy array of shape (N, 2) with predicted class probabilities.
        - 'predictions': Discrete binary predictions at standard threshold.
        - 'calibrated_predictions': Discrete binary predictions at calibrated threshold.
        - 'targets': Ground-truth binary labels array.
        - 'loss': Average evaluated loss value.
    """
    exec_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(exec_device)
    model.eval()

    if isinstance(dataloader, torch.utils.data.DataLoader):
        eval_dataloader = dataloader
    else:
        eval_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=4, shuffle=False)

    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_attns: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in eval_dataloader:
            inputs = inputs.to(exec_device, non_blocking=True)
            targets = targets.to(exec_device, non_blocking=True)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
                if len(outputs) > 1 and outputs[1] is not None:
                    all_attns.append(outputs[1].detach().cpu().numpy())
            else:
                logits = outputs

            probs = F.softmax(logits, dim=-1)

            if criterion is not None:
                loss_val = criterion(logits, targets)
                total_loss += float(loss_val.item()) * batch_size
                total_samples += batch_size

            all_targets.append(targets.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=int)
    y_prob = np.concatenate(all_probs) if all_probs else np.empty((0, 2), dtype=float)
    attention_matrix = np.concatenate(all_attns) if all_attns else np.empty((0, 0), dtype=float)

    mean_loss = (total_loss / max(total_samples, 1)) if total_samples > 0 else 0.0

    default_thresh = float(threshold if threshold is not None else 0.50)
    scores = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] >= 2) else y_prob.ravel()
    preds_standard = (scores >= default_thresh).astype(int)

    standard_metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=preds_standard,
        y_prob=y_prob,
        loss_value=mean_loss,
    )
    standard_metrics["threshold"] = default_thresh

    optimal_threshold = default_thresh
    calibrated_metrics = None
    calib_preds = preds_standard

    if calibrate_threshold:
        optimal_threshold = find_optimal_threshold(y_true, y_prob, min_thresh=0.30, max_thresh=0.60)
        calib_preds = (scores >= optimal_threshold).astype(int)
        calibrated_metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=calib_preds,
            y_prob=y_prob,
            loss_value=mean_loss,
        )
        calibrated_metrics["threshold"] = optimal_threshold

    res: Dict[str, Any] = {
        "metrics": standard_metrics,
        "optimal_threshold": optimal_threshold,
        "attention_weights": attention_matrix,
        "probabilities": y_prob,
        "predictions": preds_standard,
        "targets": y_true,
        "loss": mean_loss,
    }

    if calibrated_metrics is not None:
        res["calibrated_metrics"] = calibrated_metrics
        res["calibrated_predictions"] = calib_preds

    return res


def format_dual_level_report(
    subject_metrics: Dict[str, float],
    view_metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Format dual-level diagnostic metrics comparing Subject-Level and View-Level results.

    Benchmarking:
    - Subject-Level Soft-Voting Ensemble is compared directly against published Table 8.
    - View-Level performance (Axial, Coronal, Sagittal) reports anatomical plane contributions per Table 3.

    Args:
        subject_metrics: Computed subject-level ensemble metrics.
        view_metrics: Dictionary containing per-view metric dictionaries.

    Returns:
        Formatted pandas DataFrame for reporting.
    """
    rows = []

    # 1. Subject-level metrics
    metric_specs = [
        ("Accuracy", "accuracy", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1-Score", "f1_score", True),
        ("Specificity", "specificity", True),
        ("Balanced Accuracy", "balanced_accuracy", True),
        ("G-Mean", "g_mean", True),
        ("Composite Score", "composite_score", True),
        ("ROC-AUC", "roc_auc", False),
        ("Loss", "loss", False),
    ]

    for display_name, key, is_pct in metric_specs:
        val = subject_metrics.get(key)
        base_val = PUBLISHED_TABLE_8_BENCHMARKS.get(display_name)
        if val is not None:
            res_str = f"{val * 100:.2f}%" if is_pct else f"{val:.4f}"
        else:
            res_str = "N/A"

        if base_val is not None:
            base_str = f"{base_val * 100:.2f}%" if is_pct else f"{base_val:.4f}"
            if val is not None:
                delta = val - base_val
                delta_sign = "+" if delta >= 0 else ""
                delta_str = f"{delta_sign}{delta * 100:.2f}%" if is_pct else f"{delta_sign}{delta:.4f}"
            else:
                delta_str = "N/A"
        else:
            base_str = "N/A"
            delta_str = "N/A"

        rows.append({
            "Level": "Subject (Ensemble)",
            "Metric": display_name,
            "Result": res_str,
            "Baseline": base_str,
            "Delta": delta_str,
        })

    # 2. View-level metrics
    for view_name in ["axial", "coronal", "sagittal"]:
        v_dict = view_metrics.get(view_name, {})
        for display_name, key, is_pct in metric_specs[:4]:
            val = v_dict.get(key)
            res_str = f"{val * 100:.2f}%" if (val is not None and is_pct) else (f"{val:.4f}" if val is not None else "N/A")
            rows.append({
                "Level": f"View ({view_name.capitalize()})",
                "Metric": display_name,
                "Result": res_str,
                "Baseline": "N/A",
                "Delta": "N/A",
            })

    return pd.DataFrame(rows)


def run_evaluation_pipeline(
    model_path: Path,
    test_data: Any,
    test_labels: np.ndarray,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Execute model evaluation pipeline on held-out test data.

    Args:
        model_path: Path to serialized model checkpoint (.pt).
        test_data: DataLoader or feature tensor for evaluation.
        test_labels: Binary target labels.
        output_dir: Destination directory for saved ROC figures and reports.

    Returns:
        Dictionary of computed diagnostic evaluation metrics.
    """
    out_path = output_dir or Path("artifacts/reports")
    out_path.mkdir(parents=True, exist_ok=True)

    # If test_data is a DataLoader, evaluate batch by batch
    if hasattr(test_data, "__iter__"):
        from asd_classification.models.baseline.cnn3d import Baseline3DCNN
        from asd_classification.models.baseline.focal_loss import FocalLoss
        from asd_classification.pipelines.train import evaluate_epoch

        model = Baseline3DCNN()
        model.load(model_path)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        _, metrics, _, _, y_prob = evaluate_epoch(model, test_data, criterion)
        roc_path = out_path / "roc_curve.png"
        plot_roc_curve(test_labels, y_prob, output_path=roc_path)
        return metrics

    raise NotImplementedError("Arbitrary non-DataLoader test data evaluation is not implemented.")
