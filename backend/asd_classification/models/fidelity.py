"""
Quantitative Perturbation Fidelity Verification Suite for BrainNetCNN.

Implements graph perturbation diagnostics to evaluate explainability faithfulness:
1. Fidelity+ (Sufficiency Drop): Masks the top 10% and 20% most salient functional
   connectivity edges and measures the reduction in target diagnostic class probability.
2. Fidelity- (Retention / Stability): Preserves exclusively the top 10% and 20% salient
   edges and verifies the preservation of target diagnostic class probability.
"""

from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def _squeeze_fc(fc_matrix: np.ndarray) -> np.ndarray:
    """
    Standardize 2D, 3D, or 4D functional connectivity matrices to shape (N, N).

    Args:
        fc_matrix: Input array of shape (N, N), (1, N, N), or (1, 1, N, N).

    Returns:
        2D numpy array of shape (N, N).
    """
    fc = np.asarray(fc_matrix, dtype=np.float32)
    if fc.ndim == 4 and fc.shape[0] == 1 and fc.shape[1] == 1:
        return fc[0, 0]
    if fc.ndim == 3 and fc.shape[0] == 1:
        return fc[0]
    return fc


def _get_top_edge_indices(
    n_rois: int,
    edge_saliency: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract row and column coordinates for the top ratio fraction of salient edges.

    Args:
        n_rois: Number of nodes in the graph.
        edge_saliency: Symmetric edge saliency matrix (N, N).
        ratio: Fraction of undirected edges to select in [0.0, 1.0].

    Returns:
        Tuple of (row_indices, col_indices).
    """
    triu_indices = np.triu_indices(n_rois, k=1)
    saliencies = edge_saliency[triu_indices]
    total_edges = len(saliencies)
    k = int(np.round(np.clip(ratio, 0.0, 1.0) * total_edges))
    if k == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    top_order = np.argsort(saliencies)[-k:]
    return triu_indices[0][top_order], triu_indices[1][top_order]


def compute_edge_saliency_matrix(
    node_saliency: np.ndarray,
    fc_matrix: np.ndarray,
) -> np.ndarray:
    """
    Derive symmetric edge-level attributions from ROI node saliency and FC matrix magnitude.

    Formula:
        E_{ij} = (S_i + S_j) * |X_{ij}| for i != j, and E_{ii} = 0.

    Args:
        node_saliency: Non-negative node saliency array of shape (N,).
        fc_matrix: Functional connectivity matrix of shape (N, N).

    Returns:
        Symmetric non-negative matrix of shape (N, N) with zero diagonal.

    Raises:
        ValueError: If node_saliency length does not match the FC matrix dimensions.
    """
    s = np.asarray(node_saliency, dtype=np.float64).flatten()
    fc = _squeeze_fc(fc_matrix).astype(np.float64)

    n_rois = len(s)
    if fc.shape != (n_rois, n_rois):
        raise ValueError(f"Shape mismatch: node saliency len {n_rois} vs FC matrix {fc.shape}.")

    # Outer sum of node saliencies: (S_i + S_j)
    node_sum = s[:, np.newaxis] + s[np.newaxis, :]

    # Modulate by absolute functional connectivity weight
    edge_saliency = node_sum * np.abs(fc)

    # Ensure zero self-loops and symmetry
    np.fill_diagonal(edge_saliency, 0.0)
    edge_saliency = (edge_saliency + edge_saliency.T) / 2.0

    return edge_saliency.astype(np.float32)


def mask_top_edges(
    fc_matrix: np.ndarray,
    edge_saliency: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """
    Mask (zero out) the top ratio fraction of most salient edges symmetrically.

    Args:
        fc_matrix: Input FC matrix of shape (N, N).
        edge_saliency: Symmetric edge saliency matrix of shape (N, N).
        ratio: Fraction of total undirected edges to zero out (e.g. 0.1 for 10%).

    Returns:
        Masked FC matrix of shape (N, N).
    """
    fc = _squeeze_fc(fc_matrix).copy()
    rows, cols = _get_top_edge_indices(fc.shape[0], edge_saliency, ratio)
    if len(rows) > 0:
        fc[rows, cols] = 0.0
        fc[cols, rows] = 0.0
    return fc


def retain_top_edges(
    fc_matrix: np.ndarray,
    edge_saliency: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """
    Retain exclusively the top ratio fraction of most salient edges, zeroing all others.
    Preserves original diagonal entries.

    Args:
        fc_matrix: Input FC matrix of shape (N, N).
        edge_saliency: Symmetric edge saliency matrix of shape (N, N).
        ratio: Fraction of total undirected edges to retain (e.g. 0.1 for 10%).

    Returns:
        Retained FC matrix of shape (N, N).
    """
    fc = _squeeze_fc(fc_matrix)
    retained = np.zeros_like(fc)
    np.fill_diagonal(retained, np.diag(fc))

    rows, cols = _get_top_edge_indices(fc.shape[0], edge_saliency, ratio)
    if len(rows) > 0:
        retained[rows, cols] = fc[rows, cols]
        retained[cols, rows] = fc[cols, rows]

    return retained


class FidelityEvaluator:
    """
    Quantitative fidelity evaluation suite for BrainNetCNN models.

    Computes probability drop upon selective edge ablation (Fidelity+) and
    retention upon selective edge preservation (Fidelity-).
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def _predict_prob(self, fc_matrix: np.ndarray, target_class: int) -> float:
        """
        Evaluate class probability for a single (N, N) matrix.

        Args:
            fc_matrix: Functional connectivity matrix (N, N).
            target_class: Integer class index (0 or 1).

        Returns:
            Probability of target class as a float in [0.0, 1.0].
        """
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(fc_matrix)
            return float(probs[0, target_class])

        # Raw PyTorch model inference
        raw = getattr(self.model, "_get_raw_brain_net", lambda: self.model)()
        device = next(raw.parameters()).device
        was_training = raw.training
        raw.eval()

        with torch.no_grad():
            x_t = torch.from_numpy(fc_matrix).float().unsqueeze(0).unsqueeze(0).to(device)
            out = raw(x_t)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        raw.train(was_training)
        return float(probs[0, target_class])

    def evaluate(
        self,
        fc_matrix: Union[np.ndarray, torch.Tensor],
        saliency_weights: np.ndarray,
        target_class: int = 1,
        ratios: Sequence[float] = (0.1, 0.2),
    ) -> Dict[str, float]:
        """
        Evaluate quantitative Fidelity+ and Fidelity- metrics.

        Args:
            fc_matrix: Input FC matrix of shape (N, N) or (1, 1, N, N).
            saliency_weights: Saliency array of shape (N,).
            target_class: Target diagnostic class index (1 = ASD, 0 = TD).
            ratios: Perturbation ratios to evaluate (default 10% and 20%).

        Returns:
            Dictionary with 'drop_top_10', 'drop_top_20', 'retention_top_10', 'retention_top_20'.
        """
        if isinstance(fc_matrix, torch.Tensor):
            fc_np = fc_matrix.detach().cpu().numpy().astype(np.float32)
        else:
            fc_np = np.asarray(fc_matrix, dtype=np.float32)

        fc_np = _squeeze_fc(fc_np)

        # 1. Baseline predictive probability
        baseline_prob = self._predict_prob(fc_np, target_class)

        # 2. Derive symmetric edge attribution matrix
        edge_saliency = compute_edge_saliency_matrix(saliency_weights, fc_np)

        metrics: Dict[str, float] = {}

        for r in ratios:
            pct_label = int(np.round(r * 100))

            # Fidelity+ (Mask top salient edges -> measure drop in predicted probability)
            masked_fc = mask_top_edges(fc_np, edge_saliency, ratio=r)
            prob_masked = self._predict_prob(masked_fc, target_class)
            drop = float(baseline_prob - prob_masked)
            metrics[f"drop_top_{pct_label}"] = drop

            # Fidelity- (Preserve exclusively top salient edges -> measure retention)
            retained_fc = retain_top_edges(fc_np, edge_saliency, ratio=r)
            prob_retained = self._predict_prob(retained_fc, target_class)
            metrics[f"retention_top_{pct_label}"] = float(prob_retained)

        return metrics

    def evaluate_cohort(
        self,
        cohort_fc: np.ndarray,
        cohort_saliency: np.ndarray,
        target_class: int = 1,
        ratios: Sequence[float] = (0.1, 0.2),
    ) -> Dict[str, float]:
        """
        Evaluate mean Fidelity+ and Fidelity- metrics across a cohort.

        Args:
            cohort_fc: Array of shape (M, N, N).
            cohort_saliency: Array of shape (M, N).
            target_class: Target class index.
            ratios: Perturbation ratios.

        Returns:
            Dictionary of mean fidelity scores across subjects.
        """
        n_subjects = len(cohort_fc)
        if n_subjects == 0:
            return {}

        all_results = [
            self.evaluate(cohort_fc[i], cohort_saliency[i], target_class=target_class, ratios=ratios)
            for i in range(n_subjects)
        ]

        mean_metrics: Dict[str, float] = {}
        for k in all_results[0].keys():
            mean_metrics[k] = float(np.mean([res[k] for res in all_results]))

        return mean_metrics
