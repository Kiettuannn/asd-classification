"""
Unit tests for Ticket #32: Quantitative Fidelity+ and Fidelity- Perturbation Verification Suite.

Verifies:
- Edge saliency computation respecting graph symmetry.
- Fidelity+ perturbation: Masking top 10% and 20% salient edges produces expected probability drop.
- Fidelity- perturbation: Preserving top 10% and 20% salient edges produces stability/retention.
- Evaluator returns standardized dictionary contract matching Issue #29 and Issue #32 specs.
- Automated tests verify perturbation masks alter probabilities in expected directions.
"""

import numpy as np
import pytest
import torch

from asd_classification.models.brainnetcnn import BrainNetCNN
from asd_classification.models.fidelity import (
    FidelityEvaluator,
    compute_edge_saliency_matrix,
    mask_top_edges,
    retain_top_edges,
)


@pytest.fixture
def mock_model():
    """Create a deterministic BrainNetCNN model for perturbation fidelity tests."""
    torch.manual_seed(42)
    model = BrainNetCNN(num_rois=200, in_channels=1, num_classes=2)
    model.eval()
    return model


def test_edge_saliency_matrix_symmetry():
    """Verify edge saliency matrix is symmetric and diagonal is zero."""
    np.random.seed(42)
    node_saliency = np.random.uniform(0.1, 1.0, size=200).astype(np.float32)
    fc = np.random.randn(200, 200).astype(np.float32)
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 1.0)

    edge_saliency = compute_edge_saliency_matrix(node_saliency, fc)

    assert edge_saliency.shape == (200, 200)
    assert np.allclose(edge_saliency, edge_saliency.T, atol=1e-6)
    assert np.allclose(np.diag(edge_saliency), 0.0)
    assert np.all(edge_saliency >= 0.0)


def test_mask_and_retain_top_edges_fractions():
    """Verify that masking and retaining alters the exact specified ratio of unique edges."""
    np.random.seed(42)
    node_saliency = np.random.uniform(0.1, 1.0, size=200).astype(np.float32)
    fc = np.random.randn(200, 200).astype(np.float32)
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)

    edge_saliency = compute_edge_saliency_matrix(node_saliency, fc)

    # 10% masking
    masked_10 = mask_top_edges(fc, edge_saliency, ratio=0.1)
    assert masked_10.shape == (200, 200)
    assert np.allclose(masked_10, masked_10.T)

    triu_indices = np.triu_indices(200, k=1)
    total_edges = len(triu_indices[0])  # 19900
    expected_masked_count = int(np.round(0.1 * total_edges))  # 1990

    zeroed_count = np.sum((fc[triu_indices] != 0.0) & (masked_10[triu_indices] == 0.0))
    assert pytest.approx(expected_masked_count, abs=2) == zeroed_count

    # 10% retention
    retained_10 = retain_top_edges(fc, edge_saliency, ratio=0.1)
    assert retained_10.shape == (200, 200)
    assert np.allclose(retained_10, retained_10.T)

    retained_count = np.sum((retained_10[triu_indices] != 0.0))
    assert pytest.approx(expected_masked_count, abs=2) == retained_count


def test_fidelity_evaluation_contract(mock_model):
    """Verify evaluate_fidelity produces structured dictionary with drop and retention metrics."""
    np.random.seed(42)
    fc = np.random.randn(200, 200).astype(np.float32)
    fc = (fc + fc.T) / 2.0
    saliency = np.random.uniform(0.1, 1.0, size=200).astype(np.float32)

    evaluator = FidelityEvaluator(mock_model)
    fidelity_res = evaluator.evaluate(fc, saliency, target_class=1)

    assert "drop_top_10" in fidelity_res
    assert "drop_top_20" in fidelity_res
    assert "retention_top_10" in fidelity_res
    assert "retention_top_20" in fidelity_res

    for key, val in fidelity_res.items():
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0


def test_perturbation_alters_probability_in_expected_direction():
    """
    Verify that masking top salient edges causes predictive probability to drop (drop > 0)
    and retaining top salient edges preserves higher probability than masking.
    """
    class DirectionalTestModel:
        """Model whose positive class prediction monotonically depends on salient edge presence."""
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            fc = np.asarray(X, dtype=np.float64)
            if fc.ndim == 4:
                fc = fc[0, 0]
            elif fc.ndim == 3:
                fc = fc[0]
            # Mean edge value drives probability without numerical saturation
            pos_weight = float(np.mean(np.maximum(fc, 0.0)))
            prob_1 = 1.0 / (1.0 + np.exp(-pos_weight * 4.0))
            return np.array([[1.0 - prob_1, prob_1]])

    # Generate synthetic FC where salient subcircuit holds the positive predictive biomarker
    np.random.seed(42)
    fc = np.zeros((200, 200), dtype=np.float32)
    fc[:25, :25] = np.random.uniform(0.5, 1.0, size=(25, 25)).astype(np.float32)
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)

    # Assign high saliency to the active circuit ROIs
    saliency = np.zeros(200, dtype=np.float32)
    saliency[:25] = 2.0

    model = DirectionalTestModel()
    evaluator = FidelityEvaluator(model)

    res = evaluator.evaluate(fc, saliency, target_class=1, ratios=(0.1, 0.2))

    # Masking top 10% salient edges must cause probability drop
    assert res["drop_top_10"] > 0.0, "Expected positive drop when masking top 10% salient edges."

    # Masking top 20% salient edges should cause greater or equal probability drop
    assert res["drop_top_20"] >= res["drop_top_10"], "Expected drop_top_20 to be >= drop_top_10."

    # Retaining top 20% edges preserves the active biomarker circuit and retains higher probability than masking
    baseline_prob = model.predict_proba(fc)[0, 1]
    prob_masked_20 = baseline_prob - res["drop_top_20"]
    assert res["retention_top_20"] > prob_masked_20, (
        "Preserving top salient edges should retain higher probability than masking them."
    )
