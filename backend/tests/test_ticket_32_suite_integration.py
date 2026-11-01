"""
Integration tests for Ticket #32: End-to-End Neurobiological Explainability Engine.

Verifies:
- DualModalBrainNetModel.explain(...) adheres to contract from Issue #29 and Issue #32:
  Returns 'salient_rois', 'network_enrichment', and 'fidelity'.
- Saliency weights are non-negative and sorted in descending order of attribution.
- Network enrichment scores sum to 1.0.
- Fidelity dictionary contains drop_top_10, drop_top_20, retention_top_20.
- Visual diagnostics generation integration.
- Cohort-level consensus explainability via NeuroExplainabilitySuite.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from asd_classification.models.dual_modal_engine import DualModalBrainNetModel
from asd_classification.models.explainability import NeuroExplainabilitySuite


@pytest.fixture
def mock_trained_model():
    """Create a trained mock DualModalBrainNetModel for end-to-end explainability tests."""
    torch.manual_seed(42)
    model = DualModalBrainNetModel(
        num_rois=200,
        in_channels=1,
        num_classes=2,
        use_mock_text_backbone=True,
        epochs=1,
    )
    return model


def test_dual_modal_engine_explain_contract(mock_trained_model):
    """Verify explain(...) conforms to the unified explainability dictionary contract."""
    np.random.seed(42)
    sample_fc = np.random.randn(200, 200).astype(np.float32)
    sample_fc = (sample_fc + sample_fc.T) / 2.0

    explanation = mock_trained_model.explain(sample_fc, top_k=10, target_class=1)

    assert isinstance(explanation, dict)
    assert "salient_rois" in explanation
    assert "network_enrichment" in explanation
    assert "fidelity" in explanation

    # 1. Salient ROIs check
    salient_rois = explanation["salient_rois"]
    assert len(salient_rois) == 10
    for roi in salient_rois:
        assert "roi_id" in roi
        assert "saliency_weight" in roi
        assert "yeo_network" in roi
        assert "mni_coords" in roi
        assert roi["saliency_weight"] >= 0.0

    # Ensure sorted descending
    weights = [r["saliency_weight"] for r in salient_rois]
    assert weights == sorted(weights, reverse=True)

    # 2. Network Enrichment check
    enrichment = explanation["network_enrichment"]
    assert pytest.approx(1.0, rel=1e-5) == sum(enrichment.values())
    assert "Default Mode" in enrichment
    assert "Salience / Ventral Attention" in enrichment

    # 3. Fidelity check
    fidelity = explanation["fidelity"]
    assert "drop_top_10" in fidelity
    assert "drop_top_20" in fidelity
    assert "retention_top_20" in fidelity


def test_explain_with_plot_generation(mock_trained_model, tmp_path):
    """Verify explain(...) successfully writes publication diagnostic figures when requested."""
    np.random.seed(42)
    sample_fc = np.random.randn(200, 200).astype(np.float32)
    sample_fc = (sample_fc + sample_fc.T) / 2.0

    plot_dir = tmp_path / "explain_plots"
    explanation = mock_trained_model.explain(
        sample_fc,
        top_k=15,
        target_class=1,
        generate_plots=True,
        plot_output_dir=plot_dir,
    )

    assert "figures" in explanation
    figures = explanation["figures"]
    assert "glass_brain" in figures
    assert "network_enrichment" in figures

    glass_brain_file = Path(figures["glass_brain"])
    enrichment_file = Path(figures["network_enrichment"])

    assert glass_brain_file.exists()
    assert glass_brain_file.stat().st_size > 1000
    assert enrichment_file.exists()
    assert enrichment_file.stat().st_size > 1000


def test_cohort_consensus_explainability(mock_trained_model):
    """Verify NeuroExplainabilitySuite.explain_cohort aggregates consensus biomarkers."""
    np.random.seed(42)
    cohort_fc = np.random.randn(6, 200, 200).astype(np.float32)
    for i in range(len(cohort_fc)):
        cohort_fc[i] = (cohort_fc[i] + cohort_fc[i].T) / 2.0

    suite = NeuroExplainabilitySuite(mock_trained_model)
    cohort_res = suite.explain_cohort(cohort_fc, top_k=10, target_class=1)

    assert "cohort_salient_rois" in cohort_res
    assert "cohort_network_enrichment" in cohort_res
    assert "cohort_fidelity" in cohort_res

    assert len(cohort_res["cohort_salient_rois"]) == 10
    assert pytest.approx(1.0, rel=1e-5) == sum(cohort_res["cohort_network_enrichment"].values())
