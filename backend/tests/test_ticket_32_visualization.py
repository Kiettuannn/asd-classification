"""
Unit tests for Ticket #32: Publication-Grade Visual Diagnostics Suite.

Verifies:
- 3D glass brain ROI marker plotting via Nilearn/Matplotlib under headless backend.
- Yeo 7 network enrichment bar chart generation with DMN and Salience highlighting.
- File serialization to disk with valid non-zero image bytes.
"""

from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Enforce headless backend
import matplotlib.pyplot as plt

from asd_classification.models.atlas import get_cc200_yeo_atlas
from asd_classification.models.visualization import (
    plot_glass_brain_diagnostics,
    plot_network_enrichment,
)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for saving diagnostic plot artifacts."""
    plot_dir = tmp_path / "diagnostic_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def test_plot_glass_brain_diagnostics(tmp_output_dir):
    """Verify 3D glass brain plot generates figure and persists image file."""
    atlas = get_cc200_yeo_atlas()
    coords = atlas.get_coords()

    np.random.seed(42)
    saliency = np.random.uniform(0.0, 1.0, size=200).astype(np.float32)

    save_path = tmp_output_dir / "glass_brain_test.png"

    fig = plot_glass_brain_diagnostics(
        roi_coords=coords,
        saliency_weights=saliency,
        top_k=20,
        title="Test Glass Brain",
        output_path=save_path,
    )

    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 1000  # Valid non-empty PNG
    plt.close(fig)


def test_plot_network_enrichment_bar_chart(tmp_output_dir):
    """Verify Yeo 7 network enrichment bar chart generation and file persistence."""
    atlas = get_cc200_yeo_atlas()
    np.random.seed(42)
    saliency = np.random.uniform(0.01, 1.0, size=200).astype(np.float32)

    enrichment = atlas.compute_network_enrichment(saliency)

    save_path = tmp_output_dir / "network_enrichment_test.png"

    fig = plot_network_enrichment(
        enrichment_scores=enrichment,
        title="Test Yeo 7 Enrichment",
        output_path=save_path,
    )

    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 1000
    plt.close(fig)


def test_plot_network_enrichment_with_cohort_comparison(tmp_output_dir):
    """Verify bar chart plotting with dual subject vs cohort overlay."""
    atlas = get_cc200_yeo_atlas()
    np.random.seed(42)
    subject_saliency = np.random.uniform(0.0, 1.0, size=200).astype(np.float32)
    cohort_saliency = np.random.uniform(0.0, 1.0, size=(5, 200)).astype(np.float32)

    sub_enrichment = atlas.compute_network_enrichment(subject_saliency)
    cohort_enrichment = atlas.compute_cohort_enrichment(cohort_saliency)

    save_path = tmp_output_dir / "enrichment_comparison_test.png"

    fig = plot_network_enrichment(
        enrichment_scores=sub_enrichment,
        cohort_scores=cohort_enrichment,
        output_path=save_path,
    )

    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 1000
    plt.close(fig)
