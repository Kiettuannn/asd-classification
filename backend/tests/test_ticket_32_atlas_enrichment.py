"""
Unit tests for Ticket #32: CC200 Static Atlas and Yeo 7 Network Enrichment.

Verifies:
- 200 CC200 ROIs static mapping with valid MNI coordinates and Yeo 7 resting-state network assignments.
- Network enrichment score computation summing to 1.0 per subject.
- Specific quantification for Default Mode Network (DMN) and Salience Network.
- Cohort-wide enrichment aggregation summing to 1.0.
"""

import numpy as np
import pytest

from asd_classification.models.atlas import (
    CC200YeoAtlas,
    YEO_7_NETWORKS,
    get_cc200_yeo_atlas,
)


def test_cc200_atlas_structure_and_coverage():
    """Verify static atlas contains exactly 200 CC200 ROIs with valid MNI coordinates and networks."""
    atlas = get_cc200_yeo_atlas()
    assert isinstance(atlas, CC200YeoAtlas)
    assert len(atlas) == 200

    coords = atlas.get_coords()
    assert coords.shape == (200, 3)
    assert coords.dtype in (np.float32, np.float64)

    # Verify MNI space bounding limits (x in [-90, 90], y in [-120, 90], z in [-70, 90])
    assert np.all(coords[:, 0] >= -90) and np.all(coords[:, 0] <= 90)
    assert np.all(coords[:, 1] >= -120) and np.all(coords[:, 1] <= 90)
    assert np.all(coords[:, 2] >= -70) and np.all(coords[:, 2] <= 90)

    # Verify each ROI has all required metadata fields
    for roi_id in range(200):
        roi_meta = atlas.get_roi(roi_id)
        assert roi_meta["roi_id"] == roi_id
        assert isinstance(roi_meta["label"], str)
        assert len(roi_meta["label"]) > 0
        assert roi_meta["yeo_network"] in YEO_7_NETWORKS
        assert len(roi_meta["mni_coords"]) == 3


def test_yeo_7_canonical_networks_present():
    """Verify that all 7 canonical Yeo resting-state networks are represented in CC200."""
    atlas = get_cc200_yeo_atlas()
    network_counts = atlas.get_network_roi_counts()

    expected_networks = {
        "Visual",
        "Somatomotor",
        "Dorsal Attention",
        "Salience / Ventral Attention",
        "Limbic",
        "Frontoparietal Control",
        "Default Mode",
    }
    assert set(network_counts.keys()) == expected_networks
    assert sum(network_counts.values()) == 200

    # Ensure critical diagnostic networks (DMN and Salience) have substantial representation
    assert network_counts["Default Mode"] >= 20
    assert network_counts["Salience / Ventral Attention"] >= 15


def test_subject_network_enrichment_sums_to_one():
    """Verify normalized Network Enrichment Scores sum to 1.0 and quantify DMN/Salience concentration."""
    atlas = get_cc200_yeo_atlas()

    # Generate synthetic non-negative saliency weights
    np.random.seed(42)
    saliency = np.random.uniform(0.01, 1.0, size=200).astype(np.float32)

    enrichment = atlas.compute_network_enrichment(saliency)

    # Verify every Yeo 7 network is in the output
    assert set(enrichment.keys()) == set(YEO_7_NETWORKS)

    # Invariant: Normalized scores must sum to 1.0 within numerical tolerance
    total_score = sum(enrichment.values())
    assert pytest.approx(1.0, rel=1e-5) == total_score

    # Scores must all be non-negative
    for net, score in enrichment.items():
        assert score >= 0.0

    # Test concentrated saliency in Default Mode Network
    dmn_roi_indices = [
        roi_id for roi_id in range(200)
        if atlas.get_roi(roi_id)["yeo_network"] == "Default Mode"
    ]
    concentrated_saliency = np.zeros(200, dtype=np.float32)
    concentrated_saliency[dmn_roi_indices] = 1.0

    dmn_enrichment = atlas.compute_network_enrichment(concentrated_saliency)
    assert pytest.approx(1.0, rel=1e-5) == dmn_enrichment["Default Mode"]
    assert pytest.approx(0.0, abs=1e-6) == dmn_enrichment["Visual"]


def test_cohort_network_enrichment():
    """Verify cohort-level network enrichment calculates valid mean distributions summing to 1.0."""
    atlas = get_cc200_yeo_atlas()
    n_subjects = 10

    np.random.seed(123)
    cohort_saliency = np.random.uniform(0.0, 2.0, size=(n_subjects, 200)).astype(np.float32)

    cohort_enrichment = atlas.compute_cohort_enrichment(cohort_saliency)

    assert set(cohort_enrichment.keys()) == set(YEO_7_NETWORKS)
    assert pytest.approx(1.0, rel=1e-5) == sum(cohort_enrichment.values())
    for score in cohort_enrichment.values():
        assert score >= 0.0
