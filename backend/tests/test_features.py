"""Tests for feature extraction and graph construction."""

import pytest
from asd_classification.features.connectivity import compute_fc, vectorize_fc, threshold_fc


def test_feature_interface_definitions():
    """Verify feature extraction function signatures exist."""
    assert callable(compute_fc)
    assert callable(vectorize_fc)
    assert callable(threshold_fc)
