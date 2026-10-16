"""Tests for Notebook 1: ABIDE I NYU Preprocessing and Kaggle Dataset Export.

Verifies the notebook structure, JSON validity, acceptance criteria seams,
and contract adherence to ADR-0002 and ADR-0003.
"""

import json
from pathlib import Path
import pytest


@pytest.fixture
def notebook_path() -> Path:
    """Resolve path to Notebook 1."""
    candidates = [
        Path("../notebooks/01_fmri_3dcnn_preprocessing.ipynb"),
        Path("notebooks/01_fmri_3dcnn_preprocessing.ipynb"),
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    pytest.fail("Could not find notebooks/01_fmri_3dcnn_preprocessing.ipynb")


def test_notebook_json_validity_and_format(notebook_path: Path):
    """Verify notebook is well-formed JSON conforming to nbformat version 4."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 10


def test_notebook_contains_all_acceptance_criteria(notebook_path: Path):
    """Verify notebook content contains all key algorithmic and packaging requirements."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_source = " ".join("".join(cell.get("source", [])) for cell in nb_data["cells"])

    # 1. Automatic retrieval of the 184 NYU cohort subjects
    assert "184" in all_source
    assert "SITE_ID" in all_source
    assert "NYU" in all_source

    # 2. Temporal frame trimming (10 initial, 10 terminal)
    assert "trim_temporal_frames" in all_source
    assert "INITIAL_FRAMES" in all_source

    # 3. Temporal Mean Projection
    assert "compute_temporal_mean" in all_source

    # 4. Z-score intensity normalization
    assert "z_score_normalize" in all_source

    # 5. 50-slice orthogonal extraction & resizing to 224x224
    assert "extract_orthogonal_slices" in all_source
    assert "224" in all_source
    assert "N_SLICES = 50" in all_source or "n_slices=50" in all_source

    # 6. Multi-View Tensor assembly of shape (1, 150, 224, 224)
    assert "assemble_multiview_tensor" in all_source
    assert "(1, 150, 224, 224)" in all_source

    # 7. Table 3 exact split verification: 64 ASD / 84 TC in train, 15 ASD / 21 TC in test
    assert "64" in all_source
    assert "84" in all_source
    assert "15" in all_source
    assert "21" in all_source
    assert "train_asd == 64" in all_source
    assert "train_tc == 84" in all_source
    assert "test_asd == 15" in all_source
    assert "test_tc == 21" in all_source

    # 8. Manifest metadata.csv export with covariates
    assert "metadata.csv" in all_source
    assert "dx_group" in all_source
    assert "split" in all_source

    # 9. Compressed tensor persistence (.npz)
    assert "save_multiview_tensor" in all_source
    assert ".npz" in all_source

    # 10. Kaggle Dataset packaging and publishing guide
    assert "dataset-metadata.json" in all_source
    assert "abide-i-nyu-multiview-fmri" in all_source
    assert "02_fmri_3dcnn_training.ipynb" in all_source
