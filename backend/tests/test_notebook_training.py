"""Unit tests for Notebook 2: Kaggle GPU Training Pipeline, Evaluation, and ROC Analysis.

Verifies the notebook structure, JSON validity, acceptance criteria seams,
and contract adherence to ADR-0001, ADR-0002, and ADR-0003.
"""

import json
from pathlib import Path
import pytest


@pytest.fixture
def notebook_path() -> Path:
    """Resolve path to Notebook 2."""
    candidates = [
        Path("../notebooks/02_fmri_3dcnn_training.ipynb"),
        Path("notebooks/02_fmri_3dcnn_training.ipynb"),
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    pytest.fail("Could not find notebooks/02_fmri_3dcnn_training.ipynb")


def test_notebook_json_validity_and_format(notebook_path: Path):
    """Verify notebook is well-formed JSON conforming to nbformat version 4."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 12


def test_notebook_contains_all_acceptance_criteria(notebook_path: Path):
    """Verify notebook content contains all key algorithmic and packaging requirements for Ticket #8."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_source = " ".join("".join(cell.get("source", [])) for cell in nb_data["cells"])

    # 1. Self-contained notebook with Automatic Mixed Precision (torch.amp.autocast)
    assert "torch.amp.autocast" in all_source
    assert "GradScaler" in all_source

    # 2. Correctly loads metadata.csv and multi-view tensor inputs from Kaggle dataset paths
    assert "metadata.csv" in all_source
    assert "MultiViewDataset" in all_source
    assert "/kaggle/input/abide-i-nyu-multiview-fmri" in all_source

    # 3. Applies volume-consistent augmentation during training and disables during validation/testing
    assert "VolumeConsistentAugmenter" in all_source
    assert "augmenter.train()" in all_source
    assert "augmenter.eval()" in all_source

    # 4. Implements training loop with physical batch size 2 and gradient accumulation 4 steps (effective batch size 8)
    assert "PHYSICAL_BATCH_SIZE = 2" in all_source
    assert "ACCUMULATION_STEPS = 4" in all_source
    assert "EFFECTIVE_BATCH_SIZE" in all_source

    # 5. Evaluates against the 80/20 test split (15 ASD, 21 TC) and reports Accuracy, Precision, Recall, F1-Score, and Focal Loss
    assert "15 ASD" in all_source
    assert "21 TC" in all_source
    assert "accuracy_score" in all_source or "Accuracy" in all_source
    assert "precision_score" in all_source or "Precision" in all_source
    assert "recall_score" in all_source or "Recall" in all_source
    assert "f1_score" in all_source or "F1-Score" in all_source
    assert "FocalLoss" in all_source

    # 6. Generates and saves an ROC curve plot with AUC score to /kaggle/working/roc_curve.png
    assert "roc_curve" in all_source
    assert "roc_auc_score" in all_source
    assert "roc_curve.png" in all_source
    assert "/kaggle/working/roc_curve.png" in all_source or "ROC_CURVE_PATH" in all_source

    # 7. Provides an optional flag to run stratified 5-fold cross-validation across all 184 subjects
    assert "RUN_CROSS_VALIDATION" in all_source
    assert "StratifiedKFold" in all_source
    assert "N_SPLITS = 5" in all_source or "n_splits=5" in all_source

    # 8. Benchmarks against published Table 8 results (81.24% accuracy, 79.63% F1-score)
    assert "81.24" in all_source
    assert "79.63" in all_source
    assert "80.23" in all_source
    assert "79.05" in all_source
    assert "0.4932" in all_source

    # 9. Early stopping with patience 15
    assert "EARLY_STOPPING_PATIENCE = 15" in all_source or "patience=15" in all_source
    assert "EarlyStopping" in all_source
