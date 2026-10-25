"""Unit and structural tests for Notebook 3: Pretrained 2.5D ResNet-18 MIL Training.

Verifies the notebook structure, JSON validity, Python syntax via AST parsing,
acceptance criteria seams, and contract adherence for Issue #17.
"""

import ast
import json
from pathlib import Path
import pytest


@pytest.fixture
def notebook_path() -> Path:
    """Resolve path to Notebook 3 (03_fmri_mil_resnet_training.ipynb)."""
    candidates = [
        Path(__file__).resolve().parents[2] / "notebooks/03_fmri_mil_resnet_training.ipynb",
        Path("../notebooks/03_fmri_mil_resnet_training.ipynb"),
        Path("notebooks/03_fmri_mil_resnet_training.ipynb"),
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    pytest.fail("Could not find notebooks/03_fmri_mil_resnet_training.ipynb")


def test_notebook_json_validity_and_format(notebook_path: Path):
    """Verify notebook is well-formed JSON conforming to nbformat version 4."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 12


def test_notebook_python_cells_syntax_valid(notebook_path: Path):
    """Verify all Python code cells parse without syntax errors using ast.parse."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell for cell in nb_data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 10, "Expected at least 10 code cells in the notebook."

    for idx, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        # Filter out IPython magic commands (! or %) for pure AST parsing
        python_lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                python_lines.append(f"# {line}")
            else:
                python_lines.append(line)
        cleaned_source = "\n".join(python_lines)
        try:
            ast.parse(cleaned_source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in code cell {idx} at line {e.lineno}: {e.msg}\nCode snippet:\n{cleaned_source[:300]}")


def test_notebook_contains_all_acceptance_criteria(notebook_path: Path):
    """Verify notebook contains all key algorithmic and execution requirements for Issue #17."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_source = " ".join("".join(cell.get("source", [])) for cell in nb_data["cells"])

    # 1. Multi-GPU data parallelism (torch.nn.DataParallel) and mixed precision (torch.amp)
    assert "torch.nn.DataParallel" in all_source or "nn.DataParallel" in all_source
    assert "torch.amp.autocast" in all_source
    assert "GradScaler" in all_source
    assert "num_gpus" in all_source or "device_count" in all_source

    # 2. Kaggle dataset discovery across mount paths
    assert "metadata.csv" in all_source
    assert "/kaggle/input" in all_source
    assert "resolve_dataset_paths" in all_source

    # 3. 2.5D Tri-slice bag extraction and BagConsistentAugmenter
    assert "extract_tri_slices" in all_source
    assert "BagConsistentAugmenter" in all_source
    assert "TOTAL_BAG_SLICES" in all_source or "60" in all_source

    # 4. ResNet-18 backbone with Gated Attention MIL and frozen early layers
    assert "ResNet18MIL" in all_source
    assert "GatedAttention" in all_source
    assert "freeze_early_layers" in all_source
    assert "conv1" in all_source and "layer1" in all_source and "layer2" in all_source
    assert "layer3" in all_source and "layer4" in all_source

    # 5. Differential optimizer with 1e-5 backbone and 1e-4 head, 3-epoch warmup
    assert "configure_differential_optimizer" in all_source
    assert "BACKBONE_LR = 1e-5" in all_source or "1e-5" in all_source
    assert "HEAD_LR = 1e-4" in all_source or "1e-4" in all_source
    assert "WARMUP_EPOCHS = 3" in all_source or "warmup_epochs=3" in all_source
    assert "LABEL_SMOOTHING = 0.1" in all_source or "label_smoothing=0.1" in all_source

    # 6. Early stopping on composite score with warmup suppression
    assert "EarlyStopping" in all_source
    assert "composite_score" in all_source
    assert "EARLY_STOPPING_PATIENCE = 15" in all_source or "patience=15" in all_source

    # 7. Section 10: 80/20 Holdout evaluation with standard and calibrated threshold (tau*)
    assert "SECTION 10" in all_source or "Holdout Evaluation" in all_source
    assert "find_optimal_threshold" in all_source
    assert "optimal_threshold" in all_source
    assert "mil_roc_curve.png" in all_source or "plot_mil_roc_curve" in all_source

    # 8. Section 11: Stratified 5-Fold Cross-Validation across 184 subjects
    assert "SECTION 11" in all_source or "Stratified 5-Fold" in all_source
    assert "StratifiedKFold" in all_source
    assert "run_stratified_5fold_cv" in all_source
    assert "n_splits=5" in all_source or "n_splits = 5" in all_source

    # 9. Section 12: Top-attended slice interpretability visualizations
    assert "SECTION 12" in all_source or "Top-Attended" in all_source
    assert "render_top_attended_slices" in all_source
    assert "attention_slices.png" in all_source
    assert "Axial" in all_source and "Coronal" in all_source and "Sagittal" in all_source

    # 10. Verification Checklist and benchmark comparisons (Table 8)
    assert "81.24" in all_source
    assert "80.23" in all_source
    assert "79.05" in all_source
    assert "79.63" in all_source
    assert "NUM_WORKERS = 0" in all_source


def test_mil_notebook_components_execution_smoke(tmp_path: Path):
    """Smoke test executing extracted logic components from the notebook."""
    import numpy as np
    import torch
    from asd_classification.models.baseline.resnet_mil import GatedAttention, ResNet18MIL
    from asd_classification.models.baseline.augmentation import BagConsistentAugmenter
    from asd_classification.preprocessing.multiview import extract_tri_slices
    from asd_classification.pipelines.evaluate import find_optimal_threshold

    # 1. Test 2.5D tri-slice extraction contract
    dummy_volume = torch.randn(1, 150, 224, 224)
    bag = extract_tri_slices(dummy_volume, slices_per_view=50, central_slices=40, stride=2)
    assert bag.shape == (60, 3, 224, 224), f"Expected shape (60, 3, 224, 224), got {bag.shape}"

    # 2. Test bag data augmentation contract
    augmenter = BagConsistentAugmenter()
    augmenter.train()
    aug_bag = augmenter(bag)
    assert aug_bag.shape == (60, 3, 224, 224)

    # 3. Test Gated Attention MIL pooling contract
    model = ResNet18MIL(pretrained=False, freeze_early=True)
    batch = torch.randn(2, 60, 3, 224, 224)
    logits, attns = model(batch)

    assert logits.shape == (2, 2)
    assert attns.shape == (2, 60)
    assert torch.allclose(torch.sum(attns, dim=-1), torch.ones(2), atol=1e-5)
    assert (attns >= 0.0).all()

    # 4. Test threshold calibration contract
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.4, 0.6],
        [0.3, 0.7],
        [0.2, 0.8],
    ])
    opt_tau = find_optimal_threshold(y_true, y_prob)
    assert 0.30 <= opt_tau <= 0.60


