"""Unit and structural tests for Notebook 4: Spatio-Temporal 3D-CNN + Bi-Mamba Training.

Verifies the notebook structure, JSON validity, Python syntax via AST parsing,
acceptance criteria seams, multi-part dataset auto-discovery, and attention
trajectory visualization for Issue #28.
"""

import ast
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def notebook_path() -> Path:
    """Resolve path to Notebook 4 (04_spatiotemporal_bimamba_training.ipynb)."""
    candidates = [
        Path(__file__).resolve().parents[2] / "notebooks/04_spatiotemporal_bimamba_training.ipynb",
        Path("../notebooks/04_spatiotemporal_bimamba_training.ipynb"),
        Path("notebooks/04_spatiotemporal_bimamba_training.ipynb"),
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    pytest.fail("Could not find notebooks/04_spatiotemporal_bimamba_training.ipynb")


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
            pytest.fail(
                f"Syntax error in code cell {idx} at line {e.lineno}: {e.msg}\n"
                f"Code snippet:\n{cleaned_source[:300]}"
            )


def test_notebook_contains_all_acceptance_criteria(notebook_path: Path):
    """Verify notebook contains all key algorithmic and execution requirements for Issue #28."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_source = " ".join("".join(cell.get("source", [])) for cell in nb_data["cells"])

    # 1. Multi-part Kaggle dataset configuration and auto-discovery
    assert "PART_SITES" in all_source
    assert "NYU" in all_source
    assert "CALTECH" in all_source
    assert "discover_multisite_dataset" in all_source or "discover_multisite_data" in all_source
    assert "/kaggle/input" in all_source

    # 2. Spatio-temporal DataLoader and neuroimaging preprocessing
    assert "SpatioTemporalFMRIDataset" in all_source
    assert "trim_temporal_frames" in all_source
    assert "normalize_nonzero_voxels" in all_source
    assert "standardize_spatial_volume" in all_source
    assert "pad_temporal_sequence" in all_source
    assert "64, 64, 36" in all_source
    assert "200" in all_source

    # 3. 3D-CNN + Bi-Mamba architecture with PyTorch fallback
    assert "Spatial3DCNNEncoder" in all_source
    assert "BiMambaSequenceModel" in all_source
    assert "MambaBlock" in all_source
    assert "selective_scan_fn" in all_source
    assert "GatedTemporalAttentionPooling" in all_source
    assert "ClassificationHead" in all_source
    assert "SpatioTemporalBiMamba" in all_source

    # 4. Multi-GPU DataParallel / DDP acceleration and native BF16 mixed precision
    assert "bfloat16" in all_source
    assert "torch.amp.autocast" in all_source
    assert "DataParallel" in all_source

    # 5. Focal Loss and AdamW optimization with Cosine Annealing
    assert "FocalLoss" in all_source
    assert "AdamW" in all_source
    assert "CosineAnnealingLR" in all_source

    # 6. Joint Site-and-Class Stratified 5-Fold Cross-Validation
    assert "get_site_and_class_stratified_folds" in all_source or "Stratified" in all_source
    assert "run_spatiotemporal_kfold_cv" in all_source or "KFold" in all_source

    # 7. Aggregate ROC curve plotting with calculated AUC
    assert "roc_curve" in all_source
    assert "roc_auc_score" in all_source
    assert "spatiotemporal_roc_curve.png" in all_source

    # 8. Dynamic temporal attention trajectory extraction and visualization (ASD vs. TC)
    assert "temporal_attention_trajectories.png" in all_source
    assert "plot_temporal_attention_trajectories" in all_source or "attention_trajectories" in all_source


def test_notebook_generator_idempotency():
    """Verify that running scripts/build_notebook_4.py is idempotent."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts/build_notebook_4.py"
    target_nb = repo_root / "notebooks/04_spatiotemporal_bimamba_training.ipynb"

    assert script_path.is_file(), f"Generator script not found: {script_path}"

    # First execution
    proc1 = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc1.returncode == 0, f"Generator script failed: {proc1.stderr}"
    content1 = target_nb.read_text(encoding="utf-8")

    # Second execution
    proc2 = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, f"Second generator script run failed: {proc2.stderr}"
    content2 = target_nb.read_text(encoding="utf-8")

    assert content1 == content2, "Generator script produced different content on second run."


def test_plot_spatiotemporal_roc_curve(tmp_path: Path):
    """Verify plot_spatiotemporal_roc_curve generates valid ROC figure and PNG artifact."""
    from asd_classification.pipelines.spatiotemporal_engine import plot_spatiotemporal_roc_curve

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.1, 0.9],
    ])
    save_file = tmp_path / "test_roc.png"

    fig, ax = plot_spatiotemporal_roc_curve(
        y_true=y_true,
        y_prob=y_prob,
        save_path=save_file,
    )

    assert save_file.is_file()
    assert save_file.stat().st_size > 1000
    assert ax.get_xlabel() == "False Positive Rate (1 - Specificity)"
    assert ax.get_ylabel() == "True Positive Rate (Sensitivity)"
    plt.close(fig)


def test_plot_temporal_attention_trajectories(tmp_path: Path):
    """Verify plot_temporal_attention_trajectories generates valid ASD vs. TC trajectory plots."""
    from asd_classification.pipelines.spatiotemporal_engine import plot_temporal_attention_trajectories

    # Synthetic cohort: 4 subjects (2 ASD, 2 TC), sequence length T=50
    rng = np.random.RandomState(42)
    attns = rng.uniform(0.01, 0.05, size=(4, 50))
    # Normalize to sum to 1 per subject
    attns = attns / attns.sum(axis=1, keepdims=True)
    # Inject distinct temporal peak for ASD
    attns[0:2, 20:25] += 0.05
    attns = attns / attns.sum(axis=1, keepdims=True)

    labels = np.array([1, 1, 0, 0])
    save_file = tmp_path / "test_attention_traj.png"

    fig, (ax_main, ax_diff) = plot_temporal_attention_trajectories(
        attention_weights=attns,
        labels=labels,
        tr_seconds=2.0,
        save_path=save_file,
    )

    assert save_file.is_file()
    assert save_file.stat().st_size > 1000
    assert ax_main.get_ylabel() == "Temporal Attention Weight (alpha_t)"
    assert "Scan Acquisition Time" in ax_diff.get_xlabel()
    plt.close(fig)
