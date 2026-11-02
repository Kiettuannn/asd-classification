"""
Unit and integration tests for Ticket #33:
Self-Contained Kaggle Training Notebook Generator, CLI Automation & Validation.
"""

import ast
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import pytest

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_notebook_5 import (
    build_all,
    generate_kernel_metadata,
    generate_notebook,
    get_kaggle_cli_instructions,
)


def test_generate_kernel_metadata_creates_valid_json(tmp_path: Path) -> None:
    """Verify that kernel metadata is generated with correct Kaggle properties."""
    meta_path = tmp_path / "kernel-metadata.json"
    result_path = generate_kernel_metadata(output_path=meta_path)

    assert result_path.exists()
    assert result_path == meta_path

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required Kaggle kernel metadata attributes
    assert data["enable_gpu"] is True
    assert data["accelerator"] == "nvidiaTeslaT4x2"
    assert data["enable_internet"] is True
    assert data["kernel_type"] == "notebook"
    assert data["language"] == "python"
    assert data["is_private"] is True
    assert data["code_file"] == "05_explainable_brainnet_clip_training.ipynb"
    assert "id" in data
    assert "title" in data


def test_get_kaggle_cli_instructions() -> None:
    """Verify that documented Kaggle CLI push, monitoring, and download instructions are provided."""
    instructions = get_kaggle_cli_instructions(
        kernel_slug="kiettuan/05-explainable-brainnet-clip-training",
        notebook_dir="notebooks",
        output_dir="artifacts/kaggle_outputs",
    )

    assert isinstance(instructions, str)
    assert "kaggle kernels push -p notebooks" in instructions
    assert "kaggle kernels status kiettuan/05-explainable-brainnet-clip-training" in instructions
    assert "kaggle kernels output kiettuan/05-explainable-brainnet-clip-training -p artifacts/kaggle_outputs" in instructions


def test_generate_notebook_creates_valid_schema(tmp_path: Path) -> None:
    """Verify that generated notebook conforms to Jupyter Notebook v4 JSON schema."""
    nb_path = tmp_path / "05_explainable_brainnet_clip_training.ipynb"
    result = generate_notebook(output_path=nb_path)

    assert result == nb_path
    assert result.exists()

    with open(result, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Invariants for Jupyter notebook JSON format
    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data.get("nbformat") == 4
    assert isinstance(nb_data["cells"], list)
    assert len(nb_data["cells"]) >= 10

    for i, cell in enumerate(nb_data["cells"]):
        assert "cell_type" in cell, f"Cell {i} missing 'cell_type'"
        assert cell["cell_type"] in ("markdown", "code"), f"Cell {i} has invalid type: {cell['cell_type']}"
        assert "source" in cell, f"Cell {i} missing 'source'"
        assert "metadata" in cell, f"Cell {i} missing 'metadata'"
        if cell["cell_type"] == "code":
            assert "outputs" in cell, f"Code cell {i} missing 'outputs'"
            assert isinstance(cell["outputs"], list), f"Code cell {i} 'outputs' is not a list"
            assert "execution_count" in cell, f"Code cell {i} missing 'execution_count'"


def test_build_all_idempotency(tmp_path: Path) -> None:
    """Verify that multiple builds produce byte-for-byte identical outputs."""
    nb_path = tmp_path / "notebook.ipynb"
    meta_path = tmp_path / "kernel-metadata.json"

    res_nb1, res_meta1 = build_all(notebook_path=nb_path, metadata_path=meta_path)
    content_nb1 = nb_path.read_text(encoding="utf-8")
    content_meta1 = meta_path.read_text(encoding="utf-8")

    res_nb2, res_meta2 = build_all(notebook_path=nb_path, metadata_path=meta_path)
    content_nb2 = nb_path.read_text(encoding="utf-8")
    content_meta2 = meta_path.read_text(encoding="utf-8")

    assert res_nb1 == res_nb2
    assert res_meta1 == res_meta2
    assert content_nb1 == content_nb2
    assert content_meta1 == content_meta2


def test_code_cells_syntax_validity(tmp_path: Path) -> None:
    """Verify that every code cell in the generated notebook is syntactically valid Python."""
    nb_path = tmp_path / "notebook.ipynb"
    generate_notebook(output_path=nb_path)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell for cell in nb_data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 10, "Expected at least 10 code cells"

    for idx, cell in enumerate(code_cells):
        raw_code = "".join(cell.get("source", []))
        # Strip any ipython magics if present
        filtered_lines = [
            line for line in raw_code.splitlines()
            if not line.strip().startswith(("%", "!"))
        ]
        executable_code = "\n".join(filtered_lines)
        try:
            ast.parse(executable_code)
        except SyntaxError as e:
            pytest.fail(f"Code cell {idx} failed syntax parsing: {e}\nCode:\n{executable_code}")


def test_notebook_zero_backend_dependencies(tmp_path: Path) -> None:
    """Verify that generated notebook does not depend on local asd_classification package."""
    nb_path = tmp_path / "notebook.ipynb"
    generate_notebook(output_path=nb_path)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    for idx, cell in enumerate(nb_data["cells"]):
        if cell.get("cell_type") == "code":
            source_text = "".join(cell.get("source", []))
            assert "asd_classification" not in source_text, (
                f"Cell {idx} contains reference to local backend package 'asd_classification'"
            )


def test_notebook_contains_all_required_pipeline_components(tmp_path: Path) -> None:
    """Verify all pipeline components required by Ticket #33 are present."""
    nb_path = tmp_path / "notebook.ipynb"
    generate_notebook(output_path=nb_path)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_code = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"] if c.get("cell_type") == "code")

    required_keywords = [
        "log_msg",
        "training_progress.log",
        "discover_or_fetch_abide_dataset",
        "filter_motion_quality",
        "generate_phenotypic_prompts",
        "compute_tangent_fc",
        "BrainPhenoDataset",
        "BrainNetCNN",
        "E2EBlock",
        "E2NBlock",
        "N2GBlock",
        "PhenotypicTextEncoder",
        "DualModalLoss",
        "infonce_loss",
        "DataParallel",
        "StratifiedKFold",
        "run_stratified_10_fold_cv",
        "compute_and_plot_aggregate_metrics",
        "Balanced Accuracy",
        "Sensitivity (Recall)",
        "Specificity",
        "ROC-AUC",
        "Macro-F1",
        "aggregate_roc_curve.png",
        "build_cc200_yeo7_mapping",
        "BrainNetGradCAM",
        "compute_network_enrichment",
        "plot_yeo7_enrichment",
        "yeo7_network_enrichment.png",
        "mask_top_edges",
        "retain_top_edges",
        "evaluate_fidelity",
        "fidelity_plus_drop_10",
        "fidelity_plus_drop_20",
        "fidelity_minus_retention_10",
        "fidelity_minus_retention_20",
        "brainnetcnn_best.pt",
    ]

    for kw in required_keywords:
        assert kw in all_code, f"Required component keyword '{kw}' missing from notebook code cells."


def test_cli_execution_with_custom_paths(tmp_path: Path) -> None:
    """Verify CLI argument parsing and artifact generation."""
    import subprocess

    target_nb = tmp_path / "custom_nb.ipynb"
    target_meta = tmp_path / "custom_meta.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_notebook_5.py"),
        "--output-notebook", str(target_nb),
        "--output-metadata", str(target_meta),
        "--kernel-slug", "test_user/test-slug",
        "--accelerator", "nvidiaTeslaT4x2",
        "--print-cli",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI command failed: {res.stderr}"
    assert target_nb.exists()
    assert target_meta.exists()
    assert "SUCCESS: Generated standalone training notebook" in res.stdout
    assert "test_user/test-slug" in res.stdout


def test_notebook_components_execution_on_synthetic_data(tmp_path: Path) -> None:
    """Smoke test verifying execution of self-contained classes and functions extracted from notebook."""
    import numpy as np
    import pandas as pd
    import torch

    nb_path = tmp_path / "notebook.ipynb"
    generate_notebook(output_path=nb_path)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Collect code cells excluding installation and discovery/download cells
    selected_cells = []
    for cell in nb_data["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "install_if_missing" in src or "def main()" in src or "datasets.fetch_abide_pcp" in src:
                continue
            selected_cells.append(src)

    combined_code = "\n\n".join(selected_cells)
    namespace: Dict[str, Any] = {}
    exec(combined_code, namespace)

    # 1. Verify phenotypic prompt generation & motion filtering
    mock_df = pd.DataFrame({
        "SUB_ID": ["sub1", "sub2", "sub3", "sub4"],
        "AGE_AT_SCAN": [12.5, 14.0, 10.2, 16.5],
        "SEX": [1, 2, 1, 2],
        "FIQ": [105.0, 98.0, 112.0, 89.0],
        "VIQ": [100.0, 95.0, 110.0, 90.0],
        "ADOS_GOTHAM_SOCAFFECT": [8.0, 2.0, 9.0, 1.0],
        "ADOS_GOTHAM_RRB": [3.0, 0.0, 4.0, 1.0],
        "DX_GROUP": [1, 2, 1, 2],
        "func_mean_fd": [0.12, 0.25, 0.45, 0.60],
    })
    mock_ts = [np.random.randn(50, 200).astype(np.float32) for _ in range(4)]

    filter_fn = namespace["filter_motion_quality"]
    clean_df, clean_ts, keep_mask = filter_fn(mock_df, mock_ts, max_fd=0.5)
    assert len(clean_df) == 3
    assert len(clean_ts) == 3

    prompt_fn = namespace["generate_phenotypic_prompts"]
    prompts = prompt_fn(clean_df)
    assert len(prompts) == 3
    assert "12.5-year-old male" in prompts[0]

    # 2. Verify BrainNetCNN forward pass
    BrainNetCNN = namespace["BrainNetCNN"]
    brain_net = BrainNetCNN(num_rois=200, embed_dim=128, num_classes=2)
    dummy_fc = torch.randn(2, 200, 200)
    logits, z_brain = brain_net(dummy_fc, return_embedding=True)
    assert logits.shape == (2, 2)
    assert z_brain.shape == (2, 128)
    # Check L2 normalization
    norms = torch.norm(z_brain, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # 3. Verify PhenotypicTextEncoder forward pass
    TextEncoder = namespace["PhenotypicTextEncoder"]
    text_enc = TextEncoder(embed_dim=128)
    z_text = text_enc(prompts[:2], device=torch.device("cpu"))
    assert z_text.shape == (2, 128)

    # 4. Verify DualModalLoss
    DualModalLoss = namespace["DualModalLoss"]
    criterion = DualModalLoss(lambda_align=0.3)
    loss, ce, align = criterion(logits, torch.tensor([1, 0]), z_brain, z_text)
    assert loss.item() > 0.0
    assert ce.item() > 0.0

    # 5. Verify Atlas and Grad-CAM explainability
    build_atlas = namespace["build_cc200_yeo7_mapping"]
    atlas_records = build_atlas()
    assert len(atlas_records) == 200

    GradCAM = namespace["BrainNetGradCAM"]
    gradcam = GradCAM(brain_net)
    single_fc = torch.randn(200, 200)
    saliency = gradcam.compute_saliency(single_fc, target_class=1)
    assert saliency.shape == (200,)
    assert np.all(saliency >= 0.0)

    enrichment_fn = namespace["compute_network_enrichment"]
    enrichment = enrichment_fn(saliency, atlas_records)
    assert isinstance(enrichment, dict)
    assert len(enrichment) == 7
    assert np.isclose(sum(enrichment.values()), 1.0, atol=1e-3)

    # 6. Verify Fidelity evaluation
    eval_fid = namespace["evaluate_fidelity"]
    fid_scores = eval_fid(brain_net, single_fc.numpy(), saliency, device=torch.device("cpu"))
    assert "fidelity_plus_drop_10" in fid_scores
    assert "fidelity_plus_drop_20" in fid_scores
    assert "fidelity_minus_retention_10" in fid_scores
    assert "fidelity_minus_retention_20" in fid_scores

    # 7. Verify aggregate metrics computation
    metrics_fn = namespace["compute_and_plot_aggregate_metrics"]
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    fig_path = tmp_path / "test_roc.png"
    csv_path = tmp_path / "test_oof.csv"

    # Matplotlib non-interactive backend for headless test
    import matplotlib
    matplotlib.use("Agg")

    metrics = metrics_fn(
        y_true, y_prob, y_pred,
        output_fig_path=str(fig_path),
        output_csv_path=str(csv_path),
    )
    assert metrics["Balanced Accuracy"] == 1.0
    assert metrics["ROC-AUC"] == 1.0
    assert fig_path.exists()
    assert csv_path.exists()



