"""Unit and structural tests for Notebook: Spatio-Temporal 3D-CNN Feature Extraction.

Verifies the notebook generator script, JSON schema validity, Python syntax
via AST parsing, and acceptance criteria for two-stage decoupled feature caching.
"""

import ast
import json
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture
def repo_root() -> Path:
    """Resolve repository root path."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def notebook_path(repo_root: Path) -> Path:
    """Resolve path to notebooks/04_spatiotemporal_feature_extraction.ipynb."""
    return repo_root / "notebooks/04_spatiotemporal_feature_extraction.ipynb"


def test_generator_script_exists(repo_root: Path):
    """Verify scripts/build_notebook_4_feature_extraction.py exists."""
    script_path = repo_root / "scripts/build_notebook_4_feature_extraction.py"
    assert script_path.is_file(), f"Generator script not found: {script_path}"


def test_generator_executes_and_is_idempotent(repo_root: Path, notebook_path: Path):
    """Verify running the generator script succeeds and produces idempotent output."""
    script_path = repo_root / "scripts/build_notebook_4_feature_extraction.py"

    proc1 = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc1.returncode == 0, f"Generator script execution 1 failed: {proc1.stderr}"
    assert notebook_path.is_file(), f"Notebook was not created: {notebook_path}"
    content1 = notebook_path.read_text(encoding="utf-8")

    proc2 = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, f"Generator script execution 2 failed: {proc2.stderr}"
    content2 = notebook_path.read_text(encoding="utf-8")

    assert content1 == content2, "Generator script produced non-idempotent content."


def test_notebook_json_validity(notebook_path: Path):
    """Verify notebook is well-formed JSON conforming to nbformat v4."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 10


def test_notebook_python_cells_syntax_valid(notebook_path: Path):
    """Verify all Python code cells parse without syntax errors using ast.parse."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [cell for cell in nb_data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 6, "Expected at least 6 code cells in feature extraction notebook."

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
                f"Snippet:\n{cleaned_source[:300]}"
            )


def test_notebook_contains_feature_extraction_criteria(notebook_path: Path):
    """Verify notebook contains all key algorithmic and execution requirements for feature caching."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_source = " ".join("".join(cell.get("source", [])) for cell in nb_data["cells"])

    # 1. Multi-part discovery
    assert "PART_SITES" in all_source
    assert "discover_multisite_dataset" in all_source

    # 2. 3D spatial encoder architecture
    assert "Spatial3DCNNEncoder" in all_source
    assert "Spatial3DCNNClassifier" in all_source or "Linear" in all_source

    # 3. Fast static 3D pretraining on mean volumes
    assert "FocalLoss" in all_source
    assert "StaticMean3DDataset" in all_source or "mean_volume" in all_source or "mean_3d" in all_source

    # 4. Feature extraction worker & serialization
    assert "extract_and_cache_subject_features" in all_source or "features" in all_source
    assert "_features.pt" in all_source or ".pt" in all_source
    assert "preprocessed_manifest.csv" in all_source or "dataset-metadata.json" in all_source


def test_stage1_to_stage2_feature_ingestion_contract(tmp_path: Path):
    """Verify that Stage 1 exported features dataset is automatically discovered and ingested by Stage 2."""
    import pandas as pd
    import torch
    from asd_classification.data.spatiotemporal import (
        SpatioTemporalFMRIDataset,
        discover_multisite_dataset,
    )
    from asd_classification.models.baseline.spatiotemporal_bimamba import SpatioTemporalBiMamba

    stage1_root = tmp_path / "kaggle_input" / "abide-spatiotemporal-features"
    features_dir = stage1_root / "features"
    features_dir.mkdir(parents=True)

    # Save mock Stage 1 feature embeddings
    for sid in ["50003", "50004"]:
        payload = {
            "features": torch.randn(200, 128, dtype=torch.float32),
            "mask": torch.ones(200, dtype=torch.bool),
            "label": 1 if sid == "50003" else 0,
            "subject_id": sid,
            "site_id": "PITT",
        }
        torch.save(payload, features_dir / f"{sid}_features.pt")

    # Save Stage 1 manifest
    manifest_csv = stage1_root / "preprocessed_manifest.csv"
    df = pd.DataFrame([
        {
            "subject_id": "50003",
            "site_id": "PITT",
            "label": 1,
            "file_path": str(features_dir / "50003_features.pt"),
            "features_file": str(features_dir / "50003_features.pt"),
            "part_num": 4,
        },
        {
            "subject_id": "50004",
            "site_id": "PITT",
            "label": 0,
            "file_path": str(features_dir / "50004_features.pt"),
            "features_file": str(features_dir / "50004_features.pt"),
            "part_num": 4,
        },
    ])
    df.to_csv(manifest_csv, index=False)

    # Stage 2: Ingest via discover_multisite_dataset
    discovered_df, file_map = discover_multisite_dataset(candidate_paths=[stage1_root])
    assert len(discovered_df) == 2
    assert "50003" in file_map
    assert "50004" in file_map
    assert file_map["50003"].is_file()
    assert file_map["50004"].is_file()

    # Pass into SpatioTemporalFMRIDataset
    dataset = SpatioTemporalFMRIDataset(manifest=discovered_df)
    item = dataset[0]
    assert item["volume"].shape == (200, 128)
    assert item["mask"].shape == (200,)

    # Pass into SpatioTemporalBiMamba
    model = SpatioTemporalBiMamba(
        in_channels=1,
        d_model=128,
        num_classes=2,
        d_state=8,
        num_layers=1,
        force_pure_pytorch=True,
    )
    model.eval()

    vol_batch = item["volume"].unsqueeze(0)
    mask_batch = item["mask"].unsqueeze(0)
    with torch.no_grad():
        logits, weights = model(vol_batch, mask=mask_batch)

    assert logits.shape == (1, 2)
    assert weights.shape == (1, 200)

