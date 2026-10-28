"""Tests for Notebook: Site-Specific Full ABIDE I Raw fMRI Preprocessed Dataset Downloader.

Verifies the notebook structure, JSON validity, acceptance criteria seams,
institution mapping across all 1,112 subjects, and contract adherence for Kaggle execution.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pytest


@pytest.fixture
def notebook_path() -> Path:
    """Resolve path to the ABIDE raw download notebook."""
    candidates = [
        Path("../notebooks/01_abide_raw_download.ipynb"),
        Path("notebooks/01_abide_raw_download.ipynb"),
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    pytest.fail("Could not find notebooks/01_abide_raw_download.ipynb")


@pytest.fixture
def phenotypic_csv_path() -> Path:
    """Resolve path to the ABIDE phenotypic CSV."""
    candidates = [
        Path("notebooks/data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
        Path("notebooks/abide_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
        Path("../notebooks/data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
        Path("data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    pytest.fail("Could not find Phenotypic_V1_0b_preprocessed1.csv")


# Canonical 17 institutions and their expected subject counts
EXPECTED_INSTITUTION_COUNTS: Dict[str, int] = {
    "CALTECH": 38,
    "CMU": 27,
    "KKI": 55,
    "LEUVEN": 64,
    "MAX_MUN": 57,
    "NYU": 184,
    "OHSU": 28,
    "OLIN": 36,
    "PITT": 57,
    "SBL": 30,
    "SDSU": 36,
    "STANFORD": 40,
    "TRINITY": 49,
    "UCLA": 109,
    "UM": 145,
    "USM": 101,
    "YALE": 56,
}


def map_site_to_institution(site_id: str) -> str:
    """Canonical mapping from raw SITE_ID to the 17 institutions.

    Consolidates sub-sites UM_1/UM_2, UCLA_1/UCLA_2, and LEUVEN_1/LEUVEN_2
    into their parent institutions.
    """
    clean_site = str(site_id).strip().upper()
    sub_site_map = {
        "UM_1": "UM",
        "UM_2": "UM",
        "UCLA_1": "UCLA",
        "UCLA_2": "UCLA",
        "LEUVEN_1": "LEUVEN",
        "LEUVEN_2": "LEUVEN",
    }
    resolved = sub_site_map.get(clean_site, clean_site)
    if resolved not in EXPECTED_INSTITUTION_COUNTS:
        valid_sites = ", ".join(sorted(EXPECTED_INSTITUTION_COUNTS.keys()))
        raise ValueError(f"Unknown site '{site_id}'. Valid institutions: {valid_sites}")
    return resolved


def get_s3_url(
    file_id: str,
    pipeline: str = "cpac",
    strategy: str = "nofilt_noglobal",
    derivative: str = "func_preproc",
    base_url: str = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative",
) -> str:
    """Construct public AWS S3 download URL for an ABIDE derivative."""
    return f"{base_url}/Outputs/{pipeline}/{strategy}/{derivative}/{file_id}_{derivative}.nii.gz"


def export_dataset_artifacts(
    manifest: pd.DataFrame,
    output_dir: Path,
    target_site: str,
) -> Tuple[Path, Path]:
    """Export metadata.csv and dataset-metadata.json for native Kaggle dataset publishing."""
    site_clean = target_site.strip().upper()
    site_slug = site_clean.lower()

    csv_path = output_dir / "metadata.csv"
    manifest.to_csv(csv_path, index=False)

    dataset_title = f"ABIDE I {site_clean} Raw fMRI Dataset"
    dataset_slug = f"abide-i-{site_slug}-raw"

    kaggle_metadata = {
        "title": dataset_title,
        "id": f"antigravity/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            f"Preprocessed 4D resting-state fMRI NIfTI acquisitions (*_func_preproc.nii.gz) "
            f"for the {site_clean} institution from the ABIDE I cohort. Preprocessed with the CPAC "
            f"pipeline (nofilt_noglobal / filt_noglobal). Preserves raw 4D volumetric NIfTI format "
            f"without slice downsampling, accompanied by standardized phenotypic metadata.csv."
        ),
    }

    json_path = output_dir / "dataset-metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kaggle_metadata, f, indent=2)

    return csv_path, json_path


def test_notebook_json_validity_and_format(notebook_path: Path):
    """Verify notebook is well-formed JSON conforming to nbformat version 4."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 8


def test_institution_mapping_all_1112_subjects(phenotypic_csv_path: Path):
    """Verify that all 1,112 rows in the phenotypic CSV map to the 17 institutions."""
    df = pd.read_csv(phenotypic_csv_path)
    assert len(df) == 1112

    institutions = df["SITE_ID"].apply(map_site_to_institution)

    unique_institutions = set(institutions.unique())
    assert unique_institutions == set(EXPECTED_INSTITUTION_COUNTS.keys())
    assert len(unique_institutions) == 17

    counts = institutions.value_counts().to_dict()
    for inst, expected_count in EXPECTED_INSTITUTION_COUNTS.items():
        assert counts[inst] == expected_count, (
            f"Institution {inst} count mismatch: got {counts[inst]}, expected {expected_count}"
        )

    assert map_site_to_institution("UM_1") == "UM"
    assert map_site_to_institution("um_2") == "UM"
    assert map_site_to_institution("UCLA_1") == "UCLA"
    assert map_site_to_institution("ucla_2") == "UCLA"
    assert map_site_to_institution("LEUVEN_1") == "LEUVEN"
    assert map_site_to_institution("leuven_2") == "LEUVEN"

    with pytest.raises(ValueError, match="Valid institutions:"):
        map_site_to_institution("INVALID_SITE")


def test_s3_url_construction():
    """Verify public S3 URL format adheres to CPAC derivative structure."""
    url = get_s3_url("Pitt_0050003")
    expected = (
        "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
        "/Outputs/cpac/nofilt_noglobal/func_preproc/Pitt_0050003_func_preproc.nii.gz"
    )
    assert url == expected


def test_notebook_contains_all_acceptance_criteria(notebook_path: Path):
    """Verify notebook content contains all key configuration, download, and export logic."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_source = " ".join("".join(cell.get("source", [])) for cell in nb_data["cells"])

    assert "TARGET_SITE" in all_source
    assert "DRY_RUN" in all_source
    assert "MAX_SUBJECTS" in all_source
    assert "map_site_to_institution(TARGET_SITE)" in all_source

    assert "s3.amazonaws.com" in all_source
    assert "nofilt_noglobal" in all_source
    assert "filt_noglobal" in all_source
    assert "func_preproc" in all_source

    assert "2 ** (attempt - 1)" in all_source
    assert "time.sleep" in all_source

    assert "missing_on_s3" in all_source
    assert "no_filename" in all_source

    assert "abide_i_" in all_source
    assert "scans" in all_source

    assert "metadata.csv" in all_source
    assert "dataset-metadata.json" in all_source
    assert "abide-i-" in all_source


def test_notebook_code_cells_manifest_schema(notebook_path: Path):
    """Verify that exported manifest schema in the notebook includes all required fields."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_source = " ".join("".join(cell.get("source", [])) for cell in nb_data["cells"])

    required_columns = [
        "subject_id",
        "site_id",
        "institution",
        "file_id",
        "dx_group",
        "age",
        "sex",
        "fiq",
        "file_path",
        "file_size_mb",
        "download_status",
    ]

    for col in required_columns:
        assert col in all_source, f"Required column '{col}' missing from notebook code"


def test_export_dataset_artifacts_schema_and_metadata(tmp_path: Path):
    """Verify export_dataset_artifacts writes metadata.csv and dataset-metadata.json correctly."""
    output_dir = tmp_path / "abide_i_pitt_raw"
    output_dir.mkdir(parents=True)

    test_data = [
        {
            "subject_id": "50003",
            "site_id": "PITT",
            "institution": "PITT",
            "file_id": "Pitt_0050003",
            "dx_group": 1,
            "age": 19.3,
            "sex": 1,
            "fiq": 105.0,
            "file_path": "scans/Pitt_0050003_func_preproc.nii.gz",
            "file_size_mb": 94.2,
            "download_status": "downloaded",
        },
        {
            "subject_id": "50002",
            "site_id": "PITT",
            "institution": "PITT",
            "file_id": "no_filename",
            "dx_group": 1,
            "age": 16.7,
            "sex": 1,
            "fiq": 110.0,
            "file_path": "",
            "file_size_mb": 0.0,
            "download_status": "missing_on_s3",
        },
    ]
    manifest_df = pd.DataFrame(test_data)

    csv_path, json_path = export_dataset_artifacts(
        manifest=manifest_df,
        output_dir=output_dir,
        target_site="PITT",
    )

    assert csv_path.is_file()
    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == 2
    expected_cols = [
        "subject_id",
        "site_id",
        "institution",
        "file_id",
        "dx_group",
        "age",
        "sex",
        "fiq",
        "file_path",
        "file_size_mb",
        "download_status",
    ]
    assert list(loaded_df.columns) == expected_cols
    assert loaded_df.iloc[0]["download_status"] == "downloaded"
    assert loaded_df.iloc[1]["download_status"] == "missing_on_s3"

    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["title"] == "ABIDE I PITT Raw fMRI Dataset"
    assert meta["id"] == "antigravity/abide-i-pitt-raw"
    assert "licenses" in meta
    assert meta["licenses"][0]["name"] == "CC0-1.0"
    assert "PITT" in meta["description"]
