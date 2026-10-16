"""Unit tests for NYU cohort phenotypic loading, tensor persistence, and dataset manifest export."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from asd_classification.data.loader import load_nyu_phenotypic
from asd_classification.data.split import partition_nyu_cohort, verify_nyu_split_counts
from asd_classification.preprocessing.multiview import (
    export_dataset_manifest,
    load_multiview_tensor,
    save_multiview_tensor,
)


def test_load_nyu_phenotypic_reads_184_subjects():
    """Verify load_nyu_phenotypic extracts exactly 184 NYU subjects with complete columns."""
    df = load_nyu_phenotypic()

    assert len(df) == 184
    expected_cols = {"subject_id", "file_id", "dx_group", "age", "sex", "fiq"}
    assert expected_cols.issubset(set(df.columns))

    # Diagnostic label distribution: 79 ASD, 105 TC
    dx_counts = df["dx_group"].value_counts().to_dict()
    assert dx_counts[1] == 79
    assert dx_counts[0] == 105

    # Check that missing file_id ('no_filename') is resolved to NYU_00{sub_id}
    assert not (df["file_id"] == "no_filename").any()
    assert not df["file_id"].isna().any()

    # Covariates must have no missing values in the 184 cohort
    assert df["age"].isna().sum() == 0
    assert df["sex"].isna().sum() == 0
    assert df["fiq"].isna().sum() == 0


def test_save_and_load_multiview_tensor(tmp_path: Path):
    """Verify Multi-View Tensor can be saved as compressed .npz and loaded back losslessly."""
    tensor_shape = (1, 150, 224, 224)
    synthetic_tensor = np.ones(tensor_shape, dtype=np.float32) * 1.5

    output_path = tmp_path / "tensors" / "50952.npz"
    saved_path = save_multiview_tensor(synthetic_tensor, output_path)

    assert saved_path.is_file()
    assert saved_path.suffix == ".npz"

    loaded_tensor = load_multiview_tensor(saved_path)
    assert loaded_tensor.shape == tensor_shape
    assert loaded_tensor.dtype == np.float32
    np.testing.assert_allclose(loaded_tensor, synthetic_tensor)


def test_export_dataset_manifest(tmp_path: Path):
    """Verify export_dataset_manifest creates valid metadata.csv and dataset-metadata.json."""
    pheno_df = load_nyu_phenotypic()
    split_df = partition_nyu_cohort(pheno_df, test_size=36, random_state=42)
    verify_nyu_split_counts(split_df)

    split_df["tensor_path"] = split_df["subject_id"].apply(lambda sid: f"tensors/{sid}.npz")

    csv_path, json_path = export_dataset_manifest(
        metadata_df=split_df,
        output_dir=tmp_path / "dataset_export",
        dataset_title="ABIDE I NYU Multi-View fMRI 3D-CNN Dataset",
        dataset_slug="abide-i-nyu-multiview-fmri",
    )

    assert csv_path.is_file()
    assert json_path.is_file()

    # Verify CSV content
    read_df = pd.read_csv(csv_path)
    assert len(read_df) == 184
    assert set(read_df["split"].unique()) == {"train", "test"}
    assert (read_df["split"] == "train").sum() == 148
    assert (read_df["split"] == "test").sum() == 36

    # Verify JSON content
    with open(json_path, "r", encoding="utf-8") as f:
        descriptor = json.load(f)

    assert descriptor["id"] == "antigravity/abide-i-nyu-multiview-fmri"
    assert descriptor["title"] == "ABIDE I NYU Multi-View fMRI 3D-CNN Dataset"
