"""Unit and integration tests for Ticket #30: CC200 data ingestion, Tangent FC, and phenotypic prompts."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure repo root is in sys.path so scripts can be imported
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_pipeline_cc200 import (
    CohortDataBundle,
    filter_motion_quality,
    compute_tangent_fc,
    generate_phenotypic_prompts,
    create_joint_stratified_kfold,
    ABIDETangentDataset,
    create_dataloaders,
)


def test_quality_control_motion_filtering():
    """Verify that subjects with excessive head motion (func_mean_fd > 0.5) are removed."""
    mock_df = pd.DataFrame({
        "SUB_ID": ["sub_1", "sub_2", "sub_3", "sub_4", "sub_5"],
        "func_mean_fd": [0.15, 0.45, 0.55, 0.90, np.nan],
        "DX_GROUP": [1, 2, 1, 2, 1],
    })
    
    clean_df, keep_mask = filter_motion_quality(mock_df, max_fd=0.5)
    
    assert "sub_1" in clean_df["SUB_ID"].values
    assert "sub_2" in clean_df["SUB_ID"].values
    assert "sub_3" not in clean_df["SUB_ID"].values
    assert "sub_4" not in clean_df["SUB_ID"].values
    assert len(clean_df) <= 3


def test_phenotypic_prompt_generation_no_leakage():
    """Verify clinical prompt formatting replaces missing tokens and leaks zero diagnostic labels."""
    mock_df = pd.DataFrame({
        "SUB_ID": ["1001", "1002", "1003"],
        "AGE_AT_SCAN": [14.5, 9.2, 22.0],
        "SEX": [1, 2, np.nan],  # 1 = male, 2 = female, nan = missing
        "FIQ": [105.0, -9999.0, np.nan],
        "VIQ": [98.0, 110.0, -9999.0],
        "ADOS_TOTAL": [12.0, -9999.0, 4.0],
        "ADOS_SOCIAL": [8.0, -9999.0, np.nan],
        "ADOS_STEREO_BEHAV": [4.0, -9999.0, 1.0],
        "DX_GROUP": [1, 2, 1],  # 1 = ASD, 2 = Control
    })

    prompts = generate_phenotypic_prompts(mock_df)

    assert len(prompts) == 3
    for prompt in prompts:
        assert isinstance(prompt, str)
        assert len(prompt) > 20
        # Critical security invariant: No ground-truth label leakage in text prompts
        lower_prompt = prompt.lower()
        assert "autism" not in lower_prompt
        assert "asd" not in lower_prompt
        assert "control" not in lower_prompt
        assert "dx_group" not in lower_prompt
        assert "diagnosis" not in lower_prompt
        # Ensure grammar is valid and avoids 'individual individual'
        assert "individual individual" not in lower_prompt

    # Verify missing value replacement
    assert "not documented" in prompts[1]
    assert "male" in prompts[0]
    assert "female" in prompts[1]


def test_tangent_space_fc_computation_and_fold_fitting():
    """Verify Riemannian Tangent Space connectivity and leak-free fold-wise fitting."""
    np.random.seed(42)
    n_subjects = 6
    n_rois = 200
    n_timepoints = 80

    time_series_list = [
        np.random.randn(n_timepoints, n_rois).astype(np.float32)
        for _ in range(n_subjects)
    ]

    # Test full fitting
    fc_matrices = compute_tangent_fc(time_series_list)
    assert fc_matrices.shape == (n_subjects, n_rois, n_rois)
    assert fc_matrices.dtype == np.float32

    # Test leak-free fold-wise fitting
    train_indices = [0, 1, 2, 3]
    fc_fold = compute_tangent_fc(time_series_list, fit_indices=train_indices)
    assert fc_fold.shape == (n_subjects, n_rois, n_rois)
    assert fc_fold.dtype == np.float32

    for i in range(n_subjects):
        mat = fc_fold[i]
        assert np.allclose(mat, mat.T, atol=1e-5), f"Matrix {i} is not symmetric"
        assert not np.isnan(mat).any(), f"Matrix {i} contains NaNs"


def test_joint_stratified_10fold_splits():
    """Verify joint class and site stratified 10-fold splits provide balanced disjoint folds."""
    n_samples = 50
    mock_df = pd.DataFrame({
        "SUB_ID": [f"sub_{i}" for i in range(n_samples)],
        "DX_GROUP": [1 if i % 2 == 0 else 2 for i in range(n_samples)],
        "SITE_ID": [f"SITE_{i % 5}" for i in range(n_samples)],
    })

    splits = list(create_joint_stratified_kfold(mock_df, n_splits=10, random_state=42))

    assert len(splits) == 10

    all_val_indices = []
    for train_idx, val_idx in splits:
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        all_val_indices.extend(val_idx)

    assert sorted(all_val_indices) == list(range(n_samples))


def test_abide_tangent_dataset_and_dataloader_with_bundle():
    """Verify PyTorch Dataset and DataLoader emit expected tensor shapes and site codes."""
    n_samples = 6
    n_rois = 200
    fc_matrices = np.random.randn(n_samples, n_rois, n_rois).astype(np.float32)
    prompts = [f"Clinical summary for subject {i}" for i in range(n_samples)]
    labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    site_ids = np.array(["NYU", "UCLA", "NYU", "USM", "UCLA", "NYU"])
    site_to_idx = {"NYU": 0, "UCLA": 1, "USM": 2}
    site_codes = np.array([site_to_idx[s] for s in site_ids], dtype=np.int64)

    bundle = CohortDataBundle(
        fc_matrices=fc_matrices,
        text_prompts=prompts,
        labels=labels,
        site_ids=site_ids,
        site_codes=site_codes,
        site_to_idx=site_to_idx,
        pheno_df=pd.DataFrame({"SUB_ID": range(n_samples)}),
    )

    train_idx = np.array([0, 1, 2, 3])
    val_idx = np.array([4, 5])

    train_loader, val_loader = create_dataloaders(
        bundle=bundle,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=2,
    )

    batch = next(iter(train_loader))
    fc_batch, text_batch, label_batch, site_code_batch = batch

    assert fc_batch.shape == (2, 1, n_rois, n_rois)
    assert fc_batch.dtype == torch.float32
    assert len(text_batch) == 2
    assert isinstance(text_batch[0], str)
    assert label_batch.shape == (2,)
    assert label_batch.dtype == torch.long
    assert site_code_batch.shape == (2,)
    assert site_code_batch.dtype == torch.long
