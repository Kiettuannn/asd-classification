"""Unit tests for multi-site 4D rs-fMRI DataLoader and cross-validation pipeline.

Tests exercise public interfaces at pre-agreed seams:
1. Dataset Auto-Discovery Seam (discover_multisite_dataset)
2. Preprocessing & Sequence Standardization Seam (trim_temporal_frames, normalize_nonzero_voxels, standardize_spatial_volume, pad_temporal_sequence)
3. Joint Site-and-Class Stratified Partitioning Seam (get_site_and_class_stratified_folds, assign_stratified_folds)
4. Dataset & Batch Collation Seam (SpatioTemporalFMRIDataset, create_spatiotemporal_dataloader)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from asd_classification.data.spatiotemporal import (
    SpatioTemporalFMRIDataset,
    assign_stratified_folds,
    create_spatiotemporal_dataloader,
    discover_multisite_dataset,
    get_site_and_class_stratified_folds,
    normalize_nonzero_voxels,
    pad_temporal_sequence,
    spatiotemporal_collate_fn,
    standardize_spatial_volume,
    trim_temporal_frames,
)



def test_discover_multisite_dataset_matches_files(tmp_path: Path):
    """Verify that discover_multisite_dataset auto-discovers and pairs phenotypic records with NIfTI files.

    Checks:
    1. Locates phenotypic CSV table across nested directory structure without hardcoded paths.
    2. Locates 4D functional NIfTI files (*func_preproc.nii.gz).
    3. Accurately pairs each subject with its corresponding file path.
    4. Formats labels to binary diagnostic classification (1 = ASD, 0 = TC).
    """
    # Create mock Kaggle / local directory structure
    dataset_dir = tmp_path / "kaggle_input" / "abide_dataset"
    raw_dir = dataset_dir / "raw_preproc"
    raw_dir.mkdir(parents=True)

    # Create dummy functional NIfTI files
    file_pitt = raw_dir / "Pitt_0050003_func_preproc.nii.gz"
    file_nyu = raw_dir / "NYU_0050952_func_preproc.nii.gz"
    file_pitt.write_text("mock_pitt_nifti")
    file_nyu.write_text("mock_nyu_nifti")

    # Create mock phenotypic CSV with varying FILE_ID conventions
    phenotypic_csv = dataset_dir / "Phenotypic_V1_0b_preprocessed1.csv"
    df = pd.DataFrame(
        [
            {
                "SUB_ID": 50003,
                "FILE_ID": "Pitt_0050003",
                "SITE_ID": "PITT",
                "DX_GROUP": 1,  # ASD
                "AGE_AT_SCAN": 19.5,
                "SEX": 1,
            },
            {
                "SUB_ID": 50952,
                "FILE_ID": "no_filename",  # Convention is NYU_00{SUB_ID}
                "SITE_ID": "NYU",
                "DX_GROUP": 2,  # TC
                "AGE_AT_SCAN": 12.0,
                "SEX": 2,
            },
        ]
    )
    df.to_csv(phenotypic_csv, index=False)

    manifest_df, file_mapping = discover_multisite_dataset(candidate_paths=[dataset_dir])

    assert len(manifest_df) == 2
    assert "file_path" in manifest_df.columns
    assert set(manifest_df["subject_id"].tolist()) == {"50003", "50952"}
    assert set(file_mapping.keys()) == {"50003", "50952"}
    assert file_mapping["50003"] == file_pitt
    assert file_mapping["50952"] == file_nyu

    # Validate binary label mapping (ASD=1, TC=0)
    pitt_row = manifest_df[manifest_df["subject_id"] == "50003"].iloc[0]
    nyu_row = manifest_df[manifest_df["subject_id"] == "50952"].iloc[0]
    assert pitt_row["label"] == 1
    assert nyu_row["label"] == 0


def test_discover_multisite_dataset_not_found(tmp_path: Path):
    """Verify that discover_multisite_dataset raises FileNotFoundError on missing data."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        discover_multisite_dataset(candidate_paths=[empty_dir])


def test_trim_temporal_frames():
    """Verify that trim_temporal_frames removes first 10 and last 10 TR frames.

    Tests:
    1. Input sequence with T=100 frames is trimmed to T=80 frames.
    2. Boundary frames 0..9 and 90..99 are discarded.
    3. Proper error is raised when sequence duration T <= 20.
    """
    # 4D numpy array with shape (X, Y, Z, T)
    t_full = 100
    vol_np = np.arange(t_full, dtype=np.float32).reshape(1, 1, 1, t_full)
    vol_np = np.repeat(np.repeat(np.repeat(vol_np, 4, axis=0), 4, axis=1), 4, axis=2)

    trimmed_np = trim_temporal_frames(vol_np, trim_start=10, trim_end=10, time_axis=-1)
    assert trimmed_np.shape[-1] == 80
    assert trimmed_np[0, 0, 0, 0] == 10.0
    assert trimmed_np[0, 0, 0, -1] == 89.0

    # 4D torch Tensor with time along axis 0: (T, C, H, W, D) or (T, H, W, D)
    vol_torch = torch.arange(t_full, dtype=torch.float32).view(t_full, 1, 1, 1)
    trimmed_torch = trim_temporal_frames(vol_torch, trim_start=10, trim_end=10, time_axis=0)
    assert trimmed_torch.shape[0] == 80
    assert trimmed_torch[0, 0, 0, 0].item() == 10.0
    assert trimmed_torch[-1, 0, 0, 0].item() == 89.0

    # Test error when duration is too short to trim
    short_vol = np.zeros((4, 4, 4, 15))
    with pytest.raises(ValueError):
        trim_temporal_frames(short_vol, trim_start=10, trim_end=10, time_axis=-1)


def test_normalize_nonzero_voxels():
    """Verify that normalize_nonzero_voxels applies subject-level z-scoring to non-zero voxels.

    Invariants:
    1. Zero-valued background voxels remain strictly 0.0.
    2. Non-zero brain tissue voxels have zero mean and unit variance.
    3. Consistent behavior on both NumPy ndarray and PyTorch Tensor.
    """
    np.random.seed(42)
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    # Populate brain tissue region with random signal (mean ~ 50, std ~ 15)
    mask = np.random.rand(10, 10, 10) > 0.5
    vol[mask] = np.random.normal(loc=50.0, scale=15.0, size=mask.sum()).astype(np.float32)

    norm_vol = normalize_nonzero_voxels(vol)

    # Invariant 1: Zero voxels remain zero
    assert np.all(norm_vol[~mask] == 0.0)

    # Invariant 2: Non-zero voxels have zero mean and unit variance
    tissue_voxels = norm_vol[mask]
    assert np.isclose(tissue_voxels.mean(), 0.0, atol=1e-5)
    assert np.isclose(tissue_voxels.std(), 1.0, atol=1e-5)

    # Invariant 3: Works with PyTorch tensor
    vol_torch = torch.from_numpy(vol)
    norm_torch = normalize_nonzero_voxels(vol_torch)
    assert torch.all(norm_torch[~torch.from_numpy(mask)] == 0.0)
    assert torch.isclose(norm_torch[torch.from_numpy(mask)].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(norm_torch[torch.from_numpy(mask)].std(), torch.tensor(1.0), atol=1e-5)


def test_standardize_spatial_volume():
    """Verify that standardize_spatial_volume standardizes volumes to 64x64x36.

    Tests:
    1. Shapes larger or smaller than (64, 64, 36) are cropped or padded to exact target shape.
    2. Supports both 3D volumes (H, W, D) and 4D sequences (T, H, W, D).
    3. Handles crop_pad and resample modes.
    """
    # 3D volume with shape (61, 73, 61) standard in ABIDE I
    vol_abide = np.random.randn(61, 73, 61).astype(np.float32)

    # Mode 1: crop_pad
    standardized_cp = standardize_spatial_volume(vol_abide, target_shape=(64, 64, 36), mode="crop_pad")
    assert standardized_cp.shape == (64, 64, 36)

    # Mode 2: resample
    standardized_res = standardize_spatial_volume(vol_abide, target_shape=(64, 64, 36), mode="resample")
    assert standardized_res.shape == (64, 64, 36)

    # 4D sequence: (T=5, 61, 73, 61)
    seq_4d = torch.randn(5, 61, 73, 61)
    standardized_seq = standardize_spatial_volume(seq_4d, target_shape=(64, 64, 36), mode="crop_pad")
    assert standardized_seq.shape == (5, 64, 64, 36)


def test_pad_temporal_sequence():
    """Verify that pad_temporal_sequence pads or truncates sequences to T_max=200 with boolean masks.

    Tests:
    1. Short sequence (T=50) is padded to 200 with mask True for 0..49 and False for 50..199.
    2. Over-length sequence (T=220) is truncated to 200 with mask True for all 200 frames.
    3. Output shapes and boolean dtype contracts.
    """
    t_short = 50
    feat_dim = (1, 64, 64, 36)
    short_tensor = torch.randn(t_short, *feat_dim)

    padded, mask = pad_temporal_sequence(short_tensor, target_length=200)

    assert padded.shape == (200, *feat_dim)
    assert mask.shape == (200,)
    assert mask.dtype == torch.bool
    assert mask[:50].all()
    assert not mask[50:].any()
    assert (padded[50:] == 0.0).all()

    # Test truncation when T > 200
    t_long = 220
    long_tensor = torch.randn(t_long, *feat_dim)
    truncated, long_mask = pad_temporal_sequence(long_tensor, target_length=200)

    assert truncated.shape == (200, *feat_dim)
    assert long_mask.shape == (200,)
    assert long_mask.all()


def test_site_and_class_stratified_folds_invariants():
    """Verify that joint Site-and-Class Stratification satisfies partitioning invariants.

    Invariants:
    1. Validation partitions are mutually exclusive and collectively exhaustive.
    2. Exactly n_splits folds are yielded.
    3. Each fold contains balanced diagnostic classes (ASD vs. TC).
    4. Each fold contains balanced representation across all institutions.
    """
    # Create synthetic multi-site cohort across 17 institutions
    sites = [f"SITE_{i:02d}" for i in range(1, 18)]
    rows = []
    sub_counter = 50001

    # Populate 10 subjects (5 ASD, 5 TC) per site => 170 subjects total
    for site in sites:
        for label in [0, 1]:
            for _ in range(5):
                rows.append(
                    {
                        "subject_id": str(sub_counter),
                        "site_id": site,
                        "label": label,
                    }
                )
                sub_counter += 1

    df = pd.DataFrame(rows)
    n_splits = 5
    splits = get_site_and_class_stratified_folds(df, n_splits=n_splits, random_state=42)

    assert len(splits) == n_splits

    all_val_indices = []
    total_samples = len(df)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Invariant 1: No overlap between train and val
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(train_idx) + len(val_idx) == total_samples

        all_val_indices.extend(val_idx.tolist())

        val_df = df.iloc[val_idx]
        # Invariant 2: Class balance in each fold (exactly 17 ASD and 17 TC out of 34)
        asd_count = (val_df["label"] == 1).sum()
        tc_count = (val_df["label"] == 0).sum()
        assert asd_count == 17
        assert tc_count == 17

        # Invariant 3: Site representation in each fold (every site has exactly 2 subjects: 1 ASD, 1 TC)
        assert val_df["site_id"].nunique() == 17
        for site in sites:
            site_val = val_df[val_df["site_id"] == site]
            assert len(site_val) == 2
            assert (site_val["label"] == 1).sum() == 1
            assert (site_val["label"] == 0).sum() == 1

    # Invariant 4: Non-overlapping validation partitions that cover every subject
    assert len(all_val_indices) == total_samples
    assert len(set(all_val_indices)) == total_samples


def test_assign_stratified_folds():
    """Verify that assign_stratified_folds annotates DataFrame with fold column."""
    df = pd.DataFrame(
        {
            "subject_id": [str(i) for i in range(20)],
            "site_id": ["SITE_A"] * 10 + ["SITE_B"] * 10,
            "label": [0, 1] * 10,
        }
    )

    annotated = assign_stratified_folds(df, n_splits=5, random_state=123, fold_col="fold")
    assert "fold" in annotated.columns
    assert set(annotated["fold"].unique()) == {0, 1, 2, 3, 4}
    # Each fold should contain exactly 4 subjects (20 / 5)
    assert (annotated["fold"].value_counts() == 4).all()


def test_spatiotemporal_dataset_and_dataloader(tmp_path: Path):
    """Verify that SpatioTemporalFMRIDataset and DataLoader produce expected tensor batches.

    Verifies:
    1. Loads 4D NIfTI files from disk on the fly.
    2. Performs trimming of first and last 10 frames.
    3. Normalizes non-zero voxels and standardizes 3D spatial dimensions to 64x64x36.
    4. Pads temporal duration to T_max=200 and outputs boolean mask.
    5. Collation produces batch volume (B, T, 1, 64, 64, 36), mask (B, T), and label (B,).
    """
    import nibabel as nib

    nii_dir = tmp_path / "nifti_data"
    nii_dir.mkdir()

    # Subject 1: raw shape (61, 73, 61, 50 frames)
    # After trimming: 50 - 20 = 30 frames
    data_1 = np.random.randn(61, 73, 61, 50).astype(np.float32)
    # Background zeros
    data_1[:10, :10, :10, :] = 0.0
    img_1 = nib.Nifti1Image(data_1, affine=np.eye(4))
    file_1 = nii_dir / "Sub1_0050001_func_preproc.nii.gz"
    nib.save(img_1, str(file_1))

    # Subject 2: raw shape (64, 64, 36, 70 frames)
    # After trimming: 70 - 20 = 50 frames
    data_2 = np.random.randn(64, 64, 36, 70).astype(np.float32)
    data_2[:10, :10, :10, :] = 0.0
    img_2 = nib.Nifti1Image(data_2, affine=np.eye(4))
    file_2 = nii_dir / "Sub2_0050002_func_preproc.nii.gz"
    nib.save(img_2, str(file_2))

    manifest_df = pd.DataFrame(
        [
            {
                "subject_id": "50001",
                "site_id": "SITE_A",
                "label": 1,
                "file_path": str(file_1),
            },
            {
                "subject_id": "50002",
                "site_id": "SITE_B",
                "label": 0,
                "file_path": str(file_2),
            },
        ]
    )

    dataset = SpatioTemporalFMRIDataset(
        manifest=manifest_df,
        target_spatial_shape=(64, 64, 36),
        target_seq_len=200,
        trim_start=10,
        trim_end=10,
        normalize=True,
        cache_data=True,
    )

    assert len(dataset) == 2

    # Verify single sample from dataset
    item_0 = dataset[0]
    assert "volume" in item_0 and "mask" in item_0 and "label" in item_0
    assert item_0["volume"].shape == (200, 1, 64, 64, 36)
    assert item_0["mask"].shape == (200,)
    assert item_0["mask"].dtype == torch.bool
    # Subject 1 had 50 frames -> trimmed to 30 frames
    assert item_0["mask"][:30].all()
    assert not item_0["mask"][30:].any()
    assert item_0["label"].item() == 1

    # Verify second sample
    item_1 = dataset[1]
    # Subject 2 had 70 frames -> trimmed to 50 frames
    assert item_1["volume"].shape == (200, 1, 64, 64, 36)
    assert item_1["mask"][:50].all()
    assert not item_1["mask"][50:].any()
    assert item_1["label"].item() == 0

    # Test DataLoader batch collation
    loader = create_spatiotemporal_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert batch["volume"].shape == (2, 200, 1, 64, 64, 36)
    assert batch["mask"].shape == (2, 200)
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.long
    assert batch["mask"].dtype == torch.bool
    assert batch["volume"].dtype == torch.float32
    assert batch["subject_ids"] == ["50001", "50002"]
    assert batch["site_ids"] == ["SITE_A", "SITE_B"]


def test_discover_multisite_dataset_real_data_if_available():
    """Verify that discover_multisite_dataset matches files on real data directory if present."""
    data_dir = Path("data/ABIDE_pcp")
    if not data_dir.exists():
        pytest.skip("Local data/ABIDE_pcp directory not found")

    manifest_df, file_mapping = discover_multisite_dataset(candidate_paths=[data_dir])
    assert len(manifest_df) > 0
    assert len(file_mapping) == len(manifest_df)
    assert "file_path" in manifest_df.columns
    # Check that discovered files exist
    first_row = manifest_df.iloc[0]
    assert Path(first_row["file_path"]).exists()
    assert first_row["site_id"] in ["PITT", "NYU", "CALTECH", "UM_1", "USM", "MAX_MUN"]


def test_dataloader_to_bimamba_integration(tmp_path: Path):
    """Verify that batches from SpatioTemporal DataLoader feed directly into SpatioTemporalBiMamba."""
    import nibabel as nib
    from asd_classification.models.baseline.spatiotemporal_bimamba import SpatioTemporalBiMamba

    nii_dir = tmp_path / "nifti_integration"
    nii_dir.mkdir()

    # Create two minimal 4D NIfTI files
    # T=25 frames raw -> 5 frames after 10+10 trimming
    vol_1 = np.random.randn(64, 64, 36, 25).astype(np.float32)
    file_1 = nii_dir / "Sub1_func_preproc.nii.gz"
    nib.save(nib.Nifti1Image(vol_1, affine=np.eye(4)), str(file_1))

    vol_2 = np.random.randn(64, 64, 36, 25).astype(np.float32)
    file_2 = nii_dir / "Sub2_func_preproc.nii.gz"
    nib.save(nib.Nifti1Image(vol_2, affine=np.eye(4)), str(file_2))

    manifest_df = pd.DataFrame(
        [
            {"subject_id": "1", "site_id": "SITE_A", "label": 1, "file_path": str(file_1)},
            {"subject_id": "2", "site_id": "SITE_B", "label": 0, "file_path": str(file_2)},
        ]
    )

    dataset = SpatioTemporalFMRIDataset(
        manifest=manifest_df,
        target_spatial_shape=(64, 64, 36),
        target_seq_len=10,  # Small T_max for test speed
        trim_start=10,
        trim_end=10,
        normalize=True,
    )

    loader = create_spatiotemporal_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    x = batch["volume"]
    mask = batch["mask"]

    assert x.shape == (2, 10, 1, 64, 64, 36)
    assert mask.shape == (2, 10)
    # Recorded frames were 25 - 20 = 5 frames
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()

    # Pass through SpatioTemporalBiMamba model
    model = SpatioTemporalBiMamba(
        in_channels=1,
        d_model=128,
        num_classes=2,
        d_state=16,
        num_layers=2,
        head_dropout=0.4,
        force_pure_pytorch=True,
    )
    model.eval()

    with torch.no_grad():
        logits, attn_weights = model(x, mask=mask)

    assert logits.shape == (2, 2)
    assert attn_weights.shape == (2, 10)
    # Attention on padded frames 5..9 must be strictly 0.0
    assert torch.allclose(attn_weights[:, 5:], torch.zeros_like(attn_weights[:, 5:]))
    # Attention on valid frames must sum to 1.0
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(2), atol=1e-5)






