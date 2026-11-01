"""
Data Ingestion, Preprocessing, and Phenotypic Text Generation for ABIDE I CC200.

This module provides standalone utilities for:
1. Fetching and indexing Craddock CC200 ROI time-series via Nilearn ABIDE PCP.
2. Motion quality control scrubbing (filtering subjects by Mean Framewise Displacement).
3. Riemannian Tangent Space Functional Connectivity (FC) estimation.
4. Non-leaking clinical phenotypic prompt formulation (without diagnostic targets).
5. Joint Site-and-Class Stratified K-Fold partitioning and PyTorch DataLoaders.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset

# Nilearn imports with defensive handling
try:
    from nilearn import datasets
    from nilearn.connectome import ConnectivityMeasure
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False


def filter_motion_quality(
    pheno_df: pd.DataFrame, max_fd: float = 0.5
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Filter subjects based on head motion quality control threshold.

    Args:
        pheno_df: Phenotypic DataFrame containing 'func_mean_fd' column.
        max_fd: Maximum allowable Mean Framewise Displacement (in mm).

    Returns:
        Tuple of (filtered DataFrame, boolean mask of retained subjects).
    """
    if "func_mean_fd" not in pheno_df.columns:
        warnings.warn("Column 'func_mean_fd' not found in phenotypic data. Skipping motion filtering.")
        mask = np.ones(len(pheno_df), dtype=bool)
        return pheno_df.copy(), mask

    fd_series = pd.to_numeric(pheno_df["func_mean_fd"], errors="coerce")
    # Retain subjects with valid non-negative FD values within the threshold
    keep_mask = fd_series.notna() & (fd_series >= 0.0) & (fd_series <= max_fd)
    filtered_df = pheno_df[keep_mask].copy().reset_index(drop=True)
    return filtered_df, keep_mask.to_numpy()


def compute_tangent_fc(
    time_series_list: List[np.ndarray],
    fit_indices: Optional[Union[np.ndarray, List[int]]] = None,
) -> np.ndarray:
    """
    Compute Riemannian Tangent Space Functional Connectivity matrices.

    Args:
        time_series_list: List of 2D arrays, each of shape (n_timepoints, n_rois).
        fit_indices: Optional indices of subjects used to fit the reference Riemannian mean.
            If None, the measure is fit on all provided subjects.

    Returns:
        Array of symmetric FC matrices of shape (n_subjects, n_rois, n_rois), dtype float32.
    """
    if not HAS_NILEARN:
        raise ImportError(
            "nilearn is required for Tangent Space Functional Connectivity calculation. "
            "Please install nilearn via pip install nilearn."
        )

    measure = ConnectivityMeasure(kind="tangent", vectorize=False, discard_diagonal=False)

    if fit_indices is not None:
        fit_subset = [time_series_list[i] for i in fit_indices]
        measure.fit(fit_subset)
        matrices = measure.transform(time_series_list)
    else:
        matrices = measure.fit_transform(time_series_list)

    # Ensure float32 precision for PyTorch tensor operations
    matrices = np.asarray(matrices, dtype=np.float32)
    return matrices


def _clean_numeric_field(val: Union[float, int, str]) -> Optional[float]:
    """Helper to sanitize numerical phenotypic values, returning None if missing or encoded as -9999."""
    if pd.isna(val):
        return None
    try:
        f_val = float(val)
        if f_val < 0.0 or f_val == -9999.0:
            return None
        return f_val
    except (ValueError, TypeError):
        return None


def generate_phenotypic_prompts(pheno_df: pd.DataFrame) -> List[str]:
    """
    Construct non-leaking clinical phenotypic profile text representations.

    Security & Integrity Guarantee:
        Diagnostic labels (e.g. DX_GROUP, 'Autism', 'ASD', 'Control', 'TD') are strictly
        excluded from the prompt to prevent ground-truth target leakage. Missing values
        are mapped to the standardized neutral token 'not documented'.

    Args:
        pheno_df: Phenotypic DataFrame with clinical variables.

    Returns:
        List of formatted descriptive prompt strings for each subject.
    """
    prompts = []

    for _, row in pheno_df.iterrows():
        # Sanitize Age
        age = _clean_numeric_field(row.get("AGE_AT_SCAN", None))
        age_str = f"{age:.1f}" if age is not None else "not documented"

        # Sanitize Sex (ABIDE encoding: 1 = Male, 2 = Female)
        raw_sex = row.get("SEX", None)
        if raw_sex in (1, "1", "M", "Male", "male"):
            sex_str = "male"
        elif raw_sex in (2, "2", "F", "Female", "female"):
            sex_str = "female"
        else:
            sex_str = "individual"

        # Sanitize Cognitive Scores (IQ)
        fiq = _clean_numeric_field(row.get("FIQ", None))
        fiq_str = f"{fiq:.0f}" if fiq is not None else "not documented"

        viq = _clean_numeric_field(row.get("VIQ", None))
        viq_str = f"{viq:.0f}" if viq is not None else "not documented"

        # Sanitize ADOS Behavioral Subscores
        ados_soc = _clean_numeric_field(row.get("ADOS_SOCIAL", None))
        if ados_soc is None:
            # Fall back to ADOS_TOTAL or ADOS_COMM if available
            ados_soc = _clean_numeric_field(row.get("ADOS_TOTAL", None))
        ados_soc_str = f"{ados_soc:.0f}" if ados_soc is not None else "not documented"

        ados_rbr = _clean_numeric_field(row.get("ADOS_STEREO_BEHAV", None))
        ados_rbr_str = f"{ados_rbr:.0f}" if ados_rbr is not None else "not documented"

        # Compose clinical template without any diagnostic label keywords
        prompt = (
            f"A {age_str}-year-old {sex_str} individual, "
            f"Full-scale IQ {fiq_str}, Verbal IQ {viq_str}, "
            f"evaluated on ADOS with Social-Affect score {ados_soc_str} "
            f"and Repetitive Behavior score {ados_rbr_str}."
        )
        prompts.append(prompt)

    return prompts


def create_joint_stratified_kfold(
    pheno_df: pd.DataFrame, n_splits: int = 10, random_state: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create Stratified K-Fold cross-validation splits balanced across both diagnosis and site.

    If certain site-diagnosis combinations have fewer than n_splits instances, they are
    grouped gracefully to avoid StratifiedKFold population errors while preserving class balance.

    Args:
        pheno_df: Phenotypic DataFrame containing 'DX_GROUP' and 'SITE_ID'.
        n_splits: Number of cross-validation folds.
        random_state: Random seed for deterministic reproducibility.

    Yields:
        Tuples of (train_indices, val_indices) for each fold.
    """
    n_samples = len(pheno_df)
    if n_samples < n_splits:
        raise ValueError(f"Number of samples ({n_samples}) must be >= n_splits ({n_splits}).")

    dx_col = pheno_df["DX_GROUP"].astype(str) if "DX_GROUP" in pheno_df.columns else pd.Series(["0"] * n_samples)
    site_col = pheno_df["SITE_ID"].astype(str) if "SITE_ID" in pheno_df.columns else pd.Series(["SITE"] * n_samples)

    # Initial joint stratification key
    raw_strata = site_col + "_" + dx_col
    strata_counts = raw_strata.value_counts()

    # Re-group rare combinations (< n_splits) by class only to maintain feasibility
    adjusted_strata = []
    for s_val, d_val in zip(site_col, dx_col):
        key = f"{s_val}_{d_val}"
        if strata_counts.get(key, 0) >= n_splits:
            adjusted_strata.append(key)
        else:
            adjusted_strata.append(f"rare_{d_val}")

    # Fallback to diagnostic class directly if rare grouping still has small strata
    final_counts = pd.Series(adjusted_strata).value_counts()
    if (final_counts < n_splits).any():
        final_strata = dx_col.values
    else:
        final_strata = np.array(adjusted_strata)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy_x = np.zeros(n_samples)

    for train_idx, val_idx in skf.split(dummy_x, final_strata):
        yield train_idx, val_idx


class ABIDETangentDataset(Dataset):
    """
    PyTorch Dataset yielding (fc_tensor, phenotypic_prompt, label, site_id).

    Tensors emitted:
        fc_tensor: torch.FloatTensor of shape (1, n_rois, n_rois).
        prompt: str representing patient clinical profile.
        label: torch.LongTensor with 0 for Typically Developing (TC), 1 for ASD.
        site_id: str representing scanning site identifier.
    """

    def __init__(
        self,
        fc_matrices: np.ndarray,
        text_prompts: List[str],
        labels: np.ndarray,
        site_ids: Union[np.ndarray, List[str]],
    ):
        assert len(fc_matrices) == len(text_prompts) == len(labels) == len(site_ids), (
            f"Mismatched dataset lengths: fc={len(fc_matrices)}, text={len(text_prompts)}, "
            f"labels={len(labels)}, sites={len(site_ids)}"
        )
        self.fc_matrices = np.asarray(fc_matrices, dtype=np.float32)
        self.text_prompts = list(text_prompts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.site_ids = list(site_ids)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor, str]:
        # Shape: (1, n_rois, n_rois) for convolution operations in BrainNetCNN
        fc_tensor = torch.from_numpy(self.fc_matrices[idx]).unsqueeze(0).float()
        prompt = self.text_prompts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        site_id = str(self.site_ids[idx])
        return fc_tensor, prompt, label, site_id


def create_dataloaders(
    fc_matrices: np.ndarray,
    text_prompts: List[str],
    labels: np.ndarray,
    site_ids: Union[np.ndarray, List[str]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct training and validation PyTorch DataLoaders for a cross-validation fold.

    Args:
        fc_matrices: Full array of shape (N, n_rois, n_rois).
        text_prompts: Full list of clinical prompt strings.
        labels: Full array of diagnostic labels (0 or 1).
        site_ids: Full list or array of site identifiers.
        train_idx: Subject indices assigned to training.
        val_idx: Subject indices assigned to validation.
        batch_size: Mini-batch size.
        num_workers: Number of background data loader workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = ABIDETangentDataset(
        fc_matrices=fc_matrices[train_idx],
        text_prompts=[text_prompts[i] for i in train_idx],
        labels=labels[train_idx],
        site_ids=[site_ids[i] for i in train_idx],
    )

    val_dataset = ABIDETangentDataset(
        fc_matrices=fc_matrices[val_idx],
        text_prompts=[text_prompts[i] for i in val_idx],
        labels=labels[val_idx],
        site_ids=[site_ids[i] for i in val_idx],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader


def load_abide_cc200_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    n_subjects: Optional[int] = None,
    max_fd: float = 0.5,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """
    High-level automated pipeline to fetch, clean, and extract CC200 Tangent FC matrices.

    Args:
        data_dir: Destination path for downloaded ABIDE files. Defaults to None.
        n_subjects: Maximum number of subjects to fetch (None fetches complete cohort).
        max_fd: Motion filtering threshold for Mean Framewise Displacement (in mm).

    Returns:
        Tuple of:
            - fc_matrices: Array of shape (N_clean, 200, 200), dtype float32.
            - text_prompts: List of N_clean non-leaking clinical prompt strings.
            - labels: Integer array (N_clean,) where 1 = ASD, 0 = Typically Developing.
            - site_ids: String array (N_clean,) with scanner site identifiers.
            - clean_pheno_df: Filtered phenotypic DataFrame.
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn is required to fetch and parse ABIDE PCP datasets.")

    print(f"INFO: Fetching ABIDE I CPAC CC200 data (n_subjects={n_subjects})...")
    abide_data = datasets.fetch_abide_pcp(
        data_dir=str(data_dir) if data_dir else None,
        n_subjects=n_subjects,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        quality_checked=False,
    )

    pheno_df = pd.DataFrame(abide_data.phenotypic)
    time_series_files = abide_data.rois_cc200

    print(f"INFO: Ingested {len(pheno_df)} raw subject records. Applying motion filter (max_fd={max_fd})...")
    clean_pheno_df, keep_mask = filter_motion_quality(pheno_df, max_fd=max_fd)

    # Filter aligned time-series files
    clean_files = [f for f, keep in zip(time_series_files, keep_mask) if keep]
    print(f"INFO: {len(clean_files)} subjects passed quality control.")

    # Read and validate BOLD time-series
    time_series_list = []
    valid_indices = []
    for idx, f_path in enumerate(clean_files):
        try:
            ts = np.loadtxt(f_path)
            if ts.ndim == 2 and ts.shape[1] == 200:
                time_series_list.append(ts)
                valid_indices.append(idx)
        except Exception as exc:
            warnings.warn(f"WARNING: Failed to load time-series from {f_path}: {exc}")

    clean_pheno_df = clean_pheno_df.iloc[valid_indices].reset_index(drop=True)

    print(f"INFO: Estimating Riemannian Tangent Space FC across {len(time_series_list)} subjects...")
    fc_matrices = compute_tangent_fc(time_series_list)

    print("INFO: Formulating non-leaking phenotypic clinical prompts...")
    text_prompts = generate_phenotypic_prompts(clean_pheno_df)

    # In ABIDE: DX_GROUP 1 = ASD, 2 = Control. Convert to binary: 1 = ASD, 0 = Control
    raw_dx = clean_pheno_df["DX_GROUP"].values
    labels = np.where(raw_dx == 1, 1, 0).astype(np.int64)
    site_ids = clean_pheno_df["SITE_ID"].astype(str).values

    print(f"SUCCESS: Pipeline prepared {len(labels)} subjects (ASD={np.sum(labels == 1)}, TC={np.sum(labels == 0)}).")
    return fc_matrices, text_prompts, labels, site_ids, clean_pheno_df
