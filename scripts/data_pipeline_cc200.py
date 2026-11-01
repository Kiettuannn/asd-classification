"""
Data Ingestion, Preprocessing, and Phenotypic Text Generation for ABIDE I CC200.

This module provides standalone utilities for:
1. Fetching and indexing Craddock CC200 ROI time-series via Nilearn ABIDE PCP.
2. Motion quality control scrubbing (filtering subjects by Mean Framewise Displacement).
3. Riemannian Tangent Space Functional Connectivity (FC) estimation with fold-wise fitting.
4. Non-leaking clinical phenotypic prompt formulation (without diagnostic targets).
5. Joint Site-and-Class Stratified K-Fold partitioning and PyTorch DataLoaders.
"""

from dataclasses import dataclass
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


@dataclass
class CohortDataBundle:
    """Encapsulates aligned cohort tensors and clinical phenotypic attributes."""
    fc_matrices: np.ndarray  # Shape: (N, n_rois, n_rois)
    text_prompts: List[str]  # Length: N
    labels: np.ndarray  # Shape: (N,) binary targets (1 = ASD, 0 = TC)
    site_ids: np.ndarray  # Shape: (N,) string identifiers
    site_codes: np.ndarray  # Shape: (N,) integer encoded identifiers
    site_to_idx: Dict[str, int]  # Mapping from site name to integer code
    pheno_df: pd.DataFrame  # Filtered metadata


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

    To eliminate cross-validation data leakage, fit_indices should supply the
    indices of training subjects. The reference Riemannian mean covariance is
    estimated strictly on the training set and then projected onto all subjects.

    Args:
        time_series_list: List of 2D arrays, each of shape (n_timepoints, n_rois).
        fit_indices: Optional subject indices to fit the Riemannian mean covariance.
            If None, the measure is fit on all provided time series.

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

    return np.asarray(matrices, dtype=np.float32)


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


def _format_score(val: Optional[float]) -> str:
    """Helper to format numeric score or return standard missing token."""
    return f"{val:.0f}" if val is not None else "not documented"


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
        age = _clean_numeric_field(row.get("AGE_AT_SCAN", None))
        age_str = f"{age:.1f}" if age is not None else "not documented"

        raw_sex = row.get("SEX", None)
        if raw_sex in (1, "1", "M", "Male", "male"):
            subject_descriptor = f"{age_str}-year-old male"
        elif raw_sex in (2, "2", "F", "Female", "female"):
            subject_descriptor = f"{age_str}-year-old female"
        else:
            subject_descriptor = f"{age_str}-year-old individual"

        fiq_str = _format_score(_clean_numeric_field(row.get("FIQ", None)))
        viq_str = _format_score(_clean_numeric_field(row.get("VIQ", None)))

        ados_soc_str = _format_score(_clean_numeric_field(row.get("ADOS_SOCIAL", None)))
        ados_rep_behav_str = _format_score(_clean_numeric_field(row.get("ADOS_STEREO_BEHAV", None)))

        prompt = (
            f"A {subject_descriptor}, "
            f"Full-scale IQ {fiq_str}, Verbal IQ {viq_str}, "
            f"evaluated on ADOS with Social-Affect score {ados_soc_str} "
            f"and Repetitive Behavior score {ados_rep_behav_str}."
        )
        prompts.append(prompt)

    return prompts


def create_joint_stratified_kfold(
    pheno_df: pd.DataFrame, n_splits: int = 10, random_state: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create Stratified K-Fold cross-validation splits balanced across both diagnosis and site.

    Uses composite multi-factor stratification to balance scanner site representations
    and diagnostic classes without dropping site covariates.

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

    dx_series = pheno_df["DX_GROUP"].astype(str) if "DX_GROUP" in pheno_df.columns else pd.Series(["0"] * n_samples)
    site_series = pheno_df["SITE_ID"].astype(str) if "SITE_ID" in pheno_df.columns else pd.Series(["SITE"] * n_samples)

    # Bin sites by frequency so smaller scanner sites are aggregated by class
    site_counts = site_series.value_counts()
    major_sites = set(site_counts[site_counts >= n_splits].index)

    binned_strata = []
    for site, dx in zip(site_series, dx_series):
        if site in major_sites:
            binned_strata.append(f"{site}_{dx}")
        else:
            binned_strata.append(f"MINOR_SITE_{dx}")

    strata_counts = pd.Series(binned_strata).value_counts()
    if (strata_counts < 2).any():
        final_strata = dx_series.values
    else:
        min_count = strata_counts.min()
        if min_count < n_splits:
            final_strata = [
                s if strata_counts[s] >= n_splits else f"COMBINED_{dx}"
                for s, dx in zip(binned_strata, dx_series)
            ]
        else:
            final_strata = binned_strata

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy_x = np.zeros(n_samples)

    for train_idx, val_idx in skf.split(dummy_x, final_strata):
        yield train_idx, val_idx


class ABIDETangentDataset(Dataset):
    """
    PyTorch Dataset yielding (fc_tensor, phenotypic_prompt, label, site_code).

    Tensors emitted:
        fc_tensor: torch.FloatTensor of shape (1, n_rois, n_rois).
        prompt: str representing patient clinical profile.
        label: torch.LongTensor with 0 for Typically Developing (TC), 1 for ASD.
        site_code: torch.LongTensor representing integer-encoded scanner site.
    """

    def __init__(
        self,
        fc_matrices: np.ndarray,
        text_prompts: List[str],
        labels: np.ndarray,
        site_codes: np.ndarray,
        site_ids: Optional[Union[np.ndarray, List[str]]] = None,
    ):
        assert len(fc_matrices) == len(text_prompts) == len(labels) == len(site_codes), (
            f"Mismatched dataset lengths: fc={len(fc_matrices)}, text={len(text_prompts)}, "
            f"labels={len(labels)}, site_codes={len(site_codes)}"
        )
        self.fc_matrices = np.asarray(fc_matrices, dtype=np.float32)
        self.text_prompts = list(text_prompts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.site_codes = np.asarray(site_codes, dtype=np.int64)
        self.site_ids = list(site_ids) if site_ids is not None else [str(c) for c in site_codes]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
        fc_tensor = torch.from_numpy(self.fc_matrices[idx]).unsqueeze(0).float()
        prompt = self.text_prompts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        site_code = torch.tensor(self.site_codes[idx], dtype=torch.long)
        return fc_tensor, prompt, label, site_code


def create_dataloaders(
    bundle: CohortDataBundle,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct training and validation PyTorch DataLoaders from a CohortDataBundle.

    Args:
        bundle: Preprocessed CohortDataBundle instance.
        train_idx: Subject indices assigned to training.
        val_idx: Subject indices assigned to validation.
        batch_size: Mini-batch size.
        num_workers: Number of background data loader workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    def _make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = ABIDETangentDataset(
            fc_matrices=bundle.fc_matrices[indices],
            text_prompts=[bundle.text_prompts[i] for i in indices],
            labels=bundle.labels[indices],
            site_codes=bundle.site_codes[indices],
            site_ids=[bundle.site_ids[i] for i in indices],
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return _make_loader(train_idx, shuffle=True), _make_loader(val_idx, shuffle=False)


def load_abide_cc200_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    n_subjects: Optional[int] = None,
    max_fd: float = 0.5,
) -> CohortDataBundle:
    """
    High-level automated pipeline to fetch, clean, and extract CC200 Tangent FC matrices.

    Args:
        data_dir: Destination path for downloaded ABIDE files. Defaults to None.
        n_subjects: Maximum number of subjects to fetch (None fetches complete cohort).
        max_fd: Motion filtering threshold for Mean Framewise Displacement (in mm).

    Returns:
        CohortDataBundle with FC matrices, prompt strings, binary labels, site codes, and metadata.
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

    clean_files = [f for f, keep in zip(time_series_files, keep_mask) if keep]
    print(f"INFO: {len(clean_files)} subjects passed quality control.")

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

    raw_dx = clean_pheno_df["DX_GROUP"].values
    labels = np.where(raw_dx == 1, 1, 0).astype(np.int64)

    site_ids = clean_pheno_df["SITE_ID"].astype(str).values
    unique_sites = sorted(list(set(site_ids)))
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}
    site_codes = np.array([site_to_idx[s] for s in site_ids], dtype=np.int64)

    print(f"SUCCESS: Pipeline prepared {len(labels)} subjects (ASD={np.sum(labels == 1)}, TC={np.sum(labels == 0)}).")
    return CohortDataBundle(
        fc_matrices=fc_matrices,
        text_prompts=text_prompts,
        labels=labels,
        site_ids=site_ids,
        site_codes=site_codes,
        site_to_idx=site_to_idx,
        pheno_df=clean_pheno_df,
    )
