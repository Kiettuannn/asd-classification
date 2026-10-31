"""Multi-site 4D rs-fMRI DataLoader and cross-validation pipeline.

Provides dynamic dataset auto-discovery across Kaggle input directories and local paths,
neuroimaging preprocessing (scanner boundary frame trimming, subject-level non-zero z-scoring,
spatial standardization to 64x64x36), sequence standardization (padding to T_max=200 with
boolean attention masks), and joint Site-and-Class Stratified 5-Fold Cross-Validation.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



def discover_multisite_dataset(
    candidate_paths: Optional[Sequence[Union[str, Path]]] = None,
    recursive: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Dynamically discover phenotypic manifest and 4D functional NIfTI files.

    Searches candidate paths (including mounted /kaggle/input/ storage and local paths)
    for phenotypic CSV tables and preprocessed functional NIfTI files (*func_preproc.nii.gz),
    pairing each subject record with its corresponding volumetric scan.

    Args:
        candidate_paths: Sequence of root directory paths to inspect. If None, default
            Kaggle and local project paths are inspected.
        recursive: Whether to search subdirectories recursively.

    Returns:
        Tuple containing:
            - Standardized manifest DataFrame with paired 'file_path', 'label' (ASD=1, TC=0),
              'subject_id', and phenotypic metadata.
            - Dictionary mapping string subject IDs to resolved Path objects.

    Raises:
        FileNotFoundError: If no phenotypic CSV or functional NIfTI files are found.
        ValueError: If phenotypic CSV format is invalid or missing required columns.
    """
    if candidate_paths is None:
        search_roots = [
            Path("/kaggle/input"),
            Path("data"),
            Path("notebooks/data"),
            Path("notebooks/abide_data"),
        ]
    else:
        search_roots = [Path(p) for p in candidate_paths]

    # Resolve paths and eliminate duplicates from overlapping search trees
    seen_roots = set()
    existing_roots: List[Path] = []
    for p in search_roots:
        if p.exists():
            resolved = p.resolve()
            if resolved not in seen_roots:
                seen_roots.add(resolved)
                existing_roots.append(p)

    if not existing_roots:
        raise FileNotFoundError(
            f"None of the candidate paths exist: {[str(p) for p in search_roots]}"
        )

    # 1. Discover all candidate preprocessed .pt files (excluding checkpoints)
    pt_files_dict: Dict[str, Path] = {}
    for root in existing_roots:
        pattern = "**/*.pt" if recursive else "*.pt"
        for pt_path in root.glob(pattern):
            name = pt_path.name
            if not name.startswith("best_") and not name.startswith("checkpoint_") and not name.startswith("model_"):
                sub_key = pt_path.stem
                if sub_key.startswith("sub-"):
                    sub_key = sub_key[4:]
                if sub_key not in pt_files_dict:
                    pt_files_dict[sub_key] = pt_path.resolve()

    # 2. Locate phenotypic summary tables or preprocessed manifests across candidate locations
    discovered_csvs: List[Path] = []
    for root in existing_roots:
        if root.is_file() and root.suffix == ".csv":
            discovered_csvs.append(root)
            continue
        pattern = "**/*.csv" if recursive else "*.csv"
        csv_candidates = list(root.glob(pattern))
        for candidate in csv_candidates:
            lower_name = candidate.name.lower()
            if any(k in lower_name for k in ["phenotypic", "metadata", "manifest", "preprocessed"]):
                discovered_csvs.append(candidate)

    # Deduplicate resolved CSV paths
    unique_csvs: List[Path] = []
    seen_csv_paths = set()
    for c in discovered_csvs:
        res = c.resolve()
        if res not in seen_csv_paths:
            seen_csv_paths.add(res)
            unique_csvs.append(c)

    if not unique_csvs and not pt_files_dict:
        raise FileNotFoundError(
            "Could not locate phenotypic CSV or preprocessed .pt files across candidate paths: "
            f"{[str(p) for p in existing_roots]}"
        )

    # 3. Collect functional NIfTI acquisitions (*func_preproc.nii.gz, *.nii.gz, *.nii)
    nifti_files_set = set()
    for root in existing_roots:
        patterns = ["**/*func_preproc.nii*", "**/*.nii.gz", "**/*.nii"] if recursive else ["*func_preproc.nii*", "*.nii.gz", "*.nii"]
        for pattern in patterns:
            for nifti_path in root.glob(pattern):
                nifti_files_set.add(nifti_path.resolve())

    nifti_files: List[Path] = sorted(list(nifti_files_set))

    # Standardize manifests from all discovered CSVs (supports multi-part datasets)
    dfs: List[pd.DataFrame] = []
    for csv_path in unique_csvs:
        try:
            raw_df = pd.read_csv(csv_path)
        except Exception:
            continue

        if raw_df.empty:
            continue

        df_part = raw_df.copy()
        col_map = {col: col.upper() for col in df_part.columns}
        rev_col_map = {v: k for k, v in col_map.items()}

        sub_col = rev_col_map.get("SUB_ID") or rev_col_map.get("SUBJECT_ID")
        if sub_col is None:
            continue
        df_part["subject_id"] = df_part[sub_col].astype(str).str.strip()

        # Map ABIDE DX_GROUP (1 = Autism/ASD, 2 = Control/TC) into binary classification targets (ASD=1, TC=0)
        dx_col = rev_col_map.get("DX_GROUP") or rev_col_map.get("DX") or rev_col_map.get("LABEL")
        if dx_col is not None:
            unique_vals = set(df_part[dx_col].dropna().unique())
            if {1, 2}.issubset(unique_vals) or 2 in unique_vals:
                df_part["label"] = (df_part[dx_col] == 1).astype(int)
            else:
                df_part["label"] = df_part[dx_col].astype(int)
        else:
            continue

        site_col = rev_col_map.get("SITE_ID") or rev_col_map.get("SITE")
        df_part["site_id"] = df_part[site_col].astype(str).str.strip() if site_col else "UNKNOWN"

        file_id_col = rev_col_map.get("FILE_ID")
        if file_id_col:
            df_part["file_id"] = df_part[file_id_col].astype(str).str.strip()
        else:
            df_part["file_id"] = "no_filename"

        # Check for preprocessed pt_file column or generic file_path
        pt_col = rev_col_map.get("PT_FILE")
        fp_col = rev_col_map.get("FILE_PATH")

        resolved_paths: List[Optional[Path]] = []
        for _, r in df_part.iterrows():
            sid = str(r["subject_id"])
            candidate_target: Optional[Path] = None

            # Prioritize matching against discovered .pt files
            if sid in pt_files_dict:
                candidate_target = pt_files_dict[sid]
            elif sid.isdigit() and str(int(sid)) in pt_files_dict:
                candidate_target = pt_files_dict[str(int(sid))]
            elif pt_col and pd.notna(r[pt_col]):
                pt_val = Path(str(r[pt_col]).strip())
                cand = csv_path.parent / "scans" / pt_val.name
                if cand.exists():
                    candidate_target = cand.resolve()
                elif (csv_path.parent / pt_val.name).exists():
                    candidate_target = (csv_path.parent / pt_val.name).resolve()

            # Fallback to generic file_path if provided
            if candidate_target is None and fp_col and pd.notna(r[fp_col]):
                val = str(r[fp_col]).strip()
                if val:
                    p_val = Path(val)
                    if p_val.is_absolute() and p_val.exists():
                        candidate_target = p_val
                    elif (csv_path.parent / p_val).exists():
                        candidate_target = (csv_path.parent / p_val).resolve()
                    elif (csv_path.parent / "scans" / p_val.name).exists():
                        candidate_target = (csv_path.parent / "scans" / p_val.name).resolve()
                    elif (csv_path.parent.parent / "scans" / p_val.name).exists():
                        candidate_target = (csv_path.parent.parent / "scans" / p_val.name).resolve()

            resolved_paths.append(candidate_target)

        df_part["file_path"] = pd.Series(resolved_paths, dtype=object)
        dfs.append(df_part)

    # If no valid CSV manifest records but preprocessed .pt files exist, reconstruct manifest directly from .pt payloads
    if not dfs and pt_files_dict:
        records = []
        for sid, pt_path in sorted(pt_files_dict.items()):
            try:
                payload = torch.load(pt_path, map_location="cpu", weights_only=True)
                records.append({
                    "subject_id": str(payload.get("subject_id", sid)),
                    "site_id": str(payload.get("site_id", "UNKNOWN")),
                    "label": int(payload.get("label", 0)),
                    "file_id": "preprocessed_pt",
                    "file_path": pt_path,
                })
            except Exception:
                continue
        if records:
            dfs.append(pd.DataFrame(records))

    if not dfs:
        raise ValueError(
            "Could not extract valid subject records from candidate CSVs or .pt archives: "
            f"{[str(p) for p in existing_roots]}"
        )

    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset=["subject_id"], keep="first", inplace=True)

    # Match each subject without a resolved file_path against preprocessed .pt or NIfTI files
    file_map: Dict[str, Path] = {}
    for _, row in df.iterrows():
        sid = str(row["subject_id"])
        if row.get("file_path") is not None and pd.notna(row["file_path"]):
            file_map[sid] = Path(row["file_path"])
        elif sid in pt_files_dict:
            file_map[sid] = pt_files_dict[sid]
        elif sid.isdigit() and str(int(sid)) in pt_files_dict:
            file_map[sid] = pt_files_dict[str(int(sid))]

    if nifti_files:
        unmatched_subs = {
            str(row["subject_id"]): str(row["file_id"])
            for _, row in df.iterrows()
            if str(row["subject_id"]) not in file_map
        }
        for nifti in nifti_files:
            name = nifti.name
            matched_sid: Optional[str] = None
            for sub_id, file_id in unmatched_subs.items():
                matched = False
                if file_id != "no_filename" and file_id in name:
                    matched = True
                elif f"00{sub_id}" in name or f"_{sub_id}_" in name or name.startswith(f"{sub_id}_"):
                    matched = True
                elif re.search(rf"(?:^|[^0-9])0*{re.escape(sub_id)}(?:[^0-9]|$)", name):
                    matched = True

                if matched:
                    matched_sid = sub_id
                    break

            if matched_sid is not None:
                file_map[matched_sid] = nifti.resolve()
                del unmatched_subs[matched_sid]

    df["file_path"] = df["subject_id"].map(lambda sid: file_map.get(str(sid)))
    matched_df = df[df["file_path"].notna()].copy()
    if matched_df.empty:
        raise FileNotFoundError(
            "Could not locate matching functional NIfTI or preprocessed .pt files for subjects in manifest "
            f"across candidate paths: {[str(p) for p in existing_roots]}"
        )

    matched_df["file_path"] = matched_df["file_path"].map(lambda p: Path(p))
    matched_df.reset_index(drop=True, inplace=True)

    return matched_df, {str(row["subject_id"]): Path(row["file_path"]) for _, row in matched_df.iterrows()}




def trim_temporal_frames(
    volume: Union[np.ndarray, torch.Tensor],
    trim_start: int = 10,
    trim_end: int = 10,
    time_axis: int = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """Trim boundary temporal frames to remove scanner equilibrium and termination artifacts.

    Args:
        volume: 4D or 5D neuroimaging array or tensor containing temporal frames.
        trim_start: Number of initial frames to discard (default: 10).
        trim_end: Number of trailing frames to discard (default: 10).
        time_axis: Axis index corresponding to the temporal dimension (e.g. 0 or -1).

    Returns:
        Trimmed array or tensor with reduced temporal duration.

    Raises:
        ValueError: If total sequence duration is less than or equal to trim_start + trim_end.
    """
    total_frames = volume.shape[time_axis]
    min_required = trim_start + trim_end
    if total_frames <= min_required:
        raise ValueError(
            f"Cannot trim {trim_start} start and {trim_end} end frames: "
            f"volume temporal axis {time_axis} only has {total_frames} frames (need > {min_required})."
        )

    end_idx = total_frames - trim_end
    # Normalize time_axis to positive index
    axis = time_axis if time_axis >= 0 else volume.ndim + time_axis

    slices = [slice(None)] * volume.ndim
    slices[axis] = slice(trim_start, end_idx)

    return volume[tuple(slices)]


def normalize_nonzero_voxels(
    volume: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """Apply subject-level z-score intensity normalization strictly across non-zero voxels.

    Non-zero voxels represent anatomical brain tissue, while zero voxels represent
    background air mask. Preserving zeros prevents baseline drift and skull-stripping
    distortion.

    Args:
        volume: Neuroimaging array or tensor.
        eps: Small numerical constant to avoid division by zero.

    Returns:
        Intensity-normalized volume with zero background preserved.
    """
    if isinstance(volume, torch.Tensor):
        mask = volume != 0
        if not torch.any(mask):
            return volume.clone()

        voxels = volume[mask].to(torch.float32)
        mean_val = torch.mean(voxels)
        std_val = torch.std(voxels)
        std_denom = std_val if std_val > eps else torch.tensor(1.0, dtype=mean_val.dtype, device=mean_val.device)

        normalized = torch.zeros_like(volume, dtype=torch.float32)
        normalized[mask] = (voxels - mean_val) / std_denom
        return normalized.to(volume.dtype)

    mask_np = volume != 0
    if not np.any(mask_np):
        return volume.copy()

    voxels_np = volume[mask_np].astype(np.float32)
    mean_val_np = np.mean(voxels_np)
    std_val_np = np.std(voxels_np)
    std_denom_np = std_val_np if std_val_np > eps else 1.0

    normalized_np = np.zeros_like(volume, dtype=np.float32)
    normalized_np[mask_np] = (voxels_np - mean_val_np) / std_denom_np
    return normalized_np.astype(volume.dtype)


def standardize_spatial_volume(
    volume: Union[np.ndarray, torch.Tensor],
    target_shape: Tuple[int, int, int] = (64, 64, 36),
    mode: str = "crop_pad",
) -> Union[np.ndarray, torch.Tensor]:
    """Standardize 3D spatial dimensions of volumes or sequences to target shape.

    Spatial dimensions are assumed to be the final 3 dimensions (..., H, W, D).

    Args:
        volume: Array or tensor with spatial dimensions at the trailing 3 axes.
        target_shape: Desired 3D spatial dimensions (default: 64, 64, 36).
        mode: Standardization mode: 'crop_pad' (symmetric zero-padding and center cropping)
            or 'resample' (trilinear spatial interpolation).

    Returns:
        Standardized volume or sequence with trailing dimensions matching target_shape.
    """
    is_torch = isinstance(volume, torch.Tensor)
    original_ndim = volume.ndim
    if original_ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions for spatial volume, got shape {volume.shape}")

    current_shape = volume.shape[-3:]
    if tuple(current_shape) == tuple(target_shape):
        return volume

    if mode == "resample":
        # PyTorch F.interpolate requires 5D tensor: (N, C, D, H, W)
        tensor_vol: torch.Tensor = volume if isinstance(volume, torch.Tensor) else torch.from_numpy(volume)
        orig_dtype = tensor_vol.dtype

        # Flatten leading dimensions into batch dimension N
        leading_shape = tensor_vol.shape[:-3]
        num_vols = int(np.prod(leading_shape)) if leading_shape else 1

        reshaped = tensor_vol.reshape(num_vols, 1, *current_shape).to(torch.float32)
        resampled = F.interpolate(
            reshaped,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).to(orig_dtype)

        final_shape = (*leading_shape, *target_shape)
        output_tensor = resampled.reshape(final_shape)
        return output_tensor if is_torch else output_tensor.cpu().numpy()

    if mode == "crop_pad":
        if isinstance(volume, torch.Tensor):
            processed_t: torch.Tensor = volume
            for axis_offset in range(3):
                axis = original_ndim - 3 + axis_offset
                curr_size = processed_t.shape[axis]
                targ_size = target_shape[axis_offset]

                if curr_size == targ_size:
                    continue
                elif curr_size < targ_size:
                    pad_front = (targ_size - curr_size) // 2
                    pad_back = targ_size - curr_size - pad_front
                    pad_tuple = [0, 0] * original_ndim
                    rev_axis = original_ndim - 1 - axis
                    pad_tuple[rev_axis * 2] = pad_front
                    pad_tuple[rev_axis * 2 + 1] = pad_back
                    processed_t = F.pad(processed_t, tuple(pad_tuple), mode="constant", value=0)
                else:
                    crop_start = (curr_size - targ_size) // 2
                    crop_end = crop_start + targ_size
                    slices = [slice(None)] * original_ndim
                    slices[axis] = slice(crop_start, crop_end)
                    processed_t = processed_t[tuple(slices)]
            return processed_t

        processed_np: np.ndarray = volume
        for axis_offset in range(3):
            axis = original_ndim - 3 + axis_offset
            curr_size = processed_np.shape[axis]
            targ_size = target_shape[axis_offset]

            if curr_size == targ_size:
                continue
            elif curr_size < targ_size:
                pad_front = (targ_size - curr_size) // 2
                pad_back = targ_size - curr_size - pad_front
                pad_width = [(0, 0)] * original_ndim
                pad_width[axis] = (pad_front, pad_back)
                processed_np = np.pad(processed_np, pad_width, mode="constant", constant_values=0)
            else:
                crop_start = (curr_size - targ_size) // 2
                crop_end = crop_start + targ_size
                slices = [slice(None)] * original_ndim
                slices[axis] = slice(crop_start, crop_end)
                processed_np = processed_np[tuple(slices)]
        return processed_np

    raise ValueError(f"Unknown spatial standardization mode: '{mode}'. Expected 'crop_pad' or 'resample'.")



def pad_temporal_sequence(
    sequence: Union[np.ndarray, torch.Tensor],
    target_length: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or truncate variable temporal sequence to uniform duration with boolean mask.

    Args:
        sequence: Sequence array or tensor where axis 0 is the temporal dimension (T, ...).
        target_length: Maximum target temporal frames T_max (default: 200).

    Returns:
        Tuple containing:
            - Standardized tensor of shape (target_length, ...)
            - Boolean mask tensor of shape (target_length,) where True indicates recorded
              valid frames and False indicates zero-padded frames.
    """
    tensor_seq = sequence if isinstance(sequence, torch.Tensor) else torch.from_numpy(sequence)
    curr_t = tensor_seq.shape[0]

    if curr_t >= target_length:
        truncated = tensor_seq[:target_length]
        mask = torch.ones(target_length, dtype=torch.bool, device=tensor_seq.device)
        return truncated, mask

    # Zero-pad remaining frames
    pad_shape = (target_length - curr_t, *tensor_seq.shape[1:])
    zeros = torch.zeros(pad_shape, dtype=tensor_seq.dtype, device=tensor_seq.device)
    padded = torch.cat([tensor_seq, zeros], dim=0)

    mask = torch.cat(
        [
            torch.ones(curr_t, dtype=torch.bool, device=tensor_seq.device),
            torch.zeros(target_length - curr_t, dtype=torch.bool, device=tensor_seq.device),
        ],
        dim=0,
    )
    return padded, mask


def get_site_and_class_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    site_col: str = "site_id",
    label_col: str = "label",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate joint Site-and-Class Stratified K-Fold train and validation index splits.

    Partitions subjects such that each fold contains balanced proportions of ASD/TC
    diagnostic classes while maintaining representation across all acquisition sites/institutions.

    Args:
        df: DataFrame containing subject records.
        n_splits: Number of cross-validation folds (default: 5).
        random_state: Seed for pseudo-random shuffling reproducibility.
        site_col: Column name identifying the institution/site (default: 'site_id').
        label_col: Column name identifying binary diagnostic label (default: 'label').

    Returns:
        List of (train_indices, val_indices) numpy integer arrays for each fold.

    Raises:
        ValueError: If required columns are missing or n_splits < 2.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if site_col not in df.columns:
        raise ValueError(f"DataFrame missing site column: '{site_col}'")
    if label_col not in df.columns:
        raise ValueError(f"DataFrame missing label column: '{label_col}'")

    n_samples = len(df)
    if n_samples < n_splits:
        raise ValueError(f"Cannot partition {n_samples} samples into {n_splits} folds.")

    rng = np.random.RandomState(random_state)
    fold_assignments = np.full(n_samples, -1, dtype=int)

    # Group integer row indices by (site, label)
    groups = df.groupby([site_col, label_col], sort=True).indices

    remainder_offset = 0
    for _, indices in groups.items():
        idx_arr = np.array(indices, copy=True)
        rng.shuffle(idx_arr)
        n_items = len(idx_arr)

        # Distribute complete cycles of n_splits
        n_full = (n_items // n_splits) * n_splits
        if n_full > 0:
            fold_assignments[idx_arr[:n_full]] = np.tile(np.arange(n_splits), n_full // n_splits)

        # Distribute remaining items round-robin to preserve fold balance across strata
        remainder = n_items % n_splits
        if remainder > 0:
            rem_folds = [(remainder_offset + i) % n_splits for i in range(remainder)]
            fold_assignments[idx_arr[n_full:]] = rem_folds
            remainder_offset = (remainder_offset + remainder) % n_splits

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(n_samples)
    for fold_idx in range(n_splits):
        val_idx = all_indices[fold_assignments == fold_idx]
        train_idx = all_indices[fold_assignments != fold_idx]
        splits.append((train_idx, val_idx))

    return splits


def assign_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    site_col: str = "site_id",
    label_col: str = "label",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Annotate DataFrame with a fold column from joint Site-and-Class Stratification.

    Args:
        df: Input DataFrame containing subject records.
        n_splits: Number of cross-validation folds (default: 5).
        random_state: Seed for pseudo-random shuffling reproducibility.
        site_col: Column name identifying the site (default: 'site_id').
        label_col: Column name identifying the label (default: 'label').
        fold_col: Target column name for fold assignment (default: 'fold').

    Returns:
        New DataFrame with assigned integer fold column [0 .. n_splits-1].
    """
    splits = get_site_and_class_stratified_folds(
        df=df,
        n_splits=n_splits,
        random_state=random_state,
        site_col=site_col,
        label_col=label_col,
    )

    result_df = df.copy()
    result_df[fold_col] = -1

    for fold_idx, (_, val_idx) in enumerate(splits):
        result_df.loc[result_df.index[val_idx], fold_col] = fold_idx

    return result_df



class SpatioTemporalFMRIDataset(Dataset):
    """PyTorch Dataset for 4D rs-fMRI neuroimaging sequences.

    Applies on-the-fly preprocessing and sequence standardization:
    1. Scanner boundary frame trimming (discards first 10 and last 10 TRs).
    2. Non-zero voxel z-score intensity normalization preserving background zero air mask.
    3. Spatial standardization to target resolution (default: 64x64x36).
    4. Sequence duration standardization (pads/truncates to T_max=200 with boolean masks).
    5. Formats output volume tensor to (T_max, 1, 64, 64, 36).
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        target_spatial_shape: Tuple[int, int, int] = (64, 64, 36),
        target_seq_len: int = 200,
        trim_start: int = 10,
        trim_end: int = 10,
        normalize: bool = True,
        spatial_mode: str = "crop_pad",
        cache_data: bool = False,
    ) -> None:
        """Initialize SpatioTemporalFMRIDataset.

        Args:
            manifest: DataFrame or path to CSV table with subject metadata and 'file_path'.
            target_spatial_shape: Target 3D volume dimensions (default: 64, 64, 36).
            target_seq_len: Maximum sequence duration T_max (default: 200).
            trim_start: Number of initial boundary frames to discard (default: 10).
            trim_end: Number of trailing boundary frames to discard (default: 10).
            normalize: Whether to apply non-zero voxel z-score intensity normalization.
            spatial_mode: Standardization mode ('crop_pad' or 'resample').
            cache_data: Whether to cache processed tensors in memory across epochs.
        """
        if isinstance(manifest, (str, Path)):
            self.manifest = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.manifest = manifest.copy().reset_index(drop=True)
        else:
            raise TypeError(f"Expected DataFrame or Path for manifest, got {type(manifest)}")

        if "file_path" not in self.manifest.columns:
            raise ValueError("Manifest must contain a 'file_path' column specifying NIfTI paths.")
        if "label" not in self.manifest.columns and "dx_group" not in self.manifest.columns:
            raise ValueError("Manifest must contain a 'label' or 'dx_group' column.")

        if "label" not in self.manifest.columns:
            # Map ABIDE diagnostic codes (1 = Autism/ASD, 2 = Control/TC) to binary 1/0
            raw_dx = self.manifest["dx_group"]
            unique_dx = set(raw_dx.dropna().unique())
            if {1, 2}.issubset(unique_dx) or 2 in unique_dx:
                self.manifest["label"] = (raw_dx == 1).astype(int)
            else:
                self.manifest["label"] = raw_dx.astype(int)

        self.target_spatial_shape = target_spatial_shape

        self.target_seq_len = target_seq_len
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.normalize = normalize
        self.spatial_mode = spatial_mode
        self.cache_data = cache_data
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        """Return total number of cohort subjects in manifest."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and preprocess 4D rs-fMRI acquisition for a single subject.

        Args:
            idx: Positional index in cohort manifest.

        Returns:
            Dictionary containing:
                - 'volume': Tensor of shape (T_max, 1, 64, 64, 36) with float32 values.
                - 'mask': Boolean tensor of shape (T_max,) indicating recorded frames.
                - 'label': Int64 tensor containing binary diagnostic label (ASD=1, TC=0).
                - 'subject_id': String subject identifier.
                - 'site_id': String acquisition institution identifier.
        """
        if self.cache_data and idx in self._cache:
            return self._cache[idx]

        row = self.manifest.iloc[idx]
        file_path = Path(row["file_path"])

        if not file_path.exists():
            raise FileNotFoundError(f"fMRI file not found: {file_path}")

        # Direct loading for preprocessed PyTorch tensor artifacts (.pt)
        if file_path.suffix == ".pt":
            payload = torch.load(file_path, map_location="cpu", weights_only=True)
            sample = {
                "volume": payload["volume"].to(torch.float32),
                "mask": payload["mask"].to(torch.bool),
                "label": torch.tensor(int(payload.get("label", row["label"])), dtype=torch.long),
                "subject_id": str(payload.get("subject_id", row.get("subject_id", idx))),
                "site_id": str(payload.get("site_id", row.get("site_id", "UNKNOWN"))),
            }
            if self.cache_data:
                self._cache[idx] = sample
            return sample

        # Reorient voxel arrays to standard RAS coordinate space using canonical affine transformations
        # so that left/right, anterior/posterior, and inferior/superior axes align across scanner sites.
        import nibabel as nib

        raw_img: Any = nib.load(str(file_path))
        canonical_img: Any = nib.as_closest_canonical(raw_img)
        data = np.asarray(canonical_img.get_fdata(), dtype=np.float32)

        # Standard 4D NIfTI stores spatial voxel grids in the first 3 dimensions and time frames in axis 3.
        # Fall back to axis 0 if volumes are pre-transposed to sequence-first format.
        if data.ndim == 4:
            time_axis = 0 if data.shape[0] > data.shape[-1] and data.shape[-1] in [36, 61, 64] else -1
        else:
            raise ValueError(f"Expected 4D NIfTI scan, got shape {data.shape}")

        # Discard scanner equilibrium (first 10) and scan termination (last 10) buffer frames
        # to eliminate hemodynamic instability and scanner settling noise before downstream feature extraction.
        trimmed = trim_temporal_frames(
            data,
            trim_start=self.trim_start,
            trim_end=self.trim_end,
            time_axis=time_axis,
        )
        trimmed_np: np.ndarray = trimmed if isinstance(trimmed, np.ndarray) else trimmed.cpu().numpy()

        # Permute chronological time steps to leading sequence dimension (T, X, Y, Z) for sequence modeling
        if time_axis == -1 or time_axis == 3:
            seq_data = np.moveaxis(trimmed_np, -1, 0)
        else:
            seq_data = trimmed_np

        # Normalize subject-level non-zero brain voxels to zero mean and unit variance
        # while keeping background zero-valued air voxels intact to maintain skull-stripped anatomical boundaries.
        if self.normalize:
            normalized = normalize_nonzero_voxels(seq_data)
            seq_data = normalized if isinstance(normalized, np.ndarray) else normalized.cpu().numpy()

        # Resample or crop 3D spatial dimensions to standard 64x64x36 MNI grid
        # ensuring volumetric dimensions match the 4-block 3D-CNN receptive field.
        standardized = standardize_spatial_volume(
            seq_data,
            target_shape=self.target_spatial_shape,
            mode=self.spatial_mode,
        )
        if isinstance(standardized, torch.Tensor):
            tensor_seq = standardized.unsqueeze(1).to(torch.float32)
        else:
            tensor_seq = torch.from_numpy(standardized).unsqueeze(1).to(torch.float32)

        # Pad variable acquisition durations to uniform T_max=200 and generate boolean attention mask
        # so that non-biological zero padding tokens can be masked out during temporal pooling.
        padded_vol, temporal_mask = pad_temporal_sequence(
            tensor_seq,
            target_length=self.target_seq_len,
        )

        sample = {
            "volume": padded_vol,
            "mask": temporal_mask,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": str(row.get("subject_id", idx)),
            "site_id": str(row.get("site_id", "UNKNOWN")),
        }

        if self.cache_data:
            self._cache[idx] = sample

        return sample


def spatiotemporal_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate individual subject samples into batched PyTorch tensors.

    Args:
        batch: List of dictionaries produced by SpatioTemporalFMRIDataset.__getitem__.

    Returns:
        Dictionary containing:
            - 'volume': Tensor of shape (B, T, 1, 64, 64, 36) in float32.
            - 'mask': Boolean tensor of shape (B, T).
            - 'label': Tensor of shape (B,) in int64.
            - 'subject_ids': List of string subject identifiers.
            - 'site_ids': List of string acquisition institution identifiers.
    """
    volumes = torch.stack([item["volume"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    subject_ids = [item["subject_id"] for item in batch]
    site_ids = [item["site_id"] for item in batch]

    return {
        "volume": volumes,
        "mask": masks,
        "label": labels,
        "subject_ids": subject_ids,
        "site_ids": site_ids,
    }


def create_spatiotemporal_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a configured DataLoader using spatiotemporal batch collation.

    Args:
        dataset: SpatioTemporalFMRIDataset instance.
        batch_size: Mini-batch size (default: 2).
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Subprocesses for data loading.
        pin_memory: Whether to pin memory for accelerated GPU transfer.
        drop_last: Whether to drop incomplete trailing mini-batch.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=spatiotemporal_collate_fn,
    )



