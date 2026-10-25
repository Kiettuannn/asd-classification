"""Data loader interface for ABIDE cohort."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from asd_classification.preprocessing.multiview import extract_tri_slices


@dataclass
class CohortRecord:
    """Represents an individual subject record."""

    subject_id: str
    label: int  # 1 = ASD, 0 = TC
    site_id: str
    age: Optional[float] = None
    sex: Optional[int] = None
    fiq: Optional[float] = None
    file_id: Optional[str] = None
    file_path: Optional[Path] = None


def load_nyu_phenotypic(
    csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and standardize phenotypic records for the ABIDE I NYU cohort.

    Filters the 1,112 total ABIDE I phenotypic records down to the 184 NYU subjects
    (79 ASD, 105 TC), normalizes diagnosis to binary 0/1 labels (ASD=1, TC=0),
    standardizes subject IDs, formats missing file IDs to 'NYU_00{sub_id}', and
    extracts key phenotypic covariates (age, sex, full-scale IQ).

    Args:
        csv_path: Optional path to 'Phenotypic_V1_0b_preprocessed1.csv'. If None,
            searches standard project data directories.

    Returns:
        Standardized DataFrame containing exactly 184 NYU cohort records.

    Raises:
        FileNotFoundError: If the phenotypic CSV file cannot be located.
        ValueError: If the resulting cohort does not contain exactly 184 subjects.
    """
    if csv_path is None:
        candidate_paths = [
            Path("notebooks/data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
            Path("notebooks/abide_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
            Path("../notebooks/data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
            Path("data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
        ]
        resolved_path = None
        for candidate in candidate_paths:
            if candidate.is_file():
                resolved_path = candidate
                break
        if resolved_path is None:
            raise FileNotFoundError(
                "Could not locate Phenotypic_V1_0b_preprocessed1.csv in default data directories. "
                "Please specify csv_path explicitly."
            )
        target_path = resolved_path
    else:
        target_path = Path(csv_path)
        if not target_path.is_file():
            raise FileNotFoundError(f"Phenotypic CSV file not found at: {target_path}")

    raw_df = pd.read_csv(target_path)

    # Filter for NYU site
    nyu_df = raw_df[raw_df["SITE_ID"] == "NYU"].copy()
    if len(nyu_df) != 184:
        raise ValueError(f"Expected 184 subjects for NYU site, found {len(nyu_df)}.")

    # Standardize columns
    # ABIDE DX_GROUP: 1 = Autism, 2 = Control -> Map to 1 = ASD, 0 = TC
    nyu_df["dx_group"] = (nyu_df["DX_GROUP"] == 1).astype(int)
    nyu_df["subject_id"] = nyu_df["SUB_ID"].astype(str)

    # Resolve file_id: if 'no_filename', standard convention is NYU_00{sub_id}
    def resolve_file_id(row: pd.Series) -> str:
        fid = str(row["FILE_ID"]).strip()
        if fid == "no_filename" or fid == "" or fid == "nan":
            return f"NYU_00{row['subject_id']}"
        return fid

    nyu_df["file_id"] = nyu_df.apply(resolve_file_id, axis=1)

    # Standardize covariates
    nyu_df["age"] = pd.to_numeric(nyu_df["AGE_AT_SCAN"], errors="coerce")
    nyu_df["sex"] = pd.to_numeric(nyu_df["SEX"], errors="coerce").astype(int)
    nyu_df["fiq"] = pd.to_numeric(nyu_df["FIQ"], errors="coerce")

    # Keep clean manifest columns
    keep_cols = [
        "subject_id",
        "file_id",
        "dx_group",
        "age",
        "sex",
        "fiq",
        "SITE_ID",
    ]
    standardized_df = nyu_df[keep_cols].reset_index(drop=True)
    return standardized_df


class ABIDELoader:
    """Loads and standardizes ABIDE phenotypic records and ROI BOLD time-series."""

    def __init__(self, data_dir: Optional[Path] = None, atlas: Optional[str] = None):
        self.data_dir = data_dir
        self.atlas = atlas

    def load_phenotypic(self) -> pd.DataFrame:
        """Load phenotypic CSV and normalize diagnostic labels."""
        csv_file = self.data_dir / "Phenotypic_V1_0b_preprocessed1.csv" if self.data_dir else None
        return load_nyu_phenotypic(csv_file)

    def find_roi_files(self) -> Dict[str, Path]:
        """Index ROI time-series files by subject ID."""
        if self.data_dir is None:
            return {}
        roi_files = {}
        for file in self.data_dir.glob("*.1D"):
            sub_id = file.stem.split("_")[0]
            roi_files[sub_id] = file
        return roi_files

    def load_cohort(self) -> Tuple[List[CohortRecord], np.ndarray, np.ndarray]:
        """Loads aligned cohort records, timeseries, and labels."""
        df = self.load_phenotypic()
        records: List[CohortRecord] = []
        for _, row in df.iterrows():
            records.append(
                CohortRecord(
                    subject_id=str(row["subject_id"]),
                    label=int(row["dx_group"]),
                    site_id=str(row.get("SITE_ID", "NYU")),
                    age=float(row["age"]) if pd.notna(row["age"]) else None,
                    sex=int(row["sex"]) if pd.notna(row["sex"]) else None,
                    fiq=float(row["fiq"]) if pd.notna(row["fiq"]) else None,
                    file_id=str(row["file_id"]),
                )
            )
        labels = df["dx_group"].to_numpy()
        return records, np.empty((len(records), 0)), labels


class MultiViewDataset(Dataset):
    """PyTorch Dataset for multi-view fMRI volumetric tensors.

    Supports:
    1. Full depth-stacked multi-view tensors of shape (1, 150, 224, 224).
    2. Dynamic decomposed view extraction (Option 3: View-as-Sample) into three
       50-slice sub-volumes (Axial, Coronal, Sagittal) of shape (1, 50, 224, 224),
       tripling effective sample size while preserving anatomical spatial continuity.
    3. Single view selection ('axial', 'coronal', 'sagittal').
    4. Optional in-memory caching of .npz tensors to eliminate disk I/O bottleneck.
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        tensors_dir: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        transform: Optional[Any] = None,
        view: str = "all",
        decomposed: bool = False,
        cache_tensors: bool = False,
        slices_per_view: Optional[int] = None,
    ) -> None:
        """Initialize MultiViewDataset.

        Args:
            manifest: DataFrame or path to metadata.csv containing subject records and splits.
            tensors_dir: Optional root directory containing .npz tensor files.
            split: Optional split filter: 'train', 'test', or None/'all'.
            transform: Optional volume-consistent transformation module.
            view: Extraction view mode ('all', 'axial', 'coronal', 'sagittal', 'decomposed').
            decomposed: Boolean flag forcing dynamic 3-view decomposition (Option 3).
            cache_tensors: Boolean flag enabling in-memory tensor caching after initial read.
            slices_per_view: Optional slice count per orthogonal plane (default: 50, or inferred as D // 3).
        """
        self._manifest_dir: Optional[Path]
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            self.manifest_df = pd.read_csv(manifest_path)
            self._manifest_dir = manifest_path.parent
        elif isinstance(manifest, pd.DataFrame):
            self.manifest_df = manifest.copy()
            self._manifest_dir = None
        else:
            raise TypeError(f"Expected DataFrame or path, got {type(manifest)}")

        # Filter by split if requested
        if split is not None and split.lower() not in ("all", "none"):
            if "split" not in self.manifest_df.columns:
                raise ValueError("Cannot filter by split when 'split' column is not in manifest.")
            self.manifest_df = self.manifest_df[
                self.manifest_df["split"] == split.lower()
            ].reset_index(drop=True)
        else:
            self.manifest_df = self.manifest_df.reset_index(drop=True)

        self.tensors_dir = Path(tensors_dir) if tensors_dir is not None else None
        self.split = split
        self.transform = transform
        self.view = view.lower()
        self.decomposed = decomposed or (self.view == "decomposed")
        self.cache_tensors = cache_tensors
        self.slices_per_view = slices_per_view
        self._tensor_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        n_subjects = len(self.manifest_df)
        if self.decomposed:
            # Tripled sample count: each subject yields 3 orthogonal views
            return n_subjects * 3
        return n_subjects

    def _resolve_tensor_file(self, row: pd.Series) -> Path:
        sub_id = str(row["subject_id"])
        tensor_path_str = str(row.get("tensor_path", ""))

        candidates = []
        if self.tensors_dir is not None:
            if tensor_path_str:
                candidates.append(self.tensors_dir / tensor_path_str)
                # If tensor_path is "tensors/50001.npz" and tensors_dir is already ".../tensors":
                candidates.append(self.tensors_dir / Path(tensor_path_str).name)
            candidates.append(self.tensors_dir / f"{sub_id}.npz")
            candidates.append(self.tensors_dir / "tensors" / f"{sub_id}.npz")

        if tensor_path_str:
            candidates.append(Path(tensor_path_str))
            if self._manifest_dir is not None:
                candidates.append(self._manifest_dir / tensor_path_str)

        # Standard Kaggle and project input locations
        kaggle_base = Path("/kaggle/input/abide-i-nyu-multiview-fmri")
        candidates.extend([
            kaggle_base / "tensors" / f"{sub_id}.npz",
            kaggle_base / tensor_path_str if tensor_path_str else kaggle_base / f"{sub_id}.npz",
            Path("/kaggle/working/abide_nyu_multiview/tensors") / f"{sub_id}.npz",
            Path("output/abide_nyu_multiview/tensors") / f"{sub_id}.npz",
            Path("../output/abide_nyu_multiview/tensors") / f"{sub_id}.npz",
        ])

        for c in candidates:
            if c and c.is_file():
                return c

        raise FileNotFoundError(
            f"Could not locate tensor file for subject {sub_id}. "
            f"Checked locations: {[str(c) for c in candidates[:4]]}"
        )

    def _load_raw_tensor(self, subject_idx: int) -> np.ndarray:
        row = self.manifest_df.iloc[subject_idx]
        sub_id = str(row["subject_id"])
        if self.cache_tensors and sub_id in self._tensor_cache:
            return self._tensor_cache[sub_id]

        file_path = self._resolve_tensor_file(row)
        with np.load(file_path) as npz:
            tensor_np = npz["tensor"]

        if self.cache_tensors:
            self._tensor_cache[sub_id] = tensor_np

        return tensor_np

    def _extract_view_slice(self, tensor: torch.Tensor, view_idx: int) -> torch.Tensor:
        total_depth = tensor.shape[1]
        spv = self.slices_per_view if self.slices_per_view is not None else (total_depth // 3)
        start_idx = view_idx * spv
        end_idx = start_idx + spv
        return tensor[:, start_idx:end_idx, :, :]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.decomposed:
            subject_idx = idx // 3
            view_idx = idx % 3
        else:
            subject_idx = idx
            view_idx = -1

        row = self.manifest_df.iloc[subject_idx]
        tensor_np = self._load_raw_tensor(subject_idx)

        tensor = torch.from_numpy(tensor_np).float()
        # Ensure single channel dimension (1, D, H, W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        total_depth = tensor.shape[1]
        spv = self.slices_per_view if self.slices_per_view is not None else (total_depth // 3)

        if self.decomposed:
            tensor = self._extract_view_slice(tensor, view_idx)
        elif self.view == "axial":
            tensor = tensor[:, 0:spv, :, :]
        elif self.view == "coronal":
            tensor = tensor[:, spv:2 * spv, :, :]
        elif self.view == "sagittal":
            tensor = tensor[:, 2 * spv:3 * spv, :, :]

        if self.transform is not None:
            tensor = self.transform(tensor)
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor).float()

        label = int(row["dx_group"])
        return tensor, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> np.ndarray:
        base_labels = self.manifest_df["dx_group"].to_numpy(dtype=np.int64)
        if self.decomposed:
            return np.repeat(base_labels, 3)
        return base_labels

    def get_subject_views(self, subject_idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """Return all three orthogonal 50-slice sub-volumes and label for a subject.

        Args:
            subject_idx: 0-based subject index within manifest_df.

        Returns:
            Tuple of (dict mapping 'axial', 'coronal', 'sagittal' to 4D tensors, ground-truth label int).
        """
        row = self.manifest_df.iloc[subject_idx]
        tensor_np = self._load_raw_tensor(subject_idx)
        tensor = torch.from_numpy(tensor_np).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        total_depth = tensor.shape[1]
        spv = self.slices_per_view if self.slices_per_view is not None else (total_depth // 3)

        views = {
            "axial": tensor[:, 0:spv, :, :],
            "coronal": tensor[:, spv:2 * spv, :, :],
            "sagittal": tensor[:, 2 * spv:3 * spv, :, :],
        }
        label = int(row["dx_group"])
        return views, label

    def get_subject_views_tensor(self, subject_idx: int) -> Tuple[torch.Tensor, int]:
        """Return all three orthogonal views stacked as a 5D tensor of shape (3, 1, D_view, H, W).

        Args:
            subject_idx: 0-based subject index within manifest_df.

        Returns:
            Tuple of (stacked views tensor of shape (3, 1, D_view, H, W), ground-truth label int).
        """
        views_dict, label = self.get_subject_views(subject_idx)
        stacked = torch.stack([views_dict["axial"], views_dict["coronal"], views_dict["sagittal"]], dim=0)
        return stacked, label


class TriSliceDataset(Dataset):
    """PyTorch Dataset for multi-view 2.5D tri-slice patient bags for MIL.

    Extracts patient bags of exactly 60 tri-slices of shape (60, 3, 224, 224)
    (20 axial, 20 coronal, 20 sagittal) from depth-stacked multi-view fMRI
    volumetric tensors of shape (1, 150, 224, 224).

    Adjacent depth slices [z-1, z, z+1] are grouped into 3 channels to encode
    local 3D through-plane spatial gradients for 2D convolutional backbones
    such as ResNet-18 without 3D parameter explosion.
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        tensors_dir: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        transform: Optional[Any] = None,
        cache_tensors: bool = False,
        slices_per_view: int = 50,
        central_slices: int = 40,
        stride: int = 2,
    ) -> None:
        """Initialize TriSliceDataset.

        Args:
            manifest: DataFrame or path to metadata.csv containing subject records and splits.
            tensors_dir: Optional root directory containing .npz tensor files.
            split: Optional split filter: 'train', 'test', or None/'all'.
            transform: Optional bag-level transformation module (e.g. BagConsistentAugmenter).
            cache_tensors: Boolean flag enabling in-memory tensor caching after initial read.
            slices_per_view: Slice count per orthogonal plane (default: 50).
            central_slices: Number of central slices to sample from each plane (default: 40).
            stride: Step size between sampled slice centers (default: 2).

        Raises:
            TypeError: If manifest is not a DataFrame, str, or Path.
            ValueError: If split filtering is requested but manifest lacks a 'split' column.
        """
        self._manifest_dir: Optional[Path]
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            self.manifest_df = pd.read_csv(manifest_path)
            self._manifest_dir = manifest_path.parent
        elif isinstance(manifest, pd.DataFrame):
            self.manifest_df = manifest.copy()
            self._manifest_dir = None
        else:
            raise TypeError(f"Expected DataFrame or path, got {type(manifest)}")

        # Filter by split if requested
        if split is not None and split.lower() not in ("all", "none"):
            if "split" not in self.manifest_df.columns:
                raise ValueError("Cannot filter by split when 'split' column is not in manifest.")
            self.manifest_df = self.manifest_df[
                self.manifest_df["split"] == split.lower()
            ].reset_index(drop=True)
        else:
            self.manifest_df = self.manifest_df.reset_index(drop=True)

        self.tensors_dir = Path(tensors_dir) if tensors_dir is not None else None
        self.split = split
        self.transform = transform
        self.cache_tensors = cache_tensors
        self.slices_per_view = slices_per_view
        self.central_slices = central_slices
        self.stride = stride
        self._tensor_cache: Dict[str, np.ndarray] = {}

        # Automatically ensure evaluation mode for test or validation splits
        if self.split in ("test", "val", "validation") and isinstance(self.transform, torch.nn.Module):
            self.transform.eval()

    def __len__(self) -> int:
        """Return the number of subjects in the dataset.

        Returns:
            Total subject count within the active split.
        """
        return len(self.manifest_df)

    def _resolve_tensor_file(self, row: pd.Series) -> Path:
        """Locate the .npz tensor file on disk for a given subject row.

        Args:
            row: Subject metadata row containing 'subject_id' and optional 'tensor_path'.

        Returns:
            Resolved absolute or relative path to the existing .npz tensor file.

        Raises:
            FileNotFoundError: If the tensor file cannot be located in any searched candidate directory.
        """
        sub_id = str(row["subject_id"])
        tensor_path_str = str(row.get("tensor_path", ""))

        candidates = []
        if self.tensors_dir is not None:
            if tensor_path_str:
                candidates.append(self.tensors_dir / tensor_path_str)
                candidates.append(self.tensors_dir / Path(tensor_path_str).name)
            candidates.append(self.tensors_dir / f"{sub_id}.npz")
            candidates.append(self.tensors_dir / "tensors" / f"{sub_id}.npz")

        if tensor_path_str:
            candidates.append(Path(tensor_path_str))
            if self._manifest_dir is not None:
                candidates.append(self._manifest_dir / tensor_path_str)

        # Standard Kaggle and local paths
        kaggle_base = Path("/kaggle/input/abide-i-nyu-multiview-fmri")
        candidates.extend([
            kaggle_base / "tensors" / f"{sub_id}.npz",
            kaggle_base / tensor_path_str if tensor_path_str else kaggle_base / f"{sub_id}.npz",
            Path("/kaggle/working/abide_nyu_multiview/tensors") / f"{sub_id}.npz",
            Path("output/abide_nyu_multiview/tensors") / f"{sub_id}.npz",
            Path("../output/abide_nyu_multiview/tensors") / f"{sub_id}.npz",
        ])

        for c in candidates:
            if c and c.is_file():
                return c

        raise FileNotFoundError(
            f"Could not locate tensor file for subject {sub_id}. "
            f"Checked locations: {[str(c) for c in candidates[:4]]}"
        )

    def _load_raw_tensor(self, subject_idx: int) -> np.ndarray:
        """Load the subject's raw multi-view volumetric tensor from archive or cache.

        Args:
            subject_idx: 0-based subject index within manifest_df.

        Returns:
            Multi-view tensor array of shape (1, 150, 224, 224).
        """
        row = self.manifest_df.iloc[subject_idx]
        sub_id = str(row["subject_id"])
        if self.cache_tensors and sub_id in self._tensor_cache:
            return self._tensor_cache[sub_id]

        file_path = self._resolve_tensor_file(row)
        with np.load(file_path) as npz:
            tensor_np = npz["tensor"]

        if self.cache_tensors:
            self._tensor_cache[sub_id] = tensor_np

        return tensor_np

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and return a patient bag of 2.5D tri-slices and its diagnostic label.

        Args:
            idx: 0-based index of the subject in the active split.

        Returns:
            Tuple of (bag_tensor, label_tensor):
            - bag_tensor: Float tensor of shape (K, 3, H, W) containing 60 tri-slices
              (20 axial, 20 coronal, 20 sagittal) where channels represent [z-1, z, z+1].
            - label_tensor: Long tensor scalar with target diagnosis (1 = ASD, 0 = TC).
        """
        row = self.manifest_df.iloc[idx]
        tensor_np = self._load_raw_tensor(idx)

        bag = extract_tri_slices(
            tensor_np,
            slices_per_view=self.slices_per_view,
            central_slices=self.central_slices,
            stride=self.stride,
        )

        if self.transform is not None:
            bag = self.transform(bag)
            if not isinstance(bag, torch.Tensor):
                bag = torch.as_tensor(bag, dtype=torch.float32)

        label = int(row["dx_group"])
        return bag, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> np.ndarray:
        """Return the 1D integer diagnosis labels for all subjects in the dataset.

        Returns:
            Numpy 1D array of length N with binary diagnostic targets (1 = ASD, 0 = TC).
        """
        return self.manifest_df["dx_group"].to_numpy(dtype=np.int64)


