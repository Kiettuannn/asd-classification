"""Data loader interface for ABIDE cohort."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


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
