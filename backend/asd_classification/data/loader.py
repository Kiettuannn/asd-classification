"""Data loader interface for ABIDE cohort."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd


@dataclass
class CohortRecord:
    """Represents an individual subject record."""
    subject_id: str
    label: int  # 1 = ASD, 0 = TD
    site_id: str
    age: Optional[float] = None
    sex: Optional[int] = None
    fiq: Optional[float] = None
    file_path: Optional[Path] = None


class ABIDELoader:
    """Loads and standardizes ABIDE phenotypic records and ROI BOLD time-series."""

    def __init__(self, data_dir: Optional[Path] = None, atlas: Optional[str] = None):
        self.data_dir = data_dir
        self.atlas = atlas

    def load_phenotypic(self) -> pd.DataFrame:
        """Load phenotypic CSV and normalize diagnostic labels."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def find_roi_files(self) -> Dict[str, Path]:
        """Index ROI time-series files by subject ID."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def load_cohort(self) -> Tuple[List[CohortRecord], np.ndarray, np.ndarray]:
        """Loads aligned cohort records, timeseries, and labels."""
        raise NotImplementedError("Implementation will be added in implementation phase.")
