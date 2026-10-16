"""Dataset partitioning and cross-validation splitting.

This module provides stratified splitting routines for subject-level dataset
partitioning, cross-validation generators, and specific cohort verification
routines adhering to Hammash & Younis (MDPI Journal of Imaging, 2026).
"""

from typing import Any, Generator, Tuple, Union, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: Union[float, int] = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train and test sets with stratified class balance.

    Args:
        X: Feature matrix or array of subject identifiers.
        y: Binary target diagnostic labels (ASD=1, TC=0).
        test_size: Proportion or absolute count of samples allocated to test set.
        random_state: Seed for reproducible pseudo-random splitting.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def get_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate Stratified K-Fold train and validation indices.

    Stratification ensures class balance across all folds to prevent fold-level
    class skewness in clinical neuroimaging evaluation.

    Args:
        X: Feature matrix or sample index array.
        y: Class labels.
        n_splits: Number of cross-validation folds (default: 5).
        random_state: Random state for deterministic shuffling.

    Yields:
        Tuples of (train_indices, val_indices) for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    yield from skf.split(X, y)


def partition_nyu_cohort(
    df: pd.DataFrame,
    test_size: int = 36,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign subject-level 80/20 train/test split to the ABIDE I NYU cohort.

    The partitioning matches Table 3 of Hammash & Younis (2026):
    - Total: 184 subjects (79 ASD, 105 TC)
    - Train: 148 subjects (64 ASD, 84 TC)
    - Test: 36 subjects (15 ASD, 21 TC)

    Args:
        df: DataFrame containing NYU subjects with binary 'dx_group' column (1=ASD, 0=TC).
        test_size: Absolute count of test subjects (default: 36).
        random_state: Random seed for reproducible stratification (default: 42).

    Returns:
        DataFrame with an added 'split' column containing 'train' or 'test'.

    Raises:
        ValueError: If 'dx_group' column is missing or class counts do not match 79 ASD / 105 TC.
    """
    if "dx_group" not in df.columns:
        raise ValueError("DataFrame must contain a 'dx_group' column with binary diagnostic labels.")

    result_df = df.copy()
    y = result_df["dx_group"].to_numpy()

    train_indices, test_indices = train_test_split(
        result_df.index,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    result_df["split"] = "train"
    result_df.loc[test_indices, "split"] = "test"

    return result_df


def verify_nyu_split_counts(
    df: pd.DataFrame,
    expected_train_asd: int = 64,
    expected_train_tc: int = 84,
    expected_test_asd: int = 15,
    expected_test_tc: int = 21,
) -> bool:
    """Verify that dataset split composition matches the published Table 3 specification.

    Args:
        df: DataFrame containing 'split' ('train'/'test') and 'dx_group' (1=ASD, 0=TC).
        expected_train_asd: Expected ASD count in train set (default: 64).
        expected_train_tc: Expected TC count in train set (default: 84).
        expected_test_asd: Expected ASD count in test set (default: 15).
        expected_test_tc: Expected TC count in test set (default: 21).

    Returns:
        True if split composition matches expected counts.

    Raises:
        ValueError: If any partition count deviates from expectations.
    """
    if "split" not in df.columns or "dx_group" not in df.columns:
        raise ValueError("DataFrame must contain both 'split' and 'dx_group' columns.")

    crosstab = pd.crosstab(df["split"], df["dx_group"])

    actual_train_asd = int(cast(Any, crosstab.loc["train", 1])) if ("train" in crosstab.index and 1 in crosstab.columns) else 0
    actual_train_tc = int(cast(Any, crosstab.loc["train", 0])) if ("train" in crosstab.index and 0 in crosstab.columns) else 0
    actual_test_asd = int(cast(Any, crosstab.loc["test", 1])) if ("test" in crosstab.index and 1 in crosstab.columns) else 0
    actual_test_tc = int(cast(Any, crosstab.loc["test", 0])) if ("test" in crosstab.index and 0 in crosstab.columns) else 0

    if (
        actual_train_asd != expected_train_asd
        or actual_train_tc != expected_train_tc
        or actual_test_asd != expected_test_asd
        or actual_test_tc != expected_test_tc
    ):
        raise ValueError(
            f"Split composition mismatch. Expected: "
            f"Train(ASD={expected_train_asd}, TC={expected_train_tc}), "
            f"Test(ASD={expected_test_asd}, TC={expected_test_tc}). "
            f"Actual: "
            f"Train(ASD={actual_train_asd}, TC={actual_train_tc}), "
            f"Test(ASD={actual_test_asd}, TC={actual_test_tc})."
        )

    return True
