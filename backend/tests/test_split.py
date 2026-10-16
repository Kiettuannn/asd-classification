"""Unit tests for dataset partitioning and NYU cohort train/test splitting."""

import numpy as np
import pandas as pd
import pytest

from asd_classification.data.split import (
    get_kfold_splits,
    make_stratified_split,
    partition_nyu_cohort,
    verify_nyu_split_counts,
)


def test_make_stratified_split_preserves_class_proportions():
    """Verify make_stratified_split splits feature and label arrays proportionally."""
    X = np.arange(100).reshape(-1, 1)
    y = np.array([1] * 40 + [0] * 60)

    X_train, X_test, y_train, y_test = make_stratified_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert (y_train == 1).sum() == 32
    assert (y_train == 0).sum() == 48
    assert (y_test == 1).sum() == 8
    assert (y_test == 0).sum() == 12


def test_get_kfold_splits_yields_expected_folds():
    """Verify get_kfold_splits produces the configured number of stratified folds."""
    X = np.arange(100).reshape(-1, 1)
    y = np.array([1] * 50 + [0] * 50)

    splits = list(get_kfold_splits(X, y, n_splits=5, random_state=42))

    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) == 80
        assert len(val_idx) == 20
        # Each fold validation set must contain equal numbers of both classes
        assert (y[val_idx] == 1).sum() == 10
        assert (y[val_idx] == 0).sum() == 10


def test_partition_nyu_cohort_exact_table3_counts():
    """Verify NYU cohort partitioning yields exactly 64 ASD, 84 TC in train, and 15 ASD, 21 TC in test."""
    # Synthetic NYU DataFrame with 79 ASD and 105 TC
    labels = [1] * 79 + [0] * 105
    sub_ids = [f"NYU_{i:05d}" for i in range(len(labels))]
    df = pd.DataFrame({"subject_id": sub_ids, "dx_group": labels})

    split_df = partition_nyu_cohort(df, test_size=36, random_state=42)

    assert "split" in split_df.columns
    assert len(split_df) == 184

    # Verify counts
    is_valid = verify_nyu_split_counts(split_df)
    assert is_valid is True

    train_subset = split_df[split_df["split"] == "train"]
    test_subset = split_df[split_df["split"] == "test"]

    assert len(train_subset) == 148
    assert (train_subset["dx_group"] == 1).sum() == 64
    assert (train_subset["dx_group"] == 0).sum() == 84

    assert len(test_subset) == 36
    assert (test_subset["dx_group"] == 1).sum() == 15
    assert (test_subset["dx_group"] == 0).sum() == 21


def test_verify_nyu_split_counts_raises_on_invalid_proportions():
    """Verify that verify_nyu_split_counts raises ValueError when counts deviate from Table 3."""
    # Create invalid split DataFrame
    df = pd.DataFrame({
        "dx_group": [1] * 60 + [0] * 88 + [1] * 19 + [0] * 17,
        "split": ["train"] * 148 + ["test"] * 36,
    })

    with pytest.raises(ValueError, match="Split composition mismatch"):
        verify_nyu_split_counts(df)
