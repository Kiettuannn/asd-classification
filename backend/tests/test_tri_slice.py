"""Unit tests for 2.5D Multi-View Tri-Slice Extraction & Bag DataLoader.

Verifies:
1. Multi-view tri-slice extraction from (1, 150, 224, 224) tensors into (60, 3, 224, 224) bags.
2. Grouping of adjacent depth slices [z-1, z, z+1] into 3-channel input arrays.
3. Slice boundary integrity ensuring no cross-plane contamination across axial, coronal, and sagittal views.
4. Bag-level 2D data augmentation consistency and evaluation pass-through invariance.
5. End-to-end dataset and DataLoader batch contract.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pytest
import torch

from asd_classification.preprocessing.multiview import extract_tri_slices
from asd_classification.models.baseline.augmentation import BagConsistentAugmenter
from asd_classification.data.loader import TriSliceDataset


def test_extract_tri_slices_shape():
    """Verify that extract_tri_slices extracts exactly 60 tri-slices of shape (3, 224, 224)."""
    # Create synthetic multi-view volume of shape (1, 150, 224, 224)
    volume_4d = torch.zeros((1, 150, 224, 224), dtype=torch.float32)
    bag = extract_tri_slices(volume_4d)

    assert bag.shape == (60, 3, 224, 224)
    assert bag.dtype == torch.float32

    # Also test 3D unchanneled volume (150, 224, 224)
    volume_3d = torch.zeros((150, 224, 224), dtype=torch.float32)
    bag_3d = extract_tri_slices(volume_3d)
    assert bag_3d.shape == (60, 3, 224, 224)


def test_extract_tri_slices_channel_mapping_and_boundaries():
    """Verify that [z-1, z, z+1] maps to channels 0, 1, 2 without cross-plane boundary leakage."""
    # Create volume where each slice index z is filled with constant value z
    volume = torch.zeros((1, 150, 32, 32), dtype=torch.float32)
    for z in range(150):
        volume[0, z, :, :] = float(z)

    # 50 slices per view, 40 central slices (margin 5 each side), stride 2
    bag = extract_tri_slices(volume, slices_per_view=50, central_slices=40, stride=2)
    assert bag.shape == (60, 3, 32, 32)

    # Axial plane (slices 0..49): centers 5, 7, ..., 43 (20 slices)
    # Check first axial tri-slice (index 0, center = 5)
    assert bag[0, 0, 0, 0].item() == 4.0  # z - 1
    assert bag[0, 1, 0, 0].item() == 5.0  # z
    assert bag[0, 2, 0, 0].item() == 6.0  # z + 1

    # Check last axial tri-slice (index 19, center = 43)
    assert bag[19, 0, 0, 0].item() == 42.0  # z - 1
    assert bag[19, 1, 0, 0].item() == 43.0  # z
    assert bag[19, 2, 0, 0].item() == 44.0  # z + 1 (strictly < 50, no coronal leakage)

    # Coronal plane (slices 50..99): centers 55, 57, ..., 93 (20 slices)
    # Check first coronal tri-slice (index 20, center = 55)
    assert bag[20, 0, 0, 0].item() == 54.0  # z - 1 (strictly >= 50, no axial leakage)
    assert bag[20, 1, 0, 0].item() == 55.0  # z
    assert bag[20, 2, 0, 0].item() == 56.0  # z + 1

    # Check last coronal tri-slice (index 39, center = 93)
    assert bag[39, 0, 0, 0].item() == 92.0  # z - 1
    assert bag[39, 1, 0, 0].item() == 93.0  # z
    assert bag[39, 2, 0, 0].item() == 94.0  # z + 1 (strictly < 100, no sagittal leakage)

    # Sagittal plane (slices 100..149): centers 105, 107, ..., 143 (20 slices)
    # Check first sagittal tri-slice (index 40, center = 105)
    assert bag[40, 0, 0, 0].item() == 104.0  # z - 1
    assert bag[40, 1, 0, 0].item() == 105.0  # z
    assert bag[40, 2, 0, 0].item() == 106.0  # z + 1

    # Check last sagittal tri-slice (index 59, center = 143)
    assert bag[59, 0, 0, 0].item() == 142.0  # z - 1
    assert bag[59, 1, 0, 0].item() == 143.0  # z
    assert bag[59, 2, 0, 0].item() == 144.0  # z + 1 (strictly < 150)


def test_extract_tri_slices_invalid_input():
    """Verify that extract_tri_slices raises ValueError for invalid dimensions or depths."""
    # Invalid dimension (2D tensor)
    with pytest.raises(ValueError, match="Expected 3D, 4D, or 5D"):
        extract_tri_slices(torch.zeros((224, 224)))

    # Depth not matching 3 views
    with pytest.raises(ValueError, match="Total depth"):
        extract_tri_slices(torch.zeros((1, 50, 224, 224)), slices_per_view=50)


def test_bag_consistent_augmenter_eval_invariance():
    """Verify that augmenter acts as an exact identity pass-through in eval mode."""
    augmenter = BagConsistentAugmenter()
    augmenter.eval()

    # Generate random bag of shape (60, 3, 32, 32)
    bag = torch.randn((60, 3, 32, 32), dtype=torch.float32)
    out = augmenter(bag)

    assert torch.equal(out, bag)


def test_bag_consistent_augmenter_training_consistency():
    """Verify that sampled affine transformations are applied identically across all slices in a bag."""
    augmenter = BagConsistentAugmenter(
        max_rotation_angle=10.0,
        p_flip_horizontal=1.0,
        p_flip_vertical=1.0,
        intensity_scale_range=(1.1, 1.1),
    )
    augmenter.train()

    # Create a base slice with distinct asymmetrical pattern
    base_slice = torch.zeros((3, 32, 32), dtype=torch.float32)
    base_slice[:, 0:10, 0:5] = 1.0
    base_slice[:, 20:30, 20:25] = 2.0

    # Repeat this base slice across all 60 instances of the bag
    bag = base_slice.unsqueeze(0).repeat(60, 1, 1, 1)  # (60, 3, 32, 32)

    # Augment bag in training mode
    aug_bag = augmenter(bag)

    assert aug_bag.shape == (60, 3, 32, 32)
    # The output slice must have changed due to active augmentation
    assert not torch.equal(aug_bag[0], base_slice)

    # Because transformations are volume/bag-consistent, every slice in the bag must receive
    # the exact same transformation. Therefore, all 60 output slices must remain identical to slice 0.
    for k in range(1, 60):
        assert torch.allclose(aug_bag[k], aug_bag[0], atol=1e-6)


def test_bag_consistent_augmenter_batched():
    """Verify that batched bag inputs (B, K, 3, H, W) are processed correctly."""
    augmenter = BagConsistentAugmenter()
    augmenter.train()

    batched_bags = torch.randn((4, 60, 3, 32, 32), dtype=torch.float32)
    out = augmenter(batched_bags)

    assert out.shape == (4, 60, 3, 32, 32)


@pytest.fixture
def mock_dataset_env(tmp_path: Path) -> Dict[str, Path]:
    """Create a temporary dataset environment with mock multi-view tensors."""
    tensors_dir = tmp_path / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # 4 train subjects (3 ASD, 1 TC) and 2 test subjects (0 ASD, 2 TC)
    for i in range(6):
        sub_id = f"5000{i}"
        file_id = f"NYU_00{sub_id}"
        dx_group = 1 if i < 3 else 0
        split = "train" if i < 4 else "test"
        rel_path = f"tensors/{sub_id}.npz"

        # Multi-view tensor of shape (1, 150, 32, 32)
        tensor_data = np.full((1, 150, 32, 32), fill_value=float(i + 1), dtype=np.float32)
        np.savez_compressed(tensors_dir / f"{sub_id}.npz", tensor=tensor_data)

        records.append({
            "subject_id": sub_id,
            "file_id": file_id,
            "dx_group": dx_group,
            "split": split,
            "tensor_path": rel_path,
        })

    manifest_df = pd.DataFrame(records)
    csv_path = tmp_path / "metadata.csv"
    manifest_df.to_csv(csv_path, index=False)

    return {"manifest_path": csv_path, "tensors_dir": tensors_dir}


def test_tri_slice_dataset_contract(mock_dataset_env: Dict[str, Path]):
    """Verify that TriSliceDataset extracts bags of shape (60, 3, H, W) and valid target labels."""
    dataset = TriSliceDataset(
        manifest=mock_dataset_env["manifest_path"],
        tensors_dir=mock_dataset_env["tensors_dir"],
    )

    assert len(dataset) == 6
    bag, label = dataset[0]

    assert bag.shape == (60, 3, 32, 32)
    assert bag.dtype == torch.float32
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert label.item() == 1  # subject 50000 is ASD (dx_group=1)

    labels = dataset.get_labels()
    assert len(labels) == 6
    assert np.array_equal(labels, np.array([1, 1, 1, 0, 0, 0]))


def test_tri_slice_dataset_split_filtering(mock_dataset_env: Dict[str, Path]):
    """Verify that TriSliceDataset correctly filters subjects by train/test split."""
    train_dataset = TriSliceDataset(
        manifest=mock_dataset_env["manifest_path"],
        tensors_dir=mock_dataset_env["tensors_dir"],
        split="train",
    )
    assert len(train_dataset) == 4

    test_dataset = TriSliceDataset(
        manifest=mock_dataset_env["manifest_path"],
        tensors_dir=mock_dataset_env["tensors_dir"],
        split="test",
    )
    assert len(test_dataset) == 2


def test_tri_slice_dataset_transform_and_caching(mock_dataset_env: Dict[str, Path]):
    """Verify transform integration, evaluation invariance, and in-memory caching."""
    raw_dataset = TriSliceDataset(
        manifest=mock_dataset_env["manifest_path"],
        tensors_dir=mock_dataset_env["tensors_dir"],
    )
    raw_bag, _ = raw_dataset[0]

    augmenter = BagConsistentAugmenter(max_rotation_angle=10.0)
    augmenter.eval()  # Eval mode should pass through untouched

    dataset = TriSliceDataset(
        manifest=mock_dataset_env["manifest_path"],
        tensors_dir=mock_dataset_env["tensors_dir"],
        transform=augmenter,
        cache_tensors=True,
    )

    eval_bag, label = dataset[0]
    assert eval_bag.shape == (60, 3, 32, 32)
    # Evaluation invariance: returned bag must be strictly identical to raw unaugmented bag
    assert torch.equal(eval_bag, raw_bag)
    # Check cache is populated
    assert "50000" in dataset._tensor_cache


def test_tri_slice_dataset_test_split_enforces_eval(mock_dataset_env: Dict[str, Path]):
    """Verify that test split automatically sets transform to eval mode."""
    augmenter = BagConsistentAugmenter(max_rotation_angle=10.0)
    augmenter.train()  # Explicitly in training mode

    test_dataset = TriSliceDataset(
        manifest=mock_dataset_env["manifest_path"],
        tensors_dir=mock_dataset_env["tensors_dir"],
        split="test",
        transform=augmenter,
    )
    assert test_dataset.transform is not None
    assert not test_dataset.transform.training


def test_tri_slice_dataset_full_resolution(tmp_path: Path):
    """Verify TriSliceDataset extraction with full (1, 150, 224, 224) volumetric tensor."""
    tensors_dir = tmp_path / "full_tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    tensor_data = np.zeros((1, 150, 224, 224), dtype=np.float32)
    np.savez_compressed(tensors_dir / "50001.npz", tensor=tensor_data)

    manifest_df = pd.DataFrame([{
        "subject_id": "50001",
        "file_id": "NYU_0050001",
        "dx_group": 1,
        "split": "train",
        "tensor_path": "50001.npz",
    }])
    csv_path = tmp_path / "full_metadata.csv"
    manifest_df.to_csv(csv_path, index=False)

    dataset = TriSliceDataset(manifest=csv_path, tensors_dir=tensors_dir)
    assert len(dataset) == 1

    bag, label = dataset[0]
    assert bag.shape == (60, 3, 224, 224)
    assert bag.dtype == torch.float32
    assert label.item() == 1


def test_tri_slice_dataset_dataloader_batching(mock_dataset_env: Dict[str, Path]):
    """Verify compatibility with standard PyTorch DataLoader producing (B, 60, 3, H, W)."""
    from torch.utils.data import DataLoader

    dataset = TriSliceDataset(
        manifest=mock_dataset_env["manifest_path"],
        tensors_dir=mock_dataset_env["tensors_dir"],
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch_bags, batch_labels = next(iter(loader))
    assert batch_bags.shape == (2, 60, 3, 32, 32)
    assert batch_labels.shape == (2,)
    assert batch_labels.dtype == torch.long


