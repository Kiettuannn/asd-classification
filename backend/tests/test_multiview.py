"""Unit tests for Core Multi-View Extraction and Volumetric Tensor Assembly.

This test suite verifies the neuroimaging preprocessing routine described in
Ticket #5 and ADR-0003, ensuring that 4D fMRI acquisitions are correctly converted
into depth-stacked Multi-View Tensors of shape (1, 150, 224, 224).
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import pytest

from asd_classification.preprocessing.multiview import (
    MultiViewExtractor,
    assemble_multiview_tensor,
    compute_temporal_mean,
    extract_middle_slices,
    extract_multiview_tensor,
    extract_orthogonal_slices,
    resize_slice,
    resize_slices,
    trim_temporal_frames,
    z_score_normalize,
)


@pytest.fixture
def synthetic_4d_fmri() -> np.ndarray:
    """Generate a reproducible synthetic 4D fMRI volume array.
    
    Spatial dimensions (64, 64, 64) ensure each orthogonal dimension has at least
    50 slices for extraction, and temporal dimension 40 allows discarding 10 initial
    and 10 terminal frames with 20 frames remaining for temporal mean projection.
    """
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((64, 64, 64, 40), dtype=np.float32)


def test_trim_temporal_frames_valid(synthetic_4d_fmri: np.ndarray):
    """Verify that trimming removes exactly 10 initial and 10 terminal frames."""
    trimmed = trim_temporal_frames(synthetic_4d_fmri, initial_frames=10, terminal_frames=10, time_axis=-1)
    
    # 40 frames total - 10 initial - 10 terminal = 20 remaining frames
    assert trimmed.shape == (64, 64, 64, 20)


def test_trim_temporal_frames_discards_noise():
    """Verify scanner equilibration and termination noise are excluded from remaining signal."""
    shape = (52, 52, 52, 35)
    data = np.ones(shape, dtype=np.float32)
    
    # Inject large noise artifacts into the initial 10 and terminal 10 frames
    data[:, :, :, :10] = 999.0
    data[:, :, :, -10:] = 888.0
    
    trimmed = trim_temporal_frames(data, initial_frames=10, terminal_frames=10, time_axis=-1)
    
    # The trimmed volume should only contain the central valid signal (all 1.0)
    assert np.all(trimmed == 1.0)
    assert trimmed.shape[-1] == 15


def test_trim_temporal_frames_insufficient_frames():
    """Verify that temporal sequences with <= initial + terminal frames raise a ValueError."""
    data = np.zeros((52, 52, 52, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="Temporal length .* must be greater than"):
        trim_temporal_frames(data, initial_frames=10, terminal_frames=10)


def test_trim_temporal_frames_invalid_parameters():
    """Verify that negative frame trimming parameters are rejected."""
    data = np.zeros((52, 52, 52, 30), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be negative"):
        trim_temporal_frames(data, initial_frames=-1, terminal_frames=10)


def test_compute_temporal_mean():
    """Verify arithmetic mean calculation across the temporal dimension."""
    shape = (50, 50, 50, 5)
    # Create frames with known values 1, 2, 3, 4, 5
    data = np.stack([np.full((50, 50, 50), float(i), dtype=np.float32) for i in range(1, 6)], axis=-1)
    
    mean_vol = compute_temporal_mean(data, time_axis=-1)
    
    assert mean_vol.shape == (50, 50, 50)
    # Expected mean of [1, 2, 3, 4, 5] is 3.0
    assert np.allclose(mean_vol, 3.0, atol=1e-5)


def test_z_score_normalize_standard():
    """Verify that z-score normalization results in approximately zero mean and unit variance."""
    rng = np.random.default_rng(seed=123)
    volume = rng.normal(loc=150.0, scale=25.0, size=(50, 50, 50)).astype(np.float32)
    
    norm_vol = z_score_normalize(volume, epsilon=1e-8)
    
    assert norm_vol.dtype == np.float32
    assert np.isclose(float(norm_vol.mean()), 0.0, atol=1e-5)
    assert np.isclose(float(norm_vol.std()), 1.0, atol=1e-4)


def test_z_score_normalize_constant_volume():
    """Verify numerical stability with epsilon when input volume has zero variance."""
    constant_vol = np.full((50, 50, 50), 42.0, dtype=np.float32)
    norm_vol = z_score_normalize(constant_vol, epsilon=1e-8)
    
    # Division by epsilon should not produce NaN or Inf
    assert not np.isnan(norm_vol).any()
    assert not np.isinf(norm_vol).any()
    assert np.allclose(norm_vol, 0.0, atol=1e-5)


def test_extract_middle_slices_range():
    """Verify extraction of contiguous middle slices centered along an axis."""
    # Create volume with coordinate values along axis 2
    dim_z = 64
    volume = np.broadcast_to(np.arange(dim_z, dtype=np.float32), (60, 60, dim_z))
    
    # Midpoint of 64 is 32; 50 middle slices range from 32 - 25 = 7 to 7 + 50 = 57
    slices = extract_middle_slices(volume, n_slices=50, axis=2)
    
    assert slices.shape == (50, 60, 60)
    expected_indices = np.arange(7, 57, dtype=np.float32)
    for idx, expected_val in enumerate(expected_indices):
        assert np.all(slices[idx] == expected_val)


def test_extract_middle_slices_dimension_too_small():
    """Verify ValueError is raised when dimension size is smaller than requested slice count."""
    volume = np.zeros((40, 60, 60), dtype=np.float32)
    with pytest.raises(ValueError, match="is smaller than requested slice count"):
        extract_middle_slices(volume, n_slices=50, axis=0)


def test_extract_orthogonal_slices():
    """Verify that 50 middle slices are extracted along axial, coronal, and sagittal planes."""
    volume = np.zeros((60, 70, 80), dtype=np.float32)
    axial, coronal, sagittal = extract_orthogonal_slices(volume, n_slices=50)
    
    # Axial slices (sliced along Z axis=2): 2D slices have shape (X=60, Y=70)
    assert axial.shape == (50, 60, 70)
    # Coronal slices (sliced along Y axis=1): 2D slices have shape (X=60, Z=80)
    assert coronal.shape == (50, 60, 80)
    # Sagittal slices (sliced along X axis=0): 2D slices have shape (Y=70, Z=80)
    assert sagittal.shape == (50, 70, 80)


def test_resize_slice():
    """Verify 2D bilinear interpolation resizes arbitrary slice shapes to exactly (224, 224)."""
    slice_2d = np.ones((61, 73), dtype=np.float32)
    resized = resize_slice(slice_2d, target_shape=(224, 224))
    
    assert resized.shape == (224, 224)
    assert resized.dtype == np.float32
    assert np.allclose(resized, 1.0, atol=1e-5)


def test_resize_slices():
    """Verify batch resizing of slice stack to (N, 224, 224)."""
    slices = np.ones((50, 55, 65), dtype=np.float32)
    resized = resize_slices(slices, target_shape=(224, 224))
    
    assert resized.shape == (50, 224, 224)
    assert resized.dtype == np.float32


def test_assemble_multiview_tensor_shape_and_ordering():
    """Verify assembly into single-channel 3D volume of shape (1, 150, 224, 224) with slice ordering."""
    axial = np.full((50, 224, 224), 1.0, dtype=np.float32)
    coronal = np.full((50, 224, 224), 2.0, dtype=np.float32)
    sagittal = np.full((50, 224, 224), 3.0, dtype=np.float32)
    
    tensor = assemble_multiview_tensor(axial, coronal, sagittal)
    
    assert tensor.shape == (1, 150, 224, 224)
    assert tensor.dtype == np.float32
    
    # Verify depth stack ordering: [0:50] axial, [50:100] coronal, [100:150] sagittal
    assert np.all(tensor[0, 0:50, :, :] == 1.0)
    assert np.all(tensor[0, 50:100, :, :] == 2.0)
    assert np.all(tensor[0, 100:150, :, :] == 3.0)


def test_extract_multiview_tensor_end_to_end_numpy(synthetic_4d_fmri: np.ndarray):
    """Verify end-to-end extraction from a 4D numpy array."""
    tensor = extract_multiview_tensor(synthetic_4d_fmri)
    
    assert tensor.shape == (1, 150, 224, 224)
    assert tensor.dtype == np.float32
    assert not np.isnan(tensor).any()


def test_extract_multiview_tensor_end_to_end_nifti(synthetic_4d_fmri: np.ndarray):
    """Verify end-to-end extraction from an in-memory NIfTI image object."""
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    nii_img = nib.Nifti1Image(synthetic_4d_fmri, affine)
    
    tensor = extract_multiview_tensor(nii_img)
    
    assert tensor.shape == (1, 150, 224, 224)
    assert tensor.dtype == np.float32


def test_extract_multiview_tensor_from_file(synthetic_4d_fmri: np.ndarray, tmp_path: Path):
    """Verify end-to-end extraction from an on-disk NIfTI file path."""
    nii_file = tmp_path / "sub-01_bold.nii.gz"
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(synthetic_4d_fmri, affine), nii_file)
    
    # Test path as string and as Path object
    tensor_from_path = extract_multiview_tensor(nii_file)
    tensor_from_str = extract_multiview_tensor(str(nii_file))
    
    assert tensor_from_path.shape == (1, 150, 224, 224)
    assert tensor_from_str.shape == (1, 150, 224, 224)
    assert np.allclose(tensor_from_path, tensor_from_str)


def test_multiview_extractor_class_custom_configuration(synthetic_4d_fmri: np.ndarray):
    """Verify object-oriented MultiViewExtractor with configurable slice count and target resolution."""
    extractor = MultiViewExtractor(n_slices=10, target_shape=(112, 112))
    
    # 3 planes * 10 slices = 30 depth slices
    tensor = extractor(synthetic_4d_fmri)
    
    assert tensor.shape == (1, 30, 112, 112)
    assert tensor.dtype == np.float32


def test_extract_multiview_tensor_invalid_dimension():
    """Verify that a non-4D array raises a ValueError."""
    arr_3d = np.zeros((64, 64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="Input must be a 4D fMRI array"):
        extract_multiview_tensor(arr_3d)


def test_extract_multiview_tensor_file_not_found():
    """Verify that non-existent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        extract_multiview_tensor("non_existent_file.nii.gz")
