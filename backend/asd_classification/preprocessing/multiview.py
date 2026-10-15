"""Core Multi-View Extraction and Volumetric Tensor Assembly.

This module implements the neuroimaging extraction routine that converts 4D fMRI
acquisitions into depth-stacked Multi-View Tensors of shape (1, 150, 224, 224)
in accordance with ADR-0003 and Hammash & Younis (MDPI Journal of Imaging, 2026).
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi


def trim_temporal_frames(
    data: np.ndarray,
    initial_frames: int = 10,
    terminal_frames: int = 10,
    time_axis: int = -1,
) -> np.ndarray:
    """Discard initial and terminal temporal frames from a 4D fMRI acquisition.

    The first 10 frames are discarded to eliminate scanner equilibration noise
    and T1 saturation transient artifacts. The terminal 10 frames are discarded
    to remove subject movement and scan termination noise.

    Args:
        data: 4D numpy array containing fMRI BOLD timeseries.
        initial_frames: Number of early buffer frames to discard (default: 10).
        terminal_frames: Number of ending buffer frames to discard (default: 10).
        time_axis: Axis index representing the temporal dimension (default: -1).

    Returns:
        Trimmed 4D numpy array with buffer frames excluded.

    Raises:
        ValueError: If frame counts are negative or total frames are insufficient.
    """
    if initial_frames < 0 or terminal_frames < 0:
        raise ValueError(
            f"Frame trimming counts cannot be negative (got initial={initial_frames}, "
            f"terminal={terminal_frames})."
        )

    # Normalize negative axis to positive index
    axis = time_axis if time_axis >= 0 else data.ndim + time_axis
    total_frames = data.shape[axis]
    required_frames = initial_frames + terminal_frames

    if total_frames <= required_frames:
        raise ValueError(
            f"Temporal length ({total_frames}) along axis {axis} must be greater than "
            f"the sum of initial ({initial_frames}) and terminal ({terminal_frames}) "
            f"buffer frames to discard."
        )

    # Slice dynamically along the specified temporal axis
    slice_indices = slice(initial_frames, total_frames - terminal_frames)
    indexer = [slice(None)] * data.ndim
    indexer[axis] = slice_indices

    return np.ascontiguousarray(data[tuple(indexer)])


def compute_temporal_mean(data: np.ndarray, time_axis: int = -1) -> np.ndarray:
    """Compute the Temporal Mean Projection across fMRI time frames.

    Averaging BOLD fluctuations across retained frames produces a static 3D
    functional activation volume per subject, capturing baseline functional intensity
    while reducing high-frequency physiological noise.

    Args:
        data: 4D numpy array of fMRI acquisitions.
        time_axis: Axis index representing the temporal dimension (default: -1).

    Returns:
        3D numpy array representing the mean functional volume.
    """
    return np.mean(data, axis=time_axis, dtype=np.float32)


def z_score_normalize(volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Apply subject-level z-score intensity normalization.

    Normalizes voxel intensities according to I_norm = (I - mu) / (sigma + eps)
    to standardize signal contrast across subjects and scanners. Epsilon prevents
    division by zero in background voxels or zero-variance volumes.

    Args:
        volume: 3D numpy array of voxel intensities.
        epsilon: Small numerical constant for stability against zero variance.

    Returns:
        Normalized 3D volume of type float32 with approximately zero mean and unit variance.
    """
    mean_val = float(np.mean(volume))
    std_val = float(np.std(volume))
    normalized = (volume - mean_val) / (std_val + epsilon)
    return normalized.astype(np.float32)


def extract_middle_slices(
    volume: np.ndarray,
    n_slices: int = 50,
    axis: int = 2,
) -> np.ndarray:
    """Extract contiguous middle slices centered along an anatomical axis.

    Central brain structures contain the most discriminative cortical and subcortical
    regions for ASD classification, whereas peripheral slices contain non-brain skull
    and background air.

    Args:
        volume: 3D numpy array of shape (X, Y, Z).
        n_slices: Number of contiguous slices to extract (default: 50).
        axis: Anatomical slicing axis (0: Sagittal, 1: Coronal, 2: Axial).

    Returns:
        Array of shape (n_slices, H_view, W_view) containing the extracted 2D slices.

    Raises:
        ValueError: If dimension size along axis is smaller than n_slices.
    """
    dim_size = volume.shape[axis]
    if dim_size < n_slices:
        raise ValueError(
            f"Dimension size {dim_size} along axis {axis} is smaller than requested "
            f"slice count {n_slices}."
        )

    # Calculate centered bounding indices
    center = dim_size // 2
    start = center - (n_slices // 2)
    end = start + n_slices

    # Ensure bounds remain within array boundaries
    if end > dim_size:
        start = dim_size - n_slices
        end = dim_size
    elif start < 0:
        start = 0
        end = n_slices

    indexer = [slice(None)] * volume.ndim
    indexer[axis] = slice(start, end)
    sliced_subvolume = volume[tuple(indexer)]

    # Move the sliced axis to position 0 to produce uniform (n_slices, H, W) layout
    return np.ascontiguousarray(np.moveaxis(sliced_subvolume, axis, 0))


def extract_orthogonal_slices(
    volume: np.ndarray,
    n_slices: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract 50 middle slices along Axial, Coronal, and Sagittal planes.

    Assuming standard canonical RAS+ orientation:
    - Axial plane: slices orthogonal to Z axis (axis 2), yielding (X, Y) slices.
    - Coronal plane: slices orthogonal to Y axis (axis 1), yielding (X, Z) slices.
    - Sagittal plane: slices orthogonal to X axis (axis 0), yielding (Y, Z) slices.

    Args:
        volume: 3D numpy array of shape (X, Y, Z) in canonical orientation.
        n_slices: Number of middle slices per anatomical plane (default: 50).

    Returns:
        Tuple of (axial_slices, coronal_slices, sagittal_slices), each of shape
        (n_slices, H_view, W_view).
    """
    axial_slices = extract_middle_slices(volume, n_slices=n_slices, axis=2)
    coronal_slices = extract_middle_slices(volume, n_slices=n_slices, axis=1)
    sagittal_slices = extract_middle_slices(volume, n_slices=n_slices, axis=0)

    return axial_slices, coronal_slices, sagittal_slices


def resize_slice(
    slice_2d: np.ndarray,
    target_shape: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize a 2D anatomical slice using bilinear interpolation.

    Standardizes image resolution to (224, 224) across all views.

    Args:
        slice_2d: 2D numpy array of shape (H, W).
        target_shape: Target dimensions (target_h, target_w) (default: (224, 224)).

    Returns:
        Resized 2D array of shape target_shape and dtype float32.
    """
    curr_h, curr_w = slice_2d.shape
    target_h, target_w = target_shape

    if (curr_h, curr_w) == (target_h, target_w):
        return slice_2d.astype(np.float32)

    zoom_factors = (target_h / curr_h, target_w / curr_w)
    # Order 1 corresponds to bilinear interpolation in scipy.ndimage
    resized = ndi.zoom(slice_2d, zoom_factors, order=1).astype(np.float32)

    # Guard against float rounding discrepancies to guarantee exact target dimensions
    if resized.shape != (target_h, target_w):
        exact_output = np.zeros((target_h, target_w), dtype=np.float32)
        min_h = min(resized.shape[0], target_h)
        min_w = min(resized.shape[1], target_w)
        exact_output[:min_h, :min_w] = resized[:min_h, :min_w]
        return exact_output

    return resized


def resize_slices(
    slices: np.ndarray,
    target_shape: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize a stack of 2D slices to target resolution via bilinear interpolation.

    Args:
        slices: 3D numpy array of shape (N, H, W).
        target_shape: Target spatial dimensions (target_h, target_w) (default: (224, 224)).

    Returns:
        Resized slice stack of shape (N, target_h, target_w) and dtype float32.
    """
    n_slices, curr_h, curr_w = slices.shape
    target_h, target_w = target_shape

    if (curr_h, curr_w) == (target_h, target_w):
        return slices.astype(np.float32)

    # Perform vectorized in-plane interpolation across all slices simultaneously
    zoom_factors = (1.0, target_h / curr_h, target_w / curr_w)
    resized = ndi.zoom(slices, zoom_factors, order=1).astype(np.float32)

    # Guard against float rounding discrepancies along spatial axes
    if resized.shape != (n_slices, target_h, target_w):
        exact_output = np.zeros((n_slices, target_h, target_w), dtype=np.float32)
        min_h = min(resized.shape[1], target_h)
        min_w = min(resized.shape[2], target_w)
        exact_output[:, :min_h, :min_w] = resized[:, :min_h, :min_w]
        return exact_output

    return resized


def assemble_multiview_tensor(
    axial_slices: np.ndarray,
    coronal_slices: np.ndarray,
    sagittal_slices: np.ndarray,
) -> np.ndarray:
    """Concatenate multi-view slice stacks into a single-channel 3D volume.

    Per ADR-0003, 50 axial, 50 coronal, and 50 sagittal slices are stacked along
    the depth axis to produce a single-channel tensor of shape (1, 150, 224, 224).
    This unified volume allows 3D convolutional kernels (3x3x3) to jointly learn
    intra-plane spatial patterns and cross-plane volumetric correlations.

    Args:
        axial_slices: Array of shape (n_slices, 224, 224).
        coronal_slices: Array of shape (n_slices, 224, 224).
        sagittal_slices: Array of shape (n_slices, 224, 224).

    Returns:
        Multi-view tensor of shape (1, 3 * n_slices, 224, 224) and dtype float32.
    """
    # Depth concatenation in order: axial (0:50), coronal (50:100), sagittal (100:150)
    depth_stacked = np.concatenate([axial_slices, coronal_slices, sagittal_slices], axis=0)
    # Add single modality channel dimension (C=1, D=150, H=224, W=224)
    tensor = np.expand_dims(depth_stacked, axis=0)
    return tensor.astype(np.float32)


def extract_multiview_tensor(
    fmri_input: Union[np.ndarray, nib.spatialimages.SpatialImage, str, Path],
    n_slices: int = 50,
    target_shape: Tuple[int, int] = (224, 224),
    initial_frames: int = 10,
    terminal_frames: int = 10,
    epsilon: float = 1e-8,
    time_axis: int = -1,
) -> np.ndarray:
    """End-to-end extraction pipeline from 4D fMRI to depth-stacked Multi-View Tensor.

    Executes:
    1. Loading and standardizing orientation to canonical RAS+ space.
    2. Discarding initial 10 and terminal 10 temporal frames.
    3. Computing Temporal Mean Projection into a 3D functional volume.
    4. Subject-level z-score intensity normalization.
    5. Extracting 50 middle slices along axial, coronal, and sagittal planes.
    6. Bilinear resizing of all slices to 224x224.
    7. Concatenating slices along depth into shape (1, 150, 224, 224).

    Args:
        fmri_input: 4D numpy array, nibabel SpatialImage object, or file path.
        n_slices: Number of middle slices per plane (default: 50).
        target_shape: Target 2D slice dimensions (default: (224, 224)).
        initial_frames: Initial frames to discard (default: 10).
        terminal_frames: Terminal frames to discard (default: 10).
        epsilon: Numerical stability constant for z-score normalization (default: 1e-8).
        time_axis: Temporal axis in raw numpy array (default: -1).

    Returns:
        Standardized Multi-View Tensor of shape (1, 3 * n_slices, target_h, target_w)
        and dtype float32.

    Raises:
        FileNotFoundError: If input file path does not exist.
        ValueError: If input is not a 4D acquisition array or image.
    """
    # Handle on-disk file paths
    if isinstance(fmri_input, (str, Path)):
        file_path = Path(fmri_input)
        if not file_path.is_file():
            raise FileNotFoundError(f"Input fMRI file not found at: {file_path}")
        nii = nib.load(str(file_path))
        # Enforce canonical RAS+ orientation to guarantee predictable slice axes
        canonical_nii = nib.as_closest_canonical(nii)
        data = np.asarray(canonical_nii.get_fdata(dtype=np.float32))
        effective_time_axis = -1
    # Handle in-memory NIfTI image objects
    elif hasattr(fmri_input, "get_fdata"):
        canonical_nii = nib.as_closest_canonical(fmri_input)
        data = np.asarray(canonical_nii.get_fdata(dtype=np.float32))
        effective_time_axis = -1
    # Handle raw numpy arrays
    elif isinstance(fmri_input, np.ndarray):
        data = fmri_input.astype(np.float32, copy=False)
        effective_time_axis = time_axis
    else:
        raise TypeError(
            f"Unsupported input type '{type(fmri_input)}'. Expected np.ndarray, "
            f"nibabel SpatialImage, or Path/str."
        )

    if data.ndim != 4:
        raise ValueError(
            f"Input must be a 4D fMRI array of shape (X, Y, Z, T), got {data.ndim}D "
            f"with shape {data.shape}."
        )

    # 1. Frame trimming
    trimmed = trim_temporal_frames(
        data,
        initial_frames=initial_frames,
        terminal_frames=terminal_frames,
        time_axis=effective_time_axis,
    )

    # 2. Temporal Mean Projection
    mean_volume = compute_temporal_mean(trimmed, time_axis=effective_time_axis)

    # 3. Intensity normalization
    normalized_volume = z_score_normalize(mean_volume, epsilon=epsilon)

    # 4. Orthogonal middle slice extraction
    axial, coronal, sagittal = extract_orthogonal_slices(normalized_volume, n_slices=n_slices)

    # 5. Bilinear slice resizing
    axial_resized = resize_slices(axial, target_shape=target_shape)
    coronal_resized = resize_slices(coronal, target_shape=target_shape)
    sagittal_resized = resize_slices(sagittal, target_shape=target_shape)

    # 6. Depth concatenation and channel dimension assembly
    multiview_tensor = assemble_multiview_tensor(
        axial_resized,
        coronal_resized,
        sagittal_resized,
    )

    return multiview_tensor


class MultiViewExtractor:
    """Configurable pipeline for fMRI multi-view volumetric tensor extraction."""

    def __init__(
        self,
        n_slices: int = 50,
        target_shape: Tuple[int, int] = (224, 224),
        initial_frames: int = 10,
        terminal_frames: int = 10,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize MultiViewExtractor configuration.

        Args:
            n_slices: Number of middle slices per orthogonal plane (default: 50).
            target_shape: Target slice height and width (default: (224, 224)).
            initial_frames: Initial frames to discard (default: 10).
            terminal_frames: Terminal frames to discard (default: 10).
            epsilon: Epsilon for numerical stability during z-score normalization.
        """
        self.n_slices = n_slices
        self.target_shape = target_shape
        self.initial_frames = initial_frames
        self.terminal_frames = terminal_frames
        self.epsilon = epsilon

    def extract(
        self,
        fmri_input: Union[np.ndarray, nib.spatialimages.SpatialImage, str, Path],
        time_axis: int = -1,
    ) -> np.ndarray:
        """Execute multi-view tensor extraction on provided fMRI input.

        Args:
            fmri_input: 4D numpy array, NIfTI image, or file path.
            time_axis: Temporal axis if input is a numpy array (default: -1).

        Returns:
            Multi-view tensor of shape (1, 3 * n_slices, target_h, target_w).
        """
        return extract_multiview_tensor(
            fmri_input=fmri_input,
            n_slices=self.n_slices,
            target_shape=self.target_shape,
            initial_frames=self.initial_frames,
            terminal_frames=self.terminal_frames,
            epsilon=self.epsilon,
            time_axis=time_axis,
        )

    def __call__(
        self,
        fmri_input: Union[np.ndarray, nib.spatialimages.SpatialImage, str, Path],
        time_axis: int = -1,
    ) -> np.ndarray:
        """Callable shorthand for extract."""
        return self.extract(fmri_input=fmri_input, time_axis=time_axis)
