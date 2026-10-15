## Parent

Part of #4

## What to build

Implement and verify the neuroimaging extraction routine that converts 4D fMRI acquisitions into depth-stacked Multi-View Tensors. The pipeline trims the first 10 and last 10 temporal frames to avoid scanner equilibration and termination noise, computes the Temporal Mean Projection across the remaining frames, applies subject-level z-score intensity normalization, extracts 50 successive middle slices along each of the three orthogonal anatomical planes (axial, coronal, sagittal), resizes them to 224x224 via bilinear interpolation, and concatenates them along the depth axis into a single-channel 3D volume of shape `(1, 150, 224, 224)` per ADR-0003.

## Acceptance criteria

- [x] Pipeline accepts a 4D fMRI acquisition array or NIfTI object and produces a standardized tensor of shape `(1, 150, 224, 224)`.
- [x] Initial 10 frames and terminal 10 frames are discarded before temporal summarization.
- [x] Temporal Mean Projection is computed over the remaining frames using mean volume calculation.
- [x] Subject-level z-score normalization ($I_{norm} = (I - \mu)/\sigma$) is applied with numerical stability epsilon.
- [x] Exactly 50 successive middle slices are extracted along the axial, coronal, and sagittal planes (150 slices total).
- [x] Slices are resized to 224x224 and concatenated along the depth dimension.
- [x] Automated unit test verifies tensor dimensions, float32 dtype, and invariant properties using synthetic 4D test arrays.

## Blocked by

None (can start immediately).
