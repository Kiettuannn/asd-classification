## Parent

Part of #24

## What to build

Implement the multi-site 4D rs-fMRI data loading pipeline with dynamic Kaggle dataset auto-discovery, on-the-fly neuroimaging preprocessing, sequence standardization, and reproducible joint Site-and-Class Stratified 5-Fold Cross-Validation.

The loader scans candidate storage locations under `/kaggle/input/` and local paths, pairs 4D NIfTI files with phenotypic metadata, trims scanner boundary frames (first 10 and last 10 TRs), normalizes voxel intensities, resamples/crops spatial volumes to $64 \times 64 \times 36$, pads variable temporal sequences up to $T_{\max} = 200$ with boolean attention masks, and partitions all subjects across the 17 institutions into 5 balanced validation folds.

## Acceptance criteria

- [ ] Auto-discovers 4D functional NIfTI files (`*func_preproc.nii.gz`) and phenotypic CSV files across candidate paths under `/kaggle/input/` and local directories without hardcoded paths.
- [ ] Trims first 10 and last 10 temporal frames to remove scanner equilibrium and termination artifacts.
- [ ] Applies subject-level non-zero voxel z-score intensity normalization.
- [ ] Standardizes 3D spatial dimensions to $64 \times 64 \times 36$ in standard MNI space.
- [ ] Pads variable sequence durations ($T \in [78, 300]$) to $T_{\max} = 200$ and outputs boolean temporal masks (`True` for recorded frames, `False` for padded frames).
- [ ] Implements joint Site-and-Class Stratified 5-Fold Cross-Validation, ensuring each fold contains balanced proportions of ASD/TC subjects across all 17 institutions.
- [ ] Automated tests verify file matching, output tensor shapes `(B, T, 1, 64, 64, 36)`, mask values, and non-overlapping fold partitions.

## Blocked by

None (can start immediately)
