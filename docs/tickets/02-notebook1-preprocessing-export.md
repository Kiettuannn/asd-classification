## Parent

Part of #4

## What to build

Implement the complete, standalone preprocessing Jupyter notebook for Kaggle (`notebooks/01_fmri_3dcnn_preprocessing.ipynb`). The notebook downloads the preprocessed resting-state 4D fMRI scans for the 184 subjects of the ABIDE I NYU Cohort (79 ASD, 105 TC) with complete phenotypic data via public AWS S3 / Nilearn. It executes the core extraction pipeline from #5 to produce depth-stacked Multi-View Tensors of shape `(1, 150, 224, 224)` for each subject. It constructs a subject-level 80/20 train/test partition (148 train [64 ASD, 84 TC] and 36 test [15 ASD, 21 TC] matching Table 3), and exports the dataset manifest (`metadata.csv`) along with compressed tensors into Kaggle working storage, formatted to be published as a native Kaggle Dataset.

## Acceptance criteria

- [ ] Self-contained Jupyter notebook executable end-to-end on a standard Kaggle notebook instance.
- [ ] Automatically downloads the 184 NYU cohort subjects with complete phenotypic records.
- [ ] Applies temporal frame trimming, temporal mean projection, z-score intensity normalization, and 50-slice orthogonal extraction per subject.
- [ ] Assembles and saves Multi-View Tensors `(1, 150, 224, 224)` per subject in compressed format.
- [ ] Outputs `metadata.csv` containing `subject_id`, binary target label `dx_group` (ASD=1, TC=0), phenotypic covariates, and `split` assignment (80% train, 20% test).
- [ ] Verifies the split composition matches exactly 64 ASD and 84 TC in train, and 15 ASD and 21 TC in test.
- [ ] Includes dataset packaging instructions to upload or attach the output as an input Kaggle Dataset for training.

## Blocked by

- Blocked by: #5
