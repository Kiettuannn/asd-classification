## Parent

Part of #4

## What to build

Implement the complete, standalone training and evaluation Jupyter notebook for Kaggle GPU environments (`notebooks/02_fmri_3dcnn_training.ipynb`). The notebook consumes the preprocessed Kaggle Dataset produced in #6, initializes PyTorch DataLoaders with Volume-Consistent Augmentation (#7) for training, instantiates `Baseline3DCNN` (#7), and executes the mixed-precision training loop using Adam ($lr=10^{-5}$, L2 weight decay $0.0005$, physical batch size 2, gradient accumulation 4 steps, effective batch size 8). It tracks validation loss with early stopping (patience 15), evaluates on the 20% holdout test set (36 subjects), calculates Accuracy, Precision, Recall, F1-Score, and Focal Loss, benchmarks against the published Table 8 results (81.24% accuracy, 79.63% F1-score), exports the ROC curve plot (`roc_curve.png`), and includes a configurable switch for stratified 5-fold cross-validation.

## Acceptance criteria

- [ ] Self-contained Jupyter notebook running smoothly on Kaggle GPU (T4 / P100) with mixed precision (`torch.amp.autocast`).
- [ ] Correctly loads `metadata.csv` and multi-view tensor inputs from Kaggle dataset input paths.
- [ ] Applies volume-consistent augmentation during training and disables augmentation during validation/testing.
- [ ] Implements training loop with physical batch size 2 and gradient accumulation 4 steps (effective batch size 8).
- [ ] Evaluates against the 80/20 test split (15 ASD, 21 TC) and reports Accuracy, Precision, Recall, F1-Score, and Focal Loss.
- [ ] Generates and saves an ROC curve plot with AUC score to `/kaggle/working/roc_curve.png`.
- [ ] Provides an optional flag to run stratified 5-fold cross-validation across all 184 subjects.

## Blocked by

- Blocked by: #6, #7
