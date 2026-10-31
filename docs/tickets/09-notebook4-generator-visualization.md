## Parent

Part of #24

## What to build

Implement the generator script `scripts/build_notebook_4.py` to programmatically construct the self-contained, standalone Kaggle training Jupyter notebook `notebooks/04_spatiotemporal_bimamba_training.ipynb`.

The generated notebook executes end-to-end 5-fold cross-validation on Kaggle with 4x NVIDIA L4 GPUs, plots cross-fold ROC curves, and extracts and visualizes dynamic temporal attention trajectories ($\alpha_t$) contrasting ASD vs. TC cohorts across scan time for clinical interpretability.

## Acceptance criteria

- [ ] Generator script `scripts/build_notebook_4.py` programmatically creates `notebooks/04_spatiotemporal_bimamba_training.ipynb`.
- [ ] Generated notebook includes dataset auto-discovery, DataLoader with frame trimming, 3D-CNN + Bi-Mamba architecture with PyTorch fallback, and multi-GPU training loop.
- [ ] Executes 5-fold cross-validation on Kaggle GPU instances with zero manual path reconfiguration.
- [ ] Aggregates out-of-fold validation predictions across all 5 folds and generates an aggregate ROC curve plot with calculated AUC.
- [ ] Extracts temporal attention weights $\alpha_t$ from Gated Temporal Attention Pooling and plots mean attention trajectories contrasting ASD vs. TC cohorts across scan time.
- [ ] Automated tests verify notebook JSON schema validity, markdown/code cell syntax, and build idempotency.

## Blocked by

- Blocked by: #27
