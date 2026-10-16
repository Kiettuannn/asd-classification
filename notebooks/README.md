# Notebooks

Interactive experiment notebooks utilizing the modular `asd_classification` package:

### 3D-CNN Multi-View fMRI Pipeline (Hammash & Younis, 2026)
- `01_fmri_3dcnn_preprocessing.ipynb`: Decoupled neuroimaging preprocessing and dataset export for the ABIDE I NYU cohort (184 subjects). Extracts 50 orthogonal slices per plane, resizes to 224x224, assembles `(1, 150, 224, 224)` multi-view tensors, and exports `metadata.csv` and `dataset-metadata.json`.
- `02_fmri_3dcnn_training.ipynb`: Kaggle GPU training, holdout evaluation, and ROC curve generation for the 6-block Baseline 3D-CNN with Volume-Consistent Augmentation, Automatic Mixed Precision (`torch.amp`), and Focal Loss.
- Execution Guide: See `docs/kaggle_runbook.md` for end-to-end environment setup, dataset mounting, and verification instructions.

### Baseline & Graph Neural Network Experiments
- `01_svm_baseline.ipynb`: Phase 1 SVM pipeline and phenotypic analysis.
- `02_gnn_experiments.ipynb`: Exp 3 Graph Convolutional Network (GCN) and Graph Attention Network (GAT) implementations on ABIDE I.
