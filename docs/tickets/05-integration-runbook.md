## Parent

Part of #4

## What to build

Implement an end-to-end integration test and Kaggle execution runbook. The integration suite validates that the preprocessed output artifacts from Notebook 1 (#6) match the input interface required by Notebook 2 (#8) using mock/sample data, and verifies that one full training epoch and evaluation step runs without numerical or structural errors on CPU and GPU. The accompanying runbook provides step-by-step instructions for uploading the notebooks to Kaggle, configuring GPU accelerators (NVIDIA T4 / P100), connecting Notebook 1's output dataset to Notebook 2, and replicating the fMRI 3D-CNN benchmark from Hammash & Younis (2026).

## Acceptance criteria

- [ ] Automated integration test verifies the contract between Notebook 1's exported `metadata.csv` / tensor files and Notebook 2's dataset reader.
- [ ] A mock/smoke training execution runs for 1 epoch on CPU/GPU without errors or memory leaks.
- [ ] Reproducibility runbook document detailing Kaggle environment configuration, GPU accelerator activation, dataset mounting, and execution sequence.
- [ ] Verification checklist comparing executed run outputs against published Table 8 metrics (Accuracy ~81.24%, F1 ~79.63%).

## Blocked by

- Blocked by: #8
