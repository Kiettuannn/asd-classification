# Kaggle Execution Runbook: Decoupled fMRI 3D-CNN Pipeline for ASD Classification

This runbook provides step-by-step instructions for reproducing the resting-state fMRI 3D Convolutional Neural Network (3D-CNN) Autism Spectrum Disorder (ASD) classification benchmark published by Hammash & Younis (MDPI Journal of Imaging, 2026, 12, 109).

The architecture is cleanly partitioned into two decoupled Jupyter notebooks linked via a Kaggle Dataset artifact, conforming to ADR-0001, ADR-0002, and ADR-0003.

---

## 1. Architectural Overview and Design Decisions

Running 3D volumetric deep learning directly on raw 4D fMRI neuroimaging acquisitions within a single notebook causes severe resource contention and execution timeouts on Kaggle GPU instances.

To address these limitations, the system decouples neuroimaging preprocessing from GPU model training:

```
[ABIDE I AWS S3 Public Bucket]
              |
              v
[Notebook 1: 01_fmri_3dcnn_preprocessing.ipynb] (CPU / GPU)
  - Phenotypic filtering: NYU site (184 subjects: 79 ASD, 105 TC)
  - Subject-level 80/20 partition (148 train, 36 test matching Table 3)
  - Temporal trimming (10 initial, 10 terminal buffer frames)
  - Temporal mean projection -> 3D volume
  - Z-score intensity normalization (zero mean, unit variance)
  - 50-slice orthogonal extraction (Axial, Coronal, Sagittal)
  - Bilinear slice resizing to 224x224 pixels
  - Depth-stacking into (1, 150, 224, 224) multi-view tensors
              |
              v
[Kaggle Dataset: abide-i-nyu-multiview-fmri]
  - metadata.csv (manifest with covariates and train/test split)
  - dataset-metadata.json (Kaggle dataset descriptor)
  - tensors/{subject_id}.npz (compressed volumetric tensors)
              |
              v
[Notebook 2: 02_fmri_3dcnn_training.ipynb] (NVIDIA T4 / P100 GPU)
  - Consumes /kaggle/input/abide-i-nyu-multiview-fmri/
  - Volume-Consistent Augmentation (train only, bypassed in eval)
  - Baseline 3D-CNN (6 conv blocks: 32 -> 64 -> 128 -> 128 -> 256 -> 256)
  - Automatic Mixed Precision (torch.amp.autocast fp16)
  - Physical batch size 2, gradient accumulation 4 steps (effective batch size 8)
  - Adam optimizer (lr=1e-5, weight_decay=0.0005)
  - Focal Loss (alpha=0.25, gamma=2.0)
  - Early stopping (patience 15 epochs, max 120 epochs)
  - Evaluation on holdout test set (15 ASD, 21 TC)
  - Verification against Table 8 benchmarks (81.24% Acc, 79.63% F1)
  - Exports: /kaggle/working/best_baseline_3dcnn.pt, /kaggle/working/roc_curve.png
```

---

## 2. Kaggle Environment Specifications

### Notebook 1: Preprocessing and Dataset Export
- Environment: Python 3.10 / 3.11
- Accelerator: CPU (4 cores) or GPU T4 x2
- Internet: **ON** (Required to fetch phenotypic records and NIfTI volumes from the public ABIDE AWS S3 bucket)
- Persistence: **Variables and Files**
- Estimated Runtime: ~45-60 minutes for the complete 184 NYU subject cohort

### Notebook 2: 3D-CNN Training and Evaluation
- Environment: Python 3.10 / 3.11
- Accelerator: **GPU T4 x2** (recommended) or **GPU P100** (16 GB VRAM)
- Internet: Optional (dataset is attached locally via Kaggle storage)
- Persistence: **Variables and Files**
- Estimated Runtime: ~15-25 minutes with mixed precision and gradient accumulation

---

## 3. Step-by-Step Execution Sequence

### Step 1: Execute Preprocessing in Notebook 1
1. In your Kaggle account, click **New Notebook**.
2. Go to **File** -> **Import Notebook** and upload `notebooks/01_fmri_3dcnn_preprocessing.ipynb`.
3. In the right settings panel:
   - Set **Accelerator** to **CPU** (or **GPU T4 x2**).
   - Set **Internet** to **On**.
4. In Cell 3, verify configuration constants:
   - `INITIAL_FRAMES = 10`
   - `TERMINAL_FRAMES = 10`
   - `N_SLICES = 50`
   - `TARGET_SHAPE = (224, 224)`
   - `OUTPUT_ROOT = Path('/kaggle/working/abide_nyu_multiview')`
5. Click **Save Version** -> select **Save & Run All (Commit)**.
6. Monitor the execution log. The pipeline will:
   - Download `Phenotypic_V1_0b_preprocessed1.csv`.
   - Filter down to the 184 NYU cohort subjects (79 ASD, 105 TC).
   - Partition into 148 train subjects (64 ASD, 84 TC) and 36 test subjects (15 ASD, 21 TC).
   - Retrieve and process the 4D fMRI volumes via temporal trimming, mean projection, z-score normalization, and multi-view orthogonal slicing.
   - Assemble and compress multi-view tensors of shape `(1, 150, 224, 224)` into `tensors/{subject_id}.npz`.
   - Export `metadata.csv` and `dataset-metadata.json`.

### Step 2: Publish Preprocessed Output as a Kaggle Dataset

#### Option A: Via Kaggle Web UI (Recommended)
1. Navigate to the completed run page of Notebook 1.
2. In the right panel, scroll down to the **Output** section.
3. Locate the generated folder `/kaggle/working/abide_nyu_multiview`.
4. Click the three dots menu (`...`) and select **New Dataset**.
5. Set the Dataset Title to:
   ```
   ABIDE I NYU Multi-View fMRI 3D-CNN Dataset
   ```
6. Set the Dataset Slug to:
   ```
   abide-i-nyu-multiview-fmri
   ```
7. Click **Create**. The dataset will be compiled and made available in your Kaggle datasets library.

#### Option B: Via Kaggle CLI
If running from a terminal or VM with configured Kaggle API credentials:
```bash
kaggle datasets create -p /kaggle/working/abide_nyu_multiview -u
```

### Step 3: Configure and Attach Dataset to Notebook 2
1. In Kaggle, create a new notebook or import `notebooks/02_fmri_3dcnn_training.ipynb`.
2. In the right settings panel:
   - Set **Accelerator** to **GPU T4 x2** or **GPU P100**.
   - Ensure PyTorch GPU environment is active.
3. Click **+ Add Input** at the top right:
   - Select **Your Datasets**.
   - Search for `ABIDE I NYU Multi-View fMRI 3D-CNN Dataset` (or slug `abide-i-nyu-multiview-fmri`).
   - Click the plus (`+`) button to mount the dataset.
4. Verify mounted dataset structure:
   - Dataset manifest: `/kaggle/input/abide-i-nyu-multiview-fmri/metadata.csv`
   - Tensor archives: `/kaggle/input/abide-i-nyu-multiview-fmri/tensors/`

### Step 4: Run Training and Evaluation (Notebook 2)
1. Verify hyperparameter constants in Cell 4:
   - `PHYSICAL_BATCH_SIZE = 2`
   - `ACCUMULATION_STEPS = 4` (effective batch size 8)
   - `LEARNING_RATE = 1e-5`
   - `WEIGHT_DECAY = 0.0005`
   - `EARLY_STOPPING_PATIENCE = 15`
   - `MAX_EPOCHS = 120`
   - `RUN_CROSS_VALIDATION = False` (set `True` if stratified 5-fold CV is desired)
2. Run all cells or execute **Save Version** -> **Save & Run All (Commit)**.
3. Training Progress:
   - DataLoaders instantiate with `VolumeConsistentAugmenter` on the 148 training subjects.
   - Mixed precision forward pass executes with `torch.amp.autocast(device_type="cuda", dtype=torch.float16)`.
   - Gradients are accumulated across 4 physical batches before `scaler.step(optimizer)` and `scaler.update()`.
   - `EarlyStopping` tracks validation loss on the 36 held-out test subjects.
   - When early stopping triggers or max epochs complete, the best checkpoint is reloaded.
4. Test Evaluation & ROC Curve Generation:
   - Computes Accuracy, Precision, Recall, F1-Score, Focal Loss, and ROC-AUC on the 36 test subjects.
   - Saves `best_baseline_3dcnn.pt` to `/kaggle/working/best_baseline_3dcnn.pt`.
   - Exports ROC curve plot to `/kaggle/working/roc_curve.png`.
   - Prints benchmark comparison table against published Table 8 baselines.

---

## 4. Benchmark Verification Checklist

Compare the evaluated test metrics against Table 8 from Hammash & Younis (MDPI Journal of Imaging, 2026, 12, 109):

| Metric | Published Table 8 Baseline | Target Verification Window | Description |
|---|---|---|---|
| Accuracy | 81.24% | 78.0% - 84.0% | Overall correct binary classification rate |
| Precision | 80.23% | 77.0% - 83.5% | Positive predictive value for ASD |
| Recall (Sensitivity) | 79.05% | 76.0% - 82.5% | True positive rate for ASD detection |
| F1-Score | 79.63% | 76.5% - 83.0% | Harmonic mean of precision and recall |
| Focal Loss | 0.4932 | 0.4000 - 0.5800 | Evaluated binary focal loss (alpha=0.25, gamma=2.0) |

### Verification Checklist Criteria
- [ ] Preprocessed dataset contains exactly 184 subjects with zero missing covariate values (`age`, `sex`, `fiq`).
- [ ] Partition strictly respects Table 3: 148 train subjects (64 ASD, 84 TC) and 36 test subjects (15 ASD, 21 TC).
- [ ] Each preprocessed tensor has shape `(1, 150, 224, 224)` with float32 voxel intensities normalized to zero mean and unit variance.
- [ ] Volume-consistent augmentation applies identical rotations (+/-10 deg), flips, and intensity scaling across all 150 depth slices during training.
- [ ] Augmentation is disabled during validation and test evaluation.
- [ ] Training executes using Automatic Mixed Precision (`torch.amp.autocast`) and physical batch size 2 with 4 gradient accumulation steps.
- [ ] GPU memory utilization remains within 8-12 GB VRAM on NVIDIA T4 (no CUDA Out-Of-Memory exceptions).
- [ ] Test evaluation outputs Accuracy (~81.24%) and F1-Score (~79.63%) consistent with the published baseline.
- [ ] Output artifacts `/kaggle/working/best_baseline_3dcnn.pt` and `/kaggle/working/roc_curve.png` are generated and non-empty.

---

## 5. Troubleshooting and Operational Edge Cases

### Issue: CUDA Out of Memory (CUDA OOM)
- **Cause**: Increasing `PHYSICAL_BATCH_SIZE` above 2 on a 16 GB GPU causes memory overflow due to large volumetric activation tensors (`2 * 1 * 150 * 224 * 224 * 4 bytes` per forward layer).
- **Remedy**: Keep `PHYSICAL_BATCH_SIZE = 2` and adjust `ACCUMULATION_STEPS` to achieve the target effective batch size. Ensure `torch.amp.autocast` is enabled.

### Issue: Dataset Path Resolution Error (`FileNotFoundError`)
- **Cause**: Attaching the dataset under an alternate name or running in a local environment.
- **Remedy**: `MultiViewDataset` includes fallback path resolution. Ensure the input path `/kaggle/input/abide-i-nyu-multiview-fmri` exists, or pass `tensors_dir` explicitly when initializing the dataset.

### Issue: NaN Loss During Training
- **Cause**: Zero standard deviation during z-score normalization or extreme gradient spikes.
- **Remedy**:
  - `z_score_normalize` enforces a numerical stabilizer (`epsilon = 1e-8`).
  - PyTorch `GradScaler` skips unscaled updates when inf/nan gradients are encountered.
  - Verify that Adam optimizer learning rate is set to `1e-5` (higher rates such as `1e-3` destabilize 3D-CNN training).

### Issue: Kaggle Session Timeout during Preprocessing (Notebook 1)
- **Cause**: Network throttling when sequentially downloading 184 4D fMRI NIfTI files from AWS S3.
- **Remedy**: Run Notebook 1 using Kaggle's background commit feature (**Save Version** -> **Save & Run All**), which grants up to 9 hours of uninterrupted execution time.
