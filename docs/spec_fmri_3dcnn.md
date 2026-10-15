## Problem Statement

Researchers and machine learning practitioners investigating Autism Spectrum Disorder (ASD) neuroimaging classification need to replicate the resting-state fMRI 3D Convolutional Neural Network (3D-CNN) experiment from Hammash & Younis (MDPI Journal of Imaging, 2026, 12, 109). However, running 3D volumetric deep learning on 4D neuroimaging datasets directly in a single notebook causes severe resource bottlenecks, excessive memory usage, and notebook timeouts on Kaggle GPU instances. There is currently no modular, decoupled two-notebook pipeline that separates heavy neuroimaging preprocessing and slice extraction from GPU-accelerated model training and evaluation while adhering to the exact published multi-view experimental setup.

## Solution

Provide two cleanly decoupled Jupyter notebooks optimized for Kaggle GPU execution:
1. Preprocessing Notebook: Downloads the ABIDE I NYU cohort (184 subjects: 79 ASD, 105 TC), performs temporal mean projection (discarding initial and terminal buffer frames), applies z-score intensity normalization, extracts 50 middle slices along the axial, coronal, and sagittal orthogonal planes resized to 224x224, and exports compressed multi-view tensors of shape `(1, 150, 224, 224)` alongside a subject-level 80/20 train/test split metadata manifest.
2. Training Notebook: Consumes the preprocessed Kaggle dataset, implements volume-consistent data augmentation (in-plane rotations, flips, intensity scaling), trains the 6-block Baseline 3D-CNN architecture using PyTorch with Automatic Mixed Precision (AMP) and gradient accumulation, optimizes with Focal Loss (and Cross-Entropy toggle), evaluates against the published 80/20 benchmark (comparing against 81.24% accuracy and 79.63% F1-score), and plots ROC curves.

## User Stories

1. As a researcher, I want to download preprocessed resting-state fMRI scans for the ABIDE I NYU cohort automatically, so that I have a consistent set of 184 subjects with complete phenotypic data without manual retrieval.
2. As a neuroimaging analyst, I want to discard the first 10 and last 10 temporal frames of each 4D fMRI acquisition, so that scanner equilibration instability and termination noise do not contaminate downstream features.
3. As a researcher, I want to compute a temporal mean projection across the remaining fMRI time series, so that 4D BOLD signals are consolidated into a representative 3D functional activation volume per subject.
4. As a machine learning practitioner, I want to apply subject-level z-score intensity normalization to each 3D volume, so that inter-subject scanner intensity discrepancies are eliminated.
5. As an ML engineer, I want to extract the 50 middle slices along the axial, coronal, and sagittal planes, so that the most discriminative central brain structures are captured while discarding uninformative peripheral slices.
6. As a researcher, I want all extracted slices resized to 224x224 pixels using bilinear interpolation, so that spatial resolution is standardized across all views.
7. As a deep learning engineer, I want the 50 slices from each plane concatenated along the depth dimension to form a single-channel multi-view tensor of shape `(1, 150, 224, 224)`, so that 3D convolutional kernels can simultaneously process spatial and cross-plane volumetric correlations.
8. As a practitioner, I want the preprocessed tensors saved in compressed format along with a metadata CSV file, so that the preprocessed assets can be uploaded or mounted as a native Kaggle Dataset.
9. As an evaluator, I want the dataset partitioned into an 80/20 subject-level train/test split (64 ASD, 84 TC in train; 15 ASD, 21 TC in test) before any augmentation occurs, so that test data leakage is strictly prevented.
10. As an ML engineer, I want the training notebook to load preprocessed multi-view tensors directly from Kaggle input storage, so that model training begins immediately without re-running heavy neuroimaging preprocessing.
11. As a researcher, I want volume-consistent data augmentation applied to training samples, so that random in-plane rotations within +/-10 degrees, horizontal/vertical flips, and intensity scaling [0.9, 1.1] are applied identically across all 150 slices of a subject tensor.
12. As an engineer, I want validation and test samples evaluated without any data augmentation, so that test performance reflects realistic clinical diagnostic accuracy.
13. As an engineer, I want a 6-block Baseline 3D-CNN implemented in PyTorch, so that feature representations incrementally scale from 32 to 256 channels through 3x3x3 convolutions with Batch Normalization and LeakyReLU activations.
14. As a researcher, I want global average pooling and a 0.4 dropout rate applied before the final classification head, so that spatial features are compacted and overfitting is minimized.
15. As a Kaggle user, I want the model to train using PyTorch Automatic Mixed Precision (AMP fp16) and gradient accumulation (batch size 2 with 4 accumulation steps), so that training runs efficiently within a 16 GB GPU VRAM envelope.
16. As an ML engineer, I want the training loop optimized using Adam with an initial learning rate of 1e-5 and L2 weight decay of 0.0005, so that the optimization protocol strictly matches Table 4 of the paper.
17. As a researcher, I want Focal Loss (alpha=0.25, gamma=2) used as the primary loss function with a switch for standard Cross-Entropy, so that the impact of hard-to-classify samples can be assessed as formulated in Equation (39).
18. As a practitioner, I want early stopping monitored on validation loss with a patience of 15 epochs up to a maximum of 120 epochs, so that training terminates before overfitting occurs.
19. As an evaluator, I want classification performance reported using Accuracy, Precision, Recall, F1-Score, and Focal Loss on the test set, so that outcomes can be directly compared against the published Table 8 benchmark of 81.24% accuracy.
20. As a researcher, I want the training notebook to generate and save an ROC curve plot with calculated AUC, so that diagnostic discrimination can be visually assessed across all decision thresholds.
21. As an investigator, I want an optional configuration flag to execute stratified 5-fold cross-validation, so that cross-fold variance and generalization stability can be statistically evaluated.

## Implementation Decisions

- Architectural Decoupling: The project is divided into two distinct, standalone Kaggle notebooks:
  - Notebook 1: Data retrieval, temporal summarization, multi-view slice extraction, and dataset artifact generation.
  - Notebook 2: Dataset consumption, volume-consistent augmentation, 3D-CNN model architecture, mixed-precision training, and evaluation.
- Multi-View Volumetric Input Format: The input to the 3D-CNN is structured as a single-channel 3D volume of shape `(1, 150, 224, 224)` (Channel=1, Depth=150, Height=224, Width=224), formed by concatenating 50 axial, 50 coronal, and 50 sagittal slices along the depth axis per ADR-0003.
- Model Architecture (Baseline 3D-CNN):
  - 6 convolutional blocks with 3x3x3 kernel size, stride 1, padding 1:
    - Block 1: Conv3d(1, 32) -> BatchNorm3d(32) -> LeakyReLU(0.01) -> MaxPool3d(2)
    - Block 2: Conv3d(32, 64) -> BatchNorm3d(64) -> LeakyReLU(0.01) -> MaxPool3d(2)
    - Block 3: Conv3d(64, 128) -> BatchNorm3d(128) -> LeakyReLU(0.01) -> MaxPool3d(2)
    - Block 4: Conv3d(128, 128) -> BatchNorm3d(128) -> LeakyReLU(0.01) -> MaxPool3d(2)
    - Block 5: Conv3d(128, 256) -> BatchNorm3d(256) -> LeakyReLU(0.01)
    - Block 6: Conv3d(256, 256) -> BatchNorm3d(256) -> LeakyReLU(0.01)
  - Head: AdaptiveAvgPool3d(1) -> Flatten() -> Dropout(0.4) -> Linear(256, 2)
- Framework Selection: PyTorch with native Automatic Mixed Precision (`torch.amp.autocast`) rather than Keras (per ADR-0001) to support efficient 3D convolution processing on Kaggle 16 GB GPUs.
- Batching and Memory Strategy: Physical batch size of 2 with 4 gradient accumulation steps (effective batch size of 8), allowing training on 16 GB VRAM GPUs without memory overflow.
- Data Augmentation Contract: Volume-consistent transforms sampled once per subject volume and applied across all depth slices during training:
  - In-plane rotation: random angle theta in [-10, +10] degrees.
  - Random horizontal flip: p = 0.5.
  - Random vertical flip: p = 0.5.
  - Random intensity scaling: factor in [0.9, 1.1].
- Loss Formulation: Focal Loss with alpha=0.25 and gamma=2.0 as primary loss (Equation 39), with a configurable option for Cross-Entropy loss.
- Dataset Cohort: Exclusively resting-state fMRI from ABIDE I NYU site (184 subjects: 79 ASD, 105 TC), split at the subject level (80% train = 148 subjects, 20% test = 36 subjects).

## Testing Decisions

- Test Seams:
  - Preprocessing Seam: Verify that processing a 4D fMRI array through temporal trimming, mean volume computation, slice extraction, and resizing produces an exact shape of `(1, 150, 224, 224)` with float32 values normalized to zero mean and unit variance.
  - Augmentation Consistency Seam: Verify that augmenting a multi-view tensor applies the exact same geometric rotation and flip to the first slice (axial) and last slice (sagittal) in the depth stack.
  - Model Forward & Backward Seam: Verify that a synthetic batch of shape `(2, 1, 150, 224, 224)` passes through `Baseline3DCNN`, yields logits of shape `(2, 2)`, computes Focal Loss, and propagates non-zero gradients to all model parameters under AMP.
  - Metric Evaluation Seam: Verify that evaluation metrics (Accuracy, Precision, Recall, F1, Focal Loss, ROC-AUC) compute correctly on known ground-truth logits and binary targets.
- Good Test Criteria: Tests must exercise external input/output behavior (tensor dimensions, mathematical invariants, gradient flow) rather than internal intermediate variable names.
- Prior Art: Existing tests in `backend/tests/test_models.py` and `backend/tests/test_features.py`.

## Out of Scope

- Structural MRI (sMRI) downloading, processing, and multi-modal fusion.
- Transformer baselines (Vision Transformer, Swin Transformer).
- Temporal Convolutional Network (TCN) and Graph Neural Network (DB-APGNN) baselines.
- The paper's proposed dual-modality Bi-LSTM + 3D-HCNN cross-modality fusion architecture.
- Full multi-site ABIDE I dataset training (only NYU site is in scope).

## Further Notes

- Target published baseline metrics for fMRI multi-view 50 slices (Table 8): Accuracy: 81.24%, Precision: 80.23%, Recall: 79.05%, F1-Score: 79.63%, Focal Loss: 0.4932.
- The notebooks are designed to run seamlessly on Kaggle GPU instances with zero manual path reconfiguration.
