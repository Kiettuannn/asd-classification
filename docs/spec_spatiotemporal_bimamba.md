## Problem Statement

Researchers and machine learning practitioners classifying Autism Spectrum Disorder (ASD) from resting-state functional Magnetic Resonance Imaging (rs-fMRI) across multi-site datasets (such as the Full ABIDE I Cohort of 1,112 subjects across 17 institutions) face two major architectural limitations. First, traditional pipelines collapse the 4D BOLD physiological time series $(X, Y, Z, T)$ into a static 3D volume using Temporal Mean Projection, entirely discarding transient functional brain dynamics and temporal anomalies. Second, attempting to process 4D neuroimaging with standard sequence models either causes prohibitive memory bottlenecks and quadratic complexity $O(T^2)$ (Transformers) or suffers from limited memory horizon and gradient instability across long scans (LSTMs), while imposing unnatural causal masks on bidirectional neuroimaging recordings. Furthermore, researchers who have already downloaded and exported raw 4D rs-fMRI datasets to cloud platforms (such as Kaggle) lack a unified, multi-GPU accelerated PyTorch pipeline that can automatically discover mounted 4D NIfTI files, normalize spatial and temporal dimensions, and train a high-performance Spatio-Temporal 3D-CNN with Bidirectional Mamba (Bi-Mamba) across multiple GPUs (such as 4x NVIDIA L4 GPUs).

## Solution

Provide a high-performance, self-contained Spatio-Temporal 3D-CNN with Bidirectional Mamba training pipeline designed for multi-GPU execution on 4x NVIDIA L4 GPUs:
1. Auto-discovers pre-exported 4D rs-fMRI NIfTI files and phenotypic manifest CSVs across mounted Kaggle storage (`/kaggle/input/`) and local storage environments.
2. Implements a robust Spatio-Temporal Data Loader that trims scanner stabilization buffer frames (first 10 and last 10 frames), applies subject-level non-zero voxel z-score intensity normalization, standardizes 3D spatial resolution ($64 \times 64 \times 36$), pads variable scan sequences ($T \in [78, 300]$) to a uniform maximum duration ($T_{\max} = 200$) with boolean temporal masking, and enforces joint Site-and-Class Stratified 5-Fold Cross-Validation.
3. Constructs an integrated 3D-CNN + Bi-Mamba architecture:
   - A 4-block volumetric 3D-CNN spatial encoder extracting 128-dimensional latent representations $z_t \in \mathbb{R}^{128}$ from each 3D BOLD volume.
   - A 2-layer Bidirectional Mamba (Bi-Mamba) sequence backbone ($d_{model}=128, d_{state}=16, \text{expansion}=2$) scanning forward and backward across time with linear $O(T)$ complexity, including a pure PyTorch selective scan fallback for platform portability.
   - A Gated Temporal Attention Pooling mechanism that dynamically weights informative time frames while masking padded intervals.
   - A classification head trained with Focal Loss ($\alpha=0.25, \gamma=2.0$) and class weighting.
4. Accelerates training across 4x NVIDIA L4 GPUs using PyTorch DistributedDataParallel (DDP) / DataParallel with native BF16 mixed precision.
5. Generates comprehensive clinical validation outputs including 5-fold cross-validation metrics (Balanced Accuracy, ROC-AUC, Sensitivity, Specificity, F1-Score), ROC curves, and dynamic temporal attention interpretability maps contrasting ASD against TC subjects.

## User Stories

1. As a neuroimaging researcher, I want the training pipeline to automatically discover mounted 4D rs-fMRI datasets under `/kaggle/input/`, so that I do not have to manually reconfigure input directory paths.
2. As an ML engineer, I want the pipeline to locate phenotypic summary tables (`Phenotypic_V1_0b_preprocessed1.csv` or `metadata.csv`) dynamically, so that diagnostic labels and phenotypic covariates are matched automatically to NIfTI files.
3. As a data engineer, I want the DataLoader to automatically discard the first 10 and last 10 temporal frames of each 4D fMRI scan, so that scanner equilibration instability and termination noise do not contaminate downstream features.
4. As a researcher, I want subject-level non-zero voxel z-score intensity normalization applied to each 3D volume, so that inter-subject baseline intensity variations are eliminated.
5. As an ML practitioner, I want all 3D BOLD volumes standardized to a spatial grid of $64 \times 64 \times 36$ voxels in MNI space, so that volumetric feature shapes are uniform across all subjects.
6. As a researcher, I want variable scan lengths ($T \in [78, 300]$) padded with zeros up to $T_{\max} = 200$ with an accompanying boolean temporal mask, so that batches can be processed in parallel on multi-GPU hardware.
7. As an evaluator, I want 5-fold cross-validation partitioned via joint Site-and-Class Stratification, so that every fold preserves identical institutional and diagnostic ratios (ASD vs. TC across all 17 sites).
8. As a deep learning engineer, I want a 4-block volumetric 3D-CNN spatial encoder to extract a 128-dimensional latent representation $z_t \in \mathbb{R}^{128}$ for each chronological frame, so that spatial voxel neighborhood interactions are effectively captured.
9. As an engineer, I want the 3D-CNN to scale channel dimensions from 32 to 256 using 3x3x3 convolutions with Batch Normalization, LeakyReLU activations, and max pooling, so that spatial representations are hierarchical.
10. As a researcher, I want the sequence model to process spatial representations using Bidirectional Mamba (Bi-Mamba), so that past and future hemodynamic correlations are captured without causal masking limitations.
11. As an engineer, I want the Bi-Mamba sequence model to operate with linear $O(T)$ computational and memory complexity, so that long BOLD sequences are trained efficiently without quadratic GPU memory spikes.
12. As a practitioner, I want the Bi-Mamba module to include a pure PyTorch selective scan fallback, so that the notebook executes seamlessly even when pre-compiled `mamba-ssm` CUDA binaries are unavailable in the host environment.
13. As a researcher, I want Gated Temporal Attention Pooling to aggregate the sequence of latent vectors $[h_1, \dots, h_T]$ into a single subject representation, so that intermittent, transient functional abnormalities receive higher diagnostic weight.
14. As an analyst, I want the Gated Temporal Attention Pooling to apply a $-\infty$ attention mask over padded time frames, so that non-biological padding tokens contribute strictly zero weight to the subject representation.
15. As a clinician, I want the model to output temporal attention weight trajectories $\alpha_t \in [0, 1]$ across scan duration, so that temporal intervals contributing most strongly to the ASD diagnosis can be visualized and interpreted.
16. As a deep learning practitioner, I want a classification head consisting of Layer Normalization, 0.4 Dropout, and a linear projection, so that overfitting on small-to-medium neuroimaging cohorts is minimized.
17. As an engineer, I want the model optimized using Focal Loss with $\alpha=0.25$ and $\gamma=2.0$, so that hard-to-classify borderline subjects receive proportional focus and class imbalance is mitigated.
18. As a Kaggle user with 4x NVIDIA L4 GPUs, I want the training pipeline to support multi-GPU parallelism (DistributedDataParallel / DataParallel), so that training throughput scales efficiently across all 96 GB of aggregate VRAM.
19. As an engineer, I want native BF16 mixed precision (`torch.amp.autocast(dtype=torch.bfloat16)`) enabled on the NVIDIA L4 GPUs, so that execution is accelerated without numerical underflow or gradient scaling overhead.
20. As an ML practitioner, I want optimization driven by the AdamW optimizer with an initial learning rate of 1e-4, weight decay of 1e-4, and a cosine annealing learning rate scheduler, so that convergence is smooth and stable.
21. As an evaluator, I want validation performance evaluated across all 5 folds using Balanced Accuracy, ROC-AUC, Sensitivity, Specificity, F1-Score, and Focal Loss, so that evaluation is statistically robust and clinically informative.
22. As a researcher, I want the notebook to aggregate out-of-fold validation predictions and plot an overall ROC curve with calculated AUC, so that diagnostic discrimination can be verified across all decision thresholds.
23. As a developer, I want early stopping monitored on validation loss with configurable patience, so that models do not overfit to training folds.
24. As an engineer, I want checkpointing to save the best model weights per fold based on validation Balanced Accuracy and ROC-AUC, so that top-performing weights are preserved for inference.
25. As a researcher, I want the training pipeline designed with modular interfaces, so that future comparative models (such as CC200 Dynamic Functional Connectivity Graph Neural Networks) can be integrated and benchmarked under the exact same data partitions.

## Implementation Decisions

- Decoupled Architectural Modularity: The architecture is structured as a modular sequence `SpatioTemporalBiMamba` composed of independent, decoupled components:
  - `Spatial3DCNNEncoder`: Takes frame-level 3D volumes $(B \times T, 1, 64, 64, 36)$ and maps them to spatial embeddings $(B, T, 128)$.
  - `BiMambaSequenceModel`: Processes the sequence $(B, T, 128)$ bidirectionally via forward and reverse selective state space blocks ($d_{model}=128, d_{state}=16, \text{expansion}=2$) with an automatic dual-mode implementation (compiled `mamba-ssm` or native pure PyTorch SSM).
  - `GatedTemporalAttentionPooling`: Implements parametric attention $\alpha_t = \text{softmax}(w^T \tanh(W_a h_t) \odot \sigma(W_b h_t))$ with boolean mask support, mapping $(B, T, 128) \to (B, 128)$.
  - `ClassificationHead`: Implements `LayerNorm(128)` -> `Dropout(0.4)` -> `Linear(128, 2)`.
- Input Auto-Discovery Interface: Flexible dataset root discovery that inspects candidate directories under `/kaggle/input/` and local paths, locating NIfTI files and phenotypic tables without hardcoded user paths.
- Temporal Normalization Contract: Input 4D fMRI sequences are dynamically trimmed by 10 frames at both ends, non-zero voxels are z-scored per subject, and sequences are padded to $T_{\max} = 200$ with boolean temporal masks passed throughout the network.
- Hardware Profile and Distributed Strategy: Standardized on 4x NVIDIA L4 GPUs utilizing native BF16 mixed precision and PyTorch multi-GPU data parallelism.
- Loss Formulation: Focal Loss with $\alpha=0.25$ and $\gamma=2.0$ combined with inverse class frequency weighting.
- Cross-Validation Partitioning: Joint Site-and-Class Stratification ensuring that each of the 5 folds contains balanced distributions of both ASD/TC diagnoses and institutional sites across the 17 institutions.

## Testing Decisions

- Test Seams:
  - Synthetic End-to-End Pipeline Seam: Verify that passing a synthetic 5D batch $(B=2, T=50, C=1, H=64, W=64, D=36)$ with temporal mask $(B=2, T=50)$ through `SpatioTemporalBiMamba` produces valid logits $(2, 2)$, valid attention weights $\alpha_t \in [0, 1]$ where masked positions receive zero attention, computes Focal Loss, and backpropagates non-zero gradients under BF16 mixed precision.
  - SSM Fallback Parity Seam: Verify that the pure PyTorch selective scan fallback executes identically on CPU/GPU without depending on compiled CUDA binaries.
  - Stratified Partitioning Seam: Verify that the joint Site-and-Class Stratified 5-Fold split assigns every subject to exactly one validation fold without overlap, with each fold reflecting balanced site and diagnosis ratios.
  - Metric Evaluation Seam: Verify that evaluation metrics (Balanced Accuracy, ROC-AUC, Sensitivity, Specificity, F1-Score) compute correctly on known ground-truth logits and binary targets.
- Good Test Criteria: Tests must exercise external input/output behavior (tensor dimensions, mathematical invariants, gradient flow, and mask adherence) rather than internal intermediate variable names.
- Prior Art: Existing test suites in `backend/tests/test_cnn3d.py`, `backend/tests/test_resnet_mil.py`, and `backend/tests/test_mil_training_pipeline.py`.

## Out of Scope

- Raw fMRI downloading from AWS S3 (pre-downloaded and exported on Kaggle by the user).
- Structural MRI (sMRI / T1w) multi-modal anatomical fusion.
- Transformer and LSTM temporal baselines (Phase 1 focus is strictly 3D-CNN + Bi-Mamba).
- CC200 Dynamic Functional Connectivity Graph Neural Network (reserved for Phase 2 comparative study).

## Further Notes

- The complete training notebook will be programmatically generated via `scripts/build_notebook_4.py` and exported to `notebooks/04_spatiotemporal_bimamba_training.ipynb`.
- Hardware targeting 4x NVIDIA L4 GPUs natively supports large batch sizes and high-throughput BF16 tensor operations.
