# ASD Classification (fMRI 3D-CNN)

Context for replicating and advancing the fMRI classification experiments across the full multi-site ABIDE I dataset and the NYU cohort.

## Language

**ASD**:
Autism Spectrum Disorder, the target positive diagnostic classification label.
_Avoid_: Autistic case, disorder cohort

**TC**:
Typically Developing Control, the target negative neurotypical control label.
_Avoid_: Normal control, NC, healthy control

**Full ABIDE I Cohort**:
The multi-site neuroimaging dataset comprising 1,112 subjects (539 ASD, 573 TC) acquired across 17 international scanning institutions, characterized by heterogeneous repetition times ($TR \in [1.5s, 3.0s]$) and variable scan lengths ($T \in [78, 300]$).
_Avoid_: Full dataset, aggregate sample

**NYU Cohort**:
The subset of 184 subjects from the New York University site in the ABIDE I repository (79 ASD, 105 TC) with complete phenotypic and neuroimaging data.
_Avoid_: NYU dataset, NYU sample

**Baseline 3D-CNN**:
The 6-block volumetric convolutional neural network with 3x3x3 kernels and channel progression 32 to 256, Batch Normalization, LeakyReLU, Global Average Pooling, and Dropout 0.4.
_Avoid_: 3D ConvNet, volumetric CNN, custom 3D model

**Temporal Mean Projection**:
A static 3D volumetric representation computed by averaging 4D fMRI BOLD signals across time frames (excluding boundary buffer frames).
_Avoid_: Temporal pooling, time-collapse, mean time series

**Volume-Consistent Augmentation**:
Spatial and intensity transformations sampled once per subject volume and applied identically across all slices in the multi-view tensor.
_Avoid_: Independent slice augmentation, slice jitter

**rs-fMRI**:
Resting-state functional Magnetic Resonance Imaging, acquiring 4D BOLD physiological time series $(X, Y, Z, T)$ capturing brain hemodynamics over time.
_Avoid_: Static fMRI, anatomical MRI

**sMRI**:
Structural Magnetic Resonance Imaging (T1-weighted), acquiring high-resolution static 3D anatomical brain tissue morphology $(X, Y, Z)$ without a temporal dimension.
_Avoid_: Dynamic MRI, functional scan

**4D BOLD Spatio-Temporal Tensor**:
A 4D neuroimaging sequence preserving both spatial volume geometry and chronological acquisition time frames $(C, T, H, W, D)$ for sequence modeling.
_Avoid_: Flattened BOLD, collapsed volume

**Dynamic Functional Connectivity (dFC)**:
Time-varying statistical correlations computed across parcellated Region-of-Interest (ROI) time series over sliding temporal windows, yielding a graph sequence $G_1, G_2, \dots, G_K$.
_Avoid_: Static functional connectivity, stationary connectome

**Bidirectional Mamba (Bi-Mamba)**:
A non-causal selective state space sequence model that scans the fMRI time series in both forward and reverse directions with $O(T)$ linear computational complexity, capturing bidirectional hemodynamic dependencies across scanning time.
_Avoid_: Causal SSM, autoregressive Mamba

**Gated Temporal Attention Pooling**:
A parametric attention mechanism assigning dynamic scalar weights $\alpha_t \in [0, 1]$ across time frames $t \in [1, \dots, T]$ to focus on transient neuroimaging markers before subject-level classification.
_Avoid_: Uniform temporal pooling, last-step selection

**3D-CNN + Bi-Mamba**:
A spatio-temporal deep neural network where a volumetric 3D-CNN extracts frame-level spatial representations $z_t$ from native 3D BOLD volumes, and a Bidirectional Mamba block models sequential dependencies across time frames.
_Avoid_: 4D ConvNet, pure Mamba

**Site-Stratified Cross-Validation**:
Validation partitioning that enforces balanced distributions of both diagnostic class (ASD vs. TC) and acquisition site across folds to prevent site-confounding leakage.
_Avoid_: Random cross-validation, unstratified split
