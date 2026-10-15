# ASD Classification (fMRI 3D-CNN)

Context for replicating the fMRI 3D-CNN autism spectrum disorder classification experiment on the ABIDE I NYU cohort.

## Language

**ASD**:
Autism Spectrum Disorder, the target positive diagnostic classification label.
_Avoid_: Autistic case, disorder cohort

**TC**:
Typically Developing Control, the target negative neurotypical control label.
_Avoid_: Normal control, NC, healthy control

**Multi-View Tensor**:
A volumetric 3D tensor constructed by concatenating extracted 2D middle slices along the axial, coronal, and sagittal orthogonal anatomical planes into shape `(1, 150, 224, 224)`.
_Avoid_: Multi-angle volume, 3D projection stack

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
