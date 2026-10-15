# Depth-Stacked Multi-View Representation for 3D-CNN Input

We decided to construct the 3D input tensor by concatenating 50 axial, 50 coronal, and 50 sagittal slices along the volumetric depth axis, yielding a single-channel 3D volume of shape `(1, 150, 224, 224)` ($C=1, D=150, H=224, W=224$). 

While multi-view data is often organized into multi-channel arrays `(3, 50, 224, 224)` or separate model branches, the paper specifies a single modality channel ($C=1$) where orthogonal planes are assembled into a unified 3D volumetric tensor. This allows standard 3D convolutional kernels ($3 \times 3 \times 3$) to capture structural cross-slice correlations across all anatomical orientations within a single unified feature hierarchy.
