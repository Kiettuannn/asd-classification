## Parent

Part of #4

## What to build

Implement and verify the core deep learning components for the fMRI 3D-CNN baseline:
1. `Baseline3DCNN`: A 6-block PyTorch 3D Convolutional Neural Network scaling from 32 to 256 channels with $3 \times 3 \times 3$ kernels, stride 1, padding 1, BatchNorm3d, LeakyReLU (slope 0.01), MaxPool3d(2), AdaptiveAvgPool3d(1), Dropout(0.4), and a Linear(256, 2) classification head.
2. `FocalLoss`: Implementation of $\alpha=0.25, \gamma=2.0$ focal loss for multi-class/binary classification (Equation 39), featuring an optional toggle for standard Cross-Entropy loss.
3. `VolumeConsistentAugmenter`: PyTorch transform implementing random in-plane rotations within $\pm 10^\circ$, horizontal and vertical flips ($p=0.5$), and intensity scaling ($[0.9, 1.1]$) sampled once per subject volume and applied identically across all 150 depth slices during training only.

## Acceptance criteria

- [ ] `Baseline3DCNN` instantiates with specified channel progression (32 -> 64 -> 128 -> 128 -> 256 -> 256) and Dropout 0.4.
- [ ] Forward pass on a synthetic batch of shape `(2, 1, 150, 224, 224)` outputs logits of shape `(2, 2)`.
- [ ] `FocalLoss` computes finite loss, places higher weight on difficult samples, and behaves as expected when toggled to Cross-Entropy.
- [ ] Volume-consistent augmentation applies the exact same affine parameters and scaling across all 150 slices in a multi-view tensor.
- [ ] Forward and backward propagation runs successfully under Automatic Mixed Precision (`torch.amp.autocast`) and updates all parameter gradients.
- [ ] Unit tests verify model shapes, loss numerical properties, and augmentation slice invariance.

## Blocked by

- Blocked by: #5
