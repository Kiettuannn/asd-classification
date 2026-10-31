## Parent

Part of #24

## What to build

Implement the modular PyTorch network architecture combining a 4-block volumetric 3D-CNN spatial encoder, a 2-layer Bidirectional Mamba (Bi-Mamba) sequence backbone with pure PyTorch selective scan fallback, a Gated Temporal Attention Pooling layer with boolean mask support, and a classification head.

The model processes 5D neuroimaging batches of shape `(B, T, 1, 64, 64, 36)` with accompanying boolean temporal masks `(B, T)`, mapping each 3D frame volume to a 128-dimensional embedding, scanning temporal dynamics bidirectionally with linear O(T) complexity, dynamically weighting informative time frames while zeroing out padding frames, and outputting classification logits for ASD vs. TC.

## Acceptance criteria

- [ ] `Spatial3DCNNEncoder` extracts a 128-dimensional feature vector per 3D BOLD volume using 4 convolutional blocks (channels 32 -> 64 -> 128 -> 256) with Batch Normalization, LeakyReLU, and max pooling.
- [ ] `BiMambaSequenceModel` implements 2 bidirectional selective state space blocks ($d_{model}=128, d_{state}=16, \text{expansion}=2$) scanning forward and reverse time directions.
- [ ] Dual-mode support executes fast compiled `mamba-ssm` CUDA kernels when present, and seamlessly falls back to a pure PyTorch selective scan implementation when CUDA extensions are unmounted.
- [ ] `GatedTemporalAttentionPooling` computes dynamic attention weights $\alpha_t \in [0, 1]$ over time frames, applying a $-\infty$ pre-softmax mask to guarantee padded frames receive zero attention.
- [ ] Synthetic unit test verifies end-to-end forward pass on shape `(B=2, T=50, C=1, H=64, W=64, D=36)` producing logits `(2, 2)` and valid attention maps.
- [ ] Verifies backward gradient propagation under BF16 mixed precision without NaNs or zero-gradient stalls across all trainable parameters.

## Blocked by

None (can start immediately)
