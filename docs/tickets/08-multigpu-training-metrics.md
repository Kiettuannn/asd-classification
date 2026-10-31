## Parent

Part of #24

## What to build

Implement the multi-GPU training and evaluation engine supporting 4x NVIDIA L4 GPUs via PyTorch DistributedDataParallel (DDP) / DataParallel with native BF16 mixed precision, Focal Loss, learning rate scheduling, and comprehensive clinical validation metrics.

The training engine coordinates training batches across multiple GPUs, optimizes network parameters with AdamW and cosine annealing, handles class imbalance via weighted Focal Loss, checks early stopping criteria on validation loss, and logs out-of-fold diagnostic metrics across all 5 cross-validation folds.

## Acceptance criteria

- [ ] Implements multi-GPU training orchestration supporting 4x NVIDIA L4 GPUs with native BF16 mixed precision (`torch.amp.autocast(dtype=torch.bfloat16)`).
- [ ] Integrates Focal Loss ($\alpha=0.25, \gamma=2.0$) with class weighting calculated from fold label distributions.
- [ ] Optimizes parameters using AdamW with cosine annealing learning rate scheduler and configurable gradient clipping.
- [ ] Implements early stopping monitoring validation loss with configurable epoch patience.
- [ ] Tracks fold-level metrics: Balanced Accuracy, ROC-AUC, Sensitivity, Specificity, F1-Score, and validation Focal Loss.
- [ ] Saves best model checkpoint weights per fold based on validation Balanced Accuracy and ROC-AUC.
- [ ] Integration test executes a synthetic multi-step training and evaluation loop, confirming metric calculation and checkpoint saving.

## Blocked by

- Blocked by: #25, #26
