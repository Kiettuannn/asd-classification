"""Focal Loss implementation for handling class imbalance and hard-to-classify samples.

Formulates Focal Loss per Equation (39) of Hammash & Younis (MDPI Journal of Imaging, 2026)
and Lin et al. (ICCV 2017). Modulates standard cross-entropy loss by a factor of
(1 - p_t)^gamma to diminish the relative contribution of easy, well-classified examples
and force gradient updates to focus on hard, ambiguous borderline cases.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with alpha-balancing and hard-sample gamma focusing.

    Computes:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the ground-truth class,
    gamma is the focusing parameter, and alpha_t balances class frequencies.
    Includes an optional toggle to revert directly to standard Cross-Entropy loss.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        use_cross_entropy: bool = False,
    ) -> None:
        """Initialize the Focal Loss module.

        Args:
            alpha: Balancing factor. If float in (0, 1), assigns weight alpha to class 1
                and (1 - alpha) to class 0. If list or Tensor, specifies per-class weights.
                If None, classes are weighted equally (default: 0.25).
            gamma: Focusing exponent. Increasing gamma smoothly reduces the loss contribution
                of easy samples (p_t >> 0.5) (default: 2.0).
            reduction: Reduction mode applied to the output: 'none' | 'mean' | 'sum' (default: 'mean').
            use_cross_entropy: When True, bypasses focal modulation and evaluates standard
                Cross-Entropy loss (default: False).

        Raises:
            ValueError: If reduction mode is not one of 'mean', 'sum', or 'none'.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Unsupported reduction: '{reduction}'. Must be one of 'mean', 'sum', or 'none'."
            )

        self.gamma = float(gamma)
        self.reduction = reduction
        self.use_cross_entropy = bool(use_cross_entropy)

        # Register alpha buffer if provided as sequence or tensor
        self.alpha: Optional[Union[float, torch.Tensor]]
        if isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.to(dtype=torch.float32))
        elif isinstance(alpha, (int, float)):
            self.alpha = float(alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss or toggled Cross-Entropy loss.

        Args:
            logits: Predicted class unnormalized logits of shape (B, num_classes).
            targets: Ground-truth class labels of shape (B,) or (B, 1),
                or one-hot probability vectors of shape (B, num_classes).

        Returns:
            Computed scalar loss (for reduction 'mean' or 'sum') or per-sample loss (for 'none').
        """
        # If toggled to standard Cross-Entropy, delegate directly to PyTorch functional
        if self.use_cross_entropy:
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1).long()
            return F.cross_entropy(logits, targets, reduction=self.reduction)

        num_classes = logits.shape[-1]

        # Standardize 1D class indices or one-hot targets
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            class_targets = targets.view(-1).long()
            # Numerical stability: compute log_softmax first to prevent overflow/underflow
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            # Gather probability of true target class p_t and log(p_t)
            log_pt = log_probs.gather(dim=-1, index=class_targets.unsqueeze(1)).squeeze(1)
            pt = probs.gather(dim=-1, index=class_targets.unsqueeze(1)).squeeze(1)

            # Determine alpha_t weighting
            if self.alpha is None:
                alpha_t = torch.ones_like(pt)
            elif isinstance(self.alpha, float):
                # Binary classification convention: class 1 -> alpha, class 0 -> (1 - alpha)
                if num_classes == 2:
                    weights = torch.tensor(
                        [1.0 - self.alpha, self.alpha],
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    alpha_t = weights.gather(dim=0, index=class_targets)
                else:
                    alpha_t = torch.full_like(pt, self.alpha)
            elif isinstance(self.alpha, torch.Tensor):
                device_alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
                alpha_t = device_alpha.gather(dim=0, index=class_targets)
            else:
                alpha_t = torch.ones_like(pt)

        elif targets.ndim == 2 and targets.shape[1] == num_classes:
            # Soft / one-hot probability targets
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            log_pt = (log_probs * targets).sum(dim=-1)
            pt = (probs * targets).sum(dim=-1)

            if self.alpha is None:
                alpha_t = torch.ones_like(pt)
            elif isinstance(self.alpha, float) and num_classes == 2:
                weights = torch.tensor(
                    [1.0 - self.alpha, self.alpha],
                    dtype=logits.dtype,
                    device=logits.device,
                )
                alpha_t = (targets * weights).sum(dim=-1)
            elif isinstance(self.alpha, torch.Tensor):
                device_alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
                alpha_t = (targets * device_alpha).sum(dim=-1)
            else:
                alpha_t = torch.ones_like(pt)
        else:
            raise ValueError(
                f"Incompatible target shape {targets.shape} for logits shape {logits.shape}"
            )

        # Clamp pt for numerical stability under mixed-precision fp16
        pt = pt.clamp(min=1e-7, max=1.0 - 1e-7)

        # Modulating focal factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Compute sample focal loss
        loss = -alpha_t * focal_weight * log_pt

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
