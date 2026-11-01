"""
Loss functions for dual-modal connectome-phenotypic alignment and classification.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric temperature-scaled InfoNCE (Noise-Contrastive Estimation) loss.

    Aligns normalized connectome embeddings z_brain with clinical phenotypic embeddings z_text
    by computing cross-modal contrastive similarity distributions across mini-batch samples.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = True,
        min_temp: float = 0.01,
        max_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp

        if learnable_temp:
            # Parameterize as log-temperature to enforce strict positivity during gradient updates
            self.log_temp = nn.Parameter(torch.tensor(np.log(temperature), dtype=torch.float32))
        else:
            self.register_buffer("log_temp", torch.tensor(np.log(temperature), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        """Returns the current clamped temperature value."""
        return torch.clamp(self.log_temp.exp(), min=self.min_temp, max=self.max_temp)

    def forward(self, z_brain: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric InfoNCE contrastive loss between modalities.

        Args:
            z_brain: Brain connectome embeddings of shape (B, D).
            z_text: Phenotypic text embeddings of shape (B, D).

        Returns:
            Scalar tensor representing symmetric contrastive loss.
        """
        # Ensure L2 unit normalization
        z_b = F.normalize(z_brain, p=2, dim=-1)
        z_t = F.normalize(z_text, p=2, dim=-1)

        batch_size = z_b.size(0)
        tau = self.temperature

        # Cosine similarity matrix scaled by temperature: shape (B, B)
        similarity = torch.matmul(z_b, z_t.T) / tau

        # Targets correspond to matching diagonal indices (subject i with profile i)
        targets = torch.arange(batch_size, device=z_b.device, dtype=torch.long)

        # Cross-entropy in both directions (Brain -> Text and Text -> Brain)
        loss_b2t = F.cross_entropy(similarity, targets)
        loss_t2b = F.cross_entropy(similarity.T, targets)

        return 0.5 * (loss_b2t + loss_t2b)


class DualModalLoss(nn.Module):
    """
    Joint multi-task objective combining label-smoothed cross-entropy and InfoNCE alignment:
    L_total = L_CE(y, y_hat) + lambda * L_InfoNCE(z_brain, z_text).
    """

    def __init__(
        self,
        lambda_weight: float = 0.3,
        label_smoothing: float = 0.1,
        temperature: float = 0.07,
        learnable_temp: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_weight = lambda_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.infonce = InfoNCELoss(temperature=temperature, learnable_temp=learnable_temp)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z_brain: torch.Tensor,
        z_text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total joint multi-task loss.

        Args:
            logits: Diagnostic prediction logits of shape (B, num_classes).
            labels: Ground truth binary class labels of shape (B,).
            z_brain: Connectome embeddings of shape (B, D).
            z_text: Phenotypic profile embeddings of shape (B, D).

        Returns:
            Dictionary containing 'loss' (total objective), 'loss_ce', and 'loss_infonce'.
        """
        loss_ce = self.ce_loss(logits, labels)
        loss_infonce = self.infonce(z_brain, z_text)
        total_loss = loss_ce + self.lambda_weight * loss_infonce

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_infonce": loss_infonce,
        }
