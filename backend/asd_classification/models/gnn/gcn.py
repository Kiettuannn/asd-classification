"""Graph Convolutional Network (GCN) architecture for whole-brain graph classification."""

from pathlib import Path
from typing import Any, Dict
import numpy as np

from asd_classification.models.base import BaseBrainModel


class BrainGCN(BaseBrainModel):
    """
    3-layer GCN with BatchNorm, global pooling (mean + max), and MLP readout.
    Operates on PyTorch Geometric Data graphs constructed from brain FC matrices.
    """

    def __init__(
        self,
        in_channels: int = 200,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.model: Any = None

    def fit(self, X: Any, y: np.ndarray, **kwargs: Any) -> "BrainGCN":
        """Trains GCN using PyG DataLoader and Adam optimizer."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels for a dataset of brain graphs."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predicts softmax probabilities for TD and ASD classes."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def save(self, file_path: Path) -> None:
        """Saves PyTorch state dict to disk."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def load(self, file_path: Path) -> "BrainGCN":
        """Restores PyTorch state dict from disk."""
        raise NotImplementedError("Implementation will be added in implementation phase.")
