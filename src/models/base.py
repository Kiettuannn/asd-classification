"""Abstract base interface for brain classification models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


class BaseBrainModel(ABC):
    """Unified interface for both classical and deep learning brain connectivity models."""

    @abstractmethod
    def fit(self, X: Any, y: np.ndarray, **kwargs: Any) -> "BaseBrainModel":
        """Train the model on the provided connectivity data."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Predict binary diagnostic class labels (0 = TD, 1 = ASD)."""
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probability scores for each class."""
        pass

    @abstractmethod
    def save(self, file_path: Path) -> None:
        """Persist model weights and configuration to disk."""
        pass

    @abstractmethod
    def load(self, file_path: Path) -> "BaseBrainModel":
        """Restore model weights and state from disk."""
        pass

    def explain(self, sample: Any, top_k: int = 10) -> Dict[str, Any]:
        """
        Extract feature importances or attention edge weights for explainability.
        
        Args:
            sample: Input connectivity matrix, vector, or graph.
            top_k: Number of most influential connections to return.
            
        Returns:
            Dictionary containing salient ROI indices and attribution weights.
        """
        return {"top_connections": [], "weights": []}
