"""Support Vector Machine (SVM) pipeline for vectorized FC features."""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from asd_classification.models.base import BaseBrainModel


class SVMClassifier(BaseBrainModel):
    """SVM classifier wrapping StandardScaler, optional PCA, and SVC."""

    def __init__(self, c: float = 1.0, kernel: str = "rbf", gamma: str = "scale"):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.pipeline: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "SVMClassifier":
        """Fits StandardScaler and SVC on flattened upper-triangle FC vectors."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels (0 or 1)."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability estimates for TD and ASD classes."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def save(self, file_path: Path) -> None:
        """Saves fitted scikit-learn pipeline using joblib."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def load(self, file_path: Path) -> "SVMClassifier":
        """Loads fitted scikit-learn pipeline from disk."""
        raise NotImplementedError("Implementation will be added in implementation phase.")

    def explain(self, sample: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Calculates top discriminative functional connections using linear weights or permutation."""
        raise NotImplementedError("Implementation will be added in implementation phase.")
