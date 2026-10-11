"""Tests for model architectures and base interfaces."""

from src.models.base import BaseBrainModel
from src.models.baseline.svm import SVMClassifier
from src.models.gnn.gcn import BrainGCN
from src.models.gnn.gat import BrainGAT


def test_model_hierarchy():
    """Verify all models inherit from BaseBrainModel."""
    assert issubclass(SVMClassifier, BaseBrainModel)
    assert issubclass(BrainGCN, BaseBrainModel)
    assert issubclass(BrainGAT, BaseBrainModel)
