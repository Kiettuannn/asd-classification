"""Tests for model architectures and base interfaces."""

from asd_classification.models.base import BaseBrainModel
from asd_classification.models.baseline.svm import SVMClassifier
from asd_classification.models.gnn.gcn import BrainGCN
from asd_classification.models.gnn.gat import BrainGAT


def test_model_hierarchy():
    """Verify all models inherit from BaseBrainModel."""
    assert issubclass(SVMClassifier, BaseBrainModel)
    assert issubclass(BrainGCN, BaseBrainModel)
    assert issubclass(BrainGAT, BaseBrainModel)
