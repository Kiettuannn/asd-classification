"""
Model implementations for ASD classification.
"""

from .base import BaseBrainModel
from .baseline.svm import SVMClassifier
from .gnn.gcn import BrainGCN
from .gnn.gat import BrainGAT
from .brainnetcnn import BrainNetCNN, E2EBlock, E2NBlock, N2GBlock
from .text_encoder import PhenotypicTextEncoder
from .losses import InfoNCELoss, DualModalLoss
from .dual_modal_engine import DualModalBrainNetModel
from .atlas import CC200YeoAtlas, get_cc200_yeo_atlas, YEO_7_NETWORKS
from .gradcam import BrainNetGradCAM
from .fidelity import (
    FidelityEvaluator,
    compute_edge_saliency_matrix,
    mask_top_edges,
    retain_top_edges,
)
from .visualization import plot_glass_brain_diagnostics, plot_network_enrichment
from .explainability import NeuroExplainabilitySuite

__all__ = [
    "BaseBrainModel",
    "SVMClassifier",
    "BrainGCN",
    "BrainGAT",
    "BrainNetCNN",
    "E2EBlock",
    "E2NBlock",
    "N2GBlock",
    "PhenotypicTextEncoder",
    "InfoNCELoss",
    "DualModalLoss",
    "DualModalBrainNetModel",
    "CC200YeoAtlas",
    "get_cc200_yeo_atlas",
    "YEO_7_NETWORKS",
    "BrainNetGradCAM",
    "FidelityEvaluator",
    "compute_edge_saliency_matrix",
    "mask_top_edges",
    "retain_top_edges",
    "plot_glass_brain_diagnostics",
    "plot_network_enrichment",
    "NeuroExplainabilitySuite",
]
