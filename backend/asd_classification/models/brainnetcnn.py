"""
BrainNetCNN Architecture for Functional Connectivity Connectome Processing.

Based on Kawahara et al. (NeuroImage 2017):
"BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"

The network leverages specialized convolutional filters preserving graph topology:
1. Edge-to-Edge (E2E) cross convolutions: combines horizontal and vertical edge signals.
2. Edge-to-Node (E2N) convolutions: pools edge features connected to each node into node features.
3. Node-to-Graph (N2G) convolutions: aggregates node features into a graph-level embedding.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EBlock(nn.Module):
    """
    Edge-to-Edge (E2E) convolutional layer.

    Applies cross convolutions combining horizontal (1 x N) and vertical (N x 1) 1D filters
    to capture edge dependencies across shared brain regions while maintaining spatial dimensionality:
    (B, C_in, N, N) -> (B, C_out, N, N).
    """

    def __init__(
        self,
        num_rois: int = 200,
        in_channels: int = 1,
        out_channels: int = 32,
        negative_slope: float = 0.33,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_rois = num_rois
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Horizontal 1D convolution combining edges per row
        self.conv_h = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, num_rois),
            bias=True,
        )

        # Vertical 1D convolution combining edges per column
        self.conv_v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(num_rois, 1),
            bias=True,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for E2E block.

        Args:
            x: Connectome tensor of shape (B, C_in, N, N).

        Returns:
            Tensor of shape (B, C_out, N, N).
        """
        # (B, C_out, N, 1) + (B, C_out, 1, N) broadcast to (B, C_out, N, N)
        h = self.conv_h(x)
        v = self.conv_v(x)
        out = h + v
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class E2NBlock(nn.Module):
    """
    Edge-to-Node (E2N) convolutional layer.

    Pools edge features for each node across all incident connections via a (1 x N) convolution,
    mapping connectome matrix representation to node-level vector representations:
    (B, C_in, N, N) -> (B, C_out, N).
    """

    def __init__(
        self,
        num_rois: int = 200,
        in_channels: int = 32,
        out_channels: int = 64,
        negative_slope: float = 0.33,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_rois = num_rois
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolve across all columns for each row (incident edges)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, num_rois),
            bias=True,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for E2N block.

        Args:
            x: Edge feature tensor of shape (B, C_in, N, N).

        Returns:
            Node feature tensor of shape (B, C_out, N).
        """
        # conv yields (B, C_out, N, 1), squeeze dim -1 -> (B, C_out, N)
        out = self.conv(x).squeeze(-1)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class N2GBlock(nn.Module):
    """
    Node-to-Graph (N2G) convolutional layer.

    Convolves node-level features across all N regions into a whole-brain graph summary:
    (B, C_in, N) -> (B, C_out, 1).
    """

    def __init__(
        self,
        num_rois: int = 200,
        in_channels: int = 64,
        out_channels: int = 128,
        negative_slope: float = 0.33,
    ) -> None:
        super().__init__()
        self.num_rois = num_rois
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolve across all nodes (kernel_size = num_rois)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=num_rois,
            bias=True,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for N2G block.

        Args:
            x: Node feature tensor of shape (B, C_in, N).

        Returns:
            Graph summary tensor of shape (B, C_out, 1).
        """
        out = self.conv(x)
        # Avoid BatchNorm1d failure when DataParallel produces single-sample slices per GPU
        if self.training and out.size(0) == 1:
            out = self.activation(out)
        else:
            out = self.bn(out)
            out = self.activation(out)
        return out


class BrainNetCNN(nn.Module):
    """
    BrainNetCNN Connectome Encoder for resting-state functional connectivity matrices.

    Composes E2EBlock, E2NBlock, and N2GBlock to project (B, 1, 200, 200) FC matrices
    into latent representations z_brain in R^128 (L2 normalized) and diagnostic class logits.
    """

    def __init__(
        self,
        num_rois: int = 200,
        in_channels: int = 1,
        e2e_channels: int = 32,
        e2n_channels: int = 64,
        n2g_channels: int = 128,
        embed_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_rois = num_rois
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Connectome convolutional blocks
        self.e2e = E2EBlock(
            num_rois=num_rois,
            in_channels=in_channels,
            out_channels=e2e_channels,
            dropout=dropout,
        )
        self.e2n = E2NBlock(
            num_rois=num_rois,
            in_channels=e2e_channels,
            out_channels=e2n_channels,
            dropout=dropout,
        )
        self.n2g = N2GBlock(
            num_rois=num_rois,
            in_channels=e2n_channels,
            out_channels=n2g_channels,
        )

        # Brain projection head mapping graph summary to unified latent space (L2 normalized)
        self.proj = nn.Linear(n2g_channels, embed_dim)

        # Diagnostic classification head
        self.classifier = nn.Sequential(
            nn.Linear(n2g_channels, 64),
            nn.LeakyReLU(negative_slope=0.33, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through BrainNetCNN.

        Args:
            x: Functional connectivity tensor of shape (B, 1, N, N) or (B, N, N).
            return_embedding: If True, returns both logits and normalized latent embeddings.

        Returns:
            If return_embedding is False:
                logits: Tensor of shape (B, num_classes).
            If return_embedding is True:
                Tuple of (logits, z_brain) where z_brain is of shape (B, embed_dim).
        """
        if x.dim() == 3:
            # Add channel dimension: (B, N, N) -> (B, 1, N, N)
            x = x.unsqueeze(1)

        e2e_out = self.e2e(x)
        e2n_out = self.e2n(e2e_out)
        n2g_out = self.n2g(e2n_out)

        graph_feat = n2g_out.squeeze(-1)
        logits = self.classifier(graph_feat)

        if return_embedding:
            z_brain = self.proj(graph_feat)
            z_brain = F.normalize(z_brain, p=2, dim=-1)
            return logits, z_brain

        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized brain latent embedding vector z_brain in R^embed_dim.

        Args:
            x: Input connectome tensor (B, 1, N, N) or (B, N, N).

        Returns:
            Normalized tensor of shape (B, embed_dim).
        """
        _, z_brain = self.forward(x, return_embedding=True)
        return z_brain
