"""Baseline 3D-CNN architecture for multi-view fMRI volumetric classification.

Implements the 6-block 3D Convolutional Neural Network baseline described in
Hammash & Younis (MDPI Journal of Imaging, 2026, 12, 109) and ADR-0001 / ADR-0003.
The network processes depth-stacked multi-view functional MRI activation volumes
of shape (B, 1, 150, 224, 224) and outputs binary diagnostic class logits.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Baseline3DCNN", "SEBlock3D", "SEResBlock3D", "SEResNet3D"]


class Baseline3DCNN(nn.Module):
    """6-block 3D Convolutional Neural Network for volumetric neuroimaging classification.

    Progressively expands feature representations from 32 to 256 channels across
    six convolutional stages using 3x3x3 kernels, Batch Normalization, and LeakyReLU
    activations. Downsampling via MaxPool3d(2) is applied after blocks 1 through 4,
    preserving spatial resolution in blocks 5 and 6 before Global Average Pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.4,
        leaky_relu_slope: float = 0.01,
    ) -> None:
        """Initialize the Baseline 3D-CNN architecture.

        Args:
            in_channels: Number of input volumetric channels (default: 1 for depth-stacked multi-view).
            num_classes: Number of diagnostic target classes (default: 2 for TD vs. ASD).
            dropout: Dropout probability applied before final linear head (default: 0.4).
            leaky_relu_slope: Negative slope coefficient for LeakyReLU activations (default: 0.01).
        """
        super().__init__()
        self._channel_progression = [32, 64, 128, 128, 256, 256]
        self._dropout_rate = dropout
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: 1 -> 32 channels with spatial halving
        self.block1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 2: 32 -> 64 channels with spatial halving
        self.block2 = nn.Sequential(
            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 3: 64 -> 128 channels with spatial halving
        self.block3 = nn.Sequential(
            nn.Conv3d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 4: 128 -> 128 channels with spatial halving
        self.block4 = nn.Sequential(
            nn.Conv3d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 5: 128 -> 256 channels without pooling to retain spatial representation
        self.block5 = nn.Sequential(
            nn.Conv3d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
        )

        # Block 6: 256 -> 256 channels without pooling
        self.block6 = nn.Sequential(
            nn.Conv3d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
        )

        # Readout head: Global Average Pooling collapses 3D volume to 256-d vector
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

    @property
    def channel_progression(self) -> List[int]:
        """Return the sequence of output channels across convolutional blocks."""
        return list(self._channel_progression)

    @property
    def dropout_rate(self) -> float:
        """Return the configured dropout rate applied before classification."""
        return self._dropout_rate

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global average pooled 256-dimensional feature embeddings.

        Args:
            x: Input volumetric tensor of shape (B, in_channels, D, H, W).

        Returns:
            Normalized 2D feature tensor of shape (B, 256).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
        return self.flatten(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through convolutional blocks and classification head.

        Args:
            x: Volumetric tensor of shape (B, in_channels, D, H, W).

        Returns:
            Classification logits of shape (B, num_classes).
        """
        features = self.forward_features(x)
        dropped = self.dropout(features)
        return self.classifier(dropped)

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Predict discrete class labels for input volumes.

        Args:
            x: Volumetric tensor or numpy array.

        Returns:
            1D numpy array of predicted class indices.
        """
        self.eval()
        tensor_x = torch.as_tensor(x, dtype=torch.float32)
        if tensor_x.device != next(self.parameters()).device:
            tensor_x = tensor_x.to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.forward(tensor_x)
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu().numpy()

    def predict_proba(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Predict softmax class probability distribution.

        Args:
            x: Volumetric tensor or numpy array.

        Returns:
            2D numpy array of class probabilities of shape (B, num_classes).
        """
        self.eval()
        tensor_x = torch.as_tensor(x, dtype=torch.float32)
        if tensor_x.device != next(self.parameters()).device:
            tensor_x = tensor_x.to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.forward(tensor_x)
            probas = F.softmax(logits, dim=-1)
        return probas.cpu().numpy()

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist model weights and architectural configuration to disk.

        Args:
            file_path: Target filesystem path for model checkpoint (.pt).
        """
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "dropout": self._dropout_rate,
            "channel_progression": self._channel_progression,
        }
        torch.save(checkpoint, target)

    def load(self, file_path: Union[str, Path]) -> "Baseline3DCNN":
        """Restore model weights and state from disk.

        Args:
            file_path: Path to serialized model checkpoint (.pt).

        Returns:
            Restored self instance.
        """
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            self.load_state_dict(checkpoint)
        return self


class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation channel attention module.

    Performs volumetric global average pooling across spatial and depth dimensions
    (D, H, W) followed by two-layer bottleneck excitation gating to model
    inter-channel dependencies and selectively boost ASD-informative features.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize 3D Squeeze-and-Excitation block.

        Args:
            channels: Number of input and output feature channels.
            reduction: Channel dimension bottleneck compression factor (default: 16).
        """
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention scaling to 5D input tensor.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Channel-recalibrated tensor of shape (B, C, D, H, W).
        """
        b, c, _, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1, 1)
        return x * scale


class SEResBlock3D(nn.Module):
    """3D Residual Block with Squeeze-and-Excitation channel attention.

    Integrates two 3x3x3 convolutions with Batch Normalization, LeakyReLU,
    an SE channel attention mechanism, and a residual shortcut connection
    to guarantee smooth gradient propagation and prevent vanishing gradients.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
        leaky_relu_slope: float = 0.01,
    ) -> None:
        """Initialize 3D SE-ResNet block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            stride: Convolutional stride for downsampling (default: 1).
            reduction: SE bottleneck channel reduction ratio (default: 16).
            leaky_relu_slope: Negative slope coefficient for LeakyReLU.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels, reduction=reduction)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual path and SE attention.

        Args:
            x: Input tensor of shape (B, in_channels, D, H, W).

        Returns:
            Output tensor of shape (B, out_channels, D', H', W').
        """
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + residual)
        return out


class SEResNet3D(nn.Module):
    """Deep 3D Residual Network with Squeeze-and-Excitation Attention.

    Combines 3D residual connections across 4 hierarchical stages with SE
    channel attention at each block, addressing gradient vanishing and
    adaptively focusing on autism-relevant functional brain subvolumes.

    Channel progression:
    - Initial Stem: Conv3d(in_channels -> 32) + MaxPool3d(2)
    - Stage 1: 32 channels (stride 1)
    - Stage 2: 64 channels (downsampled stride 2)
    - Stage 3: 128 channels (downsampled stride 2)
    - Stage 4: 256 channels (downsampled stride 2)
    - Classification Head: Global Average Pooling + Dropout + Linear(256 -> num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.4,
        leaky_relu_slope: float = 0.01,
        reduction: int = 16,
    ) -> None:
        """Initialize SEResNet3D model.

        Args:
            in_channels: Number of input volumetric channels (default: 1).
            num_classes: Number of target diagnosis classes (default: 2).
            dropout: Dropout rate applied before linear classification head.
            leaky_relu_slope: Negative slope coefficient for LeakyReLU.
            reduction: Bottleneck ratio for Squeeze-and-Excitation attention.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._dropout_rate = dropout
        self.channel_progression = [32, 64, 128, 256]

        # Initial stem downsamples high-resolution inputs to reduce memory footprint
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Stage 1: 32 channels
        self.stage1 = SEResBlock3D(
            32,
            32,
            stride=1,
            reduction=reduction,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Stage 2: 32 -> 64 channels with spatial downsampling
        self.stage2 = SEResBlock3D(
            32,
            64,
            stride=2,
            reduction=reduction,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Stage 3: 64 -> 128 channels with spatial downsampling
        self.stage3 = SEResBlock3D(
            64,
            128,
            stride=2,
            reduction=reduction,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Stage 4: 128 -> 256 channels with spatial downsampling
        self.stage4 = SEResBlock3D(
            128,
            256,
            stride=2,
            reduction=reduction,
            leaky_relu_slope=leaky_relu_slope,
        )

        # Readout head: Global Average Pooling collapses 3D volume to 256-d vector
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

    @property
    def dropout_rate(self) -> float:
        """Return the configured dropout rate applied before classification."""
        return self._dropout_rate

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global average pooled 256-dimensional feature representations.

        Args:
            x: Input volumetric tensor of shape (B, in_channels, D, H, W).

        Returns:
            Normalized 2D feature tensor of shape (B, 256).
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.flatten(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through SEResNet3D stages and classification head.

        Args:
            x: Volumetric tensor of shape (B, in_channels, D, H, W).

        Returns:
            Classification logits of shape (B, num_classes).
        """
        features = self.forward_features(x)
        dropped = self.dropout(features)
        return self.classifier(dropped)

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Predict discrete class labels for input volumes."""
        self.eval()
        tensor_x = torch.as_tensor(x, dtype=torch.float32)
        if tensor_x.device != next(self.parameters()).device:
            tensor_x = tensor_x.to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.forward(tensor_x)
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu().numpy()

    def predict_proba(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Predict softmax class probability distribution."""
        self.eval()
        tensor_x = torch.as_tensor(x, dtype=torch.float32)
        if tensor_x.device != next(self.parameters()).device:
            tensor_x = tensor_x.to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.forward(tensor_x)
            probas = F.softmax(logits, dim=-1)
        return probas.cpu().numpy()

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist model weights and architectural configuration to disk."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "dropout": self._dropout_rate,
            "channel_progression": self.channel_progression,
        }
        torch.save(checkpoint, target)

    def load(self, file_path: Union[str, Path]) -> "SEResNet3D":
        """Restore model weights and state from disk."""
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            self.load_state_dict(checkpoint)
        return self

