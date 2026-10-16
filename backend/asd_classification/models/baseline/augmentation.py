"""Volume-Consistent Data Augmentation for Multi-View fMRI Volumetric Tensors.

Implements affine data augmentation that samples transformation parameters once per subject
volume and applies them identically across all 150 depth slices (axial, coronal, sagittal).
This ensures geometric consistency across views and preserves cross-plane structural
alignments in accordance with Hammash & Younis (MDPI Journal of Imaging, 2026, 12, 109).
"""

import math
import random
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VolumeConsistentAugmenter(nn.Module):
    """PyTorch module applying volume-consistent data augmentation.

    Transforms include:
    1. In-plane rotation within [-max_rotation_angle, +max_rotation_angle] degrees.
    2. Random horizontal flip (reflection along Width axis) with probability p.
    3. Random vertical flip (reflection along Height axis) with probability p.
    4. Random intensity scaling factor in [scale_min, scale_max].

    Transformation parameters are sampled once per subject volume and applied identically
    across all 150 depth slices during training. In evaluation mode (training=False),
    the module acts as an identity function.
    """

    def __init__(
        self,
        max_rotation_angle: float = 10.0,
        p_flip_horizontal: float = 0.5,
        p_flip_vertical: float = 0.5,
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
    ) -> None:
        """Initialize the volume-consistent augmenter.

        Args:
            max_rotation_angle: Maximum rotation deviation in degrees (+/- theta) (default: 10.0).
            p_flip_horizontal: Probability of horizontal flip (default: 0.5).
            p_flip_vertical: Probability of vertical flip (default: 0.5).
            intensity_scale_range: Lower and upper bounds for intensity scaling (default: (0.9, 1.1)).
        """
        super().__init__()
        self.max_rotation_angle = float(max_rotation_angle)
        self.p_flip_horizontal = float(p_flip_horizontal)
        self.p_flip_vertical = float(p_flip_vertical)
        self.intensity_scale_range = (
            float(intensity_scale_range[0]),
            float(intensity_scale_range[1]),
        )

    def sample_parameters(self) -> Dict[str, Any]:
        """Sample augmentation parameters once for a single volumetric scan.

        Returns:
            Dictionary containing rotation_angle, flip_horizontal, flip_vertical,
            and intensity_scale parameters.
        """
        angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        flip_h = random.random() < self.p_flip_horizontal
        flip_v = random.random() < self.p_flip_vertical
        scale = random.uniform(self.intensity_scale_range[0], self.intensity_scale_range[1])

        return {
            "rotation_angle": angle,
            "flip_horizontal": flip_h,
            "flip_vertical": flip_v,
            "intensity_scale": scale,
        }

    def _rotate_in_plane(self, volume: torch.Tensor, angle_degrees: float) -> torch.Tensor:
        """Apply 2D in-plane rotation identically across all depth slices.

        Args:
            volume: 4D tensor of shape (C, D, H, W).
            angle_degrees: Counter-clockwise rotation angle in degrees.

        Returns:
            Rotated 4D tensor of shape (C, D, H, W).
        """
        if abs(angle_degrees) < 1e-4:
            return volume

        c, d, h, w = volume.shape
        # PyTorch affine_grid operates on (N, C, H, W) where N is the batch dimension.
        # We reshape (C, D, H, W) -> (C * D, 1, H, W) to transform all slices in parallel.
        slices = volume.view(c * d, 1, h, w)

        # Build 2x3 affine matrix for 2D rotation.
        # In PyTorch, grid_sample maps output coordinates back to input coordinates.
        # Rotating coordinates by -theta rotates the sampled image by +theta.
        rad = math.radians(angle_degrees)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        theta = torch.tensor(
            [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]],
            dtype=slices.dtype,
            device=slices.device,
        ).unsqueeze(0).repeat(c * d, 1, 1)

        grid = F.affine_grid(theta, list(slices.size()), align_corners=False)
        rotated_slices = F.grid_sample(
            slices,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return rotated_slices.view(c, d, h, w)

    def _apply_to_volume(
        self,
        volume: torch.Tensor,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply sampled augmentation parameters consistently across a single 4D volume.

        Args:
            volume: 4D tensor of shape (C, D, H, W).
            params: Optional pre-sampled parameter dictionary for deterministic execution.

        Returns:
            Augmented 4D tensor of shape (C, D, H, W).
        """
        if params is None:
            params = self.sample_parameters()

        out = volume

        # 1. In-plane horizontal reflection (dim -1)
        if params.get("flip_horizontal", False):
            out = torch.flip(out, dims=[-1])

        # 2. In-plane vertical reflection (dim -2)
        if params.get("flip_vertical", False):
            out = torch.flip(out, dims=[-2])

        # 3. In-plane rotation applied identically to all D slices
        rotation_angle = params.get("rotation_angle", 0.0)
        out = self._rotate_in_plane(out, rotation_angle)

        # 4. Volumetric intensity scaling
        intensity_scale = params.get("intensity_scale", 1.0)
        if abs(intensity_scale - 1.0) > 1e-6:
            out = out * intensity_scale

        return out

    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        params: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Apply volume-consistent augmentation.

        Bypasses augmentation and returns input unmodified if training mode is False
        and no explicit override parameters are provided.

        Args:
            x: Input volumetric data with shape (C, D, H, W), (B, C, D, H, W),
                or unchanneled (D, H, W).
            params: Optional dictionary specifying exact transformation parameters.

        Returns:
            Augmented PyTorch tensor preserving input shape.
        """
        # In evaluation mode without explicit test override, bypass transformations
        if not self.training and params is None:
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x, dtype=torch.float32)

        is_numpy = isinstance(x, np.ndarray)
        tensor_x = torch.as_tensor(x, dtype=torch.float32)

        # Handle 3D unchanneled volume (D, H, W)
        if tensor_x.ndim == 3:
            vol_4d = tensor_x.unsqueeze(0)  # (1, D, H, W)
            augmented = self._apply_to_volume(vol_4d, params=params)
            res = augmented.squeeze(0)
            return res.cpu().numpy() if is_numpy else res

        # Handle 4D single-sample volume (C, D, H, W)
        elif tensor_x.ndim == 4:
            augmented = self._apply_to_volume(tensor_x, params=params)
            return augmented.cpu().numpy() if is_numpy else augmented

        # Handle 5D batched volumes (B, C, D, H, W)
        elif tensor_x.ndim == 5:
            b_dim = tensor_x.size(0)
            augmented_vols = []
            for b in range(b_dim):
                # Sample independent parameters per subject in batch unless explicit params passed
                batch_params = params if params is not None else self.sample_parameters()
                aug_b = self._apply_to_volume(tensor_x[b], params=batch_params)
                augmented_vols.append(aug_b)
            stacked = torch.stack(augmented_vols, dim=0)
            return stacked.cpu().numpy() if is_numpy else stacked

        else:
            raise ValueError(
                f"Expected 3D, 4D, or 5D input tensor, but received tensor with shape {tensor_x.shape}"
            )
