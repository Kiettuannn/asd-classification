"""
Grad-CAM Explainer for Functional Connectivity BrainNetCNN Models.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) hooked on the
Edge-to-Node (E2N) layer of BrainNetCNN. Computes channel importance weights (alpha_k)
via global average pooling over node dimensions and generates non-negative ROI
saliency attributions: S_i = ReLU(sum_k alpha_k A_{k, i}).
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from asd_classification.models.brainnetcnn import BrainNetCNN


class BrainNetGradCAM:
    """
    Grad-CAM explainer hooked into the E2N convolutional block of BrainNetCNN.

    Attributes:
        model: Underlying BrainNetCNN model (or DualModalBrainNetModel).
        target_layer_name: Name of target layer (defaults to 'e2n').
    """

    def __init__(
        self,
        model: Any,
        target_layer: str = "e2n",
    ) -> None:
        """
        Initialize Grad-CAM hooks on the target convolutional layer.

        Args:
            model: Model instance (BrainNetCNN or DualModalBrainNetModel).
            target_layer: Name of convolutional module (default 'e2n').

        Raises:
            AttributeError: If target_layer is not found on the model.
        """
        self.raw_model: nn.Module
        if hasattr(model, "_get_raw_brain_net"):
            self.raw_model = model._get_raw_brain_net()
        elif hasattr(model, "brain_net"):
            self.raw_model = model.brain_net
        else:
            self.raw_model = model

        self.target_layer_name = target_layer
        self.target_module = getattr(self.raw_model, target_layer, None)
        if self.target_module is None:
            raise AttributeError(f"Target layer '{target_layer}' not found on model.")

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        if self.target_module is None:
            raise AttributeError("Target module is not initialized.")

        def forward_hook(module: nn.Module, input_t: Any, output_t: torch.Tensor) -> None:
            self._activations = output_t
            if output_t.requires_grad:
                output_t.register_hook(self._save_gradient)

        handle = self.target_module.register_forward_hook(forward_hook)
        self._hook_handles.append(handle)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Save gradient of target class score with respect to activation map."""
        self._gradients = grad

    def remove_hooks(self) -> None:
        """Remove all registered PyTorch hooks to avoid memory leaks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __enter__(self) -> "BrainNetGradCAM":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def attribute(
        self,
        x: Union[np.ndarray, torch.Tensor],
        target_class: Optional[Union[int, torch.Tensor]] = 1,
    ) -> np.ndarray:
        """
        Compute Grad-CAM ROI attribution weights across CC200 regions.

        Formula:
            alpha_k = (1 / N) * sum_i (d y^c / d A_{k, i})
            S_i = ReLU(sum_k alpha_k * A_{k, i})

        Args:
            x: Input FC matrix of shape (200, 200), (1, 200, 200), (1, 1, 200, 200),
               or batch of shape (B, 200, 200) / (B, 1, 200, 200).
            target_class: Target diagnostic class index (e.g. 1 = ASD, 0 = TD).
                          If None, uses predicted class argmax(logits).

        Returns:
            Numpy array of non-negative saliency weights:
            Shape (200,) for single input, or (B, 200) for batch inputs.

        Raises:
            ValueError: If input tensor does not match expected connectome dimensions.
            RuntimeError: If hook activation or gradient capture fails during backpropagation.
        """
        device = next(self.raw_model.parameters()).device

        # Convert input to tensor
        is_single_sample = False
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.clone().detach().float()

        if x_tensor.dim() == 2:
            is_single_sample = True
            x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
        elif x_tensor.dim() == 3:
            if x_tensor.size(0) == 1 and x_tensor.size(1) == 200 and x_tensor.size(2) == 200:
                is_single_sample = True
            x_tensor = x_tensor.unsqueeze(1)
        elif x_tensor.dim() == 4:
            if x_tensor.size(0) == 1:
                is_single_sample = True
        else:
            raise ValueError(f"Unsupported FC tensor dimensions: {x_tensor.shape}.")

        x_tensor = x_tensor.to(device)
        x_tensor.requires_grad_(True)

        was_training = self.raw_model.training
        self.raw_model.eval()
        self.raw_model.zero_grad(set_to_none=True)

        # Forward pass
        forward_out = self.raw_model(x_tensor)
        logits = forward_out[0] if isinstance(forward_out, tuple) else forward_out
        batch_size = logits.size(0)

        # Determine target class score
        if target_class is None:
            class_indices = logits.argmax(dim=-1)
        elif isinstance(target_class, int):
            class_indices = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        else:
            class_indices = target_class.to(device)

        # Gather target scores
        target_score = logits.gather(1, class_indices.view(-1, 1)).squeeze(-1)

        # Backward pass to capture gradients at E2N
        self._gradients = None
        target_score.sum().backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM failed to capture activations or gradients. "
                "Ensure target layer 'e2n' is active and receives backpropagation."
            )

        activations = self._activations.detach()
        gradients = self._gradients.detach()

        # Global average pooling over nodes (N dimension) -> (B, C, 1)
        alpha = torch.mean(gradients, dim=-1, keepdim=True)

        # Weighted combination across channels: (B, C, 1) * (B, C, N) -> sum over C -> (B, N)
        saliency_tensor = torch.sum(alpha * activations, dim=1)

        # ReLU activation ensuring strictly non-negative attributions
        saliency_tensor = F.relu(saliency_tensor)

        # Move to CPU numpy array
        saliency_np = saliency_tensor.cpu().numpy().astype(np.float32)

        # Restore previous training state
        self.raw_model.train(was_training)

        if is_single_sample:
            return saliency_np[0]
        return saliency_np
