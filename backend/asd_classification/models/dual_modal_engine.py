"""
Dual-Modal BrainNetCNN and InfoNCE Optimization Engine with Multi-GPU Support.

Coordinates joint multi-task training aligning Craddock CC200 Riemannian Tangent FC matrices
with clinical phenotypic text profiles under a multi-GPU 2x Tesla T4 configuration.
Enforces strict unimodal inference at test time, requiring only fMRI matrices without text input.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from asd_classification.models.base import BaseBrainModel
from asd_classification.models.brainnetcnn import BrainNetCNN
from asd_classification.models.losses import DualModalLoss
from asd_classification.models.text_encoder import PhenotypicTextEncoder


class DualModalBrainNetModel(BaseBrainModel):
    """
    Joint multi-task deep learning engine combining BrainNetCNN connectome encoder
    with PhenotypicTextEncoder via symmetric InfoNCE and label-smoothed Cross-Entropy.

    Features:
    - Multi-GPU parallelization via torch.nn.DataParallel (e.g. Kaggle 2x Tesla T4).
    - Automatic mixed-precision FP16 training via torch.cuda.amp.
    - Decoupled AdamW optimization with CosineAnnealingLR scheduling.
    - Strict unimodal inference at test time (evaluates only the fMRI FC matrix).
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
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_mock_text_backbone: bool = False,
        freeze_text_early_layers: bool = True,
        lambda_align: float = 0.3,
        label_smoothing: float = 0.1,
        temperature: float = 0.07,
        lr_brain: float = 1e-3,
        lr_text: float = 1e-5,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        use_amp: bool = True,
        multi_gpu: bool = True,
    ) -> None:
        self.num_rois = num_rois
        self.in_channels = in_channels
        self.e2e_channels = e2e_channels
        self.e2n_channels = e2n_channels
        self.n2g_channels = n2g_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.text_model_name = text_model_name
        self.use_mock_text_backbone = use_mock_text_backbone
        self.freeze_text_early_layers = freeze_text_early_layers
        self.lambda_align = lambda_align
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.lr_brain = lr_brain
        self.lr_text = lr_text
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.history_: Dict[str, List[float]] = {}

        # Device determination
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Multi-GPU support check (e.g. 2x Tesla T4)
        self.num_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        self.multi_gpu_enabled = multi_gpu and (self.num_gpus > 1)
        self.use_amp = use_amp and (self.device.type == "cuda")

        # GradScaler setup using modern torch.amp API with backward fallback
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Instantiate sub-networks
        self.brain_net = BrainNetCNN(
            num_rois=num_rois,
            in_channels=in_channels,
            e2e_channels=e2e_channels,
            e2n_channels=e2n_channels,
            n2g_channels=n2g_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)

        self.text_encoder = PhenotypicTextEncoder(
            pretrained_model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_early_layers=freeze_text_early_layers,
            use_mock_backbone=use_mock_text_backbone,
        ).to(self.device)

        self.criterion = DualModalLoss(
            lambda_weight=lambda_align,
            label_smoothing=label_smoothing,
            temperature=temperature,
            learnable_temp=True,
        ).to(self.device)

        # Multi-GPU DataParallel wrapping
        if self.multi_gpu_enabled:
            print(f"INFO: Distributing model execution across {self.num_gpus} CUDA devices using DataParallel.")
            self.brain_net = nn.DataParallel(self.brain_net)  # type: ignore[assignment]
            self.text_encoder = nn.DataParallel(self.text_encoder)  # type: ignore[assignment]

    def _autocast_context(self):
        """Context manager for automatic mixed precision compatible across PyTorch versions."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def _get_raw_brain_net(self) -> BrainNetCNN:
        """Helper to retrieve raw underlying BrainNetCNN instance when wrapped in DataParallel."""
        if isinstance(self.brain_net, nn.DataParallel):
            return self.brain_net.module
        return self.brain_net

    def _get_raw_text_encoder(self) -> PhenotypicTextEncoder:
        """Helper to retrieve raw underlying PhenotypicTextEncoder instance when wrapped in DataParallel."""
        if isinstance(self.text_encoder, nn.DataParallel):
            return self.text_encoder.module
        return self.text_encoder

    def __getitem__(self, key: str) -> Any:
        """Allow subscript access to training history metrics."""
        return self.history_[key]

    def __contains__(self, key: str) -> bool:
        """Allow membership checks on training history metrics."""
        return key in self.history_

    def fit(
        self,
        X: Any,
        y: Optional[np.ndarray] = None,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        **kwargs: Any,
    ) -> "DualModalBrainNetModel":
        """
        Train the dual-modal architecture using AdamW and Cosine Annealing learning rate schedule.

        Args:
            X: PyTorch DataLoader yielding (fc_tensor, prompt_strings, labels, site_codes),
               or FC matrices when paired with y and phenotypic prompts in kwargs.
            y: Optional ground truth labels (when X is a raw tensor or array).
            val_loader: Optional validation DataLoader for tracking generalization loss.
            epochs: Total epochs (defaults to self.epochs).

        Returns:
            Fitted DualModalBrainNetModel instance (with training history accessible via self.history_).
        """
        if isinstance(X, DataLoader):
            train_loader = X
        else:
            raise ValueError(
                "DualModalBrainNetModel expects a DataLoader yielding "
                "(fc_tensor, prompts, labels, site_codes) for dual-modal alignment training."
            )

        total_epochs = epochs or self.epochs

        # Separate parameter groups with distinct learning rates
        raw_brain = self._get_raw_brain_net()
        raw_text = self._get_raw_text_encoder()

        brain_params = [p for p in raw_brain.parameters() if p.requires_grad]
        text_params = [p for p in raw_text.parameters() if p.requires_grad]
        loss_params = [p for p in self.criterion.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": brain_params, "lr": self.lr_brain, "weight_decay": self.weight_decay},
                {"params": text_params, "lr": self.lr_text, "weight_decay": self.weight_decay},
                {"params": loss_params, "lr": self.lr_brain, "weight_decay": 0.0},
            ]
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6
        )

        scaler = self.scaler

        history: Dict[str, List[float]] = {
            "loss": [],
            "loss_ce": [],
            "loss_infonce": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(total_epochs):
            self.brain_net.train()
            self.text_encoder.train()
            self.criterion.train()

            epoch_loss = 0.0
            epoch_loss_ce = 0.0
            epoch_loss_infonce = 0.0
            batch_count = 0

            for batch in train_loader:
                fc_tensor, prompts, labels, _ = batch

                fc_tensor = fc_tensor.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                prompt_inputs = raw_text.prepare_inputs(prompts, device=self.device)

                optimizer.zero_grad()

                with self._autocast_context():
                    logits, z_brain = self.brain_net(fc_tensor, return_embedding=True)
                    z_text = self.text_encoder(prompt_inputs)
                    loss_dict = self.criterion(logits, labels, z_brain, z_text)
                    loss = loss_dict["loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_loss_ce += loss_dict["loss_ce"].item()
                epoch_loss_infonce += loss_dict["loss_infonce"].item()
                batch_count += 1

            scheduler.step()

            avg_loss = epoch_loss / max(1, batch_count)
            avg_loss_ce = epoch_loss_ce / max(1, batch_count)
            avg_loss_infonce = epoch_loss_infonce / max(1, batch_count)

            history["loss"].append(avg_loss)
            history["loss_ce"].append(avg_loss_ce)
            history["loss_infonce"].append(avg_loss_infonce)

            # Validation evaluation if loader is supplied
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_unimodal(val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

        self.history_ = history
        return self

    def _evaluate_unimodal(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate unimodal classification accuracy and cross-entropy loss on validation data."""
        self.brain_net.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                fc_tensor = batch[0].to(self.device)
                labels = batch[2].to(self.device)

                with self._autocast_context():
                    logits = self.brain_net(fc_tensor, return_embedding=False)
                    loss = F.cross_entropy(logits, labels)

                total_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        return total_loss / max(1, total), correct / max(1, total)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict diagnostic probability distributions using strict unimodal fMRI inference.

        Security & Operational Invariant:
            Does not ingest clinical text profiles during evaluation or deployment.
            Classifies strictly from the functional connectivity matrix.

        Args:
            X: Functional connectivity tensor or array of shape (N, 1, 200, 200) or (N, 200, 200).

        Returns:
            NumPy array of shape (N, 2) containing class probabilities summing to 1.0.
        """
        self.brain_net.eval()

        if isinstance(X, np.ndarray):
            x_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            x_tensor = X.float()
        else:
            x_tensor = torch.tensor(X, dtype=torch.float32)

        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
        elif x_tensor.dim() == 3:
            x_tensor = x_tensor.unsqueeze(1)

        x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            with self._autocast_context():
                logits = self.brain_net(x_tensor, return_embedding=False)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        return probs

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict binary diagnostic class labels (0 = TD, 1 = ASD) via unimodal fMRI inference.

        Args:
            X: Functional connectivity tensor or array.

        Returns:
            NumPy integer array of shape (N,) containing class predictions.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1).astype(np.int64)

    def evaluate(self, X: Any, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate unimodal classification accuracy and cross-entropy loss strictly from fMRI matrices.

        Security & Operational Invariant:
            Does not require or accept text inputs, ensuring diagnostic evaluation
            runs strictly on the rs-fMRI connectome.

        Args:
            X: Functional connectivity tensor or array of shape (N, 1, 200, 200) or (N, 200, 200).
            y: Ground truth binary class labels of shape (N,).

        Returns:
            Dictionary containing 'loss' and 'accuracy'.
        """
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=-1)
        y_arr = np.asarray(y, dtype=np.int64)

        acc = float(np.mean(preds == y_arr))

        # Numerical log cross-entropy loss
        eps = 1e-12
        probs_clamped = np.clip(probs, eps, 1.0 - eps)
        log_p = np.log(probs_clamped[np.arange(len(y_arr)), y_arr])
        loss = float(-np.mean(log_p))

        return {"loss": loss, "accuracy": acc}

    def save(self, file_path: Path) -> None:
        """
        Persist model weights, configuration parameters, and loss state to disk.

        Args:
            file_path: Destination path for the checkpoint.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": {
                "num_rois": self.num_rois,
                "in_channels": self.in_channels,
                "e2e_channels": self.e2e_channels,
                "e2n_channels": self.e2n_channels,
                "n2g_channels": self.n2g_channels,
                "embed_dim": self.embed_dim,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "text_model_name": self.text_model_name,
                "use_mock_text_backbone": self.use_mock_text_backbone,
                "freeze_text_early_layers": self.freeze_text_early_layers,
                "lambda_align": self.lambda_align,
                "label_smoothing": self.label_smoothing,
                "temperature": self.temperature,
                "lr_brain": self.lr_brain,
                "lr_text": self.lr_text,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
            },
            "brain_net_state_dict": self._get_raw_brain_net().state_dict(),
            "text_encoder_state_dict": self._get_raw_text_encoder().state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, file_path: Path) -> "DualModalBrainNetModel":
        """
        Restore model weights and parameters from disk checkpoint.

        Args:
            file_path: Source path to saved checkpoint.

        Returns:
            Restored DualModalBrainNetModel instance.
        """
        checkpoint = torch.load(file_path, map_location=self.device)

        raw_brain = self._get_raw_brain_net()
        raw_text = self._get_raw_text_encoder()

        raw_brain.load_state_dict(checkpoint["brain_net_state_dict"])
        raw_text.load_state_dict(checkpoint["text_encoder_state_dict"])
        self.criterion.load_state_dict(checkpoint["criterion_state_dict"])

        return self
