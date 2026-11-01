"""
Lightweight Clinical Phenotypic Profile Text Encoder.

Wraps transformer backbones (such as sentence-transformers/all-MiniLM-L6-v2 or Bio_ClinicalBERT)
to project clinical profile text descriptions into a shared R^128 latent space with L2 normalization.
Supports layer freezing for transfer learning and a lightweight mock mode for offline verification.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class MockTransformerBackbone(nn.Module):
    """Deterministic lightweight backbone simulating transformer layers for fast unit testing."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.early_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.early_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.late_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.late_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.early_layer1(x))
        out = self.activation(self.early_layer2(out))
        out = self.activation(self.late_layer1(out))
        out = self.activation(self.late_layer2(out))
        return out


class PhenotypicTextEncoder(nn.Module):
    """
    Encodes clinical phenotypic prompt strings into L2-normalized vectors in R^embed_dim.

    By default, freezes early transformer layers while fine-tuning the final 2 layers
    and the linear projection head onto the unified d=128 latent space.
    """

    def __init__(
        self,
        pretrained_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_dim: int = 128,
        hidden_dim: int = 384,
        freeze_early_layers: bool = True,
        n_finetune_layers: int = 2,
        use_mock_backbone: bool = False,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.use_mock_backbone = use_mock_backbone or not HAS_TRANSFORMERS

        self.tokenizer: Any = None
        self.transformer: Any = None

        if self.use_mock_backbone:
            self.hidden_dim = hidden_dim
            # Simulate early (frozen) and late (trainable) transformer blocks
            self.backbone_early = self._build_dense_block(self.hidden_dim)
            self.backbone_late = self._build_dense_block(self.hidden_dim)

            if freeze_early_layers:
                for param in self.backbone_early.parameters():
                    param.requires_grad = False
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.transformer = AutoModel.from_pretrained(pretrained_model_name)
            self.hidden_dim = self.transformer.config.hidden_size

            if freeze_early_layers:
                self._freeze_early_layers(n_finetune_layers=n_finetune_layers)

        # Linear projection head mapping hidden_dim to unified latent space (128)
        self.proj = nn.Linear(self.hidden_dim, embed_dim)

    @staticmethod
    def _build_dense_block(dim: int) -> nn.Sequential:
        """Create a 2-layer MLP block for mock backbone representation."""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def _freeze_early_layers(self, n_finetune_layers: int = 2) -> None:
        """Freeze all initial transformer layers, keeping only the final layers trainable."""
        if self.transformer is None:
            return

        if hasattr(self.transformer, "embeddings"):
            for param in self.transformer.embeddings.parameters():
                param.requires_grad = False

        if hasattr(self.transformer, "encoder") and hasattr(self.transformer.encoder, "layer"):
            layers = self.transformer.encoder.layer
            total_layers = len(layers)
            freeze_until = max(0, total_layers - n_finetune_layers)
            for i in range(freeze_until):
                for param in layers[i].parameters():
                    param.requires_grad = False

    def _mock_tokenize_prompts(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        """Create deterministic pseudo-embeddings from text prompts for offline mock execution."""
        batch_size = len(prompts)
        embeddings = torch.zeros(batch_size, self.hidden_dim, device=device)
        for i, text in enumerate(prompts):
            # Generate deterministic seed from text content
            text_seed = sum(ord(c) * (idx + 1) for idx, c in enumerate(text[:50])) % 100000
            gen = torch.Generator().manual_seed(text_seed)
            embeddings[i] = torch.randn(self.hidden_dim, generator=gen)
        return embeddings

    def prepare_inputs(
        self,
        prompts: List[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and format text prompts into tensor representations suitable for DataParallel scatter.

        Args:
            prompts: List of clinical phenotypic profile strings.
            device: Target device for output tensors.

        Returns:
            Dictionary containing token or feature tensors partitioned along batch dimension.
        """
        dev = device or next(self.parameters()).device
        if self.use_mock_backbone:
            return {"mock_tokens": self._mock_tokenize_prompts(prompts, device=dev)}

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].to(dev),
            "attention_mask": tokenized["attention_mask"].to(dev),
        }

    def forward(
        self,
        inputs: Union[List[str], Dict[str, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass converting text prompts to L2-normalized latent embeddings.

        Args:
            inputs: Either a list of clinical prompt strings, a dictionary of tokenized tensors,
                   or a raw feature tensor.

        Returns:
            L2-normalized latent tensor of shape (B, embed_dim).
        """
        device = next(self.parameters()).device

        if isinstance(inputs, list):
            inputs = self.prepare_inputs(inputs, device=device)

        if self.use_mock_backbone:
            if isinstance(inputs, dict) and "mock_tokens" in inputs:
                feat = inputs["mock_tokens"].to(device)
            elif isinstance(inputs, torch.Tensor):
                feat = inputs.to(device)
            else:
                feat = torch.randn(len(inputs), self.hidden_dim, device=device)

            feat = self.backbone_early(feat)
            feat = self.backbone_late(feat)
        else:
            if self.transformer is None:
                raise RuntimeError("Transformer is not initialized.")

            if isinstance(inputs, dict):
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
            else:
                raise ValueError(f"Unsupported inputs type: {type(inputs)}")

            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state

            # Mean pooling weighted by attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            feat = sum_embeddings / sum_mask

        # Project and apply L2 normalization: ||z_text||_2 = 1.0
        projected = self.proj(feat)
        z_text = F.normalize(projected, p=2, dim=-1)
        return z_text
