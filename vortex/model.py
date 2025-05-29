"""
VortexModel: Fine-tuned Vision Transformer for painting year prediction.

This module implements a DINOv2-based model with LoRA adapters and CORAL loss
for ordinal regression on painting creation years (1600-1899).
"""

from transformers import AutoModel
import torch.nn as nn
import torch
from peft import get_peft_model, LoraConfig, TaskType
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from .utils import MODEL_NAME, NUM_CLASSES, get_model_embedding_size


class VortexModel(nn.Module):
    """
    Vision Transformer with LoRA adapters for painting year prediction.

    This model uses a frozen DINOv2 backbone with trainable LoRA adapters
    and a CORAL layer for ordinal regression. The architecture is designed to
    predict painting creation years while being parameter-efficient.

    Args:
        lora_r: LoRA rank parameter (higher = more parameters, default: 8)
        dropout: Dropout rate for the base model (default: 0.0)
    """

    def __init__(self, lora_r: int = 8, dropout: float = 0.0):
        super().__init__()

        # Validate parameters
        if lora_r <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")

        # Load pre-trained DINOv2 model
        vit = AutoModel.from_pretrained(MODEL_NAME)
        emb = get_model_embedding_size()

        # CORAL layer for ordinal regression (NUM_CLASSES-1 thresholds)
        self.coral = CoralLayer(emb, NUM_CLASSES - 1)

        # Configure LoRA adapters for parameter-efficient fine-tuning
        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_r * 2,  # Scaling factor (common heuristic)
            target_modules=["query", "key", "value", "dense"],  # Attention modules
        )
        self.vit = get_peft_model(vit, peft_cfg)

        # Freeze base ViT weights, only train LoRA adapters and CORAL layer
        for p in self.vit.base_model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input images as tensor (B, 3, 224, 224)

        Returns:
            CORAL logits for ordinal regression (B, NUM_CLASSES-1)
        """
        # Extract features using ViT encoder
        outputs = self.vit(pixel_values=x)
        feats = outputs.last_hidden_state[:, 0]  # CLS token embeddings

        # Apply CORAL layer for ordinal regression
        coral_outputs = self.coral(feats)
        return coral_outputs

    def get_attention_map(self, x: torch.Tensor, return_all: bool = False):
        """
        Extract attention weights for visualization.

        Args:
            x: Input images as tensor (B, 3, 224, 224)
            return_all: If True, return both features and attention weights

        Returns:
            If return_all=True: (last_hidden_state, attention_weights)
            Otherwise: last_hidden_state only
        """
        outputs = self.vit(pixel_values=x, output_attentions=True)
        if return_all:
            return outputs.last_hidden_state, outputs.attentions
        return outputs.last_hidden_state

    @staticmethod
    def coral_loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute CORAL loss for ordinal regression.

        Args:
            logits: Model predictions (B, NUM_CLASSES-1)
            y: True year offsets (B,)

        Returns:
            CORAL loss scalar
        """
        return coral_loss(logits, y)

    @staticmethod
    def decode_coral(logits: torch.Tensor) -> torch.Tensor:
        """
        Decode CORAL logits to predicted year offsets.

        Args:
            logits: CORAL predictions (B, NUM_CLASSES-1)

        Returns:
            Predicted year offsets (B,)
        """
        return (torch.sigmoid(logits) > 0.5).sum(1)
