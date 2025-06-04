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
from coral_pytorch.dataset import levels_from_labelbatch
from utils import MODEL_NAME, NUM_CLASSES, calculate_mae


def _get_model_embedding_size(model_name: str) -> int:
    """
    Get the embedding size for the specified model.

    Args:
        model_name: Name of the DINOv2 model

    Returns:
        int: Hidden size of the model
    """
    # Model-specific embedding sizes
    embedding_sizes = {
        "facebook/dinov2-small": 384,
        "facebook/dinov2-base": 768,
        "facebook/dinov2-large": 1024,
        "facebook/dinov2-giant": 1536,
    }

    if model_name not in embedding_sizes:
        raise ValueError(f"Unknown model: {model_name}")

    return embedding_sizes[model_name]


class VortexModel(nn.Module):
    """
    Vision Transformer with LoRA adapters for painting year prediction.

    This model uses a frozen DINOv2 backbone with trainable LoRA adapters
    and a CORAL layer for ordinal regression. The architecture is designed to
    predict painting creation years while being parameter-efficient.

    Args:
        lora_r: LoRA rank parameter (higher = more parameters, default: 8)
        lora_alpha: LoRA scaling parameter (default: 16, common range: rank to 4×rank)
        dropout: Dropout rate for the base model (default: 0.0)
    """

    def __init__(self, lora_r: int = 8, lora_alpha: int = 16, dropout: float = 0.0):
        super().__init__()

        # Validate parameters
        if lora_r <= 0:
            raise ValueError("LoRA rank must be positive")
        if lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")

        # Load pre-trained DINOv2 model
        vit = AutoModel.from_pretrained(MODEL_NAME)
        emb = _get_model_embedding_size(MODEL_NAME)

        # CORAL layer for ordinal regression (NUM_CLASSES-1 thresholds)
        self.coral = CoralLayer(emb, NUM_CLASSES - 1)

        # Configure LoRA adapters for parameter-efficient fine-tuning
        peft_cfg = LoraConfig(
            # task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,  # Now configurable scaling factor
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
        # convert labels into 'levels' where it's (B, NUM_CLASSES-1) as well
        levels = levels_from_labelbatch(y, NUM_CLASSES-1)
        return coral_loss(logits, levels)

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

# standalone function
def _step(vortex_model: VortexModel, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a single training/validation step.

    Args:
        model: The VortexModel instance
        batch: Tuple of (images, year_labels)

    Returns:
        Tuple of (loss, mae) where:
        - loss: CORAL loss for backpropagation
        - mae: Mean Absolute Error for monitoring
    """
    x, y = batch
    # print(x)
    logits = vortex_model(x)
    loss = vortex_model.coral_loss_fn(logits, y)

    # Calculate MAE using shared utility function
    predictions = vortex_model.decode_coral(logits)
    mae = calculate_mae(predictions, y).mean()

    return loss, mae