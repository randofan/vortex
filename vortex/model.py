import timm
import torch.nn as nn
import torch
from peft import get_peft_model, LoraConfig, TaskType
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss

NUM_CLASSES = 300  # years 1600-1899


class VortexModel(nn.Module):
    def __init__(self, lora_r: int = 8, dropout: float = 0.0):
        super().__init__()
        vit = timm.create_model("dinov2_base", pretrained=True, drop_rate=dropout)
        emb = vit.get_classifier().in_features
        vit.reset_classifier(0)
        coral = CoralLayer(emb, NUM_CLASSES - 1)
        backbone = nn.Sequential(vit, coral)

        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["qkv", "proj"],
        )
        self.model = get_peft_model(backbone, peft_cfg)

        for p in self.model.base_model.parameters():  # freeze ViT weights
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)

    # helpers
    @staticmethod
    def coral_loss_fn(logits, y):
        return coral_loss(logits, y)

    @staticmethod
    def decode_coral(logits):
        """Convert CORAL logits to integer year indices."""
        return (torch.sigmoid(logits) > 0.5).sum(1)
