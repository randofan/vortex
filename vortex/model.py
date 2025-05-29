import timm
import torch.nn as nn
import torch
from peft import get_peft_model, LoraConfig, TaskType
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss

NUM_CLASSES = 300  # 1600-1899


class VortexModel(nn.Module):
    def __init__(self, lora_r: int = 8, dropout: float = 0.0):
        super().__init__()
        vit = timm.create_model("dinov2_base", pretrained=True, drop_rate=dropout)
        emb = vit.get_classifier().in_features
        vit.reset_classifier(0)
        self.coral = CoralLayer(emb, NUM_CLASSES - 1)

        # LoRA adapters inserted only in ViT (not CORAL)
        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["qkv", "proj"],
        )
        self.vit = get_peft_model(vit, peft_cfg)

        # freeze ViT weights but leave CORAL + LoRA trainable
        for p in self.vit.base_model.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = self.vit(x)  # CLS embedding
        outputs = self.coral(feats)
        return outputs

    @staticmethod
    def coral_loss_fn(logits, y):
        return coral_loss(logits, y)

    @staticmethod
    def decode_coral(logits):
        return (torch.sigmoid(logits) > 0.5).sum(1)
