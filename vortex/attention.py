"""
vortex.attention
----------------
Utilities for attention-rollout visualisation.

Typical use:

    from vortex.model import VortexModel
    from vortex.attention import visualize

    model = VortexModel.load_from_checkpoint("best.ckpt").eval().cuda()
    visualize(model, "sample.jpg", save_path="sample_attn.png", discard_ratio=0.0)
"""

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt


def patch_vit_for_attention(vit_model):
    """Add get_attention_map method to ViT model for attention extraction."""

    def get_attention_map(self, x, return_all=False):
        attentions = []

        def hook_fn(module, input, output):
            # output is (B, num_heads, seq_len, seq_len)
            attentions.append(output.detach())

        hooks = []
        for block in self.blocks:
            hooks.append(block.attn.register_forward_hook(hook_fn))

        # Forward pass
        output = self(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if return_all:
            return output, attentions
        return output

    vit_model.get_attention_map = get_attention_map.__get__(vit_model)
    return vit_model


@torch.no_grad()
def rollout(vit_model, x, discard_ratio: float = 0.0):
    """
    Args
    ----
    vit_model : a timm ViT *with get_attention_map patched* (LoRA is fine)
    x         : tensor (B,3,224,224) in [0,1]
    discard_ratio : float in [0,1) – prune low-importance links

    Returns
    -------
    mask : (B,1,224,224) tensor in [0,1]
    """
    vit_model.eval()

    # Patch the model if it doesn't have get_attention_map
    if not hasattr(vit_model, "get_attention_map"):
        patch_vit_for_attention(vit_model)

    _, attn_list = vit_model.get_attention_map(x, return_all=True)
    result = torch.eye(attn_list[0].size(-1), device=x.device)
    for attn in attn_list:
        attn = attn.mean(1)
        if discard_ratio:
            flat = attn.view(attn.size(0), -1)
            v, _ = flat.topk(int(flat.size(1) * discard_ratio), dim=1, largest=False)
            mask = (flat <= v[:, -1].unsqueeze(-1)).view_as(attn)
            attn = attn.masked_fill(mask, 0)
        attn += torch.eye(attn.size(-1), device=x.device)
        attn /= attn.sum(-1, keepdim=True)
        result = attn @ result
    mask = result[:, 0, 1:]  # drop CLS
    s = int(mask.size(-1) ** 0.5)  # 14 for ViT-Base
    mask = mask.view(-1, 1, s, s)
    return torch.nn.functional.interpolate(
        mask, scale_factor=16, mode="bilinear", align_corners=False
    ).clamp(0, 1)


_preproc = T.Compose(
    [
        T.Resize(224, antialias=True),  # Resize longest side to 224
        T.Pad(
            (0, 0, 224, 224), fill=0, padding_mode="reflect"
        ),  # Pad to 224x224 exactly
        T.ToTensor(),
    ]
)


@torch.no_grad()
def visualize(
    model,
    img_path: str,
    save_path: str | None = None,
    discard_ratio: float = 0.0,
    alpha: float = 0.5,
):
    """
    Generate an image with attention rollout overlayed.

    Parameters
    ----------
    model         : VortexModel or PyTorch Lightning module containing VortexModel
    img_path      : path to input image
    save_path     : if given, save the PNG there; otherwise show it
    discard_ratio : see rollout()
    alpha         : blending weight mask↔original (0..1)

    Returns
    -------
    PIL.Image  (also saved / displayed as a side-effect)
    """
    # Handle Lightning module wrapper
    if hasattr(model, "model"):
        model = model.model

    device = next(model.parameters()).device
    pil = Image.open(img_path).convert("RGB")
    x = _preproc(pil).unsqueeze(0).to(device)

    # ---- obtain mask from *ViT only* ----
    vit = model.vit if hasattr(model, "vit") else model  # VortexModel or bare ViT

    # Access the base model if using PEFT
    if hasattr(vit, "base_model"):
        vit_base = vit.base_model
    else:
        vit_base = vit

    mask = rollout(vit_base, x, discard_ratio=discard_ratio)  # (1,1,224,224)

    # ---- upscale mask to original image size for nicer overlay ----
    w, h = pil.size
    mask_img = T.functional.resize(
        to_pil_image(mask.squeeze(0)),
        (h, w),
        interpolation=T.InterpolationMode.BILINEAR,
    )
    mask_img = mask_img.convert("RGB")  # greyscale → RGB

    # ---- blend ----
    orig_tensor = T.functional.to_tensor(pil)
    mask_tensor = T.functional.to_tensor(mask_img)
    blended = (1 - alpha) * orig_tensor + alpha * mask_tensor
    blended_pil = to_pil_image(blended)

    # ---- output ----
    if save_path:
        blended_pil.save(save_path)
    else:
        plt.figure(figsize=(4, 4))
        plt.imshow(blended_pil)
        plt.axis("off")
        plt.title("Attention Rollout")
        plt.show()
    return blended_pil
