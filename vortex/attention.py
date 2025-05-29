"""
Attention visualization utilities for VortexModel.

This module provides attention rollout visualization for understanding what
the Vision Transformer focuses on when predicting painting creation years.

Usage:
    python -m vortex.attention --checkpoint best.ckpt --image painting.jpg --output attention.png
"""

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
from .utils import get_image_processor


@torch.no_grad()
def rollout(
    vit_model: torch.nn.Module, x: torch.Tensor, discard_ratio: float = 0.0
) -> torch.Tensor:
    """
    Compute attention rollout for Vision Transformer visualization.

    Attention rollout traces how information flows from input patches to the
    CLS token through all transformer layers, providing interpretable heatmaps.

    Args:
        vit_model: Hugging Face DINOv2 model (with or without LoRA)
        x: Input tensor (B, 3, 224, 224) - preprocessed images
        discard_ratio: Fraction of weak attention links to prune (0-1)

    Returns:
        Attention mask tensor (B, 1, 224, 224) in range [0, 1]
    """
    vit_model.eval()

    # Extract attention weights from all transformer layers
    if hasattr(vit_model, "get_attention_map"):
        # Use custom method if available (VortexModel)
        _, attn_list = vit_model.get_attention_map(x, return_all=True)
    else:
        # Direct call for base HF model
        outputs = vit_model(pixel_values=x, output_attentions=True)
        attn_list = outputs.attentions

    # Initialize identity matrix for rollout computation
    result = torch.eye(attn_list[0].size(-1), device=x.device)

    # Roll out attention through all layers
    for attn in attn_list:
        # Average attention across all heads
        attn = attn.mean(1)  # (B, seq_len, seq_len)

        # Optional: Prune weak attention links
        if discard_ratio > 0:
            flat = attn.view(attn.size(0), -1)
            _, indices = flat.topk(
                int(flat.size(1) * discard_ratio), dim=1, largest=False
            )
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            attn = attn.masked_fill(mask.view_as(attn), 0)

        # Add residual connection and normalize
        attn = attn + torch.eye(attn.size(-1), device=x.device)
        attn = attn / attn.sum(-1, keepdim=True)

        # Accumulate attention flow
        result = attn @ result

    # Extract attention from CLS token to all patches (excluding CLS itself)
    mask = result[:, 0, 1:]  # Shape: (B, num_patches)

    # Reshape to spatial grid (14x14 for ViT with 16x16 patches)
    s = int(mask.size(-1) ** 0.5)
    mask = mask.view(-1, 1, s, s)

    # Upsample to original image resolution (224x224)
    mask = torch.nn.functional.interpolate(
        mask, scale_factor=16, mode="bilinear", align_corners=False
    )

    return mask.clamp(0, 1)


@torch.no_grad()
def visualize(
    model: torch.nn.Module,
    img_path: str,
    save_path: str | None = None,
    discard_ratio: float = 0.0,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Generate attention rollout visualization overlay.

    Creates a blended image showing the original painting with an attention
    heatmap overlay indicating which regions the model focuses on.

    Args:
        model: VortexModel or Lightning module containing VortexModel
        img_path: Path to input image file
        save_path: Optional path to save visualization PNG
        discard_ratio: Fraction of weak attention to prune (0-1)
        alpha: Blending weight for overlay (0=original, 1=attention only)

    Returns:
        PIL Image with attention overlay
    """
    # Handle Lightning module wrapper
    if hasattr(model, "model"):
        model = model.model

    # Get model device
    device = next(model.parameters()).device

    # Load and preprocess image
    try:
        pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Could not load image {img_path}: {e}")

    # Use shared processor for consistent preprocessing
    processor = get_image_processor()
    inputs = processor(pil, return_tensors="pt")
    x = inputs["pixel_values"].to(device)

    # Extract ViT component for attention computation
    vit = model.vit if hasattr(model, "vit") else model

    # Handle PEFT wrapper (access base model for attention)
    vit_base = vit.base_model if hasattr(vit, "base_model") else vit

    # Compute attention rollout
    mask = rollout(vit_base, x, discard_ratio=discard_ratio)

    # Resize attention mask to match original image dimensions
    w, h = pil.size
    mask_img = T.functional.resize(
        to_pil_image(mask.squeeze(0)),
        (h, w),
        interpolation=T.InterpolationMode.BILINEAR,
    )
    mask_img = mask_img.convert("RGB")  # Convert grayscale to RGB

    # Blend original image with attention mask
    orig_tensor = T.functional.to_tensor(pil)
    mask_tensor = T.functional.to_tensor(mask_img)
    blended = (1 - alpha) * orig_tensor + alpha * mask_tensor
    blended_pil = to_pil_image(blended)

    # Save or display result
    if save_path:
        blended_pil.save(save_path)
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(blended_pil)
        plt.axis("off")
        plt.title(f"Attention Rollout (α={alpha}, discard={discard_ratio})")
        plt.tight_layout()
        plt.show()

    return blended_pil


def main():
    """Command-line interface for attention visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize attention rollout for painting year prediction"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to PyTorch Lightning checkpoint (.ckpt file)",
    )
    parser.add_argument("--image", required=True, help="Path to input painting image")
    parser.add_argument(
        "--output",
        help="Path to save visualization (shows interactive plot if not provided)",
    )
    parser.add_argument(
        "--discard-ratio",
        type=float,
        default=0.0,
        help="Fraction of weak attention to discard (0.0-1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Attention overlay strength (0.0=original, 1.0=attention only)",
    )
    args = parser.parse_args()

    # Load model from Lightning checkpoint
    try:
        from .train import Wrapper

        wrapper = Wrapper.load_from_checkpoint(args.checkpoint)
        model = wrapper.model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {args.checkpoint}: {e}")

    # Generate visualization
    try:
        visualize(
            model,
            args.image,
            save_path=args.output,
            discard_ratio=args.discard_ratio,
            alpha=args.alpha,
        )

        result_msg = (
            f"saved to {args.output}" if args.output else "displayed interactively"
        )
        print(f"Attention visualization {result_msg}")

    except Exception as e:
        raise RuntimeError(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()
