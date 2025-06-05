"""
Single‑image inference for the fine‑tuned model **plus an attention‑rollout
heat‑map overlay**.  Uses the cumulative‑product rollout method from Abnar & Zuidema 2020.

Usage
-----
python inference_attention.py \
    --checkpoint  runs/dino_lora \
    --image       sample.jpg     \
    --out_dir     demo/
"""

import os
import math
import json
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as safe_load_file
from train import (
    DinoV2Coral,
    coral_logits_to_label,
    RESIZE_SIZE,
    processor,
    BASE_YEAR
)

# --------------------------------------------------------
# Attention rollout utilities
# --------------------------------------------------------


def compute_rollout(attentions):
    """Cumulative attention rollout (Abnar & Zuidema, 2020).

    Args
    ----
    attentions : list(torch.Tensor)
        Each tensor is (batch, heads, tokens, tokens)

    Returns
    -------
    torch.Tensor  – attention mask for CLS → patch tokens (tokens‑1,)
    """
    device = attentions[0].device
    eye = torch.eye(attentions[0].size(-1), device=device)
    rollout = eye.unsqueeze(0).repeat(attentions[0].size(0), 1, 1)

    for att in attentions:
        att_fused = att.mean(dim=1)  # average over heads => (B, T, T)
        att_added = att_fused + eye  # add residual connection
        att_norm = att_added / att_added.sum(dim=-1, keepdim=True)
        rollout = att_norm @ rollout

    # CLS token index 0 → keep only patch tokens
    return rollout[0, 0, 1:]  # shape (tokens‑1,)


# --------------------------------------------------------
# Visualisation helper
# --------------------------------------------------------


def visualize(img: Image.Image, rollout_mask: torch.Tensor, out_path: str):
    """Overlay a heat‑map (rollout_mask) onto the original image and save PNG."""
    h, w = img.size[1], img.size[0]
    # rollout_mask length == num_patches; infer grid
    grid = int(math.sqrt(rollout_mask.numel()))
    mask = rollout_mask.reshape(grid, grid).cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )

    plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------
# Main CLI
# --------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to fine‑tuned model safetensors")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument(
        "--out-dir", required=True, help="Directory to write overlay & prediction"
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file with hyperparameters (for LoraConfig, matching train.py).",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        with open(args.config, "r") as f:
            config_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config}")
        return

    # ---------------- Load model ----------------
    model = DinoV2Coral()

    lora_r_val = config_params["lora_r"]
    lora_alpha_val = lora_r_val * config_params["alpha_mult"]
    lora_dropout_val = config_params["lora_dropout"]
    lora_config = LoraConfig(
        r=lora_r_val,
        lora_alpha=lora_alpha_val,
        lora_dropout=lora_dropout_val,
        bias="none",
        target_modules=["query", "key", "value", "dense"],  # Must match train.py
    )
    model.base = get_peft_model(model.base, lora_config)
    missing, unexpected = model.load_state_dict(safe_load_file(args.checkpoint), strict=False)
    print(f"Loaded model with {len(missing)} missing keys and {len(unexpected)} unexpected keys.")
    model.eval().cuda()

    # ---------------- Preprocess image ----------------
    img = Image.open(args.image).convert("RGB")
    pixels = processor(
        img, do_resize=True, size={"shortest_edge": RESIZE_SIZE}, return_tensors="pt"
    )
    pixels = {k: v.cuda().half() for k, v in pixels.items()}

    # ---------------- Forward pass with attentions ----------------
    with torch.no_grad():
        outs = model.forward(pixels["pixel_values"].cuda(), attention=True)

    logits = outs["logits"]
    pred = coral_logits_to_label(logits)[0].item() + BASE_YEAR

    # ---------------- Rollout & visualise ----------------
    heat = compute_rollout(outs["attentions"])
    visualize(img, heat, os.path.join(args.out_dir, "attention_overlay.png"))

    with open(os.path.join(args.out_dir, "prediction.txt"), "w") as f:
        f.write(str(pred))
    print("Prediction:", pred, "| overlay saved to", args.out_dir)


if __name__ == "__main__":
    main()
