"""
Generate predictions (and optional MAE) on a held‑out CSV split using the
fine‑tuned model.

Example
-------
python test_coral_preds.py \
    --checkpoint  runs/dino_lora \
    --test_csv    splits/test.csv \
    --out_csv     results/test_preds.csv

Writes **out_csv** with columns: `path,y_true,y_pred`.
If the `year` column exists in the CSV, MAE is printed.
"""

import argparse
import os
import csv
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from train_coral_lora import (
    DinoV2Coral,
    coral_logits_to_label,
    RESIZE_SIZE,
    processor,
)
from PIL import Image
import evaluate

mae_metric = evaluate.load("mae")


def preprocess(example):
    img = Image.open(example["path"]).convert("RGB")
    pv = processor(
        img, do_resize=True, size={"shortest_edge": RESIZE_SIZE}, return_tensors="pt"
    )["pixel_values"][0]
    example["pixel_values"] = pv.half()
    return example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    # ---------------- Load dataset ----------------
    ds = (
        load_dataset("csv", data_files=args.test_csv)["train"]
        .map(preprocess)
        .with_format("torch")
    )

    has_labels = "year" in ds.column_names
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=False, collate_fn=default_data_collator
    )

    # ---------------- Model ----------------
    model = DinoV2Coral()
    ckpt = torch.load(
        os.path.join(args.checkpoint, "pytorch_model.bin"), map_location="cuda"
    )
    model.load_state_dict(ckpt)
    model.eval().cuda()

    preds = []
    labels = []
    paths = (
        ds["path"]
        if "path" in ds.column_names
        else [f"idx_{i}" for i in range(len(ds))]
    )

    with torch.no_grad():
        for batch in dl:
            pixel_values = batch["pixel_values"].cuda()
            logits = model.head(
                model.base(pixel_values=pixel_values).last_hidden_state[:, 0]
            )
            pred = coral_logits_to_label(logits).cpu().numpy().tolist()
            preds.extend(pred)
            if has_labels:
                labels.extend(batch["year"].cpu().numpy().tolist())

    # ---------------- Save CSV ----------------
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "y_true", "y_pred"])
        for i, p in enumerate(preds):
            y_true = labels[i] if has_labels else ""
            writer.writerow([paths[i], y_true, p])

    if has_labels:
        mae = mae_metric.compute(predictions=preds, references=labels)
        print("MAE on test set:", mae)
    else:
        print("Predictions written to", args.out_csv)


if __name__ == "__main__":
    main()
