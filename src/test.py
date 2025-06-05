"""
Generate predictions (and optional MAE) on a held‑out CSV split using the
fine‑tuned model.

Example
-------
python test_coral_preds.py \
    --checkpoint  runs/dino_lora \
    --test-csv    splits/test.csv \
    --out-csv     results/test_preds.csv

Writes **out-csv** with columns: `path,y_true,y_pred`.
If the `year` column exists in the CSV, MAE is printed.
"""

import argparse
import os
import csv
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from train import (
    DinoV2Coral,
    coral_logits_to_label,
    make_dataset,
    mae_metric
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    ds = make_dataset(args.test_csv)

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
            out = model(pixel_values=pixel_values)
            logits = out["logits"]
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
