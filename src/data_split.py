"""
Shuffle an ordered CSV and create **train / val / test** splits (80/10/10).
"""

import os
import argparse
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds_full = load_dataset("csv", data_files=args.csv, split="train").shuffle(
        seed=args.seed
    )
    tmp = ds_full.train_test_split(test_size=0.20, seed=args.seed)
    train_ds = tmp["train"]
    val_ds, test_ds = (
        tmp["test"].train_test_split(test_size=0.50, seed=args.seed).values()
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds.to_csv(f"{args.output_dir}/train.csv", index=False)
    val_ds.to_csv(f"{args.output_dir}/val.csv", index=False)
    test_ds.to_csv(f"{args.output_dir}/test.csv", index=False)
    print({"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)})


if __name__ == "__main__":
    main()
