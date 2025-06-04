"""
Data splitting utilities for VortexModel training pipeline.

This module provides functions to split painting datasets into train/validation/test
sets with proper shuffling to avoid sampling bias from ordered data.

Usage:
    python -m vortex.data_split --csv data.csv --output-dir splits/
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[str, str, str]:
    """
    Split dataset into train/validation/test sets with shuffling.

    Args:
        csv_path: Path to original CSV with 'path' and 'year' columns
        output_dir: Directory to save split CSV files
        train_ratio: Fraction for training set (default: 0.8)
        val_ratio: Fraction for validation set (default: 0.1)
        test_ratio: Fraction for test set (default: 0.1)
        random_seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_csv_path, val_csv_path, test_csv_path)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Load and shuffle data
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=random_seed, shuffle=True
    )

    # Second split: separate train and validation from remaining data
    val_size = val_ratio / (train_ratio + val_ratio)  # Adjust for remaining data
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_seed, shuffle=True
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save split datasets
    train_csv = output_path / "train.csv"
    val_csv = output_path / "val.csv"
    test_csv = output_path / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Print split statistics
    print(f"Dataset split completed:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df):.1%})")
    print(f"  Saved to: {output_path}")

    return str(train_csv), str(val_csv), str(test_csv)


def main():
    """Command-line interface for data splitting."""
    parser = argparse.ArgumentParser(
        description="Split painting dataset into train/val/test sets"
    )
    parser.add_argument("--csv", required=True, help="Path to original CSV file")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save split CSV files"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    split_dataset(
        args.csv,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
