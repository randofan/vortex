"""
Test evaluation script for VortexModel painting year prediction.

This script loads a trained model checkpoint, runs predictions on test data,
calculates performance metrics, and saves detailed results to CSV.

Usage:
    python -m vortex.test --checkpoint best.ckpt --test-csv data_splits/test.csv --output results.csv
"""

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset import PaintingDataset
from vortexmodel import VortexModel
from utils import BASE_YEAR, calculate_mae


@torch.no_grad()
def predict_on_dataset(
    model: VortexModel, test_csv: str, batch_size: int = 32, device: str = "auto"
) -> pd.DataFrame:
    """
    Run model predictions on test dataset.

    Args:
        model: Trained VortexModel instance
        test_csv: Path to test dataset CSV
        batch_size: Batch size for inference
        device: Device to run inference on ("auto", "cpu", "cuda")

    Returns:
        DataFrame with columns: path, year, pred_year, mae
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Create test dataset and dataloader
    test_dataset = PaintingDataset(test_csv)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # Storage for results
    results = []

    print(f"Running predictions on {len(test_dataset)} test samples...")

    for batch_idx, (images, year_offsets) in enumerate(
        tqdm(test_loader, desc="Predicting")
    ):
        images = images.to(device)
        year_offsets = year_offsets.to(device)

        # Forward pass
        logits = model(images)
        pred_offsets = model.decode_coral(logits)

        # Calculate per-sample MAE using vectorized operations
        mae_per_sample = calculate_mae(pred_offsets, year_offsets)

        # Convert to absolute years only for storage/display
        true_years = year_offsets + BASE_YEAR
        pred_years = pred_offsets + BASE_YEAR

        # Get image paths for this batch
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_dataset))
        batch_paths = [
            test_dataset.df.iloc[i]["path"] for i in range(batch_start, batch_end)
        ]

        # Store results using the calculated per-sample MAE
        for i in range(len(images)):
            results.append(
                {
                    "path": batch_paths[i],
                    "year": int(true_years[i].item()),
                    "pred_year": int(pred_years[i].item()),
                    "mae": float(mae_per_sample[i].item()),
                }
            )

    return pd.DataFrame(results)


def load_model_from_checkpoint(checkpoint_path: str) -> VortexModel:
    """
    Load VortexModel from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns:
        Loaded VortexModel instance
    """
    try:
        from train import Wrapper

        # Load Lightning wrapper
        wrapper = Wrapper.load_from_checkpoint(checkpoint_path)
        return wrapper.model

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")


def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        results_df: DataFrame with prediction results

    Returns:
        Dictionary of metrics
    """
    mae_values = results_df["mae"]

    metrics = {
        "mae_mean": float(mae_values.mean()),
        "mae_std": float(mae_values.std()),
        "mae_median": float(mae_values.median()),
        "mae_min": float(mae_values.min()),
        "mae_max": float(mae_values.max()),
        "samples_count": len(results_df),
        "accuracy_5yr": float((mae_values <= 5).mean() * 100),  # % within 5 years
        "accuracy_10yr": float((mae_values <= 10).mean() * 100),  # % within 10 years
        "accuracy_20yr": float((mae_values <= 20).mean() * 100),  # % within 20 years
    }

    return metrics


def print_metrics(metrics: dict):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 50)
    print("TEST EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples: {metrics['samples_count']}")
    print(f"\nMean Absolute Error (MAE):")
    print(f"  Mean: {metrics['mae_mean']:.2f} years")
    print(f"  Std:  {metrics['mae_std']:.2f} years")
    print(f"  Median: {metrics['mae_median']:.2f} years")
    print(f"  Range: [{metrics['mae_min']:.0f}, {metrics['mae_max']:.0f}] years")
    print(f"\nAccuracy (% within threshold):")
    print(f"  ±5 years:  {metrics['accuracy_5yr']:.1f}%")
    print(f"  ±10 years: {metrics['accuracy_10yr']:.1f}%")
    print(f"  ±20 years: {metrics['accuracy_20yr']:.1f}%")
    print("=" * 50)


def main():
    """Main test evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained VortexModel on test data"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to PyTorch Lightning checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--test-csv", required=True, help="Path to test dataset CSV file"
    )
    parser.add_argument("--output", required=True, help="Path to save results CSV file")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only print metrics without saving detailed results",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.test_csv).exists():
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint)

    # Run predictions
    results_df = predict_on_dataset(
        model, args.test_csv, batch_size=args.batch_size, device=args.device
    )

    # Calculate and display metrics
    metrics = calculate_metrics(results_df)
    print_metrics(metrics)

    # Save results
    if not args.metrics_only:
        # Save detailed results
        results_df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")

        # Save metrics summary
        metrics_path = (
            Path(args.output).parent / f"{Path(args.output).stem}_metrics.json"
        )
        import json

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics summary saved to: {metrics_path}")

    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()
