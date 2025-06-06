#!/usr/bin/env python3
"""
plot_year_errors.py

Example:
    python plot_year_errors.py --true first.csv --pred second.csv
"""

import matplotlib
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

matplotlib.use("Agg")


def load_and_merge(true_csv: str, pred_csv: str, pred2_csv: str = None) -> pd.DataFrame:
    df_true = pd.read_csv(true_csv).rename(columns={"year": "year_true"})
    df_pred = pd.read_csv(pred_csv).rename(columns={"year": "year_pred"})
    merged = pd.concat([df_true, df_pred], axis=1)
    merged["abs_error"] = (merged["year_true"] - merged["year_pred"]).abs()

    if pred2_csv:
        df_pred2 = pd.read_csv(pred2_csv).rename(columns={"prediction": "y_pred2"})
        merged = pd.concat([merged, df_pred2], axis=1)
        merged["abs_error2"] = (merged["year_true"] - merged["y_pred2"]).abs()

    return merged


def plot_violin(df: pd.DataFrame, output: Path):
    """Save a violin plot unless all errors are identical (singular KDE)."""
    has_pred2 = "abs_error2" in df.columns

    if df["abs_error"].nunique() <= 1 and (
        not has_pred2 or df["abs_error2"].nunique() <= 1
    ):
        print("Violin plot skipped (all abs_error values identical).")
        return

    plt.figure(figsize=(8, 4) if has_pred2 else (6, 4))

    # Prepare data
    error_data1 = df["abs_error"].dropna().values
    if len(error_data1) == 0:
        print("Violin plot skipped (no valid error data).")
        return

    # Create violin plots
    if has_pred2:
        error_data2 = df["abs_error2"].dropna().values
        if len(error_data2) > 0:
            parts = plt.violinplot(
                [error_data1, error_data2],
                positions=[1, 2],
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            plt.xlim(0.5, 2.5)
            plt.xticks([1, 2], ["Gemini 2.0 Flash", "Vortex"])
            all_data = np.concatenate([error_data1, error_data2])
        else:
            parts = plt.violinplot(
                [error_data1],
                positions=[1],
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            plt.xlim(0.5, 1.5)
            plt.xticks([])
            all_data = error_data1
    else:
        parts = plt.violinplot(
            [error_data1],
            positions=[1],
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        plt.xlim(0.5, 1.5)
        plt.xticks([])
        all_data = error_data1

    # Make the median and extrema lines shorter
    # Handle median lines
    if parts["cmedians"]:
        median_collection = parts["cmedians"]
        for path in median_collection.get_paths():
            vertices = path.vertices
            if len(vertices) >= 2:
                center_x = np.mean(vertices[:, 0])
                y_vals = vertices[:, 1]
                width = 0.15  # Fixed width for shorter lines
                new_vertices = np.array(
                    [[center_x - width, y_vals[0]], [center_x + width, y_vals[-1]]]
                )
                path.vertices = new_vertices

    # Handle extrema lines (min/max)
    if parts["cmaxes"]:
        max_collection = parts["cmaxes"]
        for path in max_collection.get_paths():
            vertices = path.vertices
            if len(vertices) >= 2:
                center_x = np.mean(vertices[:, 0])
                y_vals = vertices[:, 1]
                width = 0.1
                new_vertices = np.array(
                    [[center_x - width, y_vals[0]], [center_x + width, y_vals[-1]]]
                )
                path.vertices = new_vertices

    if parts["cmins"]:
        min_collection = parts["cmins"]
        for path in min_collection.get_paths():
            vertices = path.vertices
            if len(vertices) >= 2:
                center_x = np.mean(vertices[:, 0])
                y_vals = vertices[:, 1]
                width = 0.1
                new_vertices = np.array(
                    [[center_x - width, y_vals[0]], [center_x + width, y_vals[-1]]]
                )
                path.vertices = new_vertices

    # Set explicit axis limits
    plt.ylim(0, max(all_data) * 1.1)

    plt.title("Distribution of Absolute Year Prediction Error")
    plt.ylabel("Absolute Error (years)")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved violin plot ➜ {output}")
    print(f"Error data range: {error_data1.min():.2f} to {error_data1.max():.2f}")


def plot_scatter(df: pd.DataFrame, output: Path):
    has_pred2 = "abs_error2" in df.columns

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["year_true"],
        df["abs_error"],
        s=25,
        alpha=0.7,
        edgecolors="k",
        label="Gemini 2.0 Flash",
        color="blue",
    )

    if has_pred2:
        plt.scatter(
            df["year_true"],
            df["abs_error2"],
            s=25,
            alpha=0.7,
            edgecolors="k",
            label="Vortex",
            color="red",
        )
        plt.legend()

    plt.title("Absolute Year Prediction Error by True Year")
    plt.xlabel("True Year")
    plt.ylabel("Absolute Error (years)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved scatter plot ➜ {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot painting-date prediction errors")
    parser.add_argument("--true", required=True, help="CSV with columns: path, year")
    parser.add_argument(
        "--pred", required=True, help="CSV with columns: year, reasoning"
    )
    parser.add_argument("--pred2", help="CSV with columns: prediction")
    parser.add_argument(
        "--out",
        default="merged_with_errors.csv",
        help="Merged CSV output file (default: %(default)s)",
    )
    args = parser.parse_args()

    merged = load_and_merge(args.true, args.pred, args.pred2)
    merged.to_csv(args.out, index=False)

    mae = merged["abs_error"].mean()
    print(f"Mean Absolute Error (MAE) Gemini 2.0 Flash: {mae:.2f} years")

    if "abs_error2" in merged.columns:
        mae2 = merged["abs_error2"].mean()
        print(f"Mean Absolute Error (MAE) Vortex: {mae2:.2f} years")

    print(f"Merged CSV saved ➜ {args.out}")

    plot_violin(merged, Path("abs_error_violin.png"))
    plot_scatter(merged, Path("abs_error_by_year.png"))


if __name__ == "__main__":
    main()
