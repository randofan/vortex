"""
Dataset module for painting year prediction.

This module provides a PyTorch Dataset for loading and preprocessing painting images
with their corresponding creation years for training a Vision Transformer model.
"""

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_image_processor, BASE_YEAR


class PaintingDataset(Dataset):
    """
    Dataset for painting images with year labels.

    This dataset loads paintings from a CSV file and preprocesses them using
    the official Hugging Face DINOv2 image processor for consistency with
    the model's expected input format.

    Args:
        csv_path: Path to CSV file containing 'path' and 'year' columns
        base_year: Year to subtract from actual years (default: BASE_YEAR)
                  This converts absolute years to relative offsets

    Returns:
        Tuple of (preprocessed_image_tensor, year_offset)
    """

    def __init__(self, csv_path: str, base_year: int = BASE_YEAR):
        self.df = pd.read_csv(csv_path)
        self.processor = get_image_processor()
        self.base_year = base_year

        # Validate CSV format
        required_cols = {"path", "year"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, year_label):
            - image_tensor: Preprocessed image as (3, 224, 224) tensor
            - year_label: Year offset as long tensor
        """
        row = self.df.iloc[idx]

        # Load and convert image from BGR to RGB
        img = cv2.imread(row.path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {row.path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use HF processor for consistent preprocessing (resize, normalize, etc.)
        inputs = self.processor(images=img, return_tensors="pt")
        x = inputs["pixel_values"].squeeze(0)  # Remove batch dimension

        # Convert year to offset from base year
        y = torch.tensor(int(row.year) - self.base_year, dtype=torch.long)
        return x, y
