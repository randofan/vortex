"""
Shared utilities and constants for the VortexModel project.

This module contains common constants, configurations, and utility functions
used across the painting year prediction system.
"""

from transformers import AutoImageProcessor

# Model configuration
MODEL_NAME = (
    "facebook/dinov2-giant"
)
IMG_SIZE = 224  # Vision Transformer input size
NUM_CLASSES = 300  # Number of year classes (1600-1899 = 300 years)
BASE_YEAR = 1600  # Base year for relative year calculation

# Shared image processor for consistent preprocessing across all modules
_processor = None


def get_image_processor():
    """
    Get the shared image processor instance.

    This ensures all modules use the same preprocessing pipeline
    and avoids loading the processor multiple times.

    Returns:
        AutoImageProcessor: Configured for the current model
    """
    global _processor
    if _processor is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return _processor
