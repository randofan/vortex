"""
Training script for the VortexModel painting year prediction system.

This script handles the complete training pipeline including data loading,
model training with PyTorch Lightning, and checkpoint saving.

Usage:
    python -m vortex.train --csv data.csv --epochs 30 --batch 32
"""

import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import PaintingDataset
from vortexmodel import VortexModel, _step
from utils import calculate_mae  # Modified import


class Wrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for VortexModel.

    This wrapper handles the training loop, optimization, and logging
    for the painting year prediction model.
    """

    def __init__(self, cfg: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Initialize model with configuration
        self.model = VortexModel(lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha)
        self.lr = cfg.lr
        self.wd = cfg.weight_decay

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with loss and metrics logging."""
        loss, mae = _step(self.model, batch)
        self.log_dict(
            {"train_loss": loss, "train_MAE": mae}, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step with MAE logging."""
        _, mae = _step(self.model, batch)
        self.log("val_MAE", mae, prog_bar=True)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VortexModel for painting year prediction"
    )
    parser.add_argument(
        "--train-csv", required=True, help="Path to training CSV file"
    )
    parser.add_argument(
        "--val-csv", required=True, help="Path to validation CSV file"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size for training and validation"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank parameter")
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling parameter (common: rank to 4×rank)",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    cfg = cli()

    # Create separate data loaders for train and validation
    dl_train = DataLoader(
        PaintingDataset(cfg.train_csv),
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=8,
        pin_memory=True,  # Faster GPU transfer
    )
    dl_val = DataLoader(
        PaintingDataset(cfg.val_csv),
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Set up logging
    logger = CSVLogger("lightning_logs", name="vortex")

    # Configure model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_MAE",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_MAE:.3f}",
        save_last=True,
        verbose=True,
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",  # Mixed precision for faster training
        logger=logger,
        callbacks=[
            pl.callbacks.EarlyStopping("val_MAE", patience=5, mode="min"),
            checkpoint_callback,
        ],
        log_every_n_steps=50,  # Log metrics every 50 steps
    )

    # Train the model
    trainer.fit(Wrapper(cfg), dl_train, dl_val)

    # Print checkpoint locations
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Last model saved at: {checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
