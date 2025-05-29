"""
Hyperparameter tuning script for VortexModel using Optuna.

This script performs automated hyperparameter optimization using a combination
of random search followed by grid search around the best parameters.

Usage:
    python -m vortex.tune --csv data.csv --random-trials 50 --timeout 3600
"""

import argparse
import optuna
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from .dataset import PaintingDataset
from .model import VortexModel


def _step(model: VortexModel, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a single training/validation step.

    Args:
        model: The VortexModel instance
        batch: Tuple of (images, year_labels)

    Returns:
        Tuple of (loss, mae) for optimization
    """
    x, y = batch
    logits = model(x)
    loss = model.coral_loss_fn(logits, y)
    mae = (model.decode_coral(logits) - y).abs().float().mean()
    return loss, mae


class Lit(pl.LightningModule):
    """Lightning module wrapper for hyperparameter optimization."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = VortexModel(lora_r=cfg["lora_r"], dropout=cfg["dropout"])
        self.lr = cfg["lr"]
        self.wd = cfg["weight_decay"]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, mae = _step(self.model, batch)
        self.log_dict({"train_loss": loss, "train_MAE": mae})
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        _, mae = _step(self.model, batch)
        self.log("val_MAE", mae, prog_bar=True)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and scheduler."""
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sch}


def objective(csv: str, batch: int, trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        csv: Path to dataset CSV
        batch: Batch size
        trial: Optuna trial object

    Returns:
        Validation MAE to minimize
    """
    cfg = {
        "lora_r": trial.suggest_categorical("lora_r", [2, 4, 8, 16, 32]),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }

    # Create datasets (same for train/val in tuning for speed)
    train_ds = PaintingDataset(csv)
    val_ds = PaintingDataset(csv)
    dl_train = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=8)
    dl_val = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=8)

    # Quick training for hyperparameter search
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        enable_progress_bar=False,
        callbacks=[pl.callbacks.EarlyStopping("val_MAE", patience=3, mode="min")],
        logger=False,
    )
    trainer.fit(Lit(cfg), dl_train, dl_val)
    return trainer.callback_metrics["val_MAE"].item()


def main():
    """Main hyperparameter tuning function."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for VortexModel"
    )
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--random-trials", type=int, default=50, help="Number of random search trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Maximum time in seconds"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Phase 1: Random search
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=pruner
    )
    study.optimize(
        lambda t: objective(args.csv, args.batch, t),
        n_trials=args.random_trials,
        timeout=args.timeout,
    )
    best = study.best_params
    print("Best random-search params:", best)

    # Phase 2: Grid search around best parameters
    grid = {
        "lora_r": [best["lora_r"]],
        "lr": [best["lr"] * 0.5, best["lr"], best["lr"] * 2],
        "weight_decay": [
            max(0, best["weight_decay"] - 0.05),
            best["weight_decay"],
            min(0.3, best["weight_decay"] + 0.05),
        ],
        "dropout": [best["dropout"]],
    }
    study2 = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.GridSampler(grid), pruner=pruner
    )
    study2.optimize(lambda t: objective(args.csv, args.batch, t), timeout=args.timeout)
    print("Grid-search best:", study2.best_value, study2.best_params)


if __name__ == "__main__":
    main()
