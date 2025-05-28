import argparse
import optuna
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from .dataset import PaintingDataset
from .model import VortexModel


def _step(model, batch):
    x, y = batch
    logits = model(x)
    mae = (model.decode_coral(logits) - y).abs().float().mean()
    return model.coral_loss_fn(logits, y), mae


class Lit(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = VortexModel(lora_r=cfg["lora_r"], dropout=cfg["dropout"])
        self.lr, self.wd = cfg["lr"], cfg["weight_decay"]

    def training_step(self, batch, _):
        loss, mae = _step(self.model, batch)
        self.log_dict({"train_loss": loss, "train_MAE": mae})
        return loss

    def validation_step(self, batch, _):
        _, mae = _step(self.model, batch)
        self.log("val_MAE", mae, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sch}


def objective(csv, batch, trial):
    cfg = {
        "lora_r": trial.suggest_categorical("lora_r", [2, 4, 8, 16, 32]),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    train_ds = PaintingDataset(csv, True)
    val_ds = PaintingDataset(csv, False)
    dl_train = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=8)
    dl_val = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=8)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        enable_progress_bar=False,
        callbacks=[pl.callbacks.EarlyStopping("val_MAE", patience=3, mode="min")],
        logger=False,
    )
    trainer.fit(Lit(cfg), dl_train, dl_val)
    return trainer.callback_metrics["val_MAE"].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--random-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(
        lambda t: objective(args.csv, args.batch, t),
        n_trials=args.random_trials,
        timeout=args.timeout,
    )
    best = study.best_params
    print("Best random-search params:", best)

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
        direction="minimize", sampler=optuna.samplers.GridSampler(grid)
    )
    study2.optimize(lambda t: objective(args.csv, args.batch, t), timeout=args.timeout)
    print("Grid-search best:", study2.best_value, study2.best_params)


if __name__ == "__main__":
    main()
