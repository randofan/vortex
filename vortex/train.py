import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from .dataset import PaintingDataset
from .model import VortexModel


def _step(model, batch):
    x, y = batch
    logits = model(x)
    loss = model.coral_loss_fn(logits, y)
    mae = (model.decode_coral(logits) - y).abs().float().mean()
    return loss, mae


class Wrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = VortexModel(lora_r=cfg.lora_r)
        self.lr, self.wd = cfg.lr, cfg.weight_decay

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


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lora_r", type=int, default=8)
    return p.parse_args()


def main():
    cfg = cli()
    dl_train = DataLoader(
        PaintingDataset(cfg.csv, True),
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=8,
    )
    dl_val = DataLoader(
        PaintingDataset(cfg.csv, False),
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=8,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[pl.callbacks.EarlyStopping("val_MAE", patience=5, mode="min")],
    )
    trainer.fit(Wrapper(cfg), dl_train, dl_val)


if __name__ == "__main__":
    main()
