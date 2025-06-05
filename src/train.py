"""
Fine‑tune *facebook/dinov2‑large* on a single 16 GB GPU.

Key updates compared to the previous version
-------------------------------------------
* **Budget‑aware hyper‑parameter search** using Optuna’s **ASHA / Successive Halving**
  – fast exploratory runs (≈ 500 training steps) prune weak trials early.
* **Sampler** switched to `TPESampler(multivariate=True)` for smarter proposals.
* **Narrowed search‑space** (learning‑rate 1e‑5→1e‑4, LoRA r ∈ {4,8,12}, etc.)
* **Search runs just 1 epoch**; after the best config is found we re‑train for
  **4 epochs** to converge.

Memory‑saving still comes from:
* 8‑bit weight loading (`bitsandbytes`)
* fp16 activations
* gradient‑checkpointing
* batch size ≤ 2
* LoRA only on Q/V projections
"""

import os
import math
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model
import evaluate
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from PIL import Image

from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss
from coral_pytorch.dataset import levels_from_labelbatch

# ---------- constants ----------
NUM_CLASSES = 300
MODEL_NAME = "facebook/dinov2-large"
SEARCH_MIN_EPOCHS = 1  # epochs per trial during HPO
FINAL_EPOCHS = 4  # full training epochs after HPO
EVAL_STEPS = 200  # evaluation interval (steps)
PRUNER_MIN_STEPS = 500  # first ASHA rung
BASE_YEAR = 1600 # Base year for relative year calculation


# ---------- Model wrapper ----------
class DinoV2Coral(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map=None,
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
        hidden = self.base.config.hidden_size
        self.head = CoralLayer(hidden, NUM_CLASSES)
        self.criterion = CoralLoss(reduction='mean')

    def forward(self, pixel_values, labels=None):
        rep = self.base(pixel_values=pixel_values).last_hidden_state[:, 0]
        logits = self.head(rep)
        if labels is not None:
            levels = levels_from_labelbatch(labels, NUM_CLASSES).to(logits.device)
            loss = self.criterion(logits, levels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


# ---------- Data pipeline ----------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
RESIZE_SIZE = processor.size.get("shortest_edge", 224)


def preprocess(example):
    img = Image.open(example["path"]).convert("RGB")
    pv = processor(
        img, do_resize=True, size={"shortest_edge": RESIZE_SIZE}, return_tensors="pt"
    )["pixel_values"][0]
    example["pixel_values"] = pv.half()
    example["labels"] = int(example["year"]) - BASE_YEAR
    return example


def make_dataset(csv_path):
    return (
        load_dataset("csv", data_files=csv_path)["train"]
        .map(preprocess, remove_columns=["path", "year"])
        .with_format("torch")
    )


# ---------- Metrics ----------
mae_metric = evaluate.load("mae")


def coral_logits_to_label(logits):
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


def compute_metrics(p):
    logits, labels = p
    preds = coral_logits_to_label(torch.tensor(logits))
    return mae_metric.compute(predictions=preds.cpu().numpy(), references=labels)


# ---------- Optuna helpers ----------

best_hparams = {}


def model_init(trial: optuna.Trial | None = None):
    """
    If Optuna supplies `trial`, sample hyper-params.
    Otherwise re-use the best set stored in `best_hparams`.
    """
    if trial is not None:
        r = trial.suggest_categorical("lora_r", [4, 8, 12])
        alpha_m = trial.suggest_categorical("alpha_mult", [1, 2, 4])
        drp = trial.suggest_float("lora_dropout", 0.0, 0.15)
        best_hparams.update(dict(lora_r=r, alpha_mult=alpha_m, lora_dropout=drp))
    else:
        hp = {'lora_r': 8, 'alpha_mult': 2, 'lora_dropout': 0.75}
        r, alpha_m, drp = hp["lora_r"], hp["alpha_mult"], hp["lora_dropout"]

    alpha = r * alpha_m
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=drp,
        bias="none",
        target_modules=["query", "key", 'value', 'dense'],
        # task_type="FEATURE_EXTRACTION",
    )

    model = DinoV2Coral()
    model.base = get_peft_model(model.base, cfg)
    return model


def hp_space(trial):
    return {
        "per_device_train_batch_size": trial.suggest_categorical("bs", [1, 2]),
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-4, log=True),
        "num_train_epochs": SEARCH_MIN_EPOCHS,  # fixed during search
        "weight_decay": trial.suggest_float("wd", 1e-5, 1e-2, log=True),
    }


sampler = TPESampler(multivariate=True, group=True)
pruner = SuccessiveHalvingPruner(
    min_resource=PRUNER_MIN_STEPS, reduction_factor=3, min_early_stopping_rate=0
)

# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--optuna-trials", type=int, default=60)
    args = ap.parse_args()

    train_ds, val_ds = make_dataset(args.train_csv), make_dataset(args.val_csv)

    base_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="no",
        eval_steps=EVAL_STEPS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=True,
        logging_steps=EVAL_STEPS,
        metric_for_best_model="mae",
        greater_is_better=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        args=base_args,
        model_init=model_init,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    best = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=args.optuna_trials,
        sampler=sampler,
        pruner=pruner,
        compute_objective=lambda m: m["eval_mae"],
    )

    # ---------------- Final training with best hyper‑params ----------------
    final_args = base_args
    for k, v in best.hyperparameters.items():
        setattr(final_args, k, v)
    final_args.num_train_epochs = FINAL_EPOCHS  # longer run
    final_args.save_strategy = "epoch"

    trainer.args = final_args
    # trainer.create_model()
    trainer.train()
    trainer.save_model()
    trainer.save_metrics("train", {})
    print("Best trial:", best)


if __name__ == "__main__":
    main()
