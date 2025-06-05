import os
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
from peft import LoraConfig, get_peft_model  # PEFT is used here for final model setup
import evaluate
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss
from coral_pytorch.dataset import levels_from_labelbatch

# ---------- Shared Constants (used by components in this file) ----------
NUM_CLASSES = 300
BASE_YEAR = 1600
MODEL_NAME = "facebook/dinov2-base"
EVAL_STEPS = 200

# ---------- Train-specific Constants ----------
FINAL_EPOCHS = 4  # Full training epochs for the final run

# ---------- Global Initializations (for components defined in this file) ----------
# These are initialized here so they are available for functions in this module
# when train.py is imported or run.
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    RESIZE_SIZE = processor.size.get("shortest_edge", 224)
except Exception as e:
    print(
        f"Warning: Could not initialize AutoImageProcessor: {e}. Check MODEL_NAME and network."
    )
    processor = None
    RESIZE_SIZE = 224

try:
    mae_metric = evaluate.load("mae")
except Exception as e:
    print(
        f"Warning: Could not load MAE metric: {e}. Ensure 'evaluate' and 'scikit-learn' are installed."
    )
    mae_metric = None


# ---------- Model wrapper (original definition) ----------
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
        self.criterion = CoralLoss(reduction="mean")

    def forward(self, pixel_values, labels=None, attention=False):
        outs = self.base(pixel_values=pixel_values, output_attentions=True)
        rep = outs.last_hidden_state[:, 0]
        logits = self.head(rep)
        res = {"logits": logits}
        if labels is not None:
            levels = levels_from_labelbatch(labels, NUM_CLASSES).to(logits.device)
            loss = self.criterion(logits, levels)
            res["loss"] = loss
        if attention:
            res["attentions"] = outs.attentions
        return res


# ---------- Data pipeline (original definitions) ----------
def preprocess(example):
    # Accesses global 'processor', 'RESIZE_SIZE', 'BASE_YEAR' from this module
    if not processor:
        raise ValueError("Image processor is not initialized.")
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


# ---------- Metrics (original definitions) ----------
def coral_logits_to_label(logits):
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


def compute_metrics(p):
    # Accesses global 'mae_metric' from this module
    if not mae_metric:
        raise ValueError("MAE metric is not initialized.")
    logits, labels = p
    preds = coral_logits_to_label(torch.tensor(logits))
    return mae_metric.compute(predictions=preds.cpu().numpy(), references=labels)


# ---------- Main Training Function (for direct execution of train.py) ----------
def main():
    ap = argparse.ArgumentParser(
        description="Final training script for DinoV2-CORAL model."
    )
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lora_r", type=int, required=True)
    ap.add_argument("--alpha_mult", type=int, required=True)
    ap.add_argument("--lora_dropout", type=float, required=True)
    ap.add_argument("--bs", type=int, required=True, help="Batch size per device")
    ap.add_argument("--lr", type=float, required=True, help="Learning rate")
    ap.add_argument("--wd", type=float, required=True, help="Weight decay")
    args = ap.parse_args()

    if not processor or not mae_metric:
        print("Error: Processor or MAE metric not initialized. Exiting.")
        return

    train_ds, val_ds = make_dataset(args.train_csv), make_dataset(args.val_csv)

    model = DinoV2Coral()

    lora_alpha_value = args.lora_r * args.alpha_mult
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha_value,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )
    model.base = get_peft_model(model.base, lora_config)

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
        model=model,
        args=base_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    print("Starting final training with specified hyperparameters...")
    trainer.train()

    print(f"Saving best model to {args.final_output_dir}...")
    trainer.save_model()

    if trainer.state.best_metric:
        print(f"Best eval_mae achieved: {trainer.state.best_metric}")

    df = pd.DataFrame(trainer.state.log_history)
    log_csv_path = os.path.join(args.final_output_dir, "train_log.csv")
    df.to_csv(log_csv_path, index=False)

    plot_columns = [col for col in ["loss", "eval_loss"] if col in df.columns]
    if plot_columns:
        try:
            plot_df = df.dropna(subset=plot_columns).set_index(
                "step" if "step" in df.columns else df.index
            )
            plt.figure()
            plot_df[plot_columns].plot(
                title="Training/Evaluation Loss", xlabel="Step", ylabel="Loss"
            )
            loss_curve_path = os.path.join(
                args.final_output_dir, "train_loss_curve.png"
            )
            plt.savefig(loss_curve_path)
            plt.close()
            print(f"Training log and loss curve saved to {args.final_output_dir}")
        except Exception as e:
            print(f"Could not plot/save loss curve: {e}")


if __name__ == "__main__":
    main()
