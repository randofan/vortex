"""
Usage:
python tune.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --hpo-output-dir /path/to/hpo_output \
  --optuna-trials 60 \
  --best-params-json /path/to/best_params.json
"""

import argparse
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model  # PEFT is used here for HPO model setup
from train import (
    DinoV2Coral,  # The model class
    make_dataset,  # Data loading and preprocessing
    compute_metrics,  # Metrics computation
    processor,  # The image processor instance
    EVAL_STEPS,  # Shared constant for evaluation steps
)

# ---------- HPO-specific Constants ----------
SEARCH_MIN_EPOCHS = 1  # Epochs per trial during HPO
PRUNER_MIN_STEPS = 500  # First ASHA rung


# ---------- Optuna Helpers ----------
def model_init_for_hpo(trial: optuna.Trial):
    """
    Initializes the model for an HPO trial.
    It instantiates DinoV2Coral from train.py and applies LoRA based on trial suggestions.
    """
    # 1. Sample LoRA hyperparameters
    r = trial.suggest_categorical("lora_r", [4, 8, 12])
    alpha_m = trial.suggest_categorical("alpha_mult", [1, 2, 4])
    drp = trial.suggest_float("lora_dropout", 0.0, 0.15)
    lora_alpha_value = r * alpha_m

    # 2. Instantiate the base model (imported from train.py)
    model = DinoV2Coral()

    # 3. Configure LoRA and apply PEFT
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha_value,
        lora_dropout=drp,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )
    model.base = get_peft_model(model.base, lora_config)
    # model.base.print_trainable_parameters() # Optional
    return model


def hp_space_for_trainer_args(trial: optuna.Trial):
    """Defines the hyperparameter search space for TrainingArguments during HPO."""
    return {
        "per_device_train_batch_size": trial.suggest_categorical("bs", [1, 2]),
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-4, log=True),
        "num_train_epochs": SEARCH_MIN_EPOCHS,  # Fixed for HPO trials
        "weight_decay": trial.suggest_float("wd", 1e-5, 1e-2, log=True),
    }


# Optuna sampler and pruner
sampler = TPESampler(multivariate=True, group=True)
pruner = SuccessiveHalvingPruner(
    min_resource=PRUNER_MIN_STEPS, reduction_factor=3, min_early_stopping_rate=0
)


# ---------- CLI for HPO ----------
def main():
    ap = argparse.ArgumentParser(
        description="Hyperparameter tuning script using components from train.py."
    )
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument(
        "--hpo-output-dir", required=True, help="Dir for Optuna study and trial logs."
    )
    ap.add_argument("--optuna-trials", type=int, default=60)
    ap.add_argument("--best-params-json", required=True, help="Path to JSON file to write best hyperparameters")
    args = ap.parse_args()

    if not processor:  # Check if processor imported from train.py is valid
        print(
            "Error: Image processor (imported from train.py) is not initialized. Exiting tune.py."
        )
        return

    train_ds = make_dataset(args.train_csv)  # Uses make_dataset from train.py
    val_ds = make_dataset(args.val_csv)  # Uses make_dataset from train.py

    hpo_training_args = TrainingArguments(
        output_dir=args.hpo_output_dir,  # Each trial might create a subfolder here if not careful
        eval_strategy="steps",
        save_strategy="no",  # Don't save models during HPO
        eval_steps=EVAL_STEPS,  # Constant from train.py
        per_device_eval_batch_size=2,  # Fixed eval batch size for HPO trials
        fp16=True,
        logging_steps=EVAL_STEPS,  # Constant from train.py
        metric_for_best_model="mae",
        greater_is_better=False,
        max_grad_norm=1.0,
        # Other args (lr, bs, epochs, wd) will be supplied by hp_space_for_trainer_args
        report_to="none",  # Prevent integration with wandb/tensorboard for each trial unless desired
    )

    trainer = Trainer(
        args=hpo_training_args,
        model_init=model_init_for_hpo,  # Key for Optuna: re-initializes model per trial
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,  # Uses compute_metrics from train.py
        tokenizer=processor,  # Uses processor imported from train.py
    )

    print(f"Starting hyperparameter search with {args.optuna_trials} trials...")
    best_trial_results = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=hp_space_for_trainer_args,  # Provides trainer args per trial
        n_trials=args.optuna_trials,
        sampler=sampler,
        pruner=pruner,
        compute_objective=lambda metrics: metrics["eval_mae"],
    )

    print("\n--- Best Hyperparameters Found by HPO ---")
    # The hyperparameters dictionary will contain keys from model_init_for_hpo (lora_r, etc.)
    # AND keys from hp_space_for_trainer_args (bs, lr, wd).
    print(json.dumps(best_trial_results.hyperparameters, indent=4))
    print(f"Best Objective (eval_mae): {best_trial_results.objective}")

    # Write best hyperparameters to JSON file
    with open(args.best_params_json, 'w') as f:
        json.dump(best_trial_results.hyperparameters, f, indent=4)
    print(f"Best hyperparameters saved to {args.best_params_json}")


if __name__ == "__main__":
    main()
