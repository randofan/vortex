"""
Generates a single-column CSV of predictions for a test set using Hugging Face Trainer.

Example
-------
python test.py \
    --checkpoint /path/to/final_output \
    --test-csv /path/to/test.csv \
    --out-csv /path/to/results/test_predictions_column.csv \
    --config /path/to/best_params.json \
    --batch 64
"""

import argparse
import os
import csv
import json
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

try:
    from train import (
        DinoV2Coral,
        coral_logits_to_label,
        make_dataset,  # Crucial for preprocessing
        processor,  # Used by make_dataset and Trainer
    )
except ImportError as e:
    print(f"Error importing from train.py: {e}")
    print("Ensure train.py is in the same directory or in PYTHONPATH.")
    exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Prediction script for DinoV2-CORAL model using Hugging Face Trainer (single column output)."
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Directory containing the saved model (pytorch_model.bin).",
    )
    ap.add_argument(
        "--test-csv", required=True, help="Path to the CSV file for testing."
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        help="Path to save the output CSV with a single prediction column.",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file with hyperparameters (for LoraConfig, matching train.py).",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for predictions (per_device_eval_batch_size).",
    )
    args = ap.parse_args()

    # Check if essential components from train.py were initialized
    if not processor:
        print("Error: Image processor from train.py is not initialized. Exiting.")
        return

    # Load hyperparameters from JSON file (needed for LoraConfig)
    try:
        with open(args.config, "r") as f:
            config_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config}")
        return

    # --- Model Loading ---
    print("Loading model and applying LoRA configuration...")
    model = DinoV2Coral()

    try:
        lora_r_val = config_params["lora_r"]
        lora_alpha_val = lora_r_val * config_params["alpha_mult"]
        lora_dropout_val = config_params["lora_dropout"]
    except KeyError as e:
        print(
            f"Error: Missing LoRA parameter {e} in config file {args.config}. Cannot reconstruct model."
        )
        return

    lora_config = LoraConfig(
        r=lora_r_val,
        lora_alpha=lora_alpha_val,
        lora_dropout=lora_dropout_val,
        bias="none",
        target_modules=["query", "key", "value", "dense"],  # Must match train.py
    )
    model.base = get_peft_model(model.base, lora_config)

    model_path = os.path.join(args.checkpoint, "pytorch_model.bin")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint {model_path} not found.")
        return

    try:
        # Load to CPU first, Trainer will handle device placement.
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Model state_dict loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    model.eval()  # Set model to evaluation mode

    # --- Data Preparation ---
    print(f"Loading and preprocessing test dataset from {args.test_csv}...")
    try:
        # make_dataset from train.py handles image loading, transformations.
        # Row order is preserved from the input CSV.
        test_ds = make_dataset(args.test_csv)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # --- Prediction using Hugging Face Trainer ---
    # output_dir for TrainingArguments is required, can be a temporary location.
    temp_output_dir = os.path.join(
        os.path.dirname(args.out_csv) or ".", "temp_trainer_output"
    )
    os.makedirs(temp_output_dir, exist_ok=True)

    # The DinoV2Coral in train.py loads base model with torch_dtype=torch.float16
    # and preprocess converts pixel_values to half. So fp16=True is appropriate.
    training_args = TrainingArguments(
        output_dir=temp_output_dir,
        per_device_eval_batch_size=args.batch,
        do_train=False,
        do_eval=False,
        do_predict=True,
        fp16=True,  # Consistent with model dtype in train.py
        report_to="none",  # Disable external reporting like wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        tokenizer=processor,  # Image processor acts as tokenizer for image models
        compute_metrics=None,  # Not needed for this simple prediction output
    )

    print("Generating predictions using Trainer.predict()...")
    try:
        prediction_output = trainer.predict(test_dataset=test_ds)
    except Exception as e:
        print(f"Error during Trainer.predict(): {e}")
        # Clean up temporary directory on error
        try:
            if os.path.exists(temp_output_dir):
                import shutil

                shutil.rmtree(temp_output_dir)
        except Exception as e_clean:
            print(
                f"Warning: Could not clean up temporary directory {temp_output_dir} on error: {e_clean}"
            )
        return

    # Extract logits from predictions
    logits = prediction_output.predictions

    # Convert logits to final predictions (actual "year offsets" or "levels")
    # coral_logits_to_label is imported from train.py
    all_predictions = coral_logits_to_label(torch.tensor(logits)).cpu().tolist() + 1600
    print(f"Total predictions generated: {len(all_predictions)}")

    # --- Save CSV ---
    print(f"Saving predictions to {args.out_csv}...")
    os.makedirs(
        os.path.dirname(args.out_csv) or ".", exist_ok=True
    )  # Ensure dir exists
    try:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prediction"])  # Single column header
            for pred_val in all_predictions:
                writer.writerow([pred_val])
        print(f"Predictions successfully written to {args.out_csv}")
    except IOError as e:
        print(f"Error writing CSV to {args.out_csv}: {e}")

    # --- Clean up temporary directory ---
    try:
        if os.path.exists(temp_output_dir):
            import shutil

            shutil.rmtree(temp_output_dir)
            print(f"Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {temp_output_dir}: {e}")


if __name__ == "__main__":
    main()
