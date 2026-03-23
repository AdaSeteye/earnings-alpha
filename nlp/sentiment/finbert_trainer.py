"""
FinBERT fine-tuning on financial/earnings labelled data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_model_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/external/finbert_labels.csv", help="CSV with text,label")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default="models/finbert")
    args = parser.parse_args()
    cfg = load_model_config().get("finbert", {})
    epochs = args.epochs or cfg.get("fine_tune_epochs", 3)
    batch_size = args.batch_size or cfg.get("batch_size", 32)

    train_path = Path(args.train_data)
    if not train_path.exists():
        print(f"Train data not found: {train_path}. Using pretrained FinBERT only (no fine-tuning).")
        print("To fine-tune, create a CSV with columns 'text' and 'label' (positive/negative/neutral).")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        from datasets import load_dataset
        model_name = cfg.get("base_model", "ProsusAI/finbert")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        dataset = load_dataset("csv", data_files=str(train_path), split="train")
        if "label" not in dataset.column_names:
            print("CSV must have 'label' column (0=negative, 1=neutral, 2=positive).")
            return

        def tokenize(ex):
            return tokenizer(ex["text"], truncation=True, max_length=cfg.get("max_length", 256), padding="max_length")

        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        tokenized.set_format("torch")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=cfg.get("learning_rate", 2e-5),
            save_strategy="epoch",
            logging_steps=50,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
        trainer.train()
        trainer.save_model(Path(args.output_dir) / "finetuned")
        tokenizer.save_pretrained(Path(args.output_dir) / "finetuned")
        print(f"Saved fine-tuned model to {args.output_dir}/finetuned")
    except ImportError as e:
        print(f"Install transformers and datasets to fine-tune: {e}")


if __name__ == "__main__":
    main()
