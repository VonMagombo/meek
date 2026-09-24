"""Fine-tune a Shona-language multi-label toxicity classifier.

Base model: Davlan/xlm-roberta-base-finetuned-shona — an XLM-RoBERTa encoder
further pretrained (masked-language-modeling) on Shona text by the Masakhane
project. It has no classification head yet; this script adds one and trains
it for the same six Jigsaw labels as the English classifier
(ml/inference/toxicity_classifier.py), using the silver-labeled Shona data
built by ml/translation/translate_jigsaw.py (machine-translated Jigsaw
comments, original English human labels preserved).

Run: python ml/training/train_shona_classifier.py [--epochs N] [--batch-size N]
Requires: data/processed/shona/{train,val}.csv (run translate_jigsaw.py first)
Saves: data/processed/shona_classifier/ (HF model dir, loadable with
       AutoModelForSequenceClassification.from_pretrained)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE_MODEL = "Davlan/xlm-roberta-base-finetuned-shona"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DATA_DIR = ROOT / "data" / "processed" / "shona"
OUT_DIR = ROOT / "data" / "processed" / "shona_classifier"
MAX_LEN = 128


class ShonaToxicityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df["comment_text_sn"].fillna("").tolist()
        self.labels = df[LABELS].values.astype("float32")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[f"{label}_f1"] = float(f1_score(labels[:, i], preds[:, i], zero_division=0))
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        **per_label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    if not train_path.exists() or not val_path.exists():
        print(f"Missing {train_path} or {val_path}. Run ml/translation/translate_jigsaw.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"train={len(train_df)} val={len(val_df)}")

    print(f"Loading base model {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    train_dataset = ShonaToxicityDataset(train_df, tokenizer)
    val_dataset = ShonaToxicityDataset(val_df, tokenizer)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Final validation metrics:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    final_dir = OUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with open(OUT_DIR / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
