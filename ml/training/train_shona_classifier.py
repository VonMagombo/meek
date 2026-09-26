"""Train a Shona-language multi-label toxicity classifier.

Supports two training modes:
1. 'calibrated' (DEFAULT, highly recommended for CPU / low-memory hardware):
   Fast, memory-efficient (<10 MB RAM) calibrated linear ensemble (character and
   word n-gram FeatureUnion + CalibratedClassifierCV on LogisticRegression).
   Trains in seconds, eliminates memory exhaustion / OOM crashes, and prevents
   prior collapse / false positive spikes on benign Shona conversational vocabulary.
   Saves: data/processed/shona_classifier/calibrated_model.joblib

2. 'transformer' (intended for GPU / cloud training environments):
   Fine-tunes Davlan/xlm-roberta-base-finetuned-shona via PyTorch / HuggingFace Trainer.
   WARNING: Requires 10+ GB RAM when training with AdamW; do not run on CPU-only machines.
   Saves: data/processed/shona_classifier/final/

Run: python ml/training/train_shona_classifier.py [--mode calibrated|transformer]
Requires: data/processed/shona/{train,val}.csv (run build_shona_training_data.py first)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE_MODEL = "Davlan/xlm-roberta-base-finetuned-shona"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DATA_DIR = ROOT / "data" / "processed" / "shona"
OUT_DIR = ROOT / "data" / "processed" / "shona_classifier"
MAX_LEN = 128


def train_calibrated_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train a lightweight, calibrated multi-label classifier."""
    print("Training calibrated multi-label classifier (FeatureUnion + CalibratedClassifierCV)...")
    X_train = train_df["comment_text_sn"].fillna("")
    X_val = val_df["comment_text_sn"].fillna("")

    word_vec = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=3, sublinear_tf=True)
    union = FeatureUnion([("word", word_vec), ("char", char_vec)])

    print("Extracting word and character n-gram features...")
    X_tr_vec = union.fit_transform(X_train)
    X_val_vec = union.transform(X_val)

    models = {}
    per_label = {}
    preds_val = np.zeros((len(val_df), len(LABELS)))
    y_val_all = val_df[LABELS].values.astype(int)

    for idx, label in enumerate(LABELS):
        y_tr = train_df[label].values.astype(int)
        y_v = val_df[label].values.astype(int)

        base_clf = LogisticRegression(C=2.5, max_iter=300, class_weight="balanced")
        cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        cal_clf.fit(X_tr_vec, y_tr)
        models[label] = cal_clf

        probs = cal_clf.predict_proba(X_val_vec)[:, 1]
        preds = (probs >= 0.5).astype(int)
        preds_val[:, idx] = preds

        prec = float(precision_score(y_v, preds, zero_division=0))
        rec = float(recall_score(y_v, preds, zero_division=0))
        f1 = float(f1_score(y_v, preds, zero_division=0))
        auc = float(roc_auc_score(y_v, probs)) if len(np.unique(y_v)) > 1 else 0.0

        per_label[f"{label}_precision"] = prec
        per_label[f"{label}_recall"] = rec
        per_label[f"{label}_f1"] = f1
        per_label[f"{label}_auc"] = auc
        print(f"  {label:15s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")

    macro_f1 = float(f1_score(y_val_all, preds_val, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_val_all, preds_val, average="micro", zero_division=0))

    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_label": per_label,
    }
    print(f"\nValidation Summary: Macro F1={macro_f1:.3f}, Micro F1={micro_f1:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = OUT_DIR / "calibrated_model.joblib"
    bundle = {
        "vectorizer": union,
        "models": models,
        "labels": LABELS,
    }
    joblib.dump(bundle, bundle_path)
    print(f"Saved calibrated model to {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")

    metrics_path = OUT_DIR / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved train metrics to {metrics_path}")

    return metrics


def train_transformer_model(train_df: pd.DataFrame, val_df: pd.DataFrame, args) -> dict:
    """Train transformer on GPU or high-RAM machine."""
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

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
            )
            item = {k: torch.tensor(v) for k, v in enc.items()}
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
        weight_decay=0.01,
        warmup_ratio=0.1,
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
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Training transformer...")
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
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Shona toxicity classifier")
    parser.add_argument(
        "--mode",
        choices=["calibrated", "transformer"],
        default="calibrated",
        help="Training mode: 'calibrated' (fast, lightweight, CPU-safe) or 'transformer' (GPU fine-tuning)",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    if not train_path.exists() or not val_path.exists():
        print(f"Missing {train_path} or {val_path}. Run ml/preprocessing/build_shona_training_data.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"Loaded datasets: train={len(train_df)}, val={len(val_df)}")

    if args.mode == "calibrated":
        train_calibrated_model(train_df, val_df)
    else:
        train_transformer_model(train_df, val_df, args)


if __name__ == "__main__":
    main()
