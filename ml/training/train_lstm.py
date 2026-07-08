import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.models.lstm_model import LABELS, build_model  # noqa: E402
from ml.preprocessing.text_cleaner import clean_series  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "jigsaw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 20000
MAX_LEN = 150
BATCH_SIZE = 128
EPOCHS = 5


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}_dataset.csv")
    df["comment_text"] = df["comment_text"].fillna("")
    return df


def main():
    print("Loading data...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    # The Kaggle test set ships with -1 placeholders for rows excluded from scoring.
    test_df = test_df[(test_df[LABELS] != -1).all(axis=1)].reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)} test(scored)={len(test_df)}")

    print("Cleaning text...")
    train_texts = clean_series(train_df["comment_text"])
    val_texts = clean_series(val_df["comment_text"])
    test_texts = clean_series(test_df["comment_text"])

    print("Tokenizing...")
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    def to_padded(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return keras.preprocessing.sequence.pad_sequences(
            seqs, maxlen=MAX_LEN, padding="post", truncating="post"
        )

    X_train, X_val, X_test = to_padded(train_texts), to_padded(val_texts), to_padded(test_texts)
    y_train = train_df[LABELS].values.astype("float32")
    y_val = val_df[LABELS].values.astype("float32")
    y_test = test_df[LABELS].values.astype("float32")

    print("Building model...")
    model = build_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=2, restore_best_weights=True
        ),
    ]

    print("Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    model_path = OUT_DIR / "jigsaw_lstm.keras"
    tokenizer_path = OUT_DIR / "tokenizer.pkl"
    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved model to {model_path}, tokenizer to {tokenizer_path}")

    print("Evaluating on held-out test set...")
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int)

    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {
            "precision": float(precision_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test[:, i], y_prob[:, i]))
            if len(np.unique(y_test[:, i])) > 1
            else None,
            "support": int(y_test[:, i].sum()),
        }

    overall = {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(y_test, y_prob, average="macro")),
        "micro_roc_auc": float(roc_auc_score(y_test, y_prob, average="micro")),
        "hamming_loss": float(hamming_loss(y_test, y_pred)),
        "subset_accuracy": float((y_pred == y_test).all(axis=1).mean()),
        "n_test": int(len(y_test)),
    }

    metrics = {"per_label": per_label, "overall": overall}
    metrics_path = OUT_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Per-label metrics ===")
    header = f"{'label':15s} {'precision':>9s} {'recall':>9s} {'f1':>9s} {'roc_auc':>9s} {'support':>8s}"
    print(header)
    for label, m in per_label.items():
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        print(
            f"{label:15s} {m['precision']:9.4f} {m['recall']:9.4f} {m['f1']:9.4f} {auc_str:>9s} {m['support']:8d}"
        )

    print("\n=== Overall ===")
    for k, v in overall.items():
        print(f"{k:20s} {v}")

    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
