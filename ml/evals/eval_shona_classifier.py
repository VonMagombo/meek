"""Eval for the fine-tuned Shona classifier. Three parts, all weaker evidence
than the English eval (ml/evals/eval_toxicity_classifier.py) — see README's
"Shona support" and "Known limitations" for why. Deliberately not blended
into one number: each has a different, non-overlapping evidence gap.

1. Silver-labeled held-out test split (data/processed/shona/test.csv): same
   methodology as training (machine-translated Jigsaw, original English
   labels). Measures whether the model learned the training signal at all,
   not whether it understands real Shona hate speech.
2. Native-labeled held-out test split (data/processed/shona/native_test.csv):
   real Shona, natively labeled, no translation involved — but small (36
   rows), lexicon/template-driven rather than organic text, and has zero
   `threat` examples. Never read this section's `threat` row as evidence;
   there's no data to measure it.
3. Golden set: the English golden examples (ml/evals/golden_examples.py)
   translated to Shona with the SAME MT model used to build the silver
   training data (Helsinki-NLP/opus-mt-en-sn), so this shares that model's
   translation biases with the training data rather than testing independently.
   Not reviewed by a native Shona speaker. Treat results as a directional sanity
   check only, not a validated accuracy claim.

Run: python ml/evals/eval_shona_classifier.py
Writes: data/processed/shona_classifier/eval_report.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.evals.golden_examples import EXAMPLES as EN_EXAMPLES  # noqa: E402
from ml.evals.golden_examples import LABELS  # noqa: E402
from ml.inference.shona_toxicity_classifier import classify, is_available  # noqa: E402
from ml.translation.translate_jigsaw import translate_batch  # noqa: E402

DATA_DIR = ROOT / "data" / "processed" / "shona"
OUT_DIR = ROOT / "data" / "processed" / "shona_classifier"
TEST_PATH = DATA_DIR / "test.csv"
NATIVE_TEST_PATH = DATA_DIR / "native_test.csv"


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_label = {
        label: {
            "precision": float(precision_score(y_true[:, j], y_pred[:, j], zero_division=0)),
            "recall": float(recall_score(y_true[:, j], y_pred[:, j], zero_division=0)),
            "f1": float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0)),
            "support": int(y_true[:, j].sum()),
        }
        for j, label in enumerate(LABELS)
    }
    tested = [m["f1"] for m in per_label.values() if m["support"] > 0]
    return {
        "macro_f1": float(np.mean(tested)) if tested else None,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label": per_label,
    }


def _run_test_split(path: Path, missing_hint: str) -> dict:
    if not path.exists():
        return {"skipped": f"{path} not found — {missing_hint}"}
    df = pd.read_csv(path)
    y_true = df[LABELS].values.astype(int)
    y_pred = np.zeros_like(y_true)
    for i, text in enumerate(df["comment_text_sn"].fillna("")):
        result = classify(text)
        for j, label in enumerate(LABELS):
            y_pred[i, j] = label in result.flagged_labels
    return {"n_examples": len(df), **score(y_true, y_pred)}


def run_silver_test() -> dict:
    return _run_test_split(TEST_PATH, "run translate_jigsaw.py first")


def run_native_test() -> dict:
    return _run_test_split(NATIVE_TEST_PATH, "run build_shona_training_data.py first")


def run_golden_set() -> dict:
    texts_en = [ex["text"] for ex in EN_EXAMPLES]
    texts_sn = translate_batch(texts_en)

    y_true = np.zeros((len(EN_EXAMPLES), len(LABELS)))
    y_pred = np.zeros((len(EN_EXAMPLES), len(LABELS)))
    examples = []
    for i, (ex, text_sn) in enumerate(zip(EN_EXAMPLES, texts_sn)):
        result = classify(text_sn)
        for j, label in enumerate(LABELS):
            y_true[i, j] = label in ex["expected"]
            y_pred[i, j] = label in result.flagged_labels
        examples.append(
            {
                "text_en": ex["text"],
                "text_sn": text_sn,
                "expected": sorted(ex["expected"]),
                "predicted": sorted(result.flagged_labels),
                "known_limitation": ex.get("known_limitation", False),
            }
        )

    gating_true = np.array([y_true[i] for i, ex in enumerate(examples) if not ex["known_limitation"]])
    gating_pred = np.array([y_pred[i] for i, ex in enumerate(examples) if not ex["known_limitation"]])

    return {
        "n_examples": len(EN_EXAMPLES),
        **score(gating_true, gating_pred),
        "examples": examples,
    }


def main():
    if not is_available():
        print("No fine-tuned Shona model found. Run:")
        print("  python ml/translation/translate_jigsaw.py")
        print("  python ml/training/train_shona_classifier.py")
        sys.exit(1)

    print("=== Silver-labeled held-out test (data/processed/shona/test.csv) ===")
    silver = run_silver_test()
    if "skipped" in silver:
        print(f"  skipped: {silver['skipped']}")
    else:
        print(f"  n={silver['n_examples']}  macro_f1={silver['macro_f1']}  micro_f1={silver['micro_f1']:.4f}")
        for label, m in silver["per_label"].items():
            print(f"    {label:15s} f1={m['f1']:.3f} (support={m['support']})")

    print("\n=== Native-labeled held-out test (data/processed/shona/native_test.csv) ===")
    print("    (real Shona, no threat examples — ignore the threat row below)")
    native = run_native_test()
    if "skipped" in native:
        print(f"  skipped: {native['skipped']}")
    else:
        print(f"  n={native['n_examples']}  macro_f1={native['macro_f1']}  micro_f1={native['micro_f1']:.4f}")
        for label, m in native["per_label"].items():
            print(f"    {label:15s} f1={m['f1']:.3f} (support={m['support']})")

    print("\n=== Golden set, translated from English (directional sanity check only) ===")
    golden = run_golden_set()
    print(f"  n={golden['n_examples']}  macro_f1={golden['macro_f1']}  micro_f1={golden['micro_f1']:.4f}")
    for ex in golden["examples"]:
        marker = "OK" if set(ex["expected"]) == set(ex["predicted"]) else "MISS"
        print(f"  [{marker}] {ex['text_sn'][:55]!r} expected={ex['expected']} predicted={ex['predicted']}")

    report = {"silver_test": silver, "native_test": native, "golden_set": golden}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
