"""Periodic eval for the HF toxicity classifier (not a gate test — hits the
real model). Two parts:

Runtime note: the benchmark sample includes some very long Wikipedia talk-page
comments truncated to 512 tokens. Batch throughput on those is CPU-bound and
much slower than live single-comment API latency (~150ms warm, see
ml/inference/toxicity_classifier.py) — expect roughly 1-2 examples/sec on a
4-thread CPU, so the default sample size keeps a full run under ~10 minutes.

1. Golden set: hand-written examples (ml/evals/golden_examples.py) scored for
   exact flagged-label-set match and per-label precision/recall/F1.
2. Benchmark: a random sample of the same held-out Jigsaw test rows the LSTM
   baseline was scored on (data/processed/eval_metrics.json), so the two
   models are compared on identical inputs.

Run: python ml/evals/eval_toxicity_classifier.py [--sample-size N] [--seed S]
Writes: data/processed/hf_toxicity_eval.json
Exits non-zero if either pass threshold is missed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.evals.golden_examples import EXAMPLES, LABELS  # noqa: E402
from ml.inference.toxicity_classifier import DEFAULT_THRESHOLD, MODEL_NAME, get_pipeline  # noqa: E402

DATA_DIR = ROOT / "data" / "raw" / "jigsaw"
OUT_DIR = ROOT / "data" / "processed"
BASELINE_PATH = OUT_DIR / "eval_metrics.json"
OUT_PATH = OUT_DIR / "hf_toxicity_eval.json"

GOLDEN_EXACT_MATCH_THRESHOLD = 0.75
GOLDEN_LABEL_F1_THRESHOLD = 0.75


def run_golden_eval(threshold: float) -> dict:
    pipe = get_pipeline()
    texts = [ex["text"] for ex in EXAMPLES]
    raw = pipe(texts, truncation=True, max_length=512, batch_size=16)

    per_example = []
    for ex, scores in zip(EXAMPLES, raw):
        score_map = {item["label"]: item["score"] for item in scores}
        predicted = sorted(label for label in LABELS if score_map.get(label, 0.0) >= threshold)
        expected = sorted(ex["expected"])
        per_example.append(
            {
                "text": ex["text"],
                "expected": expected,
                "predicted": predicted,
                "match": predicted == expected,
                "known_limitation": ex.get("known_limitation", False),
                "limitation_note": ex.get("limitation_note"),
            }
        )

    def score(examples: list) -> dict:
        if not examples:
            return {"n_examples": 0, "exact_match_rate": None, "macro_f1": None, "per_label": {}}
        y_true = np.zeros((len(examples), len(LABELS)))
        y_pred = np.zeros((len(examples), len(LABELS)))
        for i, ex in enumerate(examples):
            for j, label in enumerate(LABELS):
                y_true[i, j] = label in ex["expected"]
                y_pred[i, j] = label in ex["predicted"]
        per_label = {
            label: {
                "precision": float(precision_score(y_true[:, j], y_pred[:, j], zero_division=0)),
                "recall": float(recall_score(y_true[:, j], y_pred[:, j], zero_division=0)),
                "f1": float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0)),
                "support": int(y_true[:, j].sum()),
            }
            for j, label in enumerate(LABELS)
        }
        # Average f1 only over labels this example set actually exercises (support > 0).
        # A label with zero positive examples has no signal either way; scoring it as a
        # flat 0 would silently drag macro_f1 down without testing anything.
        tested_f1s = [m["f1"] for m in per_label.values() if m["support"] > 0]
        macro_f1 = float(np.mean(tested_f1s)) if tested_f1s else None
        return {
            "n_examples": len(examples),
            "exact_match_rate": sum(ex["match"] for ex in examples) / len(examples),
            "macro_f1": macro_f1,
            "per_label": per_label,
        }

    gating = [ex for ex in per_example if not ex["known_limitation"]]
    known_limitations = [ex for ex in per_example if ex["known_limitation"]]

    return {
        "gating": score(gating),
        "known_limitations": score(known_limitations),
        "examples": per_example,
    }


def run_benchmark(threshold: float, sample_size: int, seed: int) -> dict:
    test_df = pd.read_csv(DATA_DIR / "test_dataset.csv")
    test_df = test_df[(test_df[LABELS] != -1).all(axis=1)].reset_index(drop=True)
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=seed).reset_index(drop=True)
    texts = sample["comment_text"].tolist()
    y_test = sample[LABELS].values.astype("float32")

    pipe = get_pipeline()
    start = time.time()
    raw = pipe(texts, truncation=True, max_length=512, batch_size=32)
    elapsed = time.time() - start

    y_prob = np.zeros((len(texts), len(LABELS)), dtype="float32")
    for i, scores in enumerate(raw):
        score_map = {item["label"]: item["score"] for item in scores}
        for j, label in enumerate(LABELS):
            y_prob[i, j] = score_map.get(label, 0.0)
    y_pred = (y_prob >= threshold).astype(int)

    per_label = {}
    for j, label in enumerate(LABELS):
        per_label[label] = {
            "precision": float(precision_score(y_test[:, j], y_pred[:, j], zero_division=0)),
            "recall": float(recall_score(y_test[:, j], y_pred[:, j], zero_division=0)),
            "f1": float(f1_score(y_test[:, j], y_pred[:, j], zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test[:, j], y_prob[:, j]))
            if len(np.unique(y_test[:, j])) > 1
            else None,
            "support": int(y_test[:, j].sum()),
        }

    overall = {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(y_test, y_prob, average="macro")),
        "micro_roc_auc": float(roc_auc_score(y_test, y_prob, average="micro")),
        "hamming_loss": float(hamming_loss(y_test, y_pred)),
        "subset_accuracy": float((y_pred == y_test).all(axis=1).mean()),
        "n_test": int(len(y_test)),
        "inference_seconds": round(elapsed, 2),
        "examples_per_second": round(len(texts) / elapsed, 1),
    }

    return {"per_label": per_label, "overall": overall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME}...")
    get_pipeline()

    print("\n=== Golden set eval (gating) ===")
    golden = run_golden_eval(args.threshold)
    gating = golden["gating"]
    print(f"exact_match_rate: {gating['exact_match_rate']:.3f} ({gating['n_examples']} examples)")
    macro_f1_str = f"{gating['macro_f1']:.4f}" if gating["macro_f1"] is not None else "n/a"
    print(f"macro_f1: {macro_f1_str}")
    for label, m in gating["per_label"].items():
        support_note = "" if m["support"] > 0 else "  (no positive examples in gating set)"
        print(
            f"  {label:15s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"f1={m['f1']:.3f}{support_note}"
        )
    for ex in golden["examples"]:
        if not ex["match"] and not ex["known_limitation"]:
            print(f"  MISMATCH: {ex['text'][:60]!r} expected={ex['expected']} predicted={ex['predicted']}")

    known = golden["known_limitations"]
    if known["n_examples"]:
        print(f"\n=== Known limitations (n={known['n_examples']}, excluded from gate) ===")
        for ex in golden["examples"]:
            if ex["known_limitation"]:
                status = "still fails as documented" if not ex["match"] else "now PASSES — update golden_examples.py"
                print(f"  {ex['text'][:65]!r}: {status}")
                print(f"    {ex['limitation_note']}")

    print(f"\n=== Benchmark vs LSTM baseline (n={args.sample_size} sampled from held-out Jigsaw test set) ===")
    benchmark = run_benchmark(args.threshold, args.sample_size, args.seed)
    print(f"HF macro_f1:  {benchmark['overall']['macro_f1']:.4f}")
    print(f"HF micro_f1:  {benchmark['overall']['micro_f1']:.4f}")
    print(f"HF speed:     {benchmark['overall']['examples_per_second']} examples/sec on CPU")

    baseline = None
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())
        print(f"\nLSTM macro_f1 (full test set, n={baseline['overall']['n_test']}): {baseline['overall']['macro_f1']:.4f}")
        print(f"LSTM micro_f1: {baseline['overall']['micro_f1']:.4f}")
        print("\n  label           HF f1    LSTM f1   delta")
        for label in LABELS:
            hf_f1 = benchmark["per_label"][label]["f1"]
            lstm_f1 = baseline["per_label"][label]["f1"]
            print(f"  {label:15s} {hf_f1:.4f}   {lstm_f1:.4f}    {hf_f1 - lstm_f1:+.4f}")

    result = {
        "model": MODEL_NAME,
        "threshold": args.threshold,
        "golden_set": golden,
        "benchmark_sample": benchmark,
        "lstm_baseline_reference": str(BASELINE_PATH) if baseline else None,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved eval report to {OUT_PATH}")

    passed = True
    if gating["exact_match_rate"] < GOLDEN_EXACT_MATCH_THRESHOLD:
        print(
            f"\nFAIL: golden exact_match_rate {gating['exact_match_rate']:.3f} "
            f"< threshold {GOLDEN_EXACT_MATCH_THRESHOLD}"
        )
        passed = False
    if gating["macro_f1"] is None or gating["macro_f1"] < GOLDEN_LABEL_F1_THRESHOLD:
        print(f"FAIL: golden macro_f1 {gating['macro_f1']} < threshold {GOLDEN_LABEL_F1_THRESHOLD}")
        passed = False
    if baseline and benchmark["overall"]["macro_f1"] <= baseline["overall"]["macro_f1"]:
        print("FAIL: HF benchmark macro_f1 did not beat the LSTM baseline")
        passed = False

    if not passed:
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
