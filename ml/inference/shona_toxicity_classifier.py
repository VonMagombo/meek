"""Shona-language toxicity inference: the fine-tuned classifier from
ml/training/train_shona_classifier.py (XLM-R fine-tuned on machine-translated,
silver-labeled Jigsaw data — see that script's docstring and README's "Known
limitations" for what that does and doesn't guarantee).

Same interface as ml/inference/toxicity_classifier.py so the backend can treat
English and Shona requests uniformly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.inference.toxicity_classifier import (
    DEFAULT_THRESHOLD,
    LABELS,
    ToxicityResult,
    _normalize_thresholds,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "processed" / "shona_classifier" / "final"
BASE_MODEL = "Davlan/xlm-roberta-base-finetuned-shona"

_load_lock = Lock()


class ModelNotFineTunedError(RuntimeError):
    """Raised when no fine-tuned checkpoint exists yet at MODEL_DIR."""


@lru_cache(maxsize=1)
def _load():
    with _load_lock:
        if not MODEL_DIR.exists():
            raise ModelNotFineTunedError(
                f"No fine-tuned Shona model at {MODEL_DIR}. Run "
                "ml/translation/translate_jigsaw.py then "
                "ml/training/train_shona_classifier.py first."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.eval()
        return model, tokenizer


@torch.inference_mode()
def classify(text: str, threshold: float | dict[str, float] = DEFAULT_THRESHOLD) -> ToxicityResult:
    threshold_map = _normalize_thresholds(threshold)
    if not isinstance(text, str) or not text.strip():
        return ToxicityResult(
            text=text or "",
            scores={label: 0.0 for label in LABELS},
            thresholds_used=threshold_map,
        )

    model, tokenizer = _load()
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits).tolist()

    scores = {label: float(prob) for label, prob in zip(LABELS, probs)}
    flagged = [label for label in LABELS if scores[label] >= threshold_map[label]]

    return ToxicityResult(
        text=text,
        scores=scores,
        flagged_labels=flagged,
        is_toxic=len(flagged) > 0,
        thresholds_used=threshold_map,
    )


@torch.inference_mode()
def classify_batch(
    texts: list[str],
    threshold: float | dict[str, float] = DEFAULT_THRESHOLD,
    batch_size: int = 16,
) -> list[ToxicityResult]:
    """Score a batch of comments using batched PyTorch tensor forward passes."""
    if not texts:
        return []

    threshold_map = _normalize_thresholds(threshold)

    valid_indices = []
    valid_texts = []
    for idx, t in enumerate(texts):
        if isinstance(t, str) and t.strip():
            valid_indices.append(idx)
            valid_texts.append(t)

    results: list[ToxicityResult | None] = [None] * len(texts)

    for idx, t in enumerate(texts):
        if idx not in valid_indices:
            results[idx] = ToxicityResult(
                text=t or "" if isinstance(t, str) else "",
                scores={label: 0.0 for label in LABELS},
                flagged_labels=[],
                is_toxic=False,
                thresholds_used=threshold_map,
            )

    if valid_texts:
        model, tokenizer = _load()
        for chunk_start in range(0, len(valid_texts), batch_size):
            chunk_texts = valid_texts[chunk_start : chunk_start + batch_size]
            chunk_indices = valid_indices[chunk_start : chunk_start + batch_size]

            inputs = tokenizer(
                chunk_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).tolist()

            for orig_idx, item_probs in zip(chunk_indices, probs):
                scores = {label: float(p) for label, p in zip(LABELS, item_probs)}
                flagged = [label for label in LABELS if scores[label] >= threshold_map[label]]
                results[orig_idx] = ToxicityResult(
                    text=texts[orig_idx],
                    scores=scores,
                    flagged_labels=flagged,
                    is_toxic=len(flagged) > 0,
                    thresholds_used=threshold_map,
                )

    return [r for r in results if r is not None]



def warm_up() -> None:
    _load()


def is_available() -> bool:
    return MODEL_DIR.exists()
