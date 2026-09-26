"""Toxicity inference backed by a pretrained Hugging Face model.

Replaces the repo's LSTM baseline (data/processed/jigsaw_lstm.keras, macro F1
0.39 on the held-out Jigsaw test set, F1 0.0 on `threat`) with
unitary/toxic-bert, a BERT model fine-tuned on the same six Jigsaw labels.
See ml/evals/eval_toxicity_classifier.py for the head-to-head benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock

from transformers import pipeline

MODEL_NAME = "unitary/toxic-bert"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DEFAULT_THRESHOLD = 0.5

_load_lock = Lock()


@dataclass(frozen=True)
class ToxicityResult:
    text: str
    scores: dict = field(default_factory=dict)
    flagged_labels: list = field(default_factory=list)
    is_toxic: bool = False
    thresholds_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "scores": self.scores,
            "flagged_labels": self.flagged_labels,
            "is_toxic": self.is_toxic,
            "thresholds_used": self.thresholds_used,
        }


def _normalize_thresholds(threshold: float | dict[str, float]) -> dict[str, float]:
    if isinstance(threshold, (int, float)):
        return {label: float(threshold) for label in LABELS}
    if isinstance(threshold, dict):
        return {label: float(threshold.get(label, DEFAULT_THRESHOLD)) for label in LABELS}
    return {label: DEFAULT_THRESHOLD for label in LABELS}


@lru_cache(maxsize=1)
def _get_pipeline():
    # lru_cache alone isn't thread-safe against concurrent first calls under
    # FastAPI's threadpool; the lock makes the one-time load exclusive.
    with _load_lock:
        return pipeline("text-classification", model=MODEL_NAME, top_k=None)


def classify(text: str, threshold: float | dict[str, float] = DEFAULT_THRESHOLD) -> ToxicityResult:
    """Score a single comment across the six Jigsaw toxicity labels."""
    threshold_map = _normalize_thresholds(threshold)
    if not isinstance(text, str) or not text.strip():
        return ToxicityResult(
            text=text or "",
            scores={label: 0.0 for label in LABELS},
            thresholds_used=threshold_map,
        )

    pipe = _get_pipeline()
    raw = pipe(text, truncation=True, max_length=512)[0]
    scores = {item["label"]: float(item["score"]) for item in raw}
    # Guarantee every known label is present even if the model output order shifts.
    scores = {label: scores.get(label, 0.0) for label in LABELS}
    flagged = [label for label in LABELS if scores[label] >= threshold_map[label]]

    return ToxicityResult(
        text=text,
        scores=scores,
        flagged_labels=flagged,
        is_toxic=len(flagged) > 0,
        thresholds_used=threshold_map,
    )


def classify_batch(
    texts: list[str],
    threshold: float | dict[str, float] = DEFAULT_THRESHOLD,
    batch_size: int = 16,
) -> list[ToxicityResult]:
    """Score a batch of comments using Hugging Face pipeline batching."""
    if not texts:
        return []

    threshold_map = _normalize_thresholds(threshold)

    # Track indices of non-empty texts to batch efficiently
    valid_indices = []
    valid_texts = []
    for idx, t in enumerate(texts):
        if isinstance(t, str) and t.strip():
            valid_indices.append(idx)
            valid_texts.append(t)

    results: list[ToxicityResult | None] = [None] * len(texts)

    # Populate empty entries immediately
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
        pipe = _get_pipeline()
        raw_outputs = pipe(valid_texts, truncation=True, max_length=512, batch_size=batch_size)
        for orig_idx, raw in zip(valid_indices, raw_outputs):
            scores = {item["label"]: float(item["score"]) for item in raw}
            scores = {label: scores.get(label, 0.0) for label in LABELS}
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
    """Force the model to load. Call at process startup to avoid a slow first request."""
    _get_pipeline()


def get_pipeline():
    """Public accessor for the cached HF pipeline, for batched eval scripts."""
    return _get_pipeline()
