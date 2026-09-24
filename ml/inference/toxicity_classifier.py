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

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "scores": self.scores,
            "flagged_labels": self.flagged_labels,
            "is_toxic": self.is_toxic,
        }


@lru_cache(maxsize=1)
def _get_pipeline():
    # lru_cache alone isn't thread-safe against concurrent first calls under
    # FastAPI's threadpool; the lock makes the one-time load exclusive.
    with _load_lock:
        return pipeline("text-classification", model=MODEL_NAME, top_k=None)


def classify(text: str, threshold: float = DEFAULT_THRESHOLD) -> ToxicityResult:
    """Score a single comment across the six Jigsaw toxicity labels."""
    if not isinstance(text, str) or not text.strip():
        return ToxicityResult(text=text or "", scores={label: 0.0 for label in LABELS})

    pipe = _get_pipeline()
    raw = pipe(text, truncation=True, max_length=512)[0]
    scores = {item["label"]: float(item["score"]) for item in raw}
    # Guarantee every known label is present even if the model output order shifts.
    scores = {label: scores.get(label, 0.0) for label in LABELS}
    flagged = [label for label in LABELS if scores[label] >= threshold]

    return ToxicityResult(
        text=text,
        scores=scores,
        flagged_labels=flagged,
        is_toxic=len(flagged) > 0,
    )


def classify_batch(texts: list, threshold: float = DEFAULT_THRESHOLD) -> list:
    return [classify(text, threshold=threshold) for text in texts]


def warm_up() -> None:
    """Force the model to load. Call at process startup to avoid a slow first request."""
    _get_pipeline()


def get_pipeline():
    """Public accessor for the cached HF pipeline, for batched eval scripts."""
    return _get_pipeline()
