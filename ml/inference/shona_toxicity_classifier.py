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

from ml.inference.toxicity_classifier import LABELS, DEFAULT_THRESHOLD, ToxicityResult

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
def classify(text: str, threshold: float = DEFAULT_THRESHOLD) -> ToxicityResult:
    if not isinstance(text, str) or not text.strip():
        return ToxicityResult(text=text or "", scores={label: 0.0 for label in LABELS})

    model, tokenizer = _load()
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits).tolist()

    scores = {label: float(prob) for label, prob in zip(LABELS, probs)}
    flagged = [label for label in LABELS if scores[label] >= threshold]

    return ToxicityResult(
        text=text,
        scores=scores,
        flagged_labels=flagged,
        is_toxic=len(flagged) > 0,
    )


def warm_up() -> None:
    _load()


def is_available() -> bool:
    return MODEL_DIR.exists()
