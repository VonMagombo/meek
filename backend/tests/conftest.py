import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (REPO_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml.inference.toxicity_classifier import LABELS, ToxicityResult  # noqa: E402


def _normalize_thresh(threshold):
    if isinstance(threshold, dict):
        return {label: float(threshold.get(label, 0.5)) for label in LABELS}
    return {label: float(threshold) for label in LABELS}


@pytest.fixture
def fake_classify(monkeypatch):
    """Stub out the real HF model so API tests stay gate-fast and deterministic."""
    from app.services import moderation_service

    def _fake(text, threshold=0.5):
        toxic_markers = ("kill", "idiot", "hate")
        is_toxic = any(marker in text.lower() for marker in toxic_markers)
        scores = {label: (0.9 if is_toxic and label == "toxic" else 0.01) for label in LABELS}
        t_map = _normalize_thresh(threshold)
        flagged = [label for label in LABELS if scores[label] >= t_map[label]]
        return ToxicityResult(
            text=text,
            scores=scores,
            flagged_labels=flagged,
            is_toxic=len(flagged) > 0,
            thresholds_used=t_map,
        )

    def _fake_batch(texts, threshold=0.5, batch_size=16):
        return [_fake(t, threshold=threshold) for t in texts]

    monkeypatch.setattr(moderation_service.en_classifier, "classify", _fake)
    monkeypatch.setattr(moderation_service.en_classifier, "classify_batch", _fake_batch)
    return _fake


@pytest.fixture
def fake_shona_classify(monkeypatch):
    """Stub out the Shona model too, and make it report as available."""
    from app.services import moderation_service

    def _fake(text, threshold=0.5):
        is_toxic = "uraya" in text.lower()
        scores = {label: (0.9 if is_toxic and label == "toxic" else 0.01) for label in LABELS}
        t_map = _normalize_thresh(threshold)
        flagged = [label for label in LABELS if scores[label] >= t_map[label]]
        return ToxicityResult(
            text=text,
            scores=scores,
            flagged_labels=flagged,
            is_toxic=len(flagged) > 0,
            thresholds_used=t_map,
        )

    def _fake_batch(texts, threshold=0.5, batch_size=16):
        return [_fake(t, threshold=threshold) for t in texts]

    monkeypatch.setattr(moderation_service.shona_toxicity_classifier, "classify", _fake)
    monkeypatch.setattr(moderation_service.shona_toxicity_classifier, "classify_batch", _fake_batch)
    monkeypatch.setattr(moderation_service.shona_toxicity_classifier, "is_available", lambda: True)
    return _fake

