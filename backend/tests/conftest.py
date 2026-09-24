import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (REPO_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml.inference.toxicity_classifier import LABELS, ToxicityResult  # noqa: E402


@pytest.fixture
def fake_classify(monkeypatch):
    """Stub out the real HF model so API tests stay gate-fast and deterministic."""
    from app.services import moderation_service

    def _fake(text, threshold=0.5):
        toxic_markers = ("kill", "idiot", "hate")
        is_toxic = any(marker in text.lower() for marker in toxic_markers)
        scores = {label: (0.9 if is_toxic and label == "toxic" else 0.01) for label in LABELS}
        flagged = ["toxic"] if is_toxic else []
        return ToxicityResult(text=text, scores=scores, flagged_labels=flagged, is_toxic=is_toxic)

    monkeypatch.setattr(moderation_service.en_classifier, "classify", _fake)
    return _fake


@pytest.fixture
def fake_shona_classify(monkeypatch):
    """Stub out the Shona model too, and make it report as available."""
    from app.services import moderation_service

    def _fake(text, threshold=0.5):
        # "uraya" = Shona root for "kill" — matches the marker convention used
        # in fake_classify's English stub, just in the other language.
        is_toxic = "uraya" in text.lower()
        scores = {label: (0.9 if is_toxic and label == "toxic" else 0.01) for label in LABELS}
        flagged = ["toxic"] if is_toxic else []
        return ToxicityResult(text=text, scores=scores, flagged_labels=flagged, is_toxic=is_toxic)

    monkeypatch.setattr(moderation_service.shona_toxicity_classifier, "classify", _fake)
    monkeypatch.setattr(moderation_service.shona_toxicity_classifier, "is_available", lambda: True)
    return _fake
