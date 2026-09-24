import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(fake_classify):
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models"]["en"] == "unitary/toxic-bert"
    assert "sn" in body["models"]


def test_moderate_flags_toxic_text(client):
    resp = client.post("/api/v1/moderate", json={"text": "I will kill you, idiot."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "en"
    assert body["is_toxic"] is True
    assert "toxic" in body["flagged_labels"]
    assert set(body["scores"].keys()) == {
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }


def test_moderate_clean_text(client):
    resp = client.post("/api/v1/moderate", json={"text": "Thanks for the thoughtful review!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_toxic"] is False
    assert body["flagged_labels"] == []


def test_moderate_defaults_to_english(client):
    resp = client.post("/api/v1/moderate", json={"text": "hello"})
    assert resp.status_code == 200
    assert resp.json()["language"] == "en"


def test_moderate_rejects_empty_text(client):
    resp = client.post("/api/v1/moderate", json={"text": ""})
    assert resp.status_code == 422


def test_moderate_rejects_missing_field(client):
    resp = client.post("/api/v1/moderate", json={})
    assert resp.status_code == 422


def test_moderate_rejects_oversized_text(client):
    resp = client.post("/api/v1/moderate", json={"text": "a" * 5001})
    assert resp.status_code == 422


def test_moderate_rejects_unknown_language(client):
    resp = client.post("/api/v1/moderate", json={"text": "hello", "language": "fr"})
    assert resp.status_code == 422


def test_frontend_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "meek" in resp.text.lower()


def test_moderate_shona_unavailable_returns_503(client, monkeypatch):
    """When no fine-tuned Shona checkpoint exists, the API should fail clearly, not 500 —
    forced deterministic regardless of whether this machine has actually run training."""
    from app.services import moderation_service

    monkeypatch.setattr(moderation_service.shona_toxicity_classifier, "is_available", lambda: False)

    def _raise(text, threshold=0.5):
        raise moderation_service.ShonaModelUnavailableError("not fine-tuned")

    monkeypatch.setattr(moderation_service.shona_toxicity_classifier, "classify", _raise)

    resp = client.post("/api/v1/moderate", json={"text": "mhoro", "language": "sn"})
    assert resp.status_code == 503


def test_moderate_shona_when_available(client, fake_shona_classify):
    resp = client.post("/api/v1/moderate", json={"text": "Ndichakuuraya", "language": "sn"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "sn"
    assert body["is_toxic"] is True
