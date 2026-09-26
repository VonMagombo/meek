import pytest

from ml.inference.shona_toxicity_classifier import (
    LABELS,
    classify,
    classify_batch,
    is_available,
    warm_up,
)


def test_is_available_and_warm_up():
    assert is_available() is True
    # warm_up should execute without error
    warm_up()


def test_empty_and_whitespace_input():
    for empty in ["", "   ", "\n\t"]:
        res = classify(empty)
        assert res.is_toxic is False
        assert res.flagged_labels == []
        assert all(res.scores[label] == 0.0 for label in LABELS)


def test_benign_conversational_phrases_not_flagged():
    """Verify regression fix: benign words and greetings must NOT be flagged."""
    clean_phrases = [
        "Mhoro, wakadii zvako?",
        "Ndinofara kukuona nhasi.",
        "shamwari",
        "munhu",
        "akanaka",
        "Uri munhu anoshamisa, tinokutendai nokuda kwokundibetsera!",
        "Maita basa zvikuru.",
        "Tinotenda nerubatsiro rwenyu.",
        "Zvakanaka chose shamwari yangu.",
    ]
    for text in clean_phrases:
        res = classify(text)
        assert (
            res.is_toxic is False
        ), f"Benign phrase '{text}' was incorrectly flagged as toxic! Scores: {res.scores}"
        assert (
            res.flagged_labels == []
        ), f"Benign phrase '{text}' had flagged labels: {res.flagged_labels}"
        assert res.scores["toxic"] < 0.40, f"Toxic score too high for '{text}': {res.scores['toxic']}"
        assert res.scores["insult"] < 0.40, f"Insult score too high for '{text}': {res.scores['insult']}"


def test_toxic_slurs_and_insults_flagged():
    toxic_cases = [
        ("Ndichakuuraya iwe benzi.", {"toxic", "insult"}),
        ("uri imbwa", {"toxic", "insult"}),
        ("Iwe svira.", {"toxic", "insult"}),
        ("chifeve", {"toxic"}),
        ("mabeche", {"toxic", "obscene"}),
    ]
    for text, expected_labels in toxic_cases:
        res = classify(text)
        assert res.is_toxic is True, f"Toxic phrase '{text}' was not flagged! Scores: {res.scores}"
        for expected in expected_labels:
            assert (
                expected in res.flagged_labels
            ), f"Expected '{expected}' in flagged labels for '{text}', got {res.flagged_labels} (scores: {res.scores})"


def test_custom_thresholds():
    text = "Ndichakuuraya iwe benzi."
    # With normal threshold, toxic is flagged
    normal = classify(text, threshold=0.5)
    assert normal.is_toxic is True

    # With very high threshold, it should not be flagged
    high = classify(text, threshold=0.9999)
    assert high.is_toxic is False
    assert high.flagged_labels == []


def test_classify_batch_consistency():
    texts = [
        "Mhoro shamwari.",
        "Ndichakuuraya iwe benzi.",
        "",
        "Uri munhu akanaka chaizvo.",
    ]
    batch_results = classify_batch(texts)
    assert len(batch_results) == len(texts)

    for text, b_res in zip(texts, batch_results):
        single_res = classify(text)
        assert b_res.text == single_res.text
        assert b_res.is_toxic == single_res.is_toxic
        assert b_res.flagged_labels == single_res.flagged_labels
        for label in LABELS:
            assert abs(b_res.scores[label] - single_res.scores[label]) < 1e-5
