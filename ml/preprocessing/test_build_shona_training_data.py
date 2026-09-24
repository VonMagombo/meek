from ml.preprocessing.build_shona_training_data import LABELS, _row_labels


def _labels_set(category, severity, is_toxic):
    result = _row_labels(category, severity, is_toxic)
    return {label for label in LABELS if result[label] == 1}


def test_non_toxic_gate_overrides_category():
    # Even a category with a mapping rule must come back all-zero when the
    # native label says non-toxic (e.g. literal_or_metalinguistic controls).
    assert _labels_set("insult", severity=2, is_toxic=False) == set()
    assert _labels_set("literal_or_metalinguistic", severity=0, is_toxic=False) == set()


def test_plain_insult_categories():
    for category in ["insult", "insult_general", "character_attack", "directed_abuse",
                      "accusation", "animal", "ableist", "political_incitement"]:
        assert _labels_set(category, severity=2, is_toxic=True) == {"toxic", "insult"}


def test_profanity_and_sexual_vulgar_map_to_obscene_not_insult():
    assert _labels_set("profanity", severity=2, is_toxic=True) == {"toxic", "obscene"}
    assert _labels_set("sexual_vulgar", severity=2, is_toxic=True) == {"toxic", "obscene"}


def test_severity_4_kicker_adds_severe_toxic_where_documented():
    assert _labels_set("sexual_vulgar", severity=4, is_toxic=True) == {"toxic", "obscene", "severe_toxic"}
    assert _labels_set("dehumanising", severity=4, is_toxic=True) == {"toxic", "insult", "severe_toxic"}
    assert _labels_set("dehumanising", severity=3, is_toxic=True) == {"toxic", "insult"}


def test_maternal_insult_only_gets_obscene_and_severe_toxic_at_severity_4():
    assert _labels_set("maternal_insult", severity=3, is_toxic=True) == {"toxic", "insult"}
    assert _labels_set("maternal_insult", severity=4, is_toxic=True) == {
        "toxic", "insult", "obscene", "severe_toxic",
    }


def test_sexual_slur_always_severe_at_severity_4():
    assert _labels_set("sexual_slur", severity=4, is_toxic=True) == {
        "toxic", "obscene", "insult", "severe_toxic",
    }


def test_ethnic_hate_maps_to_identity_hate():
    assert _labels_set("ethnic_hate", severity=2, is_toxic=True) == {"toxic", "insult", "identity_hate"}
    assert _labels_set("ethnic_hate", severity=4, is_toxic=True) == {
        "toxic", "insult", "identity_hate", "severe_toxic",
    }


def test_threat_never_set():
    # No category in this dataset maps to threat -- verify the invariant
    # directly rather than trusting the mapping table by inspection.
    for category in ["insult", "insult_general", "character_attack", "directed_abuse",
                      "accusation", "animal", "ableist", "political_incitement",
                      "dehumanising", "maternal_insult", "profanity", "sexual_vulgar",
                      "sexual_slur", "ethnic_hate", "literal_or_metalinguistic"]:
        for severity in range(5):
            result = _row_labels(category, severity, is_toxic=True)
            assert result["threat"] == 0, f"{category}@{severity} unexpectedly set threat"
