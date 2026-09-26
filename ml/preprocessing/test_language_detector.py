import pytest

from ml.preprocessing.language_detector import detect_language


def test_detect_english():
    assert detect_language("Hello, how are you doing today?") == "en"
    assert detect_language("This is a wonderful contribution to the project, thank you.") == "en"
    assert detect_language("I will find you and hurt you, idiot.") == "en"


def test_detect_shona():
    assert detect_language("Uri munhu anoshamisa, tinokutendai nokuda kwokundibetsera!") == "sn"
    assert detect_language("Mhoro shamwari, urikufamba sei nhasi?") == "sn"
    assert detect_language("Ndinokuda chaizvo amai vangu.") == "sn"
    assert detect_language("Ndichakuuraya iwe benzi.") == "sn"


def test_detect_fallback():
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"
    assert detect_language("12345 67890") == "en"
    assert detect_language("xyzabc", default="sn") == "sn"
