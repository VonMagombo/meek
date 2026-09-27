from pathlib import Path

from ml.preprocessing.validate_custom_dataset import validate_dataset


def test_validate_template_dataset():
    template_path = Path("data/raw/shona_native/custom_shona_template.csv")
    assert template_path.exists()
    assert validate_dataset(template_path) is True


def test_validate_nonexistent_dataset(tmp_path):
    fake_path = tmp_path / "does_not_exist.csv"
    assert validate_dataset(fake_path) is False


def test_validate_invalid_headers(tmp_path):
    bad_csv = tmp_path / "bad_headers.csv"
    bad_csv.write_text("sentence,is_toxic\nHello,False\n")
    assert validate_dataset(bad_csv) is False


def test_validate_invalid_labels(tmp_path):
    bad_csv = tmp_path / "bad_labels.csv"
    bad_csv.write_text("text,label,category,severity,source_lemma\nHello,invalid_label,insult,2,hello\n")
    assert validate_dataset(bad_csv) is False
