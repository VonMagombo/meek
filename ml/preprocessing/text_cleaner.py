import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize a raw comment for the LSTM tokenizer."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = _URL_RE.sub(" ", text)
    text = _IP_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_series(texts):
    return [clean_text(t) for t in texts]
