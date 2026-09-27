"""Zero-dependency language detector for English ('en') vs Shona ('sn').

Uses lexical intersection (core stopwords and function words), morphological
prefix patterns, and phonotactic markers (open syllables and characteristic
Shona digraphs like 'sv', 'zv', 'tsv', 'dzv', 'mh', 'nh').
"""

from __future__ import annotations

import re
from typing import Literal

DetectedLanguage = Literal["en", "sn"]

# High-frequency English function words / stopwords
_EN_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "am", "are", "was",
    "were", "has", "had", "been", "very", "thank", "thanks", "hello", "please"
}

# High-frequency Shona core function words and vocabulary
_SN_WORDS = {
    "uye", "asi", "nekuti", "pane", "munhu", "vanhu", "zvakanaka", "zvakaipa",
    "mhoro", "mhoroi", "kwete", "hongu", "chii", "sei", "kuti", "zve", "zvose", "ndiri",
    "uri", "ari", "isu", "imi", "ivo", "basa", "imba", "vana", "amai", "baba",
    "hama", "zuva", "gore", "nyika", "mutauro", "chero", "pano", "apo", "kuno",
    "ikoko", "iko", "ivo", "avo", "uyu", "ava", "iyi", "idzi", "ichi", "izvi",
    "ako", "edu", "enyu", "avo", "pasi", "kumusoro", "mukati", "kunze", "kure",
    "pedyo", "nguva", "chokwadi", "zvino", "nhasi", "mangwana", "nezuro",
    "ndinotenda", "unotenda", "tinotenda", "ndinokutendai", "ndinoda", "anoda",
    "zvangu", "zvako", "zvake", "zvavo", "wangu", "wako", "wake", "wavo",
    "ndiani", "riini", "nepi", "kupi", "chete", "bva", "kana", "saka", "zvikuru",
    "ndichakuuraya", "uraya", "benzi", "dofo", "mbavha", "bata", "famba", "ona",
    "shamwari", "akanaka", "yakanaka", "chakanaka", "vakanaka", "wakanaka",
    "mangwanani", "masikati", "manheru", "makadii", "wakadii", "sakadii",
    "maswerasei", "waswerasei", "kwaziwai", "ndeipi", "titambire", "titambirei",
    "mauya", "mauyai", "chisarai", "fambai", "maita", "ndatenda", "chiremba",
    "chikoro", "mufaro", "mutsvene", "chiremera", "hupenyu", "upenyu", "rudo",
    "rugare", "mwoyo", "moyo", "chingwa", "mvura", "bhora", "mutambo", "dzidzo",
    "imbwa", "gonzo", "tsotsi", "muroyi", "chifeve", "mabeche", "svira", "beche",
    "mboro", "mhata", "hure"
}

# Characteristic Shona morphological prefix patterns
_SN_PREFIX_RE = re.compile(
    r"^(?:zvak?|zvi|chik?|chak?|ndich?|vach?|uch?|ach?|tak?|mak?|mwak?|huku|kash|pa|ku|mu|aka|yaka|waka)[a-z]{3,}$"
)

# Characteristic Shona consonant clusters/digraphs
_SN_CLUSTERS = ("tsv", "dzv", "tsv", "mvh", "pwh", "sv", "zv", "mh", "nh", "zh", "rw", "mw", "ny")

_WORD_RE = re.compile(r"[a-z']+")


def detect_language(text: str, default: DetectedLanguage = "en") -> DetectedLanguage:
    """Detect whether text is English ('en') or Shona ('sn').
    
    Returns 'default' (defaulting to 'en') if text is empty or ambiguous.
    """
    if not isinstance(text, str) or not text.strip():
        return default

    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return default

    en_score = 0.0
    sn_score = 0.0

    for token in tokens:
        if token in _EN_WORDS:
            en_score += 2.0
        if token in _SN_WORDS:
            sn_score += 3.0
        elif _SN_PREFIX_RE.match(token):
            sn_score += 1.0

        for cluster in _SN_CLUSTERS:
            if cluster in token:
                sn_score += 0.5
                break

    # If Shona evidence is clearly stronger than English evidence:
    if sn_score > en_score and sn_score >= 1.0:
        return "sn"

    # If English evidence is stronger or text matches English vocabulary:
    if en_score > sn_score:
        return "en"

    return default
