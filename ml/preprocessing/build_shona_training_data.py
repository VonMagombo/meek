"""Build the final Shona training/val CSVs consumed by
ml/training/train_shona_classifier.py, by combining two sources:

1. MT-silver data (ml/translation/translate_jigsaw.py output): machine-
   translated Jigsaw comments, original English human labels preserved.
   Pristine copy kept at data/processed/shona/silver/{train,val,test}.csv.
2. Native lexicon data (data/raw/shona_native/): hand-written/lexicon-driven
   Shona sentences, natively labeled toxic/non-toxic with a category +
   severity (0-4), no English pairing. Real Shona, not translation output --
   but small (358 rows total) and, per category coverage, has NO `threat`
   examples at all. See README's "Known limitations" for what this does and
   doesn't fix.

Category -> six-label mapping (deterministic, documented here so it's
auditable and re-derivable, not a one-off judgment call buried in code):

  literal_or_metalinguistic, or label == "non-toxic": all six labels 0.
  insult, insult_general, character_attack, directed_abuse, accusation,
    animal, ableist, political_incitement: toxic + insult.
  dehumanising: toxic + insult; + severe_toxic if severity == 4.
  maternal_insult: toxic + insult; + obscene, severe_toxic if severity == 4.
  profanity: toxic + obscene.
  sexual_vulgar: toxic + obscene; + severe_toxic if severity == 4.
  sexual_slur: toxic + obscene + insult; + severe_toxic if severity == 4
    (all sexual_slur rows in this data are severity 4).
  ethnic_hate: toxic + insult + identity_hate; + severe_toxic if severity == 4.

`threat` is 0 for every native row -- there is no threat-category data here.
Do not read a `threat` improvement into any eval run that includes this data;
it isn't testing that label.

Run: python ml/preprocessing/build_shona_training_data.py
Writes: data/processed/shona/{train,val}.csv (overwritten, merged),
        data/processed/shona/native_test.csv (native test slice, kept
        separate from the silver test.csv so the two evidence qualities
        are never blended into one number).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SILVER_DIR = ROOT / "data" / "processed" / "shona" / "silver"
NATIVE_DIR = ROOT / "data" / "raw" / "shona_native"
OUT_DIR = ROOT / "data" / "processed" / "shona"

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42

# category -> set of labels to set to 1 (before the severity==4 kicker below)
CATEGORY_LABELS = {
    "insult": {"toxic", "insult"},
    "insult_general": {"toxic", "insult"},
    "character_attack": {"toxic", "insult"},
    "directed_abuse": {"toxic", "insult"},
    "accusation": {"toxic", "insult"},
    "animal": {"toxic", "insult"},
    "ableist": {"toxic", "insult"},
    "political_incitement": {"toxic", "insult"},
    "dehumanising": {"toxic", "insult"},
    "maternal_insult": {"toxic", "insult"},
    "profanity": {"toxic", "obscene"},
    "sexual_vulgar": {"toxic", "obscene"},
    "sexual_slur": {"toxic", "obscene", "insult"},
    "ethnic_hate": {"toxic", "insult", "identity_hate"},
    "literal_or_metalinguistic": set(),
}

# categories where a severity-4 row also gets obscene (on top of CATEGORY_LABELS)
SEV4_ADDS_OBSCENE = {"maternal_insult"}
# categories where a severity-4 row also gets severe_toxic
SEV4_ADDS_SEVERE_TOXIC = {"dehumanising", "maternal_insult", "sexual_vulgar", "sexual_slur", "ethnic_hate"}


def _row_labels(category: str, severity: int, is_toxic: bool) -> dict:
    if not is_toxic:
        return {label: 0 for label in LABELS}

    active = set(CATEGORY_LABELS.get(category, set()))
    active.add("toxic")
    if severity == 4:
        if category in SEV4_ADDS_OBSCENE:
            active.add("obscene")
        if category in SEV4_ADDS_SEVERE_TOXIC:
            active.add("severe_toxic")
    return {label: int(label in active) for label in LABELS}


def load_native_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(NATIVE_DIR / f"{name}.csv")
    label_rows = [
        _row_labels(row["category"], int(row["severity"]), row["label"] == "toxic")
        for _, row in df.iterrows()
    ]
    labels_df = pd.DataFrame(label_rows, columns=LABELS)
    out = pd.DataFrame(
        {
            "comment_text_en": "",
            "comment_text_sn": df["text"],
            **{label: labels_df[label] for label in LABELS},
            "source": "native_lexicon",
        }
    )
    return out


def load_silver_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(SILVER_DIR / f"{name}.csv")
    df = df.copy()
    df["source"] = "mt_silver"
    return df


def merge_split(name: str, native_name: str) -> pd.DataFrame:
    silver = load_silver_split(name)
    native = load_native_split(native_name)
    merged = pd.concat([silver, native], ignore_index=True)
    return merged.sample(frac=1, random_state=SEED).reset_index(drop=True)


def report_counts(name: str, df: pd.DataFrame) -> None:
    n_native = (df["source"] == "native_lexicon").sum()
    n_silver = (df["source"] == "mt_silver").sum()
    print(f"[{name}] {len(df)} rows total ({n_silver} mt_silver + {n_native} native_lexicon)")
    for label in LABELS:
        print(f"    {label:15s} {int(df[label].sum())}")


def main() -> None:
    for missing in [SILVER_DIR, NATIVE_DIR]:
        if not missing.exists():
            print(f"Missing {missing}")
            sys.exit(1)

    train = merge_split("train", "train")
    val = merge_split("val", "validation")
    native_test = load_native_split("test")

    report_counts("train", train)
    report_counts("val", val)
    report_counts("native_test", native_test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    native_test.to_csv(OUT_DIR / "native_test.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'train.csv'}, {OUT_DIR / 'val.csv'}, {OUT_DIR / 'native_test.csv'}")
    print("data/processed/shona/test.csv (mt_silver only) left untouched.")


if __name__ == "__main__":
    main()
