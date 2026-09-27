"""Build the final Shona training/val CSVs consumed by
ml/training/train_shona_classifier.py, by combining two sources:

1. MT-silver data (ml/translation/translate_jigsaw.py output): machine-
   translated Jigsaw comments, original English human labels preserved.
   Pristine copy kept at data/processed/shona/silver/{train,val,test}.csv.
2. Native lexicon data (data/raw/shona_native/): hand-written/lexicon-driven
   Shona sentences, natively labeled toxic/non-toxic with a category +
   severity (0-4), no English pairing. Real Shona, not translation output.
   Plus any custom_*.csv dropped in the same directory (auto-discovered by
   load_custom_datasets()) -- see docs/SHONA_DATASET_REQUIREMENTS.md for the
   collection workflow. As of the matthew_shona_dataset.csv addition, native
   `threat` and `ethnic_hate` coverage is no longer zero; see that doc's
   Phase 1 section for current row counts per gap category.

Category -> six-label mapping (deterministic, documented here so it's
auditable and re-derivable, not a one-off judgment call buried in code):

  literal_or_metalinguistic, or label == "non-toxic": all six labels 0.
  threat: toxic + threat; + severe_toxic if severity == 4.
  insult, insult_general, character_attack, directed_abuse, accusation,
    animal, political_incitement: toxic + insult.
  dehumanising: toxic + insult; + severe_toxic if severity == 4.
  maternal_insult: toxic + insult; + obscene, severe_toxic if severity == 4.
  profanity: toxic + obscene.
  sexual_vulgar: toxic + obscene; + severe_toxic if severity == 4.
  sexual_slur: toxic + obscene + insult; + severe_toxic if severity == 4
    (all sexual_slur rows in this data are severity 4).
  ethnic_hate: toxic + insult + identity_hate; + severe_toxic if severity == 4.
  ableist: toxic + insult + identity_hate; + severe_toxic if severity == 4.
    Same identity_hate treatment as ethnic_hate -- Jigsaw's identity_hate
    label is defined broadly (race, religion, gender, sexual orientation,
    disability), and disability-directed slurs are identity hate, not a
    lesser "just an insult" category. Kept as a separate category name from
    ethnic_hate so tribal/xenophobic and disability-based hate stay
    distinguishable in the raw data even though they train the same labels.

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
    "threat": {"toxic", "threat"},
    "insult": {"toxic", "insult"},
    "insult_general": {"toxic", "insult"},
    "character_attack": {"toxic", "insult"},
    "directed_abuse": {"toxic", "insult"},
    "accusation": {"toxic", "insult"},
    "animal": {"toxic", "insult"},
    "ableist": {"toxic", "insult", "identity_hate"},
    "political_incitement": {"toxic", "insult"},
    "dehumanising": {"toxic", "insult"},
    "maternal_insult": {"toxic", "insult"},
    "profanity": {"toxic", "obscene"},
    "sexual_vulgar": {"toxic", "obscene"},
    "sexual_slur": {"toxic", "obscene", "insult"},
    "ethnic_hate": {"toxic", "insult", "identity_hate"},
    "literal_or_metalinguistic": set(),
    "benign_conversational": set(),
}

# categories where a severity-4 row also gets obscene (on top of CATEGORY_LABELS)
SEV4_ADDS_OBSCENE = {"maternal_insult"}
# categories where a severity-4 row also gets severe_toxic
SEV4_ADDS_SEVERE_TOXIC = {
    "dehumanising", "maternal_insult", "sexual_vulgar", "sexual_slur", "ethnic_hate", "threat", "ableist",
}


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


def load_clean_conversational_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_path = NATIVE_DIR / "clean_conversational.csv"
    if not clean_path.exists():
        empty = pd.DataFrame(columns=["comment_text_en", "comment_text_sn"] + LABELS + ["source"])
        return empty, empty, empty

    df = pd.read_csv(clean_path)
    train_df = df.sample(frac=0.8, random_state=SEED)
    rest_df = df.drop(train_df.index)
    val_df = rest_df.sample(frac=0.5, random_state=SEED)
    test_df = rest_df.drop(val_df.index)

    def _format(split_df):
        return pd.DataFrame(
            {
                "comment_text_en": "",
                "comment_text_sn": split_df["text"],
                **{label: 0 for label in LABELS},
                "source": "native_conversational_clean",
            }
        )

    return _format(train_df), _format(val_df), _format(test_df)


def load_custom_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find any custom Shona datasets in NATIVE_DIR and split them 80/10/10."""
    custom_dfs = []
    for csv_file in NATIVE_DIR.glob("*.csv"):
        if csv_file.name in ["clean_conversational.csv", "full.csv", "train.csv", "test.csv", "validation.csv"]:
            continue
        if csv_file.name == "custom_shona_template.csv":
            try:
                if len(pd.read_csv(csv_file)) <= 15:
                    continue
            except Exception:
                continue
        try:
            df = pd.read_csv(csv_file)
            if all(col in df.columns for col in ["text", "label", "category", "severity"]):
                print(f"Found custom dataset: {csv_file.name} ({len(df)} rows)")
                label_rows = [
                    _row_labels(str(row["category"]), int(row["severity"]), str(row["label"]) == "toxic")
                    for _, row in df.iterrows()
                ]
                labels_df = pd.DataFrame(label_rows, columns=LABELS)
                formatted = pd.DataFrame(
                    {
                        "comment_text_en": "",
                        "comment_text_sn": df["text"],
                        **{label: labels_df[label] for label in LABELS},
                        "source": f"custom_{csv_file.stem}",
                    }
                )
                custom_dfs.append(formatted)
        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")

    if not custom_dfs:
        empty = pd.DataFrame(columns=["comment_text_en", "comment_text_sn"] + LABELS + ["source"])
        return empty, empty, empty

    combined = pd.concat(custom_dfs, ignore_index=True)
    train_df = combined.sample(frac=0.8, random_state=SEED)
    rest_df = combined.drop(train_df.index)
    val_df = rest_df.sample(frac=0.5, random_state=SEED)
    test_df = rest_df.drop(val_df.index)
    return train_df, val_df, test_df


def load_silver_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(SILVER_DIR / f"{name}.csv")
    df = df.copy()
    df["source"] = "mt_silver"
    return df


def merge_split(
    name: str,
    native_name: str,
    clean_df: pd.DataFrame | None = None,
    custom_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    silver = load_silver_split(name)
    native = load_native_split(native_name)
    dfs = [silver, native]
    if clean_df is not None and not clean_df.empty:
        dfs.append(clean_df)
    if custom_df is not None and not custom_df.empty:
        dfs.append(custom_df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged.sample(frac=1, random_state=SEED).reset_index(drop=True)


def report_counts(name: str, df: pd.DataFrame) -> None:
    counts = df["source"].value_counts().to_dict()
    sources_str = " + ".join(f"{cnt} {src}" for src, cnt in counts.items())
    print(f"[{name}] {len(df)} rows total ({sources_str})")
    for label in LABELS:
        print(f"    {label:15s} {int(df[label].sum())}")


def main() -> None:
    for missing in [SILVER_DIR, NATIVE_DIR]:
        if not missing.exists():
            print(f"Missing {missing}")
            sys.exit(1)

    clean_train, clean_val, clean_test = load_clean_conversational_splits()
    custom_train, custom_val, custom_test = load_custom_datasets()

    train = merge_split("train", "train", clean_train, custom_train)
    val = merge_split("val", "validation", clean_val, custom_val)
    native_test_parts = [load_native_split("test"), clean_test]
    if not custom_test.empty:
        native_test_parts.append(custom_test)

    native_test = pd.concat(native_test_parts, ignore_index=True).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)

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
