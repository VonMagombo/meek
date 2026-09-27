"""Validate a custom Shona toxicity dataset CSV before ingesting into the pipeline.

Usage:
    python ml/preprocessing/validate_custom_dataset.py [path_to_csv]

Defaults to data/raw/shona_native/custom_shona_template.csv if no path is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["text", "label", "category", "severity", "source_lemma"]
VALID_LABELS = {"toxic", "non-toxic"}
VALID_CATEGORIES = {
    "threat",
    "ethnic_hate",
    "insult",
    "insult_general",
    "character_attack",
    "directed_abuse",
    "accusation",
    "animal",
    "ableist",
    "political_incitement",
    "dehumanising",
    "maternal_insult",
    "profanity",
    "sexual_vulgar",
    "sexual_slur",
    "benign_conversational",
    "literal_or_metalinguistic",
}


def validate_dataset(csv_path: Path) -> bool:
    print(f"=== Validating: {csv_path} ===")
    if not csv_path.exists():
        print(f"❌ Error: File not found at {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Column presence
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        print("\n".join(f"  ❌ {e}" for e in errors))
        return False

    print(f"✓ Columns present: {list(df.columns)}")
    print(f"✓ Total rows: {len(df)}")

    if len(df) == 0:
        errors.append("Dataset is empty.")
        print("\n".join(f"  ❌ {e}" for e in errors))
        return False

    # 2. Null checks
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null/empty values.")

    # 3. Label validation
    invalid_labels = df[~df["label"].isin(VALID_LABELS)]
    if not invalid_labels.empty:
        errors.append(
            f"Found {len(invalid_labels)} rows with invalid 'label' (must be 'toxic' or 'non-toxic'). "
            f"Seen values: {invalid_labels['label'].unique().tolist()}"
        )

    # 4. Category validation
    invalid_cats = df[~df["category"].isin(VALID_CATEGORIES)]
    if not invalid_cats.empty:
        errors.append(
            f"Found {len(invalid_cats)} rows with invalid 'category'. "
            f"Seen invalid values: {invalid_cats['category'].unique().tolist()}"
        )

    # 5. Severity validation
    try:
        severities = df["severity"].astype(int)
        invalid_sev = df[(severities < 0) | (severities > 4)]
        if not invalid_sev.empty:
            errors.append(f"Found {len(invalid_sev)} rows with severity outside 0-4 range.")
    except Exception:
        errors.append("Severity column contains non-integer values.")

    # 6. Consistency checks
    non_toxic_with_sev = df[(df["label"] == "non-toxic") & (df["severity"] > 0)]
    if not non_toxic_with_sev.empty:
        warnings.append(
            f"{len(non_toxic_with_sev)} non-toxic rows have severity > 0. "
            "Severity for non-toxic items should usually be 0."
        )

    toxic_with_sev_zero = df[(df["label"] == "toxic") & (df["severity"] == 0)]
    if not toxic_with_sev_zero.empty:
        warnings.append(
            f"{len(toxic_with_sev_zero)} toxic rows have severity = 0. "
            "Severity for toxic items should be 1, 2, 3, or 4."
        )

    # 7. Duplicate checks
    duplicates = df[df.duplicated(subset=["text"], keep=False)]
    if not duplicates.empty:
        warnings.append(
            f"Found {len(duplicates)} duplicate text entries (e.g. '{duplicates.iloc[0]['text'][:40]}...')."
        )

    # Summary
    print("\n--- Category Breakdown ---")
    cat_counts = df["category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat:25s}: {count:4d}")

    print("\n--- Label Breakdown ---")
    lbl_counts = df["label"].value_counts()
    for lbl, count in lbl_counts.items():
        print(f"  {lbl:25s}: {count:4d}")

    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n❌ Errors found:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("\n✅ Dataset passed all validation checks and is ready for training ingestion!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Shona toxicity dataset CSV")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/shona_native/custom_shona_template.csv",
        help="Path to CSV file to validate",
    )
    args = parser.parse_args()
    path = Path(args.csv_path)
    ok = validate_dataset(path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
