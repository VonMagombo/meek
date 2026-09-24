"""Build a silver-labeled Shona toxicity dataset by machine-translating a
stratified sample of the (English-labeled) Jigsaw comments into Shona.

Why silver labels: there is no labeled Shona hate-speech dataset publicly
available (checked the HF Hub directly, including the AfriHate collection,
the standard reference for African-language hate speech — it covers 15
languages and Shona isn't one of them). Translating labeled English text
preserves the original human labels; only the language changes. The known
risk (documented in README) is that machine-translated toxicity doesn't
always reproduce how hate speech is actually phrased in the target language
-- slurs, idiom and intensity can flatten in translation.

Sampling: rare labels (threat, severe_toxic, identity_hate) get every
positive row available (capped); common labels (toxic, obscene, insult) are
capped and sampled, since translating and fine-tuning on all ~13k positive
rows in the train split isn't CPU-tractable here. Negatives are sampled to
roughly match the positive count.

Run: python ml/translation/translate_jigsaw.py [--train-size N] [--val-size N] [--test-size N]
Writes: data/processed/shona/{train,val,test}.csv (columns: comment_text_en,
comment_text_sn, toxic, severe_toxic, obscene, threat, insult, identity_hate)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "raw" / "jigsaw"
OUT_DIR = ROOT / "data" / "processed" / "shona"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-sn"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RARE_LABELS = ["threat", "severe_toxic", "identity_hate"]
MAX_CHARS = 400  # truncate before translation: MT quality/speed degrade on long inputs
BATCH_SIZE = 16

_model = None
_tokenizer = None


def _get_translator():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
        _model.eval()
    return _model, _tokenizer


@torch.inference_mode()
def translate_batch(texts: list) -> list:
    model, tok = _get_translator()
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    out = model.generate(**inputs, max_new_tokens=256, num_beams=1)
    return tok.batch_decode(out, skip_special_tokens=True)


def sample_stratified(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    """Prioritize coverage of rare labels (capped, so they can't crowd out the
    rest of the budget), then fill with common-label positives up to half the
    target size, then match that positive count with an equal number of
    negatives.

    The negative half is not optional: a multi-label classifier trained with
    zero negative examples has no contrastive signal for what "not toxic"
    looks like and degenerates into predicting every majority label positive
    regardless of input (caught via the golden-set eval after an earlier
    version of this function let uncapped rare-label positives consume the
    entire budget, leaving no room for negatives at all)."""
    rng = seed
    picked_idx = set()
    positive_budget = target_size // 2

    rare_cap = max(positive_budget // len(RARE_LABELS), 1)
    for label in RARE_LABELS:
        positives = df[(df[label] == 1) & (~df.index.isin(picked_idx))]
        if len(positives):
            take = positives.sample(n=min(rare_cap, len(positives)), random_state=rng)
            picked_idx.update(take.index)

    remaining_budget = max(positive_budget - len(picked_idx), 0)
    common_positives = df[(df[LABELS].sum(axis=1) > 0) & (~df.index.isin(picked_idx))]
    if remaining_budget and len(common_positives):
        take = common_positives.sample(n=min(remaining_budget, len(common_positives)), random_state=rng)
        picked_idx.update(take.index)

    n_positive_picked = len(picked_idx)
    negatives = df[(df[LABELS].sum(axis=1) == 0) & (~df.index.isin(picked_idx))]
    n_negative = min(n_positive_picked, len(negatives))
    if n_negative:
        neg_sample = negatives.sample(n=n_negative, random_state=rng)
        picked_idx.update(neg_sample.index)

    result = df.loc[sorted(picked_idx)].sample(frac=1, random_state=rng).reset_index(drop=True)
    return result.head(target_size) if len(result) > target_size else result


def build_split(name: str, target_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}_dataset.csv")
    df["comment_text"] = df["comment_text"].fillna("")
    if name == "test":
        df = df[(df[LABELS] != -1).all(axis=1)].reset_index(drop=True)

    sample = sample_stratified(df, target_size, seed)
    texts_en = [t[:MAX_CHARS].replace("\n", " ").strip() for t in sample["comment_text"]]

    print(f"[{name}] translating {len(texts_en)} comments ({TRANSLATION_MODEL})...")
    translations = []
    start = time.time()
    for i in range(0, len(texts_en), BATCH_SIZE):
        batch = texts_en[i : i + BATCH_SIZE]
        translations.extend(translate_batch(batch))
        done = min(i + BATCH_SIZE, len(texts_en))
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(texts_en) - done) / rate if rate > 0 else float("nan")
        print(f"  [{name}] {done}/{len(texts_en)} ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)

    out = pd.DataFrame(
        {
            "comment_text_en": texts_en,
            "comment_text_sn": translations,
            **{label: sample[label].values for label in LABELS},
        }
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _get_translator()  # load once up front

    for split_name, target, source in [
        ("train", args.train_size, "train"),
        ("val", args.val_size, "val"),
        ("test", args.test_size, "test"),
    ]:
        out_path = OUT_DIR / f"{split_name}.csv"
        result = build_split(source, target, args.seed)
        result.to_csv(out_path, index=False)
        n_positive = (result[LABELS].sum(axis=1) > 0).sum()
        print(f"[{split_name}] saved {len(result)} rows ({n_positive} positive) to {out_path}")
        for label in LABELS:
            print(f"    {label:15s} {int(result[label].sum())}")


if __name__ == "__main__":
    main()
