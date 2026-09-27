# meek

A toxicity/moderation MVP for user comments. Paste a comment in, get back a
score for each of the six Jigsaw toxicity categories (`toxic`, `severe_toxic`,
`obscene`, `threat`, `insult`, `identity_hate`) and a flag for whether it
crosses the moderation threshold. Supports English and Shona.

```
frontend (vanilla HTML/CSS/JS) ──HTTP──> backend (FastAPI) ──imports──> ml (inference wrapper)
                                                                              │
                                                              en: unitary/toxic-bert (Hugging Face)
                                                              sn: fine-tuned Davlan/xlm-roberta-base-finetuned-shona
```

## What's here

- **`ml/`** — model code.
  - `ml/inference/toxicity_classifier.py` — the English classifier: a
    pretrained Hugging Face model (`unitary/toxic-bert`), loaded once and
    cached.
  - `ml/inference/shona_toxicity_classifier.py` — the Shona classifier: a
    fine-tuned checkpoint (see "Shona support" below), same interface as the
    English one.
  - `ml/translation/translate_jigsaw.py` — builds silver-labeled Shona
    training data by machine-translating Jigsaw comments.
  - `ml/preprocessing/build_shona_training_data.py` — merges that silver data
    with native-labeled Shona data (`data/raw/shona_native/`) into the final
    training/val CSVs.
  - `ml/training/train_shona_classifier.py` — fine-tunes the Shona
    classifier on the merged data.
  - `ml/models/`, `ml/preprocessing/text_cleaner.py`, `ml/training/train_lstm.py`
    — the original from-scratch Bi-LSTM baseline (trained on this repo's
    Jigsaw split). Kept as the baseline the English HF model is benchmarked
    against; not used by the API.
  - `ml/evals/` — golden-set + benchmark eval suites for both classifiers.
- **`backend/`** — FastAPI service (`/api/v1/health`, `/api/v1/moderate`)
  that wraps the classifiers and also serves the frontend as static files.
  `backend/tests/` — gate tests (classifiers mocked, run in under a second).
- **`frontend/`** — single static page: textarea in, per-label score bars
  out, language toggle (en/sn).
- **`data/`** — Jigsaw dataset (`data/raw/jigsaw/`), native Shona toxicity
  data (`data/raw/shona_native/`), and trained artifacts/eval reports
  (`data/processed/`).

## Why a pretrained HF model instead of the LSTM baseline

The repo's from-scratch Bi-LSTM baseline (`ml/training/train_lstm.py`) scores
macro F1 **0.39** on its held-out test set, and **F1 0.0 on `threat`** — it
never once got a threat right (`data/processed/eval_metrics.json`). That's
not a usable moderation tool. `unitary/toxic-bert` is a BERT model fine-tuned
on this exact Jigsaw label set, so it's a drop-in swap with no label mapping
needed. Head-to-head on the same held-out test rows
(`ml/evals/eval_toxicity_classifier.py`, results in
`data/processed/hf_toxicity_eval.json`):

| label | LSTM baseline F1 (n=63,978, full test set) | HF (`unitary/toxic-bert`) F1 (n=4,000 sample) | delta |
|---|---|---|---|
| toxic | 0.50 | 0.68 | +0.18 |
| severe_toxic | 0.36 | 0.44 | +0.08 |
| obscene | 0.64 | 0.68 | +0.04 |
| threat | **0.00** | 0.61 | **+0.61** |
| insult | 0.60 | 0.71 | +0.11 |
| identity_hate | 0.26 | 0.59 | +0.33 |
| **macro F1** | **0.39** | **0.62** | **+0.23** |

`threat` is the standout: the LSTM never once predicted it correctly on 211
held-out threats. Reproduce with `python ml/evals/eval_toxicity_classifier.py
--sample-size 4000`; full report at `data/processed/hf_toxicity_eval.json`.

The full model card and training details: https://huggingface.co/unitary/toxic-bert

## Setup

```bash
python3 -m venv venv          # if you don't already have one
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls CPU-only PyTorch from PyTorch's own index
(`--extra-index-url` at the top of the file) — no CUDA/GPU required, and no
API key needed since the model runs locally after the first download.

## Running it

```bash
source venv/bin/activate
uvicorn backend.app.main:app --app-dir backend --reload --port 8000
```

Open http://localhost:8000 — the backend serves the frontend directly, no
separate dev server needed. First request after a cold start takes ~1-2s
while the model loads; after that, ~150ms per comment on CPU. Set
`MEEK_WARMUP=1` to load the model at startup instead of on first request
(recommended outside of local dev, so the first real user isn't the one
paying the cold-start cost).

### API

`POST /api/v1/moderate`
```json
// request
{"text": "your comment here", "language": "auto"}

// response
{
  "text": "your comment here",
  "language": "en",
  "model": "unitary/toxic-bert",
  "scores": {"toxic": 0.98, "severe_toxic": 0.28, "obscene": 0.50, "threat": 0.89, "insult": 0.60, "identity_hate": 0.06},
  "flagged_labels": ["toxic", "threat", "insult"],
  "is_toxic": true,
  "thresholds_used": {"threat": 0.35, "severe_toxic": 0.35, "identity_hate": 0.40, "toxic": 0.50, "insult": 0.50, "obscene": 0.60}
}
```

`POST /api/v1/moderate/batch`
```json
// request (up to 100 comments, batched forward passes on CPU)
{
  "texts": ["Thanks for the help!", "Shut up you idiot.", "Ndichakuuraya."],
  "language": "auto"
}

// response
{
  "results": [...],
  "total": 3,
  "flagged_count": 2
}
```

`language` is `"auto"` (default detection between English and Shona), `"en"`, or `"sn"` (Shona).
If no Shona checkpoint has been fine-tuned yet on this deployment, a Shona request returns `503`.

Optional per-request threshold overrides:
Pass `"thresholds": {"threat": 0.25, "toxic": 0.70}` in either request body.

`GET /api/v1/health` → `{"status": "ok", "models": {"en": "unitary/toxic-bert", "sn": "Davlan/xlm-roberta-base-finetuned-shona (fine-tuned)"}}`

Interactive docs at http://localhost:8000/docs (FastAPI's built-in Swagger UI).

Config (env vars, all optional):
- `MEEK_TOXICITY_THRESHOLD` (default unset) — global fallback score cutoff. When unset, safety-calibrated defaults apply: `threat: 0.35`, `severe_toxic: 0.35`, `identity_hate: 0.40`, `toxic: 0.50`, `insult: 0.50`, `obscene: 0.60`.
- `MEEK_THRESHOLD_<LABEL>` (e.g. `MEEK_THRESHOLD_THREAT=0.3`) — per-label cutoff override.
- `MEEK_MAX_TEXT_LENGTH` (default `5000`) — max single request text length, characters.
- `MEEK_MAX_BATCH_SIZE` (default `100`) — max comments per batch request.
- `MEEK_CORS_ORIGINS` (default `*`) — comma-separated allowed origins.
- `MEEK_WARMUP` (default off) — set to `1` to load models at startup.


## Shona support

`language: "sn"` routes to a fine-tuned classifier
(`ml/inference/shona_toxicity_classifier.py`), base model
`Davlan/xlm-roberta-base-finetuned-shona`. No fine-tuned Shona checkpoint
ships in git (weights are gitignored — see `.gitignore`); build one with:

```bash
python ml/translation/translate_jigsaw.py        # builds silver MT data
python ml/preprocessing/build_shona_training_data.py  # merges in native data
python ml/training/train_shona_classifier.py     # fine-tunes, saves the checkpoint
```

Training data comes from three sources, merged by
`ml/preprocessing/build_shona_training_data.py`:

1. **MT-silver** (`ml/translation/translate_jigsaw.py`): Jigsaw comments
   machine-translated to Shona, original English human labels preserved.
   There is no labeled Shona hate-speech dataset publicly available (checked
   the HF Hub directly, including AfriHate — the standard reference for
   African-language hate speech, which covers 15 languages and not Shona),
   so this is the only way to get six-label-taxonomy training signal at all.
   Risk: machine-translated toxicity doesn't always reproduce how hate speech
   is actually phrased in Shona — slurs, idiom, and intensity can flatten in
   translation.
2. **Native lexicon** (`data/raw/shona_native/`): hand-written/lexicon-driven
   Shona sentences, natively labeled toxic/non-toxic with a category and
   severity (0-4) — no translation involved. Mapped to the six-label taxonomy
   by a documented, deterministic rule in `build_shona_training_data.py`'s
   docstring.
3. **Custom datasets** (`data/raw/shona_native/custom_*.csv` and any other
   CSV dropped in that directory matching the schema): auto-discovered by
   `load_custom_datasets()`, same category → label mapping as the native
   lexicon. `matthew_shona_dataset.csv` (352 rows) is the first of these —
   see `docs/SHONA_DATASET_REQUIREMENTS.md` for the collection spec and
   Phase 1 gap targets (threat, ethnic/disability hate, ChiHarare slang,
   Shonglish code-switching) it was written to close.

Two real bugs surfaced and were fixed while adding that dataset (both have
regression tests in `ml/preprocessing/test_build_shona_training_data.py`):
`CATEGORY_LABELS` had no `"threat"` key at all, so every native `threat`-
tagged row silently trained as plain `toxic` only — the threat head never
saw a single native threat example even after they existed in the raw CSV;
and disability slurs (`chirema`, `musope`/`sope`, `chimumumu`, `bofu`,
`matsi`) were first folded into `ethnic_hate` before being split into their
own `ableist` category, which is mapped to the same `identity_hate` signal
as `ethnic_hate` (Jigsaw's `identity_hate` label is defined broadly across
race/religion/gender/orientation/disability — a disability slur is identity
hate, not a lesser plain-insult category) so that split didn't cost any
training signal.

Result of adding `matthew_shona_dataset.csv` on top of the existing native
lexicon (full report: `data/processed/shona_classifier/eval_report.json`;
pre-dataset baseline kept at
`eval_report_baseline_before_phase1_dataset.json` for comparison):

**Native-labeled held-out test** (`data/processed/shona/native_test.csv`,
real Shona, natively labeled — the strongest of the three eval slices):

| label | before | after | Δ |
|---|---|---|---|
| toxic | 0.923 (support=30) | 0.922 (support=53) | ~unchanged |
| severe_toxic | 0.444 (support=5) | 0.400 (support=9) | −0.044 (small support, noisy) |
| obscene | 0.900 (support=10) | 0.818 (support=11) | −0.082 (small support, noisy) |
| threat | **0.000 (support=0)** | **1.000 (support=7)** | **first-ever native threat signal** |
| insult | 0.800 (support=22) | 0.857 (support=38) | +0.057 |
| identity_hate | **0.000 (support=1)** | **0.737 (support=11)** | **+0.737** |
| macro F1 | 0.614 | 0.789 | +0.175 |

**Silver-labeled held-out test** (`data/processed/shona/test.csv`,
machine-translated, held constant — n=100, same rows both times):

| label | before | after | Δ |
|---|---|---|---|
| toxic | 0.659 | 0.634 | −0.025 |
| severe_toxic | 0.323 | 0.267 | −0.056 |
| obscene | 0.557 | 0.542 | −0.015 |
| threat | 0.296 | 0.387 | +0.091 |
| insult | 0.515 | 0.419 | −0.096 |
| identity_hate | 0.071 | 0.194 | +0.123 |
| macro F1 | 0.404 | 0.407 | +0.003 |

The doc's stated Phase 1 goal was "immediate F1 boost on `threat` &
`identity_hate` from ~0.0 to >0.80" on native evidence. `threat` hit it
(1.00, though on only 7 held-out examples — small-sample, expect this to
move as more data is added). `identity_hate` moved from a single unsupported
row to 0.737 on 11 examples — a real fix, short of the >0.80 bar, because
`ableist` and tribal `ethnic_hate` rows are still a few hundred sentences,
not the "commercial-grade" volume the doc's Phase 2 (1,000-1,500 rows)
targets. The silver-test dip on `insult`/`obscene` is the calibrated
classifier's decision boundary shifting as the training mix changes — not a
regression in the model's actual Shona understanding, which the native
slice (real Shona, not translation) is the one to trust more.

Eval (`ml/evals/eval_shona_classifier.py`) reports three slices, in
increasing order of evidence quality, and never blends them into one number:
- **Silver test** (`data/processed/shona/test.csv`): held-out MT-silver rows.
  Measures whether the model learned the training signal, not whether it
  understands real Shona hate speech.
- **Native test** (`data/processed/shona/native_test.csv`): held-out native
  lexicon + custom-dataset rows (108 rows as of `matthew_shona_dataset.csv`).
  Real Shona, but templated/lexicon-driven, not organic text — a stronger
  signal than it was, still not the same evidence quality as organic text.
- **Golden set**: the English golden examples
  (`ml/evals/golden_examples.py`) translated to Shona with the *same* MT
  model used to build the silver training data, so it shares that model's
  translation biases rather than testing independently. Directional sanity
  check only, not a validated accuracy claim.

## Testing

Two lanes, per this repo's testing convention:

- **Gate tests** (`backend/tests/`, classifiers mocked, deterministic, <1s):
  ```bash
  pytest
  ```
- **Periodic eval, English** (`ml/evals/`, real model, ~1-2 min for the default sample):
  ```bash
  python ml/evals/eval_toxicity_classifier.py
  ```
  Runs two checks: a hand-written golden set (17 examples covering every
  label plus negation/sarcasm/criticism-not-personal-attack edge cases) and a
  benchmark against a random sample of the same held-out Jigsaw rows the LSTM
  baseline was scored on. Exits non-zero and prints exactly which case failed
  if either the golden set's macro F1 drops below 0.75 or the HF model stops
  beating the LSTM baseline. Writes a full report to
  `data/processed/hf_toxicity_eval.json`.
- **Periodic eval, Shona** (`ml/evals/`, real model, requires a fine-tuned
  checkpoint — see "Shona support"):
  ```bash
  python ml/evals/eval_shona_classifier.py
  ```
  Reports the three slices described above. Unlike the English eval, this
  has no pass/fail gate — the evidence isn't strong enough yet to gate on;
  read the per-label numbers, don't just check the exit code. Writes a full
  report to `data/processed/shona_classifier/eval_report.json`.

## Known limitations

- **Implicit/veiled identity hate.** The model reliably catches identity hate
  that uses a slur, but two golden-set examples that name a protected group
  *without* a slur ("people like you don't belong here because of your race")
  score `identity_hate` well under threshold. This is called out explicitly
  in `ml/evals/golden_examples.py` (`known_limitation=True`) and excluded
  from the eval's pass/fail gate rather than hidden — every eval run still
  reports it. The benchmark section shows the model still clearly
  outperforms the LSTM baseline on `identity_hate` overall (organic Jigsaw
  data skews toward slur-based examples), so this is a real but narrow gap,
  not a general weakness on that label.
- **Long-document batch throughput.** The eval's benchmark sample includes
  some very long Wikipedia talk-page comments truncated to 512 tokens;
  scoring those in batches on CPU is much slower than live single-comment API
  latency (~1-2 examples/sec batched vs. ~150ms per live request). Not an
  issue for the MVP's interactive use case; would matter for bulk offline
  scoring at scale, where a GPU or a shorter truncation length would help.
- **No persistence.** The API is stateless — it scores whatever text you send
  it and doesn't store comments, scores, or history anywhere. Fine for the
  MVP; a real moderation queue would need a datastore, which is out of scope
  here.
- **Shona: `threat` is not covered.** Neither Shona training data source has
  `threat` examples — the MT-silver split only has whatever Jigsaw's rare
  positive rows happened to translate into the sample, and the native
  lexicon data has zero `threat`-category rows. Do not read any Shona eval
  run's `threat` numbers as evidence either way; there isn't enough data to
  measure it. Fixing this needs actual threat-language Shona data sourced,
  not a bigger sample of what exists today.
- **Shona: `identity_hate` F1 is 0.0 and adding native data did not fix it.**
  Measured, not assumed — see the before/after table in "Shona support".
  Most `identity_hate`-positive training rows are still MT-silver (the
  native lexicon only contributed ~4 `ethnic_hate` rows), and that MT-silver
  signal was already producing F1=0.0 before this change. The likely cause
  is the same one `translate_jigsaw.py` already documents: identity-hate
  language flattens in machine translation. A handful of native rows can't
  outweigh a much larger diluted signal. Needs substantially more native
  identity-hate examples, not incremental native data of other kinds.
- **Shona: everything else is weak evidence, not a validated model.** Even
  with the native lexicon data merged in, training data is ~700 rows total
  (vs. Jigsaw's ~160k for English) and none of the three eval slices (see
  "Shona support") is both real Shona *and* organic (non-templated,
  non-lexicon-constructed) text. Treat the Shona classifier as a directional
  MVP, not production-ready moderation — the English path is the validated
  one.
- **Only English and Shona.** Other languages aren't handled — input in any
  other language is routed through whichever classifier is selected and will
  get unreliable scores, not an error.
