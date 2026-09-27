# Shona Toxicity Dataset Requirements & Collection Guide

This guide defines the specifications, taxonomy, and collection requirements for building a production-grade native Shona toxicity dataset for **meek**.

---

## 1. Executive Summary & Objective

In low-resource African languages like Shona, **data quality and native linguistic authenticity drastically outperform machine-translated volume**. 

Machine translation (e.g. from English Wikipedia/Jigsaw) systematically flattens colloquial nuances, idioms, culturally grounded slurs, and grammatical tone. By creating a focused, natively annotated dataset of **500 to 1,500 authentic Shona sentences**, we solve the remaining blind spots in moderation:
- **Physical violence and death threats** (`threat` label — currently near 0 native examples).
- **Tribalism, xenophobia, and minority slurs** (`identity_hate` label — currently only 3 native examples).
- **Nuanced street slang (ChiHarare / contemporary youth idioms)**.
- **Code-switching / Shonglish** (blends of Shona and English common on X, WhatsApp, and Facebook).

---

## 2. Current State vs. Target Dataset Sizing

### Current Baseline Breakdown:
- **`data/processed/shona/train.csv`**: 992 rows total.
  - `native_conversational_clean`: 294 rows (greetings, courtesies, everyday speech).
  - `native_lexicon`: 298 rows (hand-curated insults, character attacks, and vulgarities).
  - `mt_silver`: 400 rows (machine-translated English Jigsaw comments).
- **Current Weak Spots**:
  - `threat`: 0 native rows (100% of threat training signal is synthetic machine translation).
  - `identity_hate`: Only 3 native rows (ethnic hate).

### Target Sizing Goals:

| Milestone | New Sentences | Target Focus Areas | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **Phase 1 (Immediate)** | **300 – 500 rows** | Threats (100), Ethnic/Tribal hate (100), Modern Slang (100), Contrastive Clean (100) | Immediate F1 boost on `threat` & `identity_hate` from ~0.0 to >0.80. |
| **Phase 2 (Production)** | **1,000 – 1,500 rows** | Shonglish code-switching, edge-case sarcasm, regional dialect coverage (Manyika, Karanga, Zezuru) | Robust commercial-grade Shona comment moderation. |

---

## 3. CSV Schema & Columns

Save your collection file at:
`data/raw/shona_native/matthew_shona_dataset.csv` (or use the starter template at [`data/raw/shona_native/custom_shona_template.csv`](file:///home/matthew-vonroy/meek/data/raw/shona_native/custom_shona_template.csv)).

| Column | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `text` | string | Any valid text | The comment/sentence in Shona or Shonglish. Wrap in quotes if it contains commas. |
| `label` | string | `toxic`, `non-toxic` | High-level binary verdict. |
| `category` | string | See Taxonomy below | The specific semantic classification. |
| `severity` | integer | `0`, `1`, `2`, `3`, `4` | `0` = clean, `1` = mild/snark, `2` = moderate, `3` = severe, `4` = extreme. |
| `source_lemma` | string | single word | The root word or key term (e.g. `kurova`, `benzi`, `muroyi`, `mhoro`). |

### Category Taxonomy:
1. `threat` — Threats of death, physical violence, arson, assault, weapon use.
2. `ethnic_hate` — Tribal slurs (Shona/Ndebele/regional), xenophobia, race, disability.
3. `insult` — Character attacks, intellectual insults, dehumanization, directed abuse.
4. `sexual_vulgar` — Vulgar body parts, explicit sexual terms, sexual degradation.
5. `sexual_slur` — Gender-based slurs, prostitution-directed insults (`hure`, `chifeve`).
6. `profanity` — General swear words and offensive curses.
7. `benign_conversational` — Clean everyday conversation, greetings, praise, inquiries.
8. `literal_or_metalinguistic` — Clean sentences that happen to mention a sensitive root word neutrally (e.g., pet dogs, folklore witches).

---

## 4. Specific Gaps to Target (With Concrete Examples)

### Gap 1: Physical Threats & Violence (`threat`)
*Goal: 100 – 150 rows. Currently 0 native examples.*

In Shona, threats often use future tense prefixes (`ndicha-`, `ticha-`, `ucha-`) combined with verbs of striking, burning, cutting, or killing.

**Examples to collect:**
- `Ndichakurova zvekuti haumuki pano.` *(I will beat you so hard you won't get up from here.)* -> `toxic,threat,3,kurova`
- `Tichakupisira imba yako nemhuri yako yese.` *(We will burn down your house and your entire family.)* -> `toxic,threat,4,kupisa`
- `Ukaonekwa panze manheru ano unotofa.` *(If you are seen outside tonight you will die.)* -> `toxic,threat,4,kufa`
- `Ndichakutsvaga ndokupedzisa iwe mbavha.` *(I will track you down and finish you, thief.)* -> `toxic,threat,4,kupedza`
- `Tichakutema nembezo / nebanga.` *(We will slash you with an axe / knife.)* -> `toxic,threat,3,kutema`
- `Uchatunda ropa nhasi.` *(You will urinate blood today - violent idiom.)* -> `toxic,threat,3,ropa`

**Contrastive Clean Counterparts (Crucial to prevent false positives):**
- `Ndichakutsvaga kana ndapedza basa rangu.` *(I will look for you once I finish my work.)* -> `non-toxic,benign_conversational,0,kutsvaga`
- `Moto uyu unopisa zvakanyanya chenjerai.` *(This fire is very hot, be careful.)* -> `non-toxic,benign_conversational,0,kupisa`
- `Kurova bhora kunonakidza chaizvo.` *(Playing soccer is very fun.)* -> `non-toxic,benign_conversational,0,kurova`

---

### Gap 2: Tribalism, Xenophobia & Group Slurs (`ethnic_hate`)
*Goal: 100 – 150 rows. Currently only 3 native examples.*

Targets derogatory statements directed at ethnic groups, nationalities, tribes, or marginalized identities.

**Examples to collect:**
- Regional/tribal slurs or demeaning stereotypes (e.g. historical epithets used pejoratively between groups).
- Xenophobic comments targeting foreign nationals or cross-border immigrants:
  - `Mabvakure aya ngaadzokere kumusha kwavo.` *(These foreigners must go back home.)* -> `toxic,ethnic_hate,3,mabvakure`
  - Demeaning foreign nationalities or minority communities.
- Pejoratives targeting persons with albinism or disabilities:
  - Using `sope` or `musope` as an insult -> `toxic,ethnic_hate,4,sope`
  - Using `chimumumu` or `chirema` as an abusive slur rather than medical descriptor -> `toxic,ethnic_hate,3,chirema`

**Contrastive Clean Counterparts:**
- `Tiri vanhu vamwe chete muZimbabwe, ngatidanane.` *(We are one people in Zimbabwe, let's love one another.)* -> `non-toxic,benign_conversational,0,vanhu`
- `Munhu wese ane kodzero dzekuremekedzwa pasinei nerudzi rwake.` *(Every human has the right to be respected regardless of tribe.)* -> `non-toxic,benign_conversational,0,kodzero`

---

### Gap 3: Contrastive Pairs (Disambiguation)
*Goal: 80 – 100 pairs.*

These teach the classifier that individual nouns are not inherently toxic when used neutrally.

| Domain | Toxic Usage | Clean Usage |
| :--- | :--- | :--- |
| **`imbwa` (dog)** | `Uri imbwa isina maturo.` *(You are a useless dog.)* | `Imbwa yangu iri kurara pamumvuri panze.` *(My dog is sleeping in the shade outside.)* |
| **`muroyi` (witch)** | `Uri muroyi chaiye, ndiwe wakauraya mwana.` *(You are a witch, you killed the child.)* | `Takadzidza nezve ngano dzevaroyi muchikoro.` *(We studied folklore stories of witches in school.)* |
| **`benzi` (fool)** | `Iwe uri benzi risingafunge nezve ramangwana.` *(You fool who doesn't think of the future.)* | `Kuseka nhamo serugare hazvirevi kupenga.` *(Laughing through hardship does not mean madness.)* |
| **`mbavha` (thief)** | `Uri mbavha yakaba mari yebasa.` *(You are a thief who stole work money.)* | `Mapurisa akasunga mbavha dzaipaza dzimba.` *(Police arrested thieves who were breaking into houses.)* |

---

### Gap 4: Modern Social Media Slang & ChiHarare
*Goal: 150 – 200 rows.*

Online discourse in Zimbabwe frequently uses contemporary street slang.

**Toxic Slang Attacks:**
- `Uyu munhu imboko chaiyo haana zvaanoziva.` *(This person is a total clown/fool, knows nothing.)* -> `toxic,insult,2,mboko`
- `Madhara embavha aya anongoba chete.` *(These thieving old men only steal.)* -> `toxic,insult,2,madhara`
- `Hure remunhu risinganyare.` *(A shameless whore of a person.)* -> `toxic,sexual_slur,4,hure`
- `Marara emunhu arikungotaura zvisina basa.` *(Trash of a person talking nonsense.)* -> `toxic,insult,2,marara`
- `Zvimbwanana zvavanhu zvisina pfungwa.` *(Puppies of people without brains.)* -> `toxic,insult,2,mbwa`

**Clean Street Slang:**
- `Ndeipi shamwari, zviri nani here?` *(What's up friend, is everything good?)* -> `non-toxic,benign_conversational,0,ndeipi`
- `Mudhara uyu ane moyo wakanaka zvikuru.` *(This elder/gentleman has a very good heart.)* -> `non-toxic,benign_conversational,0,moyo`
- `Mbinga iyi inobatsira vanoshaya.` *(This wealthy person helps the needy.)* -> `non-toxic,benign_conversational,0,mbinga`
- `Basa riri kufamba bhoo chose nhasi.` *(Work is moving great today.)* -> `non-toxic,benign_conversational,0,bhoo`

---

### Gap 5: Code-Switching / Shonglish
*Goal: 100 rows.*

Combinations of English and Shona common on platforms like X and WhatsApp.

**Toxic Shonglish:**
- `This guy is a muroyi chaiye, he ruined everything.` -> `toxic,insult,3,muroyi`
- `Stop behaving like a benzi and think properly.` -> `toxic,insult,2,benzi`
- `You are such an idiot remunhu.` -> `toxic,insult,2,idiot`
- `I will deal nawe zvekuti you will cry.` -> `toxic,threat,3,kudeal`

**Clean Shonglish:**
- `Thanks shamwari for the quick delivery.` -> `non-toxic,benign_conversational,0,thanks`
- `That was a great presentation yamaita nhasi.` -> `non-toxic,benign_conversational,0,presentation`
- `Happy birthday mukoma wangu, enjoy your day.` -> `non-toxic,benign_conversational,0,birthday`

---

## 5. Workflow: Adding and Training Your Data

### Step 1: Add Rows to the Template
Open [`data/raw/shona_native/custom_shona_template.csv`](file:///home/matthew-vonroy/meek/data/raw/shona_native/custom_shona_template.csv) in LibreOffice Calc, Excel, or VS Code, and add your rows following the schema.

### Step 2: Validate the Dataset
Run the automatic schema and integrity validator:
```bash
python ml/preprocessing/validate_custom_dataset.py data/raw/shona_native/custom_shona_template.csv
```
This checks for:
- Required column headers.
- Invalid categories or labels.
- Duplicate sentences.
- Imbalance warnings across categories.

### Step 3: Rebuild and Retrain
Once validated, re-run the build pipeline and retrain the classifier:
```bash
# 1. Merges raw data into balanced train/val/test splits
python ml/preprocessing/build_shona_training_data.py

# 2. Retrains the calibrated classifier in ~3 seconds
python ml/training/train_shona_classifier.py --mode calibrated

# 3. Runs the evaluation suite and prints precision/recall per label
python ml/evals/eval_shona_classifier.py
```
Your new words, threats, and slurs will immediately be learned by the model with zero code refactoring required.
