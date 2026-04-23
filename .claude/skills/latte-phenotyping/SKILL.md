---
name: latte-phenotyping
description: Run the LATTE semi-supervised GRU pipeline to phenotype longitudinal disease outcomes from EHR visit sequences. Use AFTER MAP has identified the study cohort. Supports two modes: incident timing (when did disease first start?) and recurring events (did the patient have a disease activity event in each period?).
---

# LATTE Phenotyping

LATTE is a semi-supervised GRU that predicts, for each patient and each time window, the probability of a disease outcome. It extends MAP (cross-sectional ever/never) to longitudinal outcome phenotyping.

## When to Use LATTE

| MAP | LATTE |
|---|---|
| Does this patient have disease X? (ever/never) | Did disease X happen *at this time window*? |
| Cross-sectional, one label per patient | Longitudinal, one label per patient-period |
| No gold labels needed | Requires ~100 gold-labeled patients |

Use LATTE when you need **timing** or **disease activity** — not just presence.

## Two Modes

### Mode 1 — Incident Timing
*When did disease first start for this patient?*

Gold labels: Gemini identifies the **first** admission where the disease appeared.
`labels_to_latte()` creates: Y=0 before incident window, Y=1 at/after.

### Mode 2 — Recurring Events
*Did the patient have a disease activity event in this time window?*

Gold labels: Gemini identifies **every** admission that qualifies as an event.
`recurring_labels_to_latte()` creates: Y=1 at windows with events, Y=0 at stable windows.

For Mode 2 within a MAP cohort (all patients have the disease), `n_controls=0` is acceptable — Y=0 comes from stable windows.

## Silver Label

The silver label anchors LATTE's unsupervised pretraining via `key_codes`. Choose a code that is ordered *in response to* the event (not on a fixed schedule):

| Disease / outcome | Silver label anchor | Reasoning |
|---|---|---|
| HF incident | `PheCode:428.1` | HF diagnosis code presence |
| HF decompensation | `LOINC:33762-6`, `ShortName:BNP` | BNP ordered when decompensation suspected |
| MS relapse | MRI procedure code | MRI ordered when relapse suspected |

## Pipeline

```
map_results.parquet
        ↓
map_prefilter()            ← sample gold label candidates from MAP cases
        ↓
run_gemini_labeling()      ← Gemini reviews discharge notes
        ↓
parse_gemini_results()         ← Mode 1
parse_gemini_recurring_results() ← Mode 2
        ↓
labels_to_latte()              ← Mode 1
recurring_labels_to_latte()   ← Mode 2
        ↓
format_latte_input() + build_cooccurrence_embeddings()
        ↓
run_latte()
        ↓
latte_predictions.parquet
```

---

## Step 1 — map_prefilter

Pre-check which patients have discharge notes before sampling, to avoid silent skips:

```python
import pandas as pd

map_results = pd.read_parquet(out / "data" / "map_results.parquet")
notes_raw   = pd.read_parquet(out / "data" / "notes_raw.parquet", columns=["subject_id"])
valid_sids  = set(notes_raw["subject_id"].astype(str).unique())

# Prioritise patients already in the Gemini cache to avoid re-labeling
cached_sids = get_cached_subject_ids(CACHE_JSONL)

pools = map_prefilter(
    map_results   = map_results,
    n_cases       = 120,         # MAP cases to send to Gemini
    n_controls    = 0,           # use 0 for recurring events within a MAP cohort
    seed          = 42,
    valid_sids    = valid_sids,       # only sample patients with notes
    preferred_sids = cached_sids,     # reuse cached Gemini responses first
)
cases_pool     = pools["cases_pool"]
unlabeled_pool = pools["unlabeled_pool"]
```

## Step 2 — Gemini Labeling

Load notes from the cached `notes_raw.parquet` (no new BigQuery fetch needed):

```python
notes_all = pd.read_parquet(out / "data" / "notes_raw.parquet")
notes_df = notes_all[notes_all["subject_id"].astype(str).isin(cases_pool)].copy()

# Mode 1 — incident
n_labeled = run_gemini_labeling(
    notes_df=notes_df, subject_ids=cases_pool,
    cache_jsonl=CACHE_JSONL, config=HF_DISEASE_CONFIG,
    model_name=MODEL, project_id=PROJECT_ID, location=LOCATION,
    max_notes_per_patient=60, retry_delay_seconds=5.0,
)

# Mode 2 — recurring events (pass additional builders)
n_labeled = run_gemini_labeling(
    notes_df=notes_df, subject_ids=cases_pool,
    cache_jsonl=CACHE_JSONL, config=HF_DECOMP_DISEASE_CONFIG,
    model_name=MODEL, project_id=PROJECT_ID, location=LOCATION,
    max_notes_per_patient=60, retry_delay_seconds=5.0,
    record_builder=build_result_record_recurring,
    system_instruction_builder=build_system_instruction_recurring,
)
```

## Step 3 — Parse Results and Convert to LATTE Labels

```python
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")

# Mode 1
results     = parse_gemini_results(CACHE_JSONL, baseline_date=BASELINE_DATE, month_window=MONTH_WINDOW)
gold_labels = labels_to_latte(results, obs_log, BASELINE_DATE, MONTH_WINDOW)

# Mode 2
results     = parse_gemini_recurring_results(CACHE_JSONL, baseline_date=BASELINE_DATE, month_window=MONTH_WINDOW)
gold_labels = recurring_labels_to_latte(results, obs_log, BASELINE_DATE, MONTH_WINDOW)
```

## Step 4 — Format Input and Run LATTE

Load feature codes from `once_features_meta.json` (saved by the feature matrix script):

```python
import json, os, random

meta = json.loads((out / "data" / "once_features_meta.json").read_text())
feature_codes = meta.get("feature_codes", meta["codified_list"])
anchor = meta["anchor"]

# Cap unlabeled pool to avoid OOM in TF
MAX_UNLABELED = 10_000
unlabeled_df  = pd.read_parquet(out / "data" / "unlabeled_pool.parquet")
unlabeled_ids = unlabeled_df["subject_id"].astype(str).tolist()
if len(unlabeled_ids) > MAX_UNLABELED:
    random.seed(42)
    unlabeled_ids = random.sample(unlabeled_ids, MAX_UNLABELED)

# Key codes for silver label initialisation — fall back to anchor if not in ONCE features
raw_key_codes = HF_DECOMP_DISEASE_CONFIG.key_codes  # e.g. ["LOINC:33762-6"]
key_codes = [c for c in raw_key_codes if c in feature_codes] or [anchor]
key_codes_str = ",".join(key_codes)

train_df, test_df, unlabeled_df = format_latte_input(
    obs_log=obs_log, gold_labels=gold_labels, feature_codes=feature_codes,
    baseline_date=BASELINE_DATE, month_window=MONTH_WINDOW,
    unlabeled_ids=unlabeled_ids, train_frac=0.8, seed=42,
)

embedding_df = build_cooccurrence_embeddings(obs_log, feature_codes, n_components=EMBEDDING_DIM)

# Adapt this parameters according to your usecase and computational resources, and refine based on execution results
predictions_df = run_latte(
    latte_dir=LATTE_DIR, data_dir=latte_data_dir,
    embedding_file=latte_data_dir + "embedding.csv",
    key_codes=key_codes_str,
    feature_col_start=3, feature_col_end=3 + len(feature_codes),
    save_dir=latte_results_dir, results_filename="results.csv",
    epochs=EPOCHS, epoch_silver=EPOCH_SILVER, embedding_dim=EMBEDDING_DIM,
    layers_incident=LAYERS_INCIDENT, month_window=MONTH_WINDOW, max_visits=25,
    weight_prevalence=0.2, weight_unlabel=0.2,
    weight_contrastive=0.1, weight_smooth=0.1,
)
```

---

## Silver Label

The silver label anchors LATTE's unsupervised pretraining via `key_codes`. Choose a code that is ordered *in response to* the event, not on a fixed schedule:

| Outcome | Silver label | Reasoning |
|---|---|---|
| HF incident | `PheCode:428.1` | HF diagnosis code presence |
| HF decompensation | `LOINC:33762-6`, `ShortName:BNP` | BNP ordered when decompensation suspected |
| MS relapse | MRI procedure code | MRI ordered when relapse suspected |

If the chosen codes are not in `feature_codes`, fall back to the anchor PheCode.

---

## Hyperparameter Guide

Adapt depending on usecase, data distribution, amount of labels and computational resources.

| Parameter | Value |
|---|---|
| `n_cases` (gold labels) | 100-150 |
| `EMBEDDING_DIM` | 20-50 |
| `EPOCHS` | 30-50 |
| `EPOCH_SILVER` | 5-10 |
| `LAYERS_INCIDENT` | `"40"`-`"80,80"` |
| `MAX_UNLABELED` | 10,000 |

## Running Scripts

LATTE scripts can take 30 min – 2 h. Always execute with `run_in_background: true`:

```bash
uv run python output/<study>/scripts/06_latte_phenotyping.py \
  > output/<study>/scripts/06_latte_phenotyping.log 2>&1
```

Read the `.log` file after the notification arrives to confirm success or diagnose errors.

## File Locations

```
src/latte/
├── latte.py           — format_latte_input, run_latte
├── embeddings.py      — build_cooccurrence_embeddings
├── gemini.py          — run_gemini_labeling, parse_gemini_results,
│                        parse_gemini_recurring_results, get_cached_subject_ids
└── labeler_utils.py   — map_prefilter, labels_to_latte, recurring_labels_to_latte,
                         HF_DISEASE_CONFIG, HF_DECOMP_DISEASE_CONFIG, DiseaseConfig
```

## Common Pitfalls

| Problem | Fix |
|---|---|
| Key codes not in feature_codes | Fall back to anchor; check ONCE file coverage |
| Empty gold_labels | `baseline_date` mismatch between scripts; verify all use the same value |
| All Y=1 in recurring mode | Gemini over-labeling; review `diagnostic_criteria` in DiseaseConfig |
| `parse_error=True` in results | Delete that patient's line from cache JSONL and re-run |
| Gemini 404 | Use `location="global"` for gemini-3.1-flash-lite-preview |
| OOM during LATTE training | Reduce `MAX_UNLABELED`; cap at 10,000 |
