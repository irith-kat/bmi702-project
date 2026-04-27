---
name: latte-phenotyping
description: Run the LATTE semi-supervised GRU pipeline to phenotype longitudinal disease outcomes from EHR visit sequences. Use AFTER MAP has identified the study cohort. Supports two modes: incident timing (when did disease first start?) and recurring events (did the patient have a disease activity event in each period?).
---

# LATTE Phenotyping

LATTE is a semi-supervised GRU that predicts, for each patient and each time window, the probability of a disease outcome. It extends MAP (cross-sectional ever/never) to longitudinal outcome phenotyping.

There is an example pipeline in `examples/latte`

## When to Use LATTE

| MAP | LATTE |
|---|---|
| Does this patient have disease X? (ever/never) | Did disease X happen *at this time window*? |
| Cross-sectional, one label per patient | Longitudinal, one label per patient-period |
| No gold labels needed | Requires ~100 gold-labeled patients |

Use LATTE when you need **timing** or **disease activity** — not just presence. YOu need to ensure the time aggregation is defined, if not ask the user while planning.

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
Script: 0x_gold_labels.py
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
        Save to disk.

Script: 0x_latte_phenotyping.py
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

# ── Mode 1 (incident timing): include controls so Gemini distinguishes true cases
# from ICD miscodes.  Target ~60% case rate:
#   n_controls ≈ n_cases × 0.67
pools = map_prefilter(
    map_results    = map_results,
    n_cases        = ~100,         # MAP phenotype=1 → probable true cases
    n_controls     = ~67,          # MAP phenotype=0 + ICD-coded → probable miscodes
    seed           = 42,
    valid_sids     = valid_sids,
    preferred_sids = cached_sids,
)
cases_pool    = pools["cases_pool"]
controls_pool = pools["controls_pool"]   # present when n_controls > 0
unlabeled_pool = pools["unlabeled_pool"]

# ── Mode 2 (recurring events within a disease cohort): all patients already have
# the disease, so Y=0 comes from stable windows — no patient-level controls needed.
pools = map_prefilter(
    map_results    = map_results,
    n_cases        = ~120,
    n_controls     = 0, # use 0 for recurring events within a MAP cohort
    seed           = 42,
    valid_sids     = valid_sids,
    preferred_sids = cached_sids,
)
cases_pool     = pools["cases_pool"]
unlabeled_pool = pools["unlabeled_pool"]
```

## Step 2 — Gemini Labeling

Load notes from the cached file if available, fetch from database if not:

```python
notes_all = pd.read_parquet(out / "data" / "notes_raw.parquet")

# ── Mode 1 — incident: label BOTH cases and controls (Gemini adjudicates true HF
# vs ICD miscode).  Pass the union of both pools as subject_ids.
all_labeled_ids = list(set(cases_pool) | set(controls_pool))
notes_df = notes_all[notes_all["subject_id"].astype(str).isin(all_labeled_ids)].copy()
n_labeled = run_gemini_labeling(
    notes_df=notes_df, subject_ids=all_labeled_ids,
    cache_jsonl=CACHE_JSONL, config=HF_DISEASE_CONFIG,
    model_name=MODEL, project_id=PROJECT_ID, location=LOCATION,
    max_notes_per_patient=60, retry_delay_seconds=5.0,
)

# ── Mode 2 — recurring events: label cases only; use extra builders so Gemini
# returns one row per admission rather than one per patient.
notes_df = notes_all[notes_all["subject_id"].astype(str).isin(cases_pool)].copy()
n_labeled = run_gemini_labeling(
    notes_df=notes_df, subject_ids=cases_pool,
    cache_jsonl=CACHE_JSONL, config=HF_DECOMP_DISEASE_CONFIG,
    model_name=MODEL, project_id=PROJECT_ID, location=LOCATION,
    max_notes_per_patient=60, retry_delay_seconds=5.0,
    record_builder=build_result_record_recurring,
    system_instruction_builder=build_system_instruction_recurring,
)
```

Build your own `xxx_DISEASE_CONFIG` for your use case. Use the built-ins as templates:
`HF_DISEASE_CONFIG` — identifies the **first** admission with true HF (incident timing, Mode 1).
`HF_DECOMP_DISEASE_CONFIG` — identifies **every** admission with ADHF decompensation (recurring, Mode 2).

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

# Key codes for silver label initialisation — use the config that matches your mode.
# Mode 1 (incident):  DISEASE_CONFIG = HF_DISEASE_CONFIG
# Mode 2 (recurring): DISEASE_CONFIG = HF_DECOMP_DISEASE_CONFIG
raw_key_codes = DISEASE_CONFIG.key_codes
key_codes = [c for c in raw_key_codes if c in feature_codes] or [anchor]
key_codes_str = ",".join(key_codes)

# Single 80/20 split (single-run tuning or inference):
train_df, test_df, unlabeled_df = format_latte_input(
    obs_log=obs_log, gold_labels=gold_labels, feature_codes=feature_codes,
    baseline_date=BASELINE_DATE, month_window=MONTH_WINDOW,
    unlabeled_ids=unlabeled_ids, train_frac=0.8, seed=42,
)

# 5-fold CV (pass explicit patient lists; see 07_latte_cv.py for a full example):
train_df, test_df, unlabeled_df = format_latte_input(
    obs_log=obs_log, gold_labels=gold_labels, feature_codes=feature_codes,
    baseline_date=BASELINE_DATE, month_window=MONTH_WINDOW,
    unlabeled_ids=unlabeled_ids,
    train_patients=train_pats,   # list[str] — overrides train_frac when provided
    test_patients=test_pats,     # list[str]
)

embedding_df = build_cooccurrence_embeddings(obs_log, feature_codes, n_components=EMBEDDING_DIM)

# Adapt these parameters to your use case; see Hyperparameter Guide below.
predictions_df = run_latte(
    latte_dir=LATTE_DIR, data_dir=latte_data_dir,
    embedding_file=latte_data_dir + "embedding.csv",
    key_codes=key_codes_str,
    feature_col_start=3, feature_col_end=3 + len(feature_codes),
    save_dir=latte_results_dir, results_filename="results.csv",
    epochs=EPOCHS, epoch_silver=EPOCH_SILVER, embedding_dim=EMBEDDING_DIM,
    layers_incident=LAYERS_INCIDENT, month_window=MONTH_WINDOW, max_visits=25,
    weight_prevalence=0.2,
    weight_unlabel=0.015,       # CRITICAL: scale to n_labeled/n_unlabeled (see guide)
    weight_contrastive=0.1, weight_smooth=0.1,
    weight_additional=0.1, flag_train_augment=1,
)
```

Run this script multiple times while tuning hyperparameters. Once satisfied or after 15 iterations, use the best configuration for inference on the full cohort. For a stable AUC estimate over the full labeled set, use 5-fold stratified CV (`07_latte_cv.py` is the reference implementation; `format_latte_input` accepts `train_patients`/`test_patients` to drive explicit fold splits).

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

Adapt depending on use case, data distribution, label count, and computational resources.

| Parameter | Recommended | Notes |
|---|---|---|
| `n_cases` (gold labels) | 100–150 | Paper used 172; diminishing returns over ~100 |
| `n_controls` | 0 (Mode 2) / `n_cases × 0.60` (Mode 1) | Mode 1: controls = ICD miscodes; target ~60% case rate. Mode 2: Y=0 comes from stable windows — no patient-level controls. |
| `EMBEDDING_DIM` | 50 | Clamped to matrix rank automatically |
| `EPOCHS` | 35 | 50 overfits on small label sets (best checkpoint typically at ~35–43) |
| `EPOCH_SILVER` | 8 | 10 reduces joint training time with no benefit |
| `LAYERS_INCIDENT` | `"80"` for <150 labels; `"80,80"` for >200 | Single layer critical for small labeled sets — reduces overfitting |
| `MAX_UNLABELED` | 10,000 | Keep full pool; tune via `weight_unlabel` instead of cutting |
| `weight_unlabel` | `n_labeled / n_unlabeled` | **Most impactful param.** Default 0.2 causes gradient collapse with longitudinal data. Target effective weight ≈ 1.0 (see below) |
| `max_visits` | 25 | Set to 95th percentile of patient visit counts; default 115 wastes memory |

**`weight_unlabel` formula** — with `month_window=3` each patient contributes ~4–6 rows,
so the row ratio is much higher than the patient ratio:

```
effective_weight = (n_unlabeled × avg_visits / n_labeled × avg_visits) × weight_unlabel
                 ≈ (n_unlabeled / n_labeled) × weight_unlabel

# Target effective_weight ≈ 1.0:
weight_unlabel ≈ n_labeled / n_unlabeled
# Example: 120 labeled, 10k unlabeled → weight_unlabel = 0.012 → use 0.015
```

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
| OOM during LATTE training | Reduce `MAX_UNLABELED` to 10,000; reduce `max_visits` to 25 |
| Constant predictions (~0.485–0.5 for all patients, AUC≈0.5) | **Gradient collapse** from `weight_unlabel` too high. With `month_window=3`, each patient contributes ~5 rows, so the row ratio is ~5× the patient ratio. Set `weight_unlabel ≈ n_labeled / n_unlabeled` (e.g. 0.015 for 120 labeled / 10k unlabeled). Default 0.2 is calibrated for the simulation data only. |
