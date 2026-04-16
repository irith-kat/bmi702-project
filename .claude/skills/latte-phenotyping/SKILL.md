---
name: latte-phenotyping
description: Run the LATTE semi-supervised GRU pipeline to phenotype longitudinal disease outcomes from EHR visit sequences. Use AFTER MAP has identified the study cohort. Supports two modes: incident timing (when did disease first start?) and recurring events (did the patient have a disease activity event in each period?).
---

# LATTE Phenotyping

LATTE is a semi-supervised GRU that predicts, for each patient and each time window, the probability of a disease outcome. It extends MAP (cross-sectional ever/never) to longitudinal outcome phenotyping.

## When to Use LATTE

**LATTE answers a different question than MAP.**

| MAP | LATTE |
|---|---|
| Does this patient have disease X? (ever/never) | Did disease X happen *at this time window*? |
| Cross-sectional, one label per patient | Longitudinal, one label per patient-period |
| No gold labels needed | Requires ~100 gold-labeled patient-periods |

Use LATTE when you need **timing** or **disease activity** — not just presence.

## Two Modes

### Mode 1 — Incident Timing
*When did the disease first start for this patient?*

Gold labels: Gemini identifies the **first** admission where the disease appeared.
`labels_to_latte()` creates: Y=0 before incident window, Y=1 at/after.

Use when: you want to detect disease onset earlier than the first ICD code appears.

### Mode 2 — Recurring Events
*Did the patient have a disease activity event in this 3-month window?*

Gold labels: Gemini identifies **every** admission that qualifies as an event.
`recurring_labels_to_latte()` creates: Y=1 at windows with events, Y=0 at stable windows.

Use when: the disease is chronic and you want to phenotype activity (relapses, decompensations, flares).

**Controls (label=0 patients) strengthen the supervised loss** — map_prefilter's `n_controls` samples MAP-rejected ICD patients as hard negatives. For Mode 2 (recurring events within a disease cohort), controls may be sparse; `n_controls=0` is acceptable when all cohort patients have the disease and Y=0 comes from stable windows.

## Silver Label

The silver label anchors LATTE's unsupervised pretraining via `key_codes`. Choose a code that is ordered *in response to* the event (not on a fixed schedule):

| Disease / outcome | Silver label anchor | Reasoning |
|---|---|---|
| HF incident | `PheCode:428.1` | HF diagnosis code presence |
| HF decompensation | `LOINC:33762-6`, `ShortName:BNP` | BNP ordered when decompensation suspected |
| MS relapse | MRI procedure code | MRI ordered when relapse suspected |

## Pipeline

```
MAP cohort (map_results.parquet)
        ↓
map_prefilter()            ← sample gold label candidates from MAP cases
        ↓
run_gemini_labeling()      ← Gemini reviews discharge notes
                               Mode 1: identify incident admission
                               Mode 2: identify all event admissions
        ↓
labels_to_latte()              ← Mode 1 (incident)
recurring_labels_to_latte()   ← Mode 2 (recurring events)
        ↓
format_latte_input()       ← obs_log + gold_labels → train/test/unlabeled CSVs
build_cooccurrence_embeddings()
run_latte()
        ↓
latte_predictions.parquet  ← per-patient, per-window outcome probability
```

## Key Code Examples

### Mode 1 — Incident
```python
from latte.labeler_utils import HF_DISEASE_CONFIG, labels_to_latte, map_prefilter
from latte.gemini import run_gemini_labeling, parse_gemini_results

pools = map_prefilter(map_results, n_cases=100, n_controls=0)
run_gemini_labeling(..., config=HF_DISEASE_CONFIG)
results = parse_gemini_results(cache_jsonl, baseline_date)
gold_labels = labels_to_latte(results, obs_log, baseline_date)
```

### Mode 2 — Recurring Events
```python
from latte.labeler_utils import (HF_DECOMP_DISEASE_CONFIG, recurring_labels_to_latte,
                                  map_prefilter, build_result_record_recurring,
                                  build_system_instruction_recurring)
from latte.gemini import run_gemini_labeling, parse_gemini_recurring_results

pools = map_prefilter(map_results, n_cases=100, n_controls=0)
run_gemini_labeling(..., config=HF_DECOMP_DISEASE_CONFIG,
                    record_builder=build_result_record_recurring,
                    system_instruction_builder=build_system_instruction_recurring)
results = parse_gemini_recurring_results(cache_jsonl, baseline_date)
gold_labels = recurring_labels_to_latte(results, obs_log, baseline_date)
```

## Hyperparameter Guide

| Parameter | Test run | Full run |
|---|---|---|
| `n_cases` (gold labels) | 30 | 100–150 |
| `epochs` | 30 | 50 |
| `epoch_silver` | 5 | 8–10 |
| `embedding_dim` | 20 | 50 |
| `layers_incident` | `"40"` | `"80,80"` |

## File Locations

```
src/latte/
├── latte.py           — format_latte_input, run_latte, compute_abcgain
├── embeddings.py      — build_cooccurrence_embeddings
├── gemini.py          — run_gemini_labeling, parse_gemini_results, parse_gemini_recurring_results
└── labeler_utils.py   — map_prefilter, labels_to_latte, recurring_labels_to_latte,
                         HF_DISEASE_CONFIG, HF_DECOMP_DISEASE_CONFIG, DiseaseConfig
```

## Common Pitfalls

| Problem | Fix |
|---|---|
| specific key_codes not in feature_codes | Fall back to anchor; check ONCE file coverage |
| Empty gold_labels | `baseline_date` mismatch between scripts; verify all use same anchor |
| All Y=1 in recurring mode | Gemini over-labeling; review `diagnostic_criteria` in DiseaseConfig |
| `parse_error=True` in results | Delete that patient's line from cache JSONL and re-run |
| Gemini 404 | Use `location="global"` for gemini-3.1-flash-lite-preview |
