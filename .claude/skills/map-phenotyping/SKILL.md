---
name: map-phenotyping
description: Run the MAP (Multimodal Automated Phenotyping) algorithm on a prepared observation log to produce per-patient phenotype scores and binary case/control labels. Use after loading ONCE features and building an observation log from structured EHR data.
---

# MAP Phenotyping

MAP fits a Poisson mixture model over ONCE-selected co-features to assign each patient a posterior probability of having the target phenotype. It is more specific than raw ICD filtering because it requires co-feature support.

## When to Use This Skill

- Building a validated disease cohort from MIMIC-IV
- Improving specificity over a raw ICD baseline
- Producing per-patient probability scores for downstream analysis

## Pipeline Position

```
get_once_features()          ← ONCE files (codified + narrative CSVs)
        ↓
build_obs_log() / icd_to_events() + notes_to_events()   ← observation log
        ↓
preprocess_map()             ← mat_df + note_df
        ↓
run_map()                    ← scores + phenotype labels
```

## Step 1 — Load ONCE Features

ONCE feature files are **user-provided** and should be placed in `input/`. Find them with glob — do not generate or download them.

```python
import glob
from once import get_once_features

# ONCE files follow naming conventions — discover them rather than hardcoding
codified_files  = glob.glob("input/ONCE_*PheCode*.csv")   # codified features
narrative_files = glob.glob("input/ONCE_*_C[0-9]*.csv")  # narrative/CUI features

once_features = get_once_features(codified_files[0], narrative_files[0])
# Keys used by MAP:
#   once_features["codified_list"]     — prefixed feature strings e.g. "PheCode:714.1"
#   once_features["nlp_target_cuis"]   — [{"term": ..., "cui": ...}, ...]
```

## Step 2 — Build Observation Log

See the **mimic-preprocessing** and **mimic-note-preprocessing** skills. The obs_log must have columns: `subject_id, event_type, event, value, datetime`.

## Step 3 — preprocess_map

```python
from map import preprocess_map

mat_df, note_df = preprocess_map(
    obs_log       = obs_log,        # output of build_obs_log()
    admissions_df = admissions_df,  # MIMIC-IV admissions (for note count proxy)
    once_features = once_features,
    main_phecode  = "714.1",        # anchor PheCode (without prefix)
    subject_col   = "subject_id",   # default
    min_nonzero   = 20,             # drop features seen in <20 patients (prevents EM failure)
)
# mat_df  : wide patient × feature count matrix; index = subject_id
#           first column is always the anchor e.g. "PheCode:714.1"
# note_df : per-patient note count (Poisson denominator); column "note_count"
```

## Step 4 — run_map

```python
from map import run_map

anchor_col = f"PheCode:{MAIN_PHECODE}"   # must match a column in mat_df
map_results = run_map(mat_df, note_df, anchor_col)

# Returns DataFrame with columns:
#   patient_id  — subject_id
#   score       — posterior probability (0–1)
#   phenotype   — 1 = case, 0 = control
```

## Key Constraints

| Constraint | Detail |
|---|---|
| Anchor must be in ONCE features | `"PheCode:714.1"` must appear in `once_features["codified_list"]` |
| Requires R + MAP package | `run_map` shells out to `Rscript map_runner.R` |
| `min_nonzero=20` | Features seen in fewer patients are dropped to prevent EM returning NA |
| Note count ≥ 1 | Patients absent from `admissions_df` receive `note_count=1` |
| MAP is a specificity filter without NLP | Patients with anchor=0 will almost never be cases; NLP adds sensitivity |

## Interpreting Results

- **High score (>0.9)** — strong co-feature support, likely true cases
- **Low score (<0.1)** — little co-feature support, likely controls or miscoded
- **`icd_only` group** — patients ICD-coded but MAP-rejected → probable false positives
- **`map_only` group** — patients MAP found that ICD missed → requires NLP to be non-zero
