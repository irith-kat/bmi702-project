---
name: map-phenotyping
description: Run the MAP (Multimodal Automated Phenotyping) algorithm on a prepared observation log to produce per-patient phenotype scores and binary case/control labels. Use after loading ONCE features and building an observation log from structured EHR data.
---

# MAP Phenotyping

MAP fits a Poisson mixture model over ONCE-selected co-features to assign each patient a posterior probability of having the target phenotype. It is more specific than raw ICD filtering because it requires co-feature support.

## When to Use

- Building a validated disease cohort from EHR data
- Improving specificity over a raw ICD baseline
- Producing per-patient probability scores for downstream analysis or LATTE

## Pipeline Position

```
get_once_features()          ← ONCE files (codified + narrative CSVs)
        ↓
build_obs_log()              ← observation log (structured ± NLP)
        ↓
preprocess_map()             ← mat_df + note_df
        ↓
run_map()                    ← scores + phenotype labels
```

## Imports

```python
import glob, json
from pathlib import Path
import pandas as pd
from preprocessing.nlp import get_once_features
from map import preprocess_map, run_map
```

---

## Step 1 — Load ONCE Features

ONCE files are **user-provided** in `input/`. Use glob to find them. Do not generate them, ask the user to upload the correct files based on their study design.

```python
codified_files  = sorted(glob.glob("input/ONCE_*PheCode*.csv"))
narrative_files = sorted(glob.glob("input/ONCE_*_C[0-9]*.csv"))

once_features = get_once_features(codified_files[0], narrative_files[0])
# Keys:
#   once_features["codified_list"]    — prefixed codes e.g. "PheCode:714.1"
#   once_features["nlp_target_cuis"]  — [{"term": ..., "cui": ...}, ...]
#   once_features["nlp_list"]         — CUI strings for embedding

```

## Step 2 — Build Observation Log

See the **mimic-preprocessing** and **mimic-note-preprocessing** skills. The obs_log must have columns: `subject_id, event_type, event, value, datetime`.

NLP CUI events (from `cui_obs.parquet`) are concatenated into the obs_log before calling `preprocess_map`:

```python
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
cui_obs_path = out / "data" / "cui_obs.parquet"
if cui_obs_path.exists():
    obs_log = pd.concat([obs_log, pd.read_parquet(cui_obs_path)], ignore_index=True)
```

## Step 3 — preprocess_map

```python
mat_df, note_df = preprocess_map(
    obs_log       = obs_log,
    admissions_df = admissions_df,   # for note count proxy; pass None if unavailable
    once_features = once_features,
    main_phecode  = "714.1",        # anchor PheCode (without prefix)
    subject_col   = "subject_id",   # default
    min_nonzero   = 20,             # drop features seen in <20 patients (prevents EM failure)
)
# mat_df  : wide patient × feature count matrix; index = subject_id
# note_df : per-patient note count (Poisson denominator)
```

`min_nonzero=20` is the correct default for full datasets. Use `5` for demo/pilot runs.

## Step 4 — run_map

```python
map_results = run_map(mat_df, note_df, anchor)

# map_results columns: patient_id, score (0–1), phenotype (1=case, 0=control)

# MAP (via R) returns patient_id as int; normalize to str to match BigQuery string IDs
map_results["patient_id"] = map_results["patient_id"].astype(str)
```

---

## Interpreting Results

```python
anchor_flag = (mat_df[anchor] > 0).rename("icd_coded").reset_index()
anchor_flag.columns = ["patient_id", "icd_coded"]
anchor_flag["patient_id"] = anchor_flag["patient_id"].astype(str)
map_results = map_results.merge(anchor_flag, on="patient_id", how="left")

icd_only  = ((map_results["icd_coded"]) & (map_results["phenotype"] == 0)).sum()
map_only  = (~map_results["icd_coded"] & (map_results["phenotype"] == 1)).sum()
```

- **`icd_only`** — ICD-coded but MAP-rejected: probable false positives
- **`map_only`** — MAP cases not ICD-coded: requires NLP co-features to be non-zero
- **Score > 0.8** — high-confidence cases
- **Score < 0.2** — high-confidence controls

---

## Script Structure

For a MAP-only study:

```
01_cohort_definition.py  ← fetch raw tables; cache parquets
02_notes_nlp.py          ← (optional) MedSpaCy NER → cui_obs.parquet
03_feature_matrix.py     ← build obs_log; rollup labs; concat NLP; preprocess_map; save mat_df + meta
04_map_phenotyping.py    ← run_map; save map_results.parquet
```

For MAP + LATTE, add:

```
05_gold_labels.py        ← Gemini labeling → gold_labels.parquet
06_latte_phenotyping.py  ← LATTE training → latte_predictions.parquet
```

refer to the latte-phenotyping skill for details on steps 5–6.

## Running Scripts

MAP scripts can take 30 min – 2 h. Always execute with `run_in_background: true`:

```bash
uv run python output/<study>/scripts/04_map_phenotyping.py \
  > output/<study>/scripts/04_map_phenotyping.log 2>&1
```

Read the `.log` file after the notification arrives to confirm success or diagnose errors.

## Key Constraints

| Constraint | Detail |
|---|---|
| Anchor must be in ONCE codified_list | Assert before running |
| Requires R + MAP package | `run_map` shells out to `Rscript` |
| `min_nonzero=20` | Prevents flexmix EM from returning NA log-likelihood on sparse features |
| Normalize patient_id to str | MAP (R) returns int; join targets are typically str from BigQuery |

## Reporting After MAP Runs

- **CONSORT flow:** total subjects → candidates (≥1 ONCE codified event) → MAP cases → MAP controls
- **Score distribution:** bimodal histogram (cases cluster near 1, controls near 0)
- **Feature prevalence:** top ONCE features cases vs. controls as a bar chart

### Protocol Template — Phenotyping Approach Section

When drafting a research protocol for a MAP study, include this section:

```markdown
### Phenotyping Approach
**Method:** MAP (Multimodal Automated Phenotyping)
**ONCE files:** [codified file name, narrative file name]
**NLP:** [Yes — clinical notes / No — structured EHR only]
**Anchor PheCode:** [e.g. 455 — Hemorrhoids]
```
