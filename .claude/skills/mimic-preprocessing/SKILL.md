---
name: mimic-preprocessing
description: Preprocess structured MIMIC-IV data (diagnoses, prescriptions, procedures) into a standardized observation log using icd_to_events, rxnorm_to_events, cpt_to_events, and build_obs_log. Use at the start of any phenotyping workflow before running MAP.
---

# MIMIC Structured Data Preprocessing

All functions in `src/preprocessing/preprocessing.py` produce a standardized **observation log** — a long-format DataFrame with five columns:

| Column | Type | Description |
|---|---|---|
| `subject_id` | int | Patient identifier |
| `event_type` | str | Modality: `"phecode"`, `"rxnorm"`, `"ccs"`, `"cui"` |
| `event` | str | Prefixed code: `"PheCode:714.1"`, `"RXNORM:1049630"`, `"CCS:3"` |
| `value` | float\|None | Numeric result (labs); None for categorical events |
| `datetime` | datetime | When the observation occurred |

## Setup

```python
import sys
sys.path.insert(0, "path/to/src/preprocessing")
from preprocessing import icd_to_events, rxnorm_to_events, cpt_to_events, build_obs_log
```

## ICD Diagnoses → PheCodes

Rolls up ICD-9/10 codes to PheCodes. Rows with no mapping are silently dropped.
Requires MIMIC-IV `diagnoses_icd` joined with `admissions` for dates.

```python
# Join diagnoses with admissions to get dates
diagnoses_with_dates = diagnoses_df.merge(
    admissions_df[["hadm_id", "admittime"]], on="hadm_id", how="left"
)

icd_obs = icd_to_events(
    df           = diagnoses_with_dates,
    icd_col      = "icd_code",
    date_col     = "admittime",
    subject_col  = "subject_id",         # default
    mapping_file = "Phecode_map_v1_2_icd9_icd10cm.csv",
)
# event format: "PheCode:714.1"
```

## RxNorm Prescriptions

No rollup applied — codes are used at the RxNorm concept level as-is.

```python
rx_obs = rxnorm_to_events(
    df          = prescriptions_df,
    rxnorm_col  = "drug_rxnorm",
    date_col    = "starttime",
    subject_col = "subject_id",   # default
)
# event format: "RXNORM:1049630"
# Rows with null RxNorm codes are dropped.
```

## CPT/HCPCS Procedures → CCS

Rolls up CPT/HCPCS codes to AHRQ CCS categories using alphanumeric range matching.

```python
cpt_obs = cpt_to_events(
    df           = procedures_df,
    cpt_col      = "hcpcs_cd",
    date_col     = "chartdate",
    subject_col  = "subject_id",   # default
    mapping_file = "CCS_Services_Procedures_v2025-1.csv",
)
# event format: "CCS:3"
```

## Combined: build_obs_log

Builds the full observation log from any subset of modalities in one call.

```python
obs_log = build_obs_log(
    icd_df          = diagnoses_with_dates,
    icd_col         = "icd_code",
    icd_date_col    = "admittime",
    rxnorm_df       = prescriptions_df,
    rxnorm_col      = "drug_rxnorm",
    rxnorm_date_col = "starttime",
    # cpt_df, notes_df — omit if not needed
    subject_col      = "subject_id",
    icd_mapping_file = "Phecode_map_v1_2_icd9_icd10cm.csv",
)
```

Omit any modality argument to skip it. Raises `ValueError` if all are None.

## Filtering to ONCE Features (optional — required only for MAP)

ONCE feature files are **user-provided** and already present in `src/`. If using MAP, filter the obs_log to patients with at least one ONCE-selected feature — these are the **candidates**:

```python
from once import get_once_features

# Files are already in src/ — pass absolute paths
once_features = get_once_features(codified_file, narrative_file)
once_events   = set(once_features["codified_list"])   # e.g. {"PheCode:714.1", "RXNORM:614391", ...}

candidate_ids = set(icd_obs[icd_obs["event"].isin(once_events)]["subject_id"])
```

`preprocess_map` performs this filtering internally, but pre-filtering to candidates is useful for scoping note loading (NLP should only run on candidates).

If using a **rule-based filter instead of MAP**, ONCE is not needed — filter the obs_log directly by the anchor event.

## Mapping Files

Both files are **user-provided** and already present in `src/`. Pass absolute paths to avoid CWD dependency.

| File | Used by | Notes |
|---|---|---|
| `Phecode_map_v1_2_icd9_icd10cm.csv` | `icd_to_events` | ICD-9 and ICD-10-CM → PheCode |
| `CCS_Services_Procedures_v2025-1.csv` | `cpt_to_events` | AHRQ CCS alphanumeric range matching |
