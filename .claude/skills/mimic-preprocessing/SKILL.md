---
name: mimic-preprocessing
description: Preprocess structured MIMIC-IV data (diagnoses, prescriptions, procedures, lab events) into a standardized observation log using icd_to_events, drug_to_events, cpt_to_events, lab_to_events, and build_obs_log. Use at the start of any phenotyping workflow before running MAP.
---

# MIMIC Structured Data Preprocessing

All functions in `src/preprocessing/structured/preprocessing.py` produce a standardized **observation log** — a long-format DataFrame with five columns:

| Column | Type | Description |
|---|---|---|
| `subject_id` | int | Patient identifier |
| `event_type` | str | Modality: `"phecode"`, `"rxnorm"`, `"ccs"`, `"loinc"`, `"cui"` |
| `event` | str | Prefixed code: `"PheCode:714.1"`, `"RXNORM:956874"`, `"CCS:3"`, `"LOINC:11555-0"` |
| `value` | float\|None | Numeric result (labs); None for categorical events |
| `datetime` | datetime | When the observation occurred |

## Setup

```python
import sys
from src.preprocessing.structured.preprocessing import icd_to_events, drug_to_events, cpt_to_events, lab_to_events, build_obs_log
from src.preprocessing.structured.rollup import rollup_itemid_to_loinc
```

## ICD Diagnoses → PheCodes

Rolls up ICD-9/10 codes to PheCodes. Rows with no mapping are silently dropped.
Requires MIMIC-IV `diagnoses_icd` joined with `admissions` for dates.

```python
diagnoses_with_dates = diagnoses_df.merge(
    admissions_df[["hadm_id", "admittime"]], on="hadm_id", how="left"
)

icd_obs = icd_to_events(
    df           = diagnoses_with_dates,
    icd_col      = "icd_code",
    date_col     = "admittime",
    subject_col  = "subject_id",                                    # default
    mapping_file = "mapping_dicts/Phecode_map_v1_2_icd9_icd10cm.csv",
)
# event format: "PheCode:714.1"
```

## Prescriptions → RxNorm Ingredients

Rolls up NDC codes to RxNorm ingredient level internally. NDC coverage in MIMIC-IV
is ~97.7%; the `drug` column is used as a case-insensitive fallback for the remaining
~2.3%. Filter to `drug_type == 'MAIN'` before passing to avoid double-counting
multi-component drugs (MIMIC rows also have BASE and ADDITIVE entries).

```python
prescriptions_main = prescriptions_df[prescriptions_df["drug_type"] == "MAIN"].copy()

rx_obs = drug_to_events(
    df                     = prescriptions_main,
    ndc_col                = "ndc",
    date_col               = "starttime",
    subject_col            = "subject_id",    # default
    drug_col               = "drug",          # optional fallback via gcpt_drug_ndc
    ndc_mapping_file       = "mapping_dicts/ndc_to_rxnorm_ingredient.csv",
    drug_name_mapping_file = "mapping_dicts/drug_name_to_rxnorm_ingredient.csv",
)
# event format: "RXNORM:956874"
# Rows where neither NDC nor drug-name lookup resolves are dropped.
```

## CPT/HCPCS Procedures → CCS

Rolls up CPT/HCPCS codes to AHRQ CCS categories using alphanumeric range matching.

```python
cpt_obs = cpt_to_events(
    df           = procedures_df,
    cpt_col      = "hcpcs_cd",
    date_col     = "chartdate",
    subject_col  = "subject_id",    # default
    mapping_file = "mapping_dicts/CCS_Services_Procedures_v2025-1_052425.csv",
)
# event format: "CCS:3"
```

## Lab Events → LOINC

Rolls up MIMIC-IV `labevents` itemids to LOINC codes using the MIT-LCP OMOP mapping
(`mapping_dicts/d_labitems_to_loinc.csv`). Call `rollup_itemid_to_loinc()` first to
join LOINC codes, then pass the result to `lab_to_events()`. Rows with no LOINC mapping
are silently dropped.

```python
from rollup import rollup_itemid_to_loinc

labevents_with_loinc = rollup_itemid_to_loinc(
    df             = labevents_df,
    itemid_column  = "itemid",
    mapping_file   = "mapping_dicts/d_labitems_to_loinc.csv",
)

lab_obs = lab_to_events(
    df          = labevents_with_loinc,
    loinc_col   = "loinc_code",
    date_col    = "charttime",
    value_col   = "valuenum",     # default
    subject_col = "subject_id",   # default
)
# event format: "LOINC:11555-0"
# value: numeric measurement (float); rows with no LOINC match are dropped
```

## Combined: build_obs_log

Builds the full observation log from any subset of modalities in one call.
Internally calls `icd_to_events`, `drug_to_events`, `cpt_to_events`, and
`notes_to_events` — no need to call those separately when using `build_obs_log`.

```python
obs_log = build_obs_log(
    # ── ICD diagnoses (optional) ──────────────────────────────────────────────
    icd_df          = diagnoses_with_dates,   # None to skip
    icd_col         = "icd_code",             # required if icd_df provided
    icd_date_col    = "admittime",            # required if icd_df provided

    # ── Prescriptions / NDC → RxNorm (optional) ───────────────────────────────
    drug_df         = prescriptions_main,     # None to skip; filter to drug_type='MAIN' first
    drug_ndc_col    = "ndc",                  # required if drug_df provided
    drug_date_col   = "starttime",            # required if drug_df provided
    drug_col        = "drug",                 # optional fallback for unresolved NDCs

    # ── CPT/HCPCS procedures (optional) ───────────────────────────────────────
    cpt_df          = procedures_df,          # None to skip
    cpt_col         = "hcpcs_cd",             # required if cpt_df provided
    cpt_date_col    = "chartdate",            # required if cpt_df provided

    # ── Clinical notes / NLP CUIs (optional) ──────────────────────────────────
    notes_df        = notes_df,               # None to skip
    notes_text_col  = "text",                 # required if notes_df provided
    notes_date_col  = "chartdate",            # required if notes_df provided
    target_cuis     = once_features["nlp_target_cuis"],  # required if notes_df provided

    # ── Shared options ─────────────────────────────────────────────────────────
    subject_col      = "subject_id",          # default; must match across all tables
    icd_mapping_file = "mapping_dicts/Phecode_map_v1_2_icd9_icd10cm.csv",          # default
    cpt_mapping_file = "mapping_dicts/CCS_Services_Procedures_v2025-1_052425.csv", # default
    ndc_mapping_file = "mapping_dicts/ndc_to_rxnorm_ingredient.csv",               # default
    drug_name_mapping_file = "mapping_dicts/drug_name_to_rxnorm_ingredient.csv",   # default
)
```

Raises `ValueError` if all modality DataFrames are None.

## Mapping Files

All mapping files live in `mapping_dicts/`. Pass absolute paths to avoid CWD dependency.

| File | Used by | Notes |
|---|---|---|
| `mapping_dicts/Phecode_map_v1_2_icd9_icd10cm.csv` | `icd_to_events` | ICD-9 and ICD-10-CM → PheCode |
| `mapping_dicts/CCS_Services_Procedures_v2025-1_052425.csv` | `cpt_to_events` | AHRQ CCS alphanumeric range matching |
| `mapping_dicts/ndc_to_rxnorm_ingredient.csv` | `drug_to_events` | 11-digit NDC → RxNorm ingredient (~97.7% MIMIC coverage) |
| `mapping_dicts/drug_name_to_rxnorm_ingredient.csv` | `drug_to_events` | Drug name fallback via gcpt_drug_ndc (case-insensitive) |
| `mapping_dicts/d_labitems_to_loinc.csv` | `rollup_itemid_to_loinc` → `lab_to_events` | MIMIC-IV itemid → LOINC (MIT-LCP OMOP mapping; 1400/1630 itemids covered) |
