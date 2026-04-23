---
name: mimic-preprocessing
description: Preprocess structured MIMIC-IV data (diagnoses, prescriptions, procedures, lab events) into a standardized observation log using build_obs_log and rollup_itemid_to_loinc. Use at the start of any MIMIC-IV phenotyping workflow before running MAP.
---

# MIMIC Structured Data Preprocessing

All preprocessing functions produce a standardized **observation log** — a long-format DataFrame with five columns:

| Column | Type | Description |
|---|---|---|
| `subject_id` | str | Patient identifier |
| `event_type` | str | `"phecode"`, `"rxnorm"`, `"ccs"`, `"loinc"`, `"cui"` |
| `event` | str | Prefixed code: `"PheCode:714.1"`, `"RXNORM:956874"`, `"CCS:3"`, `"LOINC:11555-0"` |
| `value` | float\|None | Numeric result (labs only); None for all other types |
| `datetime` | datetime | When the observation occurred |

## Imports

```python
from m4 import execute_query, set_dataset
from m4.config import set_active_backend
from preprocessing.structured import build_obs_log, rollup_itemid_to_loinc
from preprocessing.nlp import get_once_features
```

---

## Cohort Fetching Pattern

### Inline CTE (preferred for MIMIC-IV)

The m4 SecurityError fires on query strings over ~10k tokens. A Python `IN (...)` list of 7-digit subject IDs exhausts this at ~3,500 IDs. Avoid it entirely by embedding a cohort-defining CTE into every query so no patient list is ever passed from Python to SQL.

```python
COHORT_CTE = """
    WITH cohort_subjects AS (
        SELECT DISTINCT d.subject_id
        FROM mimiciv_hosp.diagnoses_icd d
        WHERE
            (d.icd_version = 10 AND d.icd_code LIKE 'I50%')
            OR
            (d.icd_version = 9  AND d.icd_code LIKE '428%')
    ),
    admission_counts AS (
        SELECT subject_id, COUNT(DISTINCT hadm_id) AS n_admissions
        FROM mimiciv_hosp.admissions
        GROUP BY subject_id
    ),
    cohort AS (
        SELECT h.subject_id
        FROM cohort_subjects h
        JOIN admission_counts a USING (subject_id)
        WHERE a.n_admissions >= 2
    )
"""

# Embed the CTE in every subsequent query:
diagnoses_raw = execute_query(f"""
    {COHORT_CTE}
    SELECT
        CAST(d.subject_id AS STRING) AS subject_id,
        CAST(d.hadm_id    AS STRING) AS hadm_id,
        d.icd_code,
        d.icd_version,
        a.admittime
    FROM mimiciv_hosp.diagnoses_icd d
    INNER JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.subject_id IN (SELECT subject_id FROM cohort)
""")
```

Define the CTE once at the top of the cohort script and embed it in every fetch query.

### Cache Recycling

Each raw table should be cached as parquet. Re-run individual fetches by deleting the file.

```python
_path = out / "data" / "diagnoses_raw.parquet"
if _path.exists():
    print("Pulling diagnoses... [cached]")
    diagnoses_raw = pd.read_parquet(_path)
else:
    print("Pulling diagnoses...")
    diagnoses_raw = execute_query(f"{COHORT_CTE} SELECT ...")
    diagnoses_raw["admittime"] = pd.to_datetime(diagnoses_raw["admittime"])
    diagnoses_raw.to_parquet(_path, index=False)
```

Apply this pattern to every table: patients, admissions, diagnoses, prescriptions, procedures, labevents, notes.

---

## Lab Pre-Filtering (Important for Performance)

The full `labevents` table is 40–60M rows. Fetching it unfiltered is slow and wasteful. Pre-filter to only the LOINC codes ONCE requires by reverse-mapping them to MIMIC itemids before querying.

```python
# 1. Load ONCE features to get the LOINC codes MAP needs
once_features = get_once_features(codified_file, narrative_file)
loinc_codes = {f[len("LOINC:"):] for f in once_features["codified_list"] if f.startswith("LOINC:")}

# 2. Reverse-lookup: LOINC codes → MIMIC itemids
import pandas as pd
map_df = pd.read_csv(
    MAPPING_ROOT / "d_labitems_to_loinc.csv",
    usecols=["itemid (omop_source_code)", "omop_concept_code", "omop_vocabulary_id"],
    dtype=str,
).rename(columns={"itemid (omop_source_code)": "itemid", "omop_concept_code": "loinc_code"})
map_df = map_df[map_df["omop_vocabulary_id"] == "LOINC"]
needed_itemids = sorted(int(r) for r in map_df[map_df["loinc_code"].isin(loinc_codes)]["itemid"])
itemid_sql = ", ".join(str(i) for i in needed_itemids)
print(f"ONCE LOINC codes: {len(loinc_codes)} → {len(needed_itemids)} MIMIC itemids")

# 3. Fetch only the needed itemids (reduces fetch from ~50M to ~1-3M rows)
labevents_raw = execute_query(f"""
    {COHORT_CTE}
    SELECT
        CAST(l.subject_id AS STRING) AS subject_id,
        l.itemid,
        l.charttime,
        l.valuenum
    FROM mimiciv_hosp.labevents l
    WHERE l.subject_id IN (SELECT subject_id FROM cohort)
      AND l.valuenum IS NOT NULL
      AND l.itemid IN ({itemid_sql})
""")
```

---

## Discharge Notes Fetching

`mimiciv_note.discharge` lives in a separate dataset with no access to the `mimiciv_hosp` CTE. Use batched `IN (...)` queries instead.

```python
set_dataset("mimic-iv-note")
BATCH_SIZE = 400  # ~3,600 tokens per query, well under the m4 limit
subject_ids = cohort_df["subject_id"].tolist()
batches = [subject_ids[i:i + BATCH_SIZE] for i in range(0, len(subject_ids), BATCH_SIZE)]

chunks = []
for i, batch in enumerate(batches, 1):
    id_list = ", ".join(str(sid) for sid in batch)
    chunk = execute_query(f"""
        SELECT
            CAST(subject_id AS STRING) AS subject_id,
            CAST(hadm_id    AS STRING) AS hadm_id,
            note_id, charttime, text
        FROM mimiciv_note.discharge
        WHERE subject_id IN ({id_list})
    """)
    chunks.append(chunk)
    print(f"  Batch {i}/{len(batches)}: {len(chunk):,} notes", flush=True)

notes_raw = pd.concat(chunks, ignore_index=True)
notes_raw["charttime"] = pd.to_datetime(notes_raw["charttime"])
set_dataset("mimic-iv")  # restore
```

---

## Building the Observation Log

After fetching and caching raw tables, call `rollup_itemid_to_loinc` first for labs, then `build_obs_log` to produce the structured obs_log. NLP CUI events (from the notes NLP script) are concatenated separately after.

```python
from preprocessing.structured import build_obs_log, rollup_itemid_to_loinc

# Rollup lab itemids → LOINC before passing to build_obs_log
labevents_loinc = rollup_itemid_to_loinc(
    labevents_raw,
    itemid_column="itemid",
    mapping_file=str(MAPPING_ROOT / "d_labitems_to_loinc.csv"),
)

obs_log = build_obs_log(
    icd_df          = diagnoses_raw,
    icd_col         = "icd_code",
    icd_date_col    = "admittime",

    drug_df         = prescriptions_raw,   # pre-filter to drug_type='MAIN'
    drug_ndc_col    = "ndc",
    drug_date_col   = "starttime",
    drug_col        = "drug",              # fallback for unresolved NDCs

    cpt_df          = procedures_raw,
    cpt_col         = "hcpcs_cd",
    cpt_date_col    = "chartdate",

    notes_df        = None,                # NLP events come from cui_obs.parquet, not here

    lab_df          = labevents_loinc,
    lab_loinc_col   = "loinc_code",
    lab_date_col    = "charttime",
    lab_value_col   = "valuenum",

    icd_mapping_file      = str(MAPPING_ROOT / "Phecode_map_v1_2_icd9_icd10cm.csv"),
    cpt_mapping_file      = str(MAPPING_ROOT / "CCS_Services_Procedures_v2025-1_052425.csv"),
    ndc_mapping_file      = str(MAPPING_ROOT / "ndc_to_rxnorm_ingredient.csv"),
    drug_name_mapping_file= str(MAPPING_ROOT / "drug_name_to_rxnorm_ingredient.csv"),
)

# Append NLP CUI events if the notes NLP script has been run
cui_obs_path = out / "data" / "cui_obs.parquet"
if cui_obs_path.exists():
    cui_obs = pd.read_parquet(cui_obs_path)
    obs_log = pd.concat([obs_log, cui_obs], ignore_index=True)
```

Pass `None` for any modality you don't have. Raises `ValueError` if all are None.

## Prescriptions Note

Always filter to `drug_type='MAIN'` before passing to `build_obs_log`. MIMIC prescriptions also have `BASE` and `ADDITIVE` rows that cause double-counting.

```python
prescriptions_raw = prescriptions_raw[prescriptions_raw["drug_type"] == "MAIN"].copy()
```

## Mapping Files

All mapping files live in `mapping_dicts/` at the repo root. Pass absolute paths.

| File | Purpose |
|---|---|
| `Phecode_map_v1_2_icd9_icd10cm.csv` | ICD-9/10 → PheCode |
| `CCS_Services_Procedures_v2025-1_052425.csv` | CPT/HCPCS → AHRQ CCS |
| `ndc_to_rxnorm_ingredient.csv` | NDC → RxNorm ingredient |
| `drug_name_to_rxnorm_ingredient.csv` | Drug name fallback |
| `d_labitems_to_loinc.csv` | MIMIC itemid → LOINC (used by `rollup_itemid_to_loinc` and lab pre-filtering) |
