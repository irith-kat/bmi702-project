---
name: mimic-table-relationships
description: Understand MIMIC-IV table relationships, join patterns, and identifier hierarchy. Use for correct data linkage, avoiding duplicates, and proper temporal joins.
---

# MIMIC-IV Table Relationships

Understanding the identifier hierarchy and table relationships is essential for correct query construction. Incorrect joins can cause data duplication or missing records.

## When to Use This Skill

- Writing complex queries joining multiple tables
- Linking clinical notes to structured EHR data
- Understanding why queries return unexpected row counts
- Debugging duplicate or missing data issues
- Learning MIMIC-IV data structure

## Identifier Hierarchy

```
subject_id (patient)
    └── hadm_id (hospital admission)
    │       └── stay_id (ICU stay)
    │               └── Events (chartevents, labevents, etc.)
    └── note_id (clinical note — mimiciv_note dataset)
            links via subject_id and hadm_id
```

### subject_id
- **Unique per patient**
- Persists across all hospitalizations and ICU stays
- Links to: `patients` table

### hadm_id
- **Unique per hospital admission**
- One patient can have multiple hadm_ids (readmissions)
- Links to: `admissions`, `diagnoses_icd`, `prescriptions`, most lab/hospital tables

### stay_id
- **Unique per ICU stay**
- One hospital admission can have multiple stay_ids (ICU readmission)
- Links to: `icustays`, `chartevents`, ICU-specific tables

## Core Table Relationships

### Hospital Module (mimiciv_hosp)
```sql
patients             -- 1 row per subject_id
    |
    +-- admissions   -- 1 row per hadm_id
    |       |
    |       +-- diagnoses_icd
    |       +-- procedures_icd
    |       +-- prescriptions
    |       +-- labevents
    |       +-- microbiologyevents
    |
    +-- transfers    -- Multiple per hadm_id (ward movements)
```

### ICU Module (mimiciv_icu)
```sql
icustays            -- 1 row per stay_id
    |
    +-- chartevents  -- Vitals, assessments
    +-- inputevents  -- Medications, fluids
    +-- outputevents -- Urine, drains
    +-- procedureevents
    +-- datetimeevents
```

### Notes Module (mimiciv_note — separate dataset)

**Important:** Notes live in a different dataset (`mimic-iv-note`). You must call `set_dataset("mimic-iv-note")` before querying them, then switch back to `mimic-iv` for structured data.

```sql
-- mimiciv_note.discharge    -- 1 row per note_id (discharge summaries)
-- mimiciv_note.radiology    -- Radiology reports
-- mimiciv_note.nursing      -- Nursing notes (if available)
```

Note schema (from observed data):
| Column | Type | Description |
|--------|------|-------------|
| `note_id` | string | Unique note identifier (e.g. `10011466-DS-18`) |
| `subject_id` | int | Patient identifier — joins to `mimiciv_hosp.patients` |
| `hadm_id` | int | Admission identifier — joins to `mimiciv_hosp.admissions` |
| `note_type` | string | `DS` = discharge summary, `RR` = radiology |
| `note_seq` | int | Sequence number within note type for this admission |
| `charttime` | timestamp | When the note was charted |
| `storetime` | timestamp | When the note was stored in the system |
| `text` | string | Full note text |

**Cardinality:** `hadm_id : discharge notes = 1 : many` (multiple versions/addenda per admission). Use `note_seq` to select the final version or aggregate all.

## Common Join Patterns

### Patient -> Hospital -> ICU
```sql
SELECT p.subject_id, a.hadm_id, ie.stay_id
FROM mimiciv_hosp.patients p
INNER JOIN mimiciv_hosp.admissions a
    ON p.subject_id = a.subject_id
INNER JOIN mimiciv_icu.icustays ie
    ON a.hadm_id = ie.hadm_id;
```

### Labs to ICU Stay (Time-Bounded)
```sql
-- Labs drawn during ICU stay only
SELECT ie.stay_id, le.charttime, le.valuenum
FROM mimiciv_icu.icustays ie
INNER JOIN mimiciv_hosp.labevents le
    ON ie.hadm_id = le.hadm_id
    AND le.charttime >= ie.intime
    AND le.charttime <= ie.outtime;
```

### Notes to Structured Data

Notes share `subject_id` and `hadm_id` with structured tables but live in a **different M4 dataset** (`mimic-iv-note`). The join must happen in Python after querying both datasets separately.

**CRITICAL: Never fetch the full notes table.** The `text` column is large — fetching all discharge summaries will exhaust memory. Always filter by the study cohort's identifiers inside the SQL `WHERE` clause before pulling `text`.

```python
from m4 import set_dataset, execute_query
import pandas as pd

# Step 1: define the cohort from structured data (mimic-iv)
set_dataset("mimic-iv")
cohort = execute_query("""
    SELECT subject_id, hadm_id, hospital_expire_flag
    FROM mimiciv_hosp.admissions
    WHERE hospital_expire_flag = 1
""")

# Step 2: pass cohort identifiers into the notes query as a filter
subject_ids = cohort["subject_id"].tolist()
id_list = ", ".join(str(s) for s in subject_ids)

set_dataset("mimic-iv-note")
notes = execute_query(f"""
    SELECT subject_id, hadm_id, note_id, note_type, note_seq, charttime, text
    FROM mimiciv_note.discharge
    WHERE subject_id IN ({id_list})
""")

# Step 3: join in pandas
cohort_notes = cohort.merge(notes, on=["subject_id", "hadm_id"], how="inner")
```

**Getting only the final discharge summary per admission** (highest `note_seq`):
```python
set_dataset("mimic-iv-note")
final_notes = execute_query(f"""
    SELECT subject_id, hadm_id, note_id, charttime, text
    FROM mimiciv_note.discharge
    WHERE subject_id IN ({id_list})
      AND note_seq = (
          SELECT MAX(note_seq)
          FROM mimiciv_note.discharge d2
          WHERE d2.hadm_id = mimiciv_note.discharge.hadm_id
      )
""")
```

**Counting notes per patient** (for MAP `note_df` denominator — metadata only, no `text`):
```python
set_dataset("mimic-iv-note")
note_counts = execute_query(f"""
    SELECT subject_id, COUNT(*) AS note_count
    FROM mimiciv_note.discharge
    WHERE subject_id IN ({id_list})
    GROUP BY subject_id
""")
set_dataset("mimic-iv")  # switch back for structured queries
```

**Rule of thumb:** Define the cohort first (`subject_id` list), then fetch notes filtered to that cohort. For large cohorts (>10k patients), process notes in batches of 1000–2000 subjects.

### Labs Within N Hours of ICU Admission
```sql
-- First 24 hours
SELECT ie.stay_id, le.charttime, le.valuenum
FROM mimiciv_icu.icustays ie
INNER JOIN mimiciv_hosp.labevents le
    ON ie.hadm_id = le.hadm_id
    AND le.charttime >= ie.intime
    AND le.charttime <= DATETIME_ADD(ie.intime, INTERVAL 24 HOUR);
```

## Critical Join Warnings

### 1. Hospital Labs Duplicate Across ICU Stays
If a patient has multiple ICU stays in one hospitalization, joining labs by `hadm_id` only will duplicate lab values:

```sql
-- WRONG: Duplicates labs for patients with multiple ICU stays
SELECT ie.stay_id, le.*
FROM mimiciv_icu.icustays ie
INNER JOIN mimiciv_hosp.labevents le
    ON ie.hadm_id = le.hadm_id;  -- No time filter!

-- CORRECT: Add time bounds
SELECT ie.stay_id, le.*
FROM mimiciv_icu.icustays ie
INNER JOIN mimiciv_hosp.labevents le
    ON ie.hadm_id = le.hadm_id
    AND le.charttime BETWEEN ie.intime AND ie.outtime;
```

### 2. Derived Tables Already Filtered
Many `mimiciv_derived` tables are pre-joined to ICU stays:
```sql
-- These already have stay_id and time-bounded data
SELECT * FROM mimiciv_derived.vitalsign;  -- Already per stay_id
SELECT * FROM mimiciv_derived.chemistry;  -- Has subject_id and hadm_id
```

### 3. Multiple Measurements Per Time Point
Aggregate or select appropriately:
```sql
-- Get worst GCS per hour
SELECT stay_id,
       DATETIME_TRUNC(charttime, HOUR) AS hour,
       MIN(gcs) AS worst_gcs
FROM mimiciv_derived.gcs
GROUP BY stay_id, DATETIME_TRUNC(charttime, HOUR);
```

## Cardinality Reference

| Relationship | Cardinality |
|-------------|-------------|
| subject_id : hadm_id | 1 : many |
| hadm_id : stay_id | 1 : many |
| stay_id : chartevents | 1 : many |
| hadm_id : labevents | 1 : many |
| hadm_id : diagnoses_icd | 1 : many |
| stay_id : derived tables | 1 : many (usually) |
| subject_id : discharge notes | 1 : many (across admissions) |
| hadm_id : discharge notes | 1 : many (versions/addenda) |

## Example: Verify Join Correctness

```sql
-- Check for unexpected duplicates
WITH joined AS (
    SELECT ie.stay_id, COUNT(*) AS n_labs
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_hosp.labevents le
        ON ie.hadm_id = le.hadm_id
    GROUP BY ie.stay_id
)
SELECT
    COUNT(*) AS n_stays,
    AVG(n_labs) AS avg_labs_per_stay,
    MAX(n_labs) AS max_labs  -- Very high = possible duplication
FROM joined;
```

## Backend Syntax Differences

Key syntax differences between backends (table names use the same canonical `schema.table` format on both):

| BigQuery-style | DuckDB-style |
|----------|--------|
| `DATETIME_ADD(x, INTERVAL '1' HOUR)` | `x + INTERVAL '1 hour'` |
| `DATETIME_DIFF(a, b, HOUR)` | `EXTRACT(EPOCH FROM (a - b))/3600` |
| `DATETIME_TRUNC(x, HOUR)` | `DATE_TRUNC('hour', x)` |

Table names like `mimiciv_hosp.patients` and `mimiciv_icu.icustays` work on both backends.

## References

- MIMIC-IV Documentation: https://mimic.mit.edu/docs/iv/
- Johnson AEW et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific Data. 2023.
