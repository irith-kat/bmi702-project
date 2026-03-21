---
name: mimic-note-preprocessing
description: Extract CUI mentions from MIMIC-IV discharge notes using MedSpaCy and convert them to observation log rows. Use when adding NLP signal to the MAP pipeline or building a notes-based feature set.
---

# MIMIC Notes Preprocessing

Extracts NLP CUI mentions from clinical notes using MedSpaCy and returns them in the standard observation log format (`event_type="cui"`, `event="CUI:C0003873"`). Requires user-provided ONCE files to specify target CUIs.

Negated and family-history mentions are excluded automatically by MedSpaCy's context module.

## Setup

```python
import sys
sys.path.insert(0, "path/to/src/preprocessing")
from preprocessing import notes_to_events
from once import get_once_features
from m4 import set_dataset, execute_query
from m4.config import set_active_backend
```

## Step 1 — Get Target CUIs from ONCE

ONCE files are **user-provided** and should be placed in `input/`.

```python
once_features = get_once_features(
    codified_file  = "input/ONCE_codified.csv", # Exact name can vary
    narrative_file = "input/ONCE_narrative.csv", # Exact name can vary
)
target_cuis = once_features["nlp_target_cuis"]
# List of {"term": "rheumatoid arthritis", "cui": "C0003873"} dicts
```

## Step 2 — Identify Candidates from obs_log

Before loading notes, filter the observation log (built in the mimic-preprocessing step) to patients who have at least one ONCE codified event. This is the candidate set — patients plausibly related to the phenotype of interest. Running NLP on the full cohort is expensive and unnecessary.

```python
once_events   = set(once_features["codified_list"])
candidate_ids = set(obs_log[obs_log["event"].isin(once_events)]["subject_id"])
print(f"Candidates: {len(candidate_ids)}")
```

## Step 3 — Load Notes from BigQuery (Candidates Only)

> **Backend switch required.** Tabular EHR data (diagnoses, admissions, procedures) uses the local DuckDB backend. Clinical notes live in BigQuery. Explicitly switch backends when crossing between them — forgetting this is a common source of silent errors.
>
> ```python
> # Tabular EHR (local DuckDB) — used in mimic-preprocessing
> set_active_backend("duckdb")
> set_dataset("mimic-iv-demo")  # or "mimic-iv"
>
> # Switch to BigQuery for notes
> set_active_backend("bigquery")
> set_dataset("mimic-iv-note")
> ```

Discharge notes are in `mimic-iv-note` on BigQuery. Always filter to candidate
patients in the SQL query — the full notes table is too large to pull wholesale.

**M4 token limit:** M4 enforces a 10k-token query limit. A plain `IN (...)` with
7-digit subject IDs hits this limit at ~3,500 IDs. Always use batched queries:

```python
import pandas as pd

set_active_backend("bigquery")
set_dataset("mimic-iv-note")

candidate_ids = sorted(candidate_ids)
BATCH_SIZE = 400   # safe margin under the 10k token limit
batches = [candidate_ids[i:i+BATCH_SIZE] for i in range(0, len(candidate_ids), BATCH_SIZE)]

chunks = []
for batch in batches:
    id_list = ", ".join(str(sid) for sid in batch)
    chunk = execute_query(f"""
        SELECT subject_id, text, charttime
        FROM mimiciv_note.discharge
        WHERE subject_id IN ({id_list})
    """)
    chunks.append(chunk)

notes_df = pd.concat(chunks, ignore_index=True).dropna(subset=["text"])
print(f"Notes fetched: {len(notes_df)} ({notes_df['subject_id'].nunique()} patients)")
```

## Step 4 — notes_to_events

```python
nlp_obs = notes_to_events(
    notes_df          = notes_df,
    text_col          = "text",
    date_col          = "charttime",
    target_cuis       = target_cuis,
    subject_col       = "subject_id",    # default
    max_note_chars    = 10_000,          # truncate per note for speed (None = no truncation)
    notes_per_patient = 3,               # N most recent notes per patient (None = all)
    n_process         = 1,               # keep at 1 in Jupyter (multiprocessing can deadlock)
)
# Returns obs_log rows: subject_id, event_type="cui", event="CUI:C0003873", value=None, datetime
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `max_note_chars` | None | Truncate each note before NLP. `10_000` covers PMH + meds (~86% of avg MIMIC note). Speeds up MedSpaCy ~4–6×. |
| `notes_per_patient` | None | Keep N most recent notes per patient. Use to ensure all candidates get coverage instead of a flat total cap. `3` is a good starting point. |
| `n_process` | 1 | Workers for `nlp.pipe()`. **In scripts**, use `os.cpu_count()` for a 4–8× speedup. Keep at 1 in Jupyter notebooks (fork-based multiprocessing deadlocks). |
| `batch_size` | 256 | Texts buffered per spaCy batch. 256 is a good default; no recall impact. |

## Runtime Estimates (MIMIC-IV, `max_note_chars=10_000`)

Observed rate: ~1.9 notes/sec on a single core (WSL2, `n_process=1`).

| Notes | Approx time |
|---|---|
| 300 | ~3 min |
| 2,600 (1/patient × 2.6K candidates) | ~23 min |
| 5,100 (3/patient × 2.6K candidates) | ~45 min |
| 12,000 (1/patient × 12K candidates) | ~1.75 hr |
| 37,000 (3/patient × 12K candidates) | ~5 hr |

## Effect on MAP

Without NLP, MAP can only **reject** weak ICD cases (specificity gain). With NLP, MAP can also **find** patients under-coded in ICD (sensitivity gain). `map_only` cases in the comparison summary are the patients NLP adds.
