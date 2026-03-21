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

Discharge notes are in `mimic-iv-note` on BigQuery. Always filter to candidate
patients in the SQL query — the full notes table is too large to pull wholesale.
The `IN (...)` list approach is practical up to ~5K subjects; for larger sets
use a temp table or `JOIN`, and work in batches if needed.

```python
set_active_backend("bigquery")
set_dataset("mimic-iv-note")

subject_id_list = ", ".join(str(sid) for sid in sorted(candidate_ids))
notes_df = execute_query(f"""
    SELECT subject_id, text, charttime
    FROM mimiciv_note.discharge
    WHERE subject_id IN ({subject_id_list})
""")
notes_df = notes_df.dropna(subset=["text"])
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
| `n_process` | 1 | Workers for `nlp.pipe()`. Keep at 1 in Jupyter to avoid deadlocks. |

## Runtime Estimates (MIMIC-IV, `max_note_chars=10_000`)

| Notes | Approx time |
|---|---|
| 300 | ~1 min |
| 3,000 (1/patient × 3K candidates) | ~10 min |
| 12,000 (1/patient × 12K candidates) | ~40 min |
| 37,000 (3/patient × 12K candidates) | ~2 hr |

## Effect on MAP

Without NLP, MAP can only **reject** weak ICD cases (specificity gain). With NLP, MAP can also **find** patients under-coded in ICD (sensitivity gain). `map_only` cases in the comparison summary are the patients NLP adds.
