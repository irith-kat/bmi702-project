---
name: mimic-note-preprocessing
description: Extract CUI mentions from MIMIC-IV discharge notes using MedSpaCy and convert them to observation log rows. Use when adding NLP signal to the MAP pipeline or building a notes-based feature set.
---

# MIMIC Notes Preprocessing

Extracts NLP CUI mentions from clinical notes using MedSpaCy and returns them in the standard observation log format (`event_type="cui"`, `event="CUI:C0003873"`).

Negated and family-history mentions are excluded automatically by MedSpaCy's context module.

## Setup

```python
import sys
sys.path.insert(0, "path/to/src/preprocessing")
from preprocessing import notes_to_events
from once import get_once_features
```

## Step 1 — Get Target CUIs from ONCE (optional — skip if not using NLP)

ONCE files are **user-provided** and already in `src/`. NLP extraction is only needed when adding CUI features to MAP or building a note-based feature set.

```python
once_features = get_once_features(codified_file, narrative_file)
target_cuis = once_features["nlp_target_cuis"]
# List of {"term": "rheumatoid arthritis", "cui": "C0003873"} dicts
```

## Step 2 — Load Notes (Candidates Only)

Always filter to candidate patients before loading — the full notes file is too large.

```python
note_chunks = []
for chunk in pd.read_csv(
    "mimiciv/note/discharge.csv.gz",
    usecols=["subject_id", "text", "charttime"],
    parse_dates=["charttime"],
    chunksize=5_000,
):
    sub = chunk[chunk["subject_id"].isin(candidate_ids)].copy()
    if len(sub):
        note_chunks.append(sub)

notes_df = pd.concat(note_chunks, ignore_index=True).dropna(subset=["text"])
```

## Step 3 — notes_to_events

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

## Adding NLP to build_obs_log

Pass `notes_df` directly to `build_obs_log` — it calls `notes_to_events` internally:

```python
obs_log = build_obs_log(
    icd_df            = diagnoses_with_dates,
    icd_col           = "icd_code",
    icd_date_col      = "admittime",
    notes_df          = notes_df,
    notes_text_col    = "text",
    notes_date_col    = "charttime",
    target_cuis       = target_cuis,
    max_note_chars    = 10_000,
    notes_per_patient = 3,
)
```

## Effect on MAP

Without NLP, MAP can only **reject** weak ICD cases (specificity gain). With NLP, MAP can also **find** patients under-coded in ICD (sensitivity gain). `map_only` cases in the comparison summary are the patients NLP adds.
