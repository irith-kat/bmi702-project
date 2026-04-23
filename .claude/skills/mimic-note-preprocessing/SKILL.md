---
name: mimic-note-preprocessing
description: Extract CUI mentions from MIMIC-IV discharge notes using MedSpaCy and convert them to observation log rows. Use when adding NLP signal to a cohort pipeline or building a notes-based feature set.
---

# MIMIC Notes Preprocessing

Extracts NLP CUI mentions from clinical notes using MedSpaCy. Output rows use `event_type="cui"`, `event="CUI:C0003873"` and are concatenated into the observation log alongside structured events.

Negated and family-history mentions are excluded automatically by MedSpaCy's context module.

## Imports

```python
from preprocessing.nlp import get_once_features
from preprocessing.structured import notes_to_events
```

---

## Two-Step Approach

Notes preprocessing is split across two scripts:

1. **Cohort script** (`01_cohort_definition.py`) — fetch and cache `notes_raw.parquet` for rellevant cohort patients. See the `mimic-preprocessing` skill for the batched fetch pattern. This fetching can be placed in 01 or later in the pipeline, decide at your discretion based on which list of patients you want to fetch notes for (all cohort, ONCE candidates, MAP gold...). More info below.

2. **NLP script** (`02_notes_nlp.py`) — read from cache, run MedSpaCy, save `cui_obs.parquet`. The feature matrix script then concatenates this into the observation log.

This split means the expensive BigQuery fetch is cached independently of the NLP run.

---

## Step 1 — Get Target CUIs from ONCE

ONCE files are **user-provided** in `input/`. Use glob to find them.

```python
import glob
from preprocessing.nlp import get_once_features

codified_files = sorted(glob.glob("input/ONCE_*PheCode*.csv"))
narrative_files = sorted(glob.glob("input/ONCE_*_C[0-9]*.csv"))

once_features = get_once_features(codified_files[0], narrative_files[0])
target_cuis = once_features["nlp_target_cuis"]
# List of {"term": "...", "cui": "C0003873"} dicts
```

---

## Step 2 — NLP Script (reads from cache)

The NLP script should exit early if `cui_obs.parquet` already exists.

```python
import os
import time
from pathlib import Path
import pandas as pd
from preprocessing.nlp import get_once_features
from preprocessing.structured import notes_to_events

out = Path(__file__).resolve().parent.parent

# Cache check — delete cui_obs.parquet to force a re-run
_out_path = out / "data" / "cui_obs.parquet"
if _out_path.exists():
    print("cui_obs.parquet already exists — skipping NLP extraction.")
    raise SystemExit(0)

# Load ONCE features
once_features = get_once_features(codified_file, narrative_file)
target_cuis = once_features["nlp_target_cuis"]

# Load cached notes
notes_raw = pd.read_parquet(out / "data" / "notes_raw.parquet")

# Run MedSpaCy NER
NOTES_PER_PATIENT = 3        # most recent N discharge notes per patient
MAX_NOTE_CHARS    = 10_000   # truncate per note; good speed/recall tradeoff
N_PROCESS         = os.cpu_count()  # use 1 in Jupyter (fork deadlock risk)

t0 = time.time()
cui_obs = notes_to_events(
    notes_df          = notes_raw,
    text_col          = "text",
    date_col          = "charttime",
    target_cuis       = target_cuis,
    notes_per_patient = NOTES_PER_PATIENT,
    max_note_chars    = MAX_NOTE_CHARS,
    n_process         = N_PROCESS,
    batch_size        = 256,
)
print(f"Done in {(time.time()-t0)/60:.1f} min — {len(cui_obs):,} CUI events, "
      f"{cui_obs['subject_id'].nunique():,} patients")

cui_obs.to_parquet(_out_path, index=False)
```

---

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `notes_per_patient` | 3 | N most recent notes per patient. Ensures all patients get coverage. |
| `max_note_chars` | 10_000 | Truncate each note before NLP. Covers PMH + meds (~86% of avg MIMIC note). Speeds MedSpaCy ~4–6×. |
| `n_process` | `os.cpu_count()` | Workers for `nlp.pipe()`. Use `1` in Jupyter notebooks only. |
| `batch_size` | 256 | Texts buffered per spaCy batch. No recall impact. |

---

## When to Fetch All Cohort Notes vs. Candidates Only

Fetch notes for **all cohort patients** when LATTE gold labeling will be needed downstream — Gemini reviews discharge notes for MAP cases and needs the full set available.

Fetch notes for **candidates only** (patients with ≥1 ONCE codified event) when the study is MAP-only and notes are purely for adding NLP sensitivity. The candidate filter reduces the note volume significantly:

```python
once_events   = set(once_features["codified_list"])
candidate_ids = set(obs_log[obs_log["event"].isin(once_events)]["subject_id"])
```

When in doubt, use your criteria to determine which patients in the cohort are likely to need notes-based features (ONCE codified, all, MAP gold...). You have discretion to flip the order of steps to fetch notes for whatever cohort you deem fit.
