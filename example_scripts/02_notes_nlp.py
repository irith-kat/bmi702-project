"""01_5 — Extract CUI mentions from discharge notes via MedSpaCy.

Study: HF_test_run_v1
Reads : data/notes_raw.parquet  (produced by 01_cohort_definition.py)
Writes: data/cui_obs.parquet    (obs_log-format NLP events, ready to concat)

Cache behaviour: skipped if cui_obs.parquet already exists.
Delete that file to force a re-run (e.g. after changing NOTES_PER_PATIENT).

Run:
    uv run python output/HF_test_run_v1/scripts/01_5_notes_nlp.py
"""

import glob
import os
import time
from pathlib import Path

import pandas as pd
from preprocessing.nlp import get_once_features
from preprocessing.structured import notes_to_events

REPO_ROOT = Path(__file__).resolve().parents[3]

out = Path(__file__).resolve().parent.parent

# ── Tunable limits ─────────────────────────────────────────────────────────────
NOTES_PER_PATIENT = 3  # most recent N discharge notes per patient
MAX_NOTE_CHARS = 10_000  # truncate per note; good speed/recall tradeoff
N_PROCESS = os.cpu_count()  # safe in script context; use 1 in Jupyter

# ── Cache check ────────────────────────────────────────────────────────────────
_out_path = out / "data" / "cui_obs.parquet"
if _out_path.exists():
    print("cui_obs.parquet already exists — skipping NLP extraction.")
    print(f"  Delete {_out_path} to force a re-run.")
    cui_obs = pd.read_parquet(_out_path)
    print(
        f"  Cached: {len(cui_obs):,} CUI events, {cui_obs['subject_id'].nunique():,} patients"
    )
    raise SystemExit(0)

# ── 1. Load ONCE features ──────────────────────────────────────────────────────
print("Loading ONCE features...")
codified_files = sorted(glob.glob(str(REPO_ROOT / "input" / "ONCE_*PheCode*.csv")))
narrative_files = sorted(glob.glob(str(REPO_ROOT / "input" / "ONCE_*_C[0-9]*.csv")))

codified_file = next(
    f for f in codified_files if "428" in f and "heart failure" in f.lower()
)
narrative_file = next(
    f for f in narrative_files if "heart failure" in f.lower() and "C0018802" in f
)

once_features = get_once_features(codified_file, narrative_file)
target_cuis = once_features["nlp_target_cuis"]
print(f"  NLP target CUIs: {len(target_cuis)}")

# ── 2. Load notes ──────────────────────────────────────────────────────────────
print("\nLoading discharge notes...")
notes_raw = pd.read_parquet(out / "data" / "notes_raw.parquet")
print(
    f"  Notes: {len(notes_raw):,} rows, {notes_raw['subject_id'].nunique():,} patients"
)

# ── 3. Run MedSpaCy NER ───────────────────────────────────────────────────────
print(
    f"\nRunning MedSpaCy NER "
    f"(notes_per_patient={NOTES_PER_PATIENT}, max_note_chars={MAX_NOTE_CHARS}, "
    f"n_process={N_PROCESS})..."
)
t0 = time.time()

cui_obs = notes_to_events(
    notes_df=notes_raw,
    text_col="text",
    date_col="charttime",
    target_cuis=target_cuis,
    notes_per_patient=NOTES_PER_PATIENT,
    max_note_chars=MAX_NOTE_CHARS,
    n_process=N_PROCESS,
    batch_size=256,
)

elapsed = time.time() - t0
print(f"  Done in {elapsed / 60:.1f} min")
print(
    f"  CUI events : {len(cui_obs):,} rows, {cui_obs['subject_id'].nunique():,} patients"
)
print(f"  Unique CUIs: {cui_obs['event'].nunique()}")
print(f"  Top CUIs:\n{cui_obs['event'].value_counts().head(10).to_string()}")

# ── 4. Save ────────────────────────────────────────────────────────────────────
cui_obs.to_parquet(_out_path, index=False)
print("\nDone. Saved cui_obs.parquet")
print(
    f"  Settings: notes_per_patient={NOTES_PER_PATIENT}, max_note_chars={MAX_NOTE_CHARS}"
)
