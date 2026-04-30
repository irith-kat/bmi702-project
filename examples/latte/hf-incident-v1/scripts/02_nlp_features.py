"""02 — Extract CUI mentions from discharge notes via MedSpaCy.

Study: hf-incident-v1
Reads : data/notes_raw.parquet   (from 01b_notes_pull.py)
Writes: data/cui_obs.parquet     (obs_log-format NLP events)

Run:
    cd output/hf-incident-v1
    uv run python scripts/02_nlp_features.py
"""

import glob
import os
import time
from pathlib import Path

import pandas as pd
from preprocessing.nlp import get_once_features
from preprocessing.structured import notes_to_events

REPO_ROOT = Path(__file__).resolve().parents[4]
out = Path(__file__).resolve().parent.parent

NOTES_PER_PATIENT = 3  # most recent N discharge notes per patient
MAX_NOTE_CHARS = 10_000
N_PROCESS = os.cpu_count()

_out_path = out / "data" / "cui_obs.parquet"
if _out_path.exists():
    print("cui_obs.parquet already exists — skipping.")
    cui_obs = pd.read_parquet(_out_path)
    print(
        f"  {len(cui_obs):,} CUI events, {cui_obs['subject_id'].nunique():,} patients"
    )
    raise SystemExit(0)

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

print("\nLoading discharge notes...")
notes_raw = pd.read_parquet(out / "data" / "notes_raw.parquet")
print(f"  {len(notes_raw):,} notes, {notes_raw['subject_id'].nunique():,} patients")

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

print(f"  Done in {(time.time() - t0) / 60:.1f} min")
print(f"  CUI events : {len(cui_obs):,}, {cui_obs['subject_id'].nunique():,} patients")
print(f"  Unique CUIs: {cui_obs['event'].nunique()}")
print(f"  Top CUIs:\n{cui_obs['event'].value_counts().head(10).to_string()}")

cui_obs.to_parquet(_out_path, index=False)
print("\nSaved cui_obs.parquet")
