"""01b — Pull discharge notes from BigQuery for HF candidates.

Study: hf-incident-v1
Reads : data/hf_patients.parquet  (produced by 01_cohort_data.py)
Writes: data/notes_raw.parquet

Notes are in a separate BigQuery dataset (mimic-iv-note) so this script
switches the backend and uses batched subject_id IN (...) queries.

Run:
    cd output/hf-incident-v1
    uv run python scripts/01b_notes_pull.py
"""

from pathlib import Path

import pandas as pd
from m4 import execute_query, set_dataset
from m4.config import set_active_backend

out = Path(__file__).resolve().parent.parent

_path = out / "data" / "notes_raw.parquet"
if _path.exists():
    print("Notes... [cached]")
    notes_raw = pd.read_parquet(_path)
    print(f"  {len(notes_raw):,} notes, {notes_raw['subject_id'].nunique():,} patients")
    raise SystemExit(0)

print("Loading HF patient list...")
hf_patients_df = pd.read_parquet(out / "data" / "hf_patients.parquet")
subject_ids = hf_patients_df["subject_id"].tolist()
print(f"  {len(subject_ids):,} patients")

set_active_backend("bigquery")
set_dataset("mimic-iv-note")

BATCH_SIZE = 400  # ~3,600 tokens per query, well under m4 limit
batches = [
    subject_ids[i : i + BATCH_SIZE] for i in range(0, len(subject_ids), BATCH_SIZE)
]
print(f"\nFetching notes: {len(batches)} batches × {BATCH_SIZE} patients...")

chunks = []
for i, batch in enumerate(batches, 1):
    id_list = ", ".join(str(sid) for sid in batch)
    chunk = execute_query(f"""
        SELECT
            CAST(subject_id AS STRING) AS subject_id,
            CAST(hadm_id    AS STRING) AS hadm_id,
            note_id,
            charttime,
            text
        FROM mimiciv_note.discharge
        WHERE subject_id IN ({id_list})
    """)
    chunks.append(chunk)
    if i % 10 == 0 or i == len(batches):
        print(
            f"  Batch {i}/{len(batches)}: {sum(len(c) for c in chunks):,} notes so far",
            flush=True,
        )

notes_raw = pd.concat(chunks, ignore_index=True)
notes_raw["charttime"] = pd.to_datetime(notes_raw["charttime"])
notes_raw.to_parquet(_path, index=False)

print(
    f"\nDone. notes_raw.parquet: {len(notes_raw):,} notes, "
    f"{notes_raw['subject_id'].nunique():,} patients"
)
print(f"  Median notes/patient: {notes_raw.groupby('subject_id').size().median():.1f}")
