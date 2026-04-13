"""Fetch a few discharge summaries from MIMIC-IV and save to a text file."""

from pathlib import Path
from m4 import execute_query, set_dataset
from m4.config import set_active_backend

N = 5
OUT = Path("discharge_summaries.txt")

set_active_backend("bigquery")
set_dataset("mimic-iv-note")

df = execute_query(f"""
    SELECT note_id, subject_id, hadm_id, charttime, text
    FROM mimiciv_note.discharge
    LIMIT {N}
""")

with OUT.open("w") as f:
    for _, row in df.iterrows():
        f.write(
            f"--- note_id={row['note_id']}  subject_id={row['subject_id']}  hadm_id={row['hadm_id']}  charttime={row['charttime']} ---\n"
        )
        f.write(row["text"])
        f.write("\n\n")

print(f"Saved {len(df)} notes to {OUT}")
