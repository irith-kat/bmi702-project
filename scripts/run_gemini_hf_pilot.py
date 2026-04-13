"""
run_gemini_hf_pilot.py
----------------------
End-to-end pilot: fetch 5 real MIMIC-IV patients with documented Heart Failure
ICD codes, run Gemini incident phenotyping on their discharge summaries, and
print a structured summary of the gold labels.

Requires:
  - gcloud auth application-default login
  - uv run python scripts/run_gemini_hf_pilot.py

Results are cached to /tmp/gemini_hf_pilot.jsonl — re-runs skip already-labeled patients.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from m4 import execute_query, set_dataset
from m4.config import set_active_backend

from latte.gemini import parse_gemini_results, run_gemini_labeling
from latte.labeler_utils import HF_DISEASE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_PATIENTS = 5
CACHE_JSONL = "/tmp/gemini_hf_pilot.jsonl"
PROJECT_ID = "just-duality-438820-n4"
LOCATION = "global"
MODEL = "publishers/google/models/gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# Step 1: Find patients with a documented HF ICD code
# ---------------------------------------------------------------------------

logger.info(
    "Step 1: Querying MIMIC-IV for patients with HF ICD codes (I50.x / 428.x)..."
)

set_active_backend("bigquery")
set_dataset("mimic-iv")

# Get patients with at least one HF ICD code and at least 2 admissions.
# No upper cap on admissions — max_notes_per_patient handles context budget.
hf_patients_df = execute_query(f"""
    WITH hf_subjects AS (
        SELECT DISTINCT
            CAST(d.subject_id AS STRING) AS subject_id
        FROM mimiciv_hosp.diagnoses_icd d
        WHERE
            (d.icd_version = 10 AND d.icd_code LIKE 'I50%')
            OR
            (d.icd_version = 9  AND d.icd_code LIKE '428%')
    ),
    admission_counts AS (
        SELECT
            CAST(subject_id AS STRING) AS subject_id,
            COUNT(DISTINCT hadm_id) AS n_admissions
        FROM mimiciv_hosp.admissions
        GROUP BY subject_id
    )
    SELECT
        h.subject_id,
        a.n_admissions
    FROM hf_subjects h
    JOIN admission_counts a USING (subject_id)
    WHERE a.n_admissions >= 2
    ORDER BY RAND()
    LIMIT {N_PATIENTS}
""")

subject_ids = hf_patients_df["subject_id"].tolist()
logger.info(
    "Selected %d patients. Admission counts: %s",
    len(subject_ids),
    hf_patients_df.set_index("subject_id")["n_admissions"].to_dict(),
)

# ---------------------------------------------------------------------------
# Step 2: Fetch their discharge summaries
# ---------------------------------------------------------------------------

logger.info("Step 2: Fetching discharge summaries...")

set_dataset("mimic-iv-note")

sid_sql = ", ".join(f"'{s}'" for s in subject_ids)
notes_df = execute_query(f"""
    SELECT
        note_id,
        CAST(subject_id AS STRING) AS subject_id,
        CAST(hadm_id    AS STRING) AS hadm_id,
        charttime,
        text
    FROM mimiciv_note.discharge
    WHERE CAST(subject_id AS STRING) IN ({sid_sql})
    ORDER BY subject_id, charttime
""")

notes_df["charttime"] = pd.to_datetime(notes_df["charttime"])

logger.info(
    "Fetched %d notes for %d patients (median %.1f notes/patient).",
    len(notes_df),
    notes_df["subject_id"].nunique(),
    notes_df.groupby("subject_id").size().median(),
)

# ---------------------------------------------------------------------------
# Step 3: Run Gemini labeling
# ---------------------------------------------------------------------------

logger.info("Step 3: Running Gemini incident phenotyping...")

n_labeled = run_gemini_labeling(
    notes_df=notes_df,
    subject_ids=subject_ids,
    cache_jsonl=CACHE_JSONL,
    config=HF_DISEASE_CONFIG,
    model_name=MODEL,
    project_id=PROJECT_ID,
    location=LOCATION,
    # Budget: stay under 200K input tokens (~600K chars).
    # System prompt ~9K chars; at median 9.5K chars/note → ~62 notes max.
    max_notes_per_patient=60,
)

logger.info("Newly labeled this run: %d", n_labeled)

# ---------------------------------------------------------------------------
# Step 4: Parse and display results
# ---------------------------------------------------------------------------

results = parse_gemini_results(CACHE_JSONL)

print("\n" + "=" * 70)
print("GEMINI HF INCIDENT PHENOTYPING — PILOT RESULTS")
print("=" * 70)
print(f"Model   : {MODEL}")
print(f"Patients: {len(results)}")
print(f"Cases   : {(results['label'] == 1).sum()}")
print(f"Controls: {(results['label'] == 0).sum()}")
print(f"Errors  : {results['parse_error'].sum()}")
print("=" * 70)

for _, row in results.iterrows():
    print(f"\n{'─' * 50}")
    print(f"Patient      : {row['subject_id']}")
    n_admissions = notes_df[notes_df["subject_id"] == row["subject_id"]].shape[0]
    print(f"Admissions   : {n_admissions}")

    if row["parse_error"]:
        print("Result       : PARSE ERROR")
        continue

    label_str = "CASE (HF present)" if row["label"] == 1 else "CONTROL (no HF)"
    print(f"Label        : {row['label']}  →  {label_str}")

    if row["label"] == 1:
        print(f"Incident hadm: {row['incident_hadm_id']}")
        print(f"Incident date: {row['incident_charttime']}")

    # Print per-admission timeline summary
    try:
        timeline = json.loads(row["timeline_json"])
        print(f"Timeline     : {len(timeline)} admissions reviewed")
        for visit in timeline:
            status_sym = "✓ HF" if visit.get("status") == 1 else "✗   "
            conf = visit.get("confidence", "?")
            evidence = visit.get("evidence", "")[:80]
            print(
                f"  [{status_sym} | {conf:8s}] Adm {visit.get('visit_id')} — {evidence}"
            )
    except (json.JSONDecodeError, TypeError):
        pass

print(f"\n{'─' * 50}")
print(f"\nFull results saved to: {CACHE_JSONL}")
