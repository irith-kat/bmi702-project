"""
test_gemini_labeler.py
----------------------
Smoke-test for gemini.py using two synthetic patients:
  - patient_001: clear HF language in notes → expect label=1
  - patient_002: no HF language         → expect label=0

Sends real API calls to Vertex AI (gemini-3.1-flash-lite-preview).
Caches results to /tmp/gemini_test_cache.jsonl so re-runs are free.

Usage:
    uv run python tests/integration/test_gemini_labeler.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from latte.gemini import parse_gemini_results, run_gemini_labeling
from latte.labeler_utils import HF_DISEASE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ---------------------------------------------------------------------------
# Synthetic discharge summaries
# ---------------------------------------------------------------------------

NOTES = [
    # patient_001: HF case — should come back label=1
    {
        "note_id": "n001",
        "subject_id": "patient_001",
        "hadm_id": "hadm_001",
        "charttime": pd.Timestamp("2120-03-15"),
        "text": (
            "DISCHARGE SUMMARY\n"
            "Chief Complaint: Progressive dyspnoea on exertion, orthopnoea.\n"
            "Assessment: Patient presents with acute decompensated heart failure. "
            "Echo shows EF of 30% consistent with HFrEF. "
            "BNP 1200 pg/mL. Started on IV furosemide with good diuretic response. "
            "Bilateral lower extremity pitting oedema, S3 gallop on auscultation.\n"
            "Diagnosis: Systolic heart failure (HFrEF), new diagnosis.\n"
            "Discharge plan: Continue furosemide 40mg daily, cardiology follow-up."
        ),
    },
    # patient_002: control — should come back label=0
    {
        "note_id": "n002",
        "subject_id": "patient_002",
        "hadm_id": "hadm_002",
        "charttime": pd.Timestamp("2121-07-04"),
        "text": (
            "DISCHARGE SUMMARY\n"
            "Chief Complaint: Right knee pain and swelling.\n"
            "Assessment: Patient presents with right knee osteoarthritis flare. "
            "No chest pain, no dyspnoea, no oedema. "
            "Cardiovascular exam unremarkable. Normal sinus rhythm on ECG. "
            "Treated with NSAIDs and physical therapy referral.\n"
            "Diagnosis: Right knee osteoarthritis.\n"
            "Discharge plan: Follow up with orthopaedics in 4 weeks."
        ),
    },
]

NOTES_DF = pd.DataFrame(NOTES)

CACHE = "/tmp/gemini_test_cache.jsonl"
PROJECT = "just-duality-438820-n4"
LOCATION = "global"
MODEL = "publishers/google/models/gemini-3.1-flash-lite-preview"


def main():
    print(f"\n{'=' * 60}")
    print("Gemini labeler smoke test")
    print(f"  model   : {MODEL}")
    print(f"  project : {PROJECT}")
    print(f"  cache   : {CACHE}")
    print(f"  patients: {NOTES_DF['subject_id'].tolist()}")
    print(f"{'=' * 60}\n")

    n = run_gemini_labeling(
        notes_df=NOTES_DF,
        subject_ids=["patient_001", "patient_002"],
        cache_jsonl=CACHE,
        config=HF_DISEASE_CONFIG,
        model_name=MODEL,
        project_id=PROJECT,
        location=LOCATION,
    )
    print(f"\nNewly labeled: {n}")

    results = parse_gemini_results(CACHE)
    print("\nResults DataFrame:")
    print(
        results[
            [
                "subject_id",
                "label",
                "incident_hadm_id",
                "incident_charttime",
                "parse_error",
            ]
        ].to_string(index=False)
    )

    # Basic assertions
    assert len(results) == 2, f"Expected 2 rows, got {len(results)}"
    assert not results["parse_error"].any(), "Parse errors found — check logs above"

    hf = results[results["subject_id"] == "patient_001"].iloc[0]
    ctrl = results[results["subject_id"] == "patient_002"].iloc[0]

    print(f"\npatient_001 label={hf['label']}  (expected 1)")
    print(f"patient_002 label={ctrl['label']}  (expected 0)")

    if hf["label"] == 1 and ctrl["label"] == 0:
        print("\n✓ Both labels correct — gemini.py is working.")
    else:
        print("\n⚠  Unexpected labels — review model reasoning in the cache file.")
        print(f"   cat {CACHE}")


if __name__ == "__main__":
    main()
