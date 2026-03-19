"""01 — Pull all subjects, diagnoses, and procedures from MIMIC-IV-demo (local DuckDB)."""

import sys
from pathlib import Path
from m4 import set_dataset, execute_query


# m4-pheno is not an installable package; add it to the path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "m4-pheno"))

set_dataset("mimic-iv-demo")

out = Path(__file__).resolve().parent.parent

# --- Subjects + demographics ---
subjects = execute_query("""
    SELECT
        p.subject_id,
        p.gender,
        p.anchor_age AS age,
        p.anchor_year,
        p.dod
    FROM mimiciv_hosp.patients p
""")
print(f"Subjects: {len(subjects)}")
subjects.to_parquet(out / "data" / "subjects.parquet", index=False)

# --- Hospital admissions (for note-count proxy + race) ---
admissions = execute_query("""
    SELECT
        subject_id,
        hadm_id,
        race,
        admittime,
        dischtime,
        hospital_expire_flag
    FROM mimiciv_hosp.admissions
""")
print(f"Admissions: {len(admissions)}")
admissions.to_parquet(out / "data" / "admissions.parquet", index=False)

# --- ICD diagnoses ---
diagnoses = execute_query("""
    SELECT
        subject_id,
        hadm_id,
        icd_code,
        icd_version
    FROM mimiciv_hosp.diagnoses_icd
""")
print(f"Diagnosis rows: {len(diagnoses)}")
diagnoses.to_parquet(out / "data" / "diagnoses.parquet", index=False)

# --- ICD procedures (used for CCS rollup) ---
procedures = execute_query("""
    SELECT
        subject_id,
        hadm_id,
        icd_code,
        icd_version
    FROM mimiciv_hosp.procedures_icd
""")
print(f"Procedure rows: {len(procedures)}")
procedures.to_parquet(out / "data" / "procedures.parquet", index=False)

print("Script 01 complete.")
