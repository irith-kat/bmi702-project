"""01 — Pull HF candidate EHR tables from DuckDB (codified data).

Study: hf-incident-v1
Cohort: all patients with ≥1 HF ICD code (I50% / 428%)
Backend: DuckDB (local MIMIC-IV) for all structured EHR tables.
         Notes are pulled separately in 01b_notes_pull.py via BigQuery.

Cache: each parquet is skipped if already present. Delete to re-pull.

Run:
    cd output/hf-incident-v1
    uv run python scripts/01_cohort_data.py
"""

import glob as _glob
from pathlib import Path

import pandas as pd
from m4 import execute_query, set_dataset
from m4.config import set_active_backend
from preprocessing.nlp import get_once_features as _get_once_features

set_active_backend("duckdb")
set_dataset("mimic-iv")

out = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
MAPPING_ROOT = REPO_ROOT / "mapping_dicts"
(out / "data").mkdir(exist_ok=True)

# Inline CTE — avoids passing large patient ID lists; inlined into every query.
# No ≥2-admissions filter here: the washup rule (≥2 two-month periods before
# first HF code) is applied in script 03 after obs_log is built.
HF_CTE = """
    WITH hf_subjects AS (
        SELECT DISTINCT CAST(subject_id AS VARCHAR) AS subject_id
        FROM mimiciv_hosp.diagnoses_icd
        WHERE (icd_version = 10 AND icd_code LIKE 'I50%')
           OR (icd_version = 9  AND icd_code LIKE '428%')
    )
"""

# ── 1. HF patient list ─────────────────────────────────────────────────────────
_path = out / "data" / "hf_patients.parquet"
if _path.exists():
    print("HF patients... [cached]")
    hf_patients_df = pd.read_parquet(_path)
else:
    print("Pulling HF patients from DuckDB...")
    hf_patients_df = execute_query(f"""
        {HF_CTE}
        SELECT h.subject_id
        FROM hf_subjects h
        ORDER BY h.subject_id
    """)
    hf_patients_df.to_parquet(_path, index=False)
print(f"  HF patients: {len(hf_patients_df):,}")

# ── 2. Admissions ──────────────────────────────────────────────────────────────
_path = out / "data" / "admissions.parquet"
if _path.exists():
    print("Admissions... [cached]")
    admissions = pd.read_parquet(_path)
else:
    print("Pulling admissions...")
    admissions = execute_query(f"""
        {HF_CTE}
        SELECT
            CAST(a.subject_id AS VARCHAR) AS subject_id,
            CAST(a.hadm_id    AS VARCHAR) AS hadm_id,
            a.admittime,
            a.dischtime,
            a.hospital_expire_flag
        FROM mimiciv_hosp.admissions a
        WHERE CAST(a.subject_id AS VARCHAR) IN (SELECT subject_id FROM hf_subjects)
    """)
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    admissions.to_parquet(_path, index=False)
print(
    f"  Admissions: {len(admissions):,} rows, {admissions['subject_id'].nunique():,} patients"
)

# ── 3. Diagnoses ───────────────────────────────────────────────────────────────
_path = out / "data" / "diagnoses_raw.parquet"
if _path.exists():
    print("Diagnoses... [cached]")
    diagnoses_raw = pd.read_parquet(_path)
else:
    print("Pulling diagnoses...")
    diagnoses_raw = execute_query(f"""
        {HF_CTE}
        SELECT
            CAST(d.subject_id AS VARCHAR) AS subject_id,
            CAST(d.hadm_id    AS VARCHAR) AS hadm_id,
            d.icd_code,
            d.icd_version,
            d.seq_num,
            a.admittime
        FROM mimiciv_hosp.diagnoses_icd d
        INNER JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
        WHERE CAST(d.subject_id AS VARCHAR) IN (SELECT subject_id FROM hf_subjects)
    """)
    diagnoses_raw["admittime"] = pd.to_datetime(diagnoses_raw["admittime"])
    diagnoses_raw.to_parquet(_path, index=False)
print(
    f"  Diagnoses: {len(diagnoses_raw):,} rows, {diagnoses_raw['subject_id'].nunique():,} patients"
)

# ── 4. Prescriptions ───────────────────────────────────────────────────────────
_path = out / "data" / "prescriptions_raw.parquet"
if _path.exists():
    print("Prescriptions... [cached]")
    prescriptions_raw = pd.read_parquet(_path)
else:
    print("Pulling prescriptions...")
    prescriptions_raw = execute_query(f"""
        {HF_CTE}
        SELECT
            CAST(p.subject_id AS VARCHAR) AS subject_id,
            CAST(p.hadm_id    AS VARCHAR) AS hadm_id,
            p.ndc,
            p.drug,
            p.starttime
        FROM mimiciv_hosp.prescriptions p
        WHERE p.drug_type = 'MAIN'
          AND CAST(p.subject_id AS VARCHAR) IN (SELECT subject_id FROM hf_subjects)
    """)
    prescriptions_raw["starttime"] = pd.to_datetime(prescriptions_raw["starttime"])
    prescriptions_raw.to_parquet(_path, index=False)
print(
    f"  Prescriptions: {len(prescriptions_raw):,} rows, {prescriptions_raw['subject_id'].nunique():,} patients"
)

# ── 5. Procedures ──────────────────────────────────────────────────────────────
_path = out / "data" / "procedures_raw.parquet"
if _path.exists():
    print("Procedures... [cached]")
    procedures_raw = pd.read_parquet(_path)
else:
    print("Pulling procedures (HCPCS)...")
    procedures_raw = execute_query(f"""
        {HF_CTE}
        SELECT
            CAST(e.subject_id AS VARCHAR) AS subject_id,
            CAST(e.hadm_id    AS VARCHAR) AS hadm_id,
            e.hcpcs_cd,
            e.chartdate
        FROM mimiciv_hosp.hcpcsevents e
        WHERE CAST(e.subject_id AS VARCHAR) IN (SELECT subject_id FROM hf_subjects)
    """)
    procedures_raw["chartdate"] = pd.to_datetime(procedures_raw["chartdate"])
    procedures_raw.to_parquet(_path, index=False)
print(f"  Procedures: {len(procedures_raw):,} rows")

# ── 6. Lab events (ONCE-filtered, LOINC codes only) ────────────────────────────
_codified_files = sorted(_glob.glob(str(REPO_ROOT / "input" / "ONCE_*PheCode*.csv")))
_narrative_files = sorted(_glob.glob(str(REPO_ROOT / "input" / "ONCE_*_C[0-9]*.csv")))
_codified_file = next(
    f for f in _codified_files if "428" in f and "heart failure" in f.lower()
)
_narrative_file = next(
    f for f in _narrative_files if "heart failure" in f.lower() and "C0018802" in f
)

_once = _get_once_features(_codified_file, _narrative_file)
_loinc_codes = {
    f[len("LOINC:") :] for f in _once["codified_list"] if f.startswith("LOINC:")
}
print(f"\nONCE LOINC codes for MAP: {len(_loinc_codes)}")

_map_df = pd.read_csv(
    MAPPING_ROOT / "d_labitems_to_loinc.csv",
    usecols=["itemid (omop_source_code)", "omop_concept_code", "omop_vocabulary_id"],
    dtype=str,
).rename(
    columns={"itemid (omop_source_code)": "itemid", "omop_concept_code": "loinc_code"}
)
_map_df = _map_df[_map_df["omop_vocabulary_id"] == "LOINC"]
_needed_itemids = sorted(
    int(r) for r in _map_df[_map_df["loinc_code"].isin(_loinc_codes)]["itemid"]
)
print(f"  Mapped to {len(_needed_itemids)} MIMIC itemids")

# labevents is not available in the local DuckDB MIMIC-IV (table absent).
# Only 3 LOINC codes / 2 itemids from ONCE were mapped here — minor contribution.
# Lab features are excluded; MAP runs on diagnoses, prescriptions, procedures + NLP.
labevents_raw = None
print("  Lab events: skipped (labevents not present in local DuckDB)")

print("\nDone. DuckDB raw tables saved.")
print(f"  hf_patients         : {len(hf_patients_df):,}")
print(f"  admissions          : {admissions['subject_id'].nunique():,} patients")
print(f"  diagnoses_raw       : {diagnoses_raw['subject_id'].nunique():,} patients")
print(f"  prescriptions_raw   : {prescriptions_raw['subject_id'].nunique():,} patients")
print(f"  procedures_raw      : {procedures_raw['subject_id'].nunique():,} patients")
print("  labevents_raw       : skipped (not available in local DuckDB)")
