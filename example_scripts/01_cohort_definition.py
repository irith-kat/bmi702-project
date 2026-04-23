"""01 — Pull HF candidates and raw EHR tables from MIMIC-IV (BigQuery).

Study: HF_test_run_v1
Cohort: patients with ≥1 Heart Failure ICD code (I50.x / 428.x) and ≥2 admissions.
BigQuery is used throughout because Step 02 also needs discharge notes from BigQuery.

All queries use inline CTEs to avoid passing large patient ID lists back to SQL
(the m4 SecurityError fires at >10K tokens in the query string).

Cache behaviour: each output parquet is skipped if already present in data/.
Re-run individual fetches by deleting the corresponding parquet file.
"""

import glob as _glob
from pathlib import Path

import pandas as pd
from m4 import execute_query, set_dataset
from m4.config import set_active_backend
from preprocessing.nlp import get_once_features as _get_once_features

set_active_backend("bigquery")
set_dataset("mimic-iv")

out = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
MAPPING_ROOT = REPO_ROOT / "mapping_dicts"
(out / "data").mkdir(exist_ok=True)

# Shared CTE that identifies HF subjects with ≥2 admissions.
# Inlined into every query so we never pass a Python list back to SQL.
HF_SUBJECTS_CTE = """
    WITH hf_subjects AS (
        SELECT DISTINCT d.subject_id
        FROM mimiciv_hosp.diagnoses_icd d
        WHERE
            (d.icd_version = 10 AND d.icd_code LIKE 'I50%')
            OR
            (d.icd_version = 9  AND d.icd_code LIKE '428%')
    ),
    admission_counts AS (
        SELECT subject_id, COUNT(DISTINCT hadm_id) AS n_admissions
        FROM mimiciv_hosp.admissions
        GROUP BY subject_id
    ),
    hf_cohort AS (
        SELECT h.subject_id
        FROM hf_subjects h
        JOIN admission_counts a USING (subject_id)
        WHERE a.n_admissions >= 2
    )
"""

# ── 1. Identify HF patients ────────────────────────────────────────────────────
_path = out / "data" / "hf_patients.parquet"
if _path.exists():
    print("Identifying HF patients... [cached]")
    hf_patients_df = pd.read_parquet(_path)
else:
    print("Identifying HF patients...")
    hf_patients_df = execute_query(f"""
        {HF_SUBJECTS_CTE}
        SELECT
            CAST(h.subject_id AS STRING) AS subject_id,
            a.n_admissions
        FROM hf_cohort h
        JOIN (
            SELECT subject_id, COUNT(DISTINCT hadm_id) AS n_admissions
            FROM mimiciv_hosp.admissions GROUP BY subject_id
        ) a USING (subject_id)
        ORDER BY h.subject_id
    """)
    hf_patients_df.to_parquet(_path, index=False)
print(f"  HF patients (≥2 admissions): {len(hf_patients_df):,}")
print(f"  Admission count median: {hf_patients_df['n_admissions'].median():.1f}")

# ── 2. Admissions ──────────────────────────────────────────────────────────────
_path = out / "data" / "admissions.parquet"
if _path.exists():
    print("\nPulling admissions... [cached]")
    admissions = pd.read_parquet(_path)
else:
    print("\nPulling admissions...")
    admissions = execute_query(f"""
        {HF_SUBJECTS_CTE}
        SELECT
            CAST(a.subject_id AS STRING) AS subject_id,
            CAST(a.hadm_id    AS STRING) AS hadm_id,
            a.admittime,
            a.dischtime,
            a.hospital_expire_flag
        FROM mimiciv_hosp.admissions a
        WHERE a.subject_id IN (SELECT subject_id FROM hf_cohort)
    """)
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    admissions.to_parquet(_path, index=False)
print(
    f"  Admissions: {len(admissions):,} rows, {admissions['subject_id'].nunique():,} patients"
)

# ── 3. Diagnoses (with admittime joined) ───────────────────────────────────────
_path = out / "data" / "diagnoses_raw.parquet"
if _path.exists():
    print("\nPulling diagnoses... [cached]")
    diagnoses_raw = pd.read_parquet(_path)
else:
    print("\nPulling diagnoses...")
    diagnoses_raw = execute_query(f"""
        {HF_SUBJECTS_CTE}
        SELECT
            CAST(d.subject_id AS STRING) AS subject_id,
            CAST(d.hadm_id    AS STRING) AS hadm_id,
            d.icd_code,
            d.icd_version,
            d.seq_num,
            a.admittime
        FROM mimiciv_hosp.diagnoses_icd d
        INNER JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
        WHERE d.subject_id IN (SELECT subject_id FROM hf_cohort)
    """)
    diagnoses_raw["admittime"] = pd.to_datetime(diagnoses_raw["admittime"])
    diagnoses_raw.to_parquet(_path, index=False)
print(
    f"  Diagnoses: {len(diagnoses_raw):,} rows, {diagnoses_raw['subject_id'].nunique():,} patients"
)

# ── 4. Prescriptions (MAIN drug_type only) ─────────────────────────────────────
_path = out / "data" / "prescriptions_raw.parquet"
if _path.exists():
    print("\nPulling prescriptions... [cached]")
    prescriptions_raw = pd.read_parquet(_path)
else:
    print("\nPulling prescriptions...")
    prescriptions_raw = execute_query(f"""
        {HF_SUBJECTS_CTE}
        SELECT
            CAST(p.subject_id AS STRING) AS subject_id,
            CAST(p.hadm_id    AS STRING) AS hadm_id,
            p.ndc,
            p.drug,
            p.starttime
        FROM mimiciv_hosp.prescriptions p
        WHERE p.drug_type = 'MAIN'
          AND p.subject_id IN (SELECT subject_id FROM hf_cohort)
    """)
    prescriptions_raw["starttime"] = pd.to_datetime(prescriptions_raw["starttime"])
    prescriptions_raw.to_parquet(_path, index=False)
print(
    f"  Prescriptions: {len(prescriptions_raw):,} rows, {prescriptions_raw['subject_id'].nunique():,} patients"
)

# ── 5. Procedures (HCPCS/CPT) ─────────────────────────────────────────────────
_path = out / "data" / "procedures_raw.parquet"
if _path.exists():
    print("\nPulling procedures... [cached]")
    procedures_raw = pd.read_parquet(_path)
else:
    print("\nPulling procedures (HCPCS)...")
    procedures_raw = execute_query(f"""
        {HF_SUBJECTS_CTE}
        SELECT
            CAST(e.subject_id AS STRING) AS subject_id,
            CAST(e.hadm_id    AS STRING) AS hadm_id,
            e.hcpcs_cd,
            e.chartdate
        FROM mimiciv_hosp.hcpcsevents e
        WHERE e.subject_id IN (SELECT subject_id FROM hf_cohort)
    """)
    procedures_raw["chartdate"] = pd.to_datetime(procedures_raw["chartdate"])
    procedures_raw.to_parquet(_path, index=False)
print(
    f"  Procedures: {len(procedures_raw):,} rows, {procedures_raw['subject_id'].nunique():,} patients"
)

# ── 6. Lab events (ONCE-filtered) ─────────────────────────────────────────────
# Rather than fetching all lab rows (~40–60M), we resolve which LOINC codes ONCE
# selected for HF, reverse-look them up to MIMIC itemids, and filter the query.
# This typically reduces the fetch from ~40M rows to ~1–3M.

# Load ONCE features to get the LOINC codes MAP needs
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
print(f"\nONCE LOINC codes needed for MAP: {len(_loinc_codes)}")

# Reverse-lookup: LOINC codes → MIMIC itemids via d_labitems_to_loinc.csv
_map_df = pd.read_csv(
    MAPPING_ROOT / "d_labitems_to_loinc.csv",
    usecols=["itemid (omop_source_code)", "omop_concept_code", "omop_vocabulary_id"],
    dtype=str,
).rename(
    columns={"itemid (omop_source_code)": "itemid", "omop_concept_code": "loinc_code"}
)
_map_df = _map_df[_map_df["omop_vocabulary_id"] == "LOINC"]
_needed_itemids = sorted(
    int(row) for row in _map_df[_map_df["loinc_code"].isin(_loinc_codes)]["itemid"]
)
print(f"  Mapped to {len(_needed_itemids)} MIMIC itemids")

_itemid_sql = ", ".join(str(i) for i in _needed_itemids)

_path = out / "data" / "labevents_raw.parquet"
if _path.exists():
    print("Pulling lab events... [cached]")
    labevents_raw = pd.read_parquet(_path)
else:
    print("Pulling lab events (itemid-filtered)...")
    labevents_raw = execute_query(f"""
        {HF_SUBJECTS_CTE}
        SELECT
            CAST(l.subject_id AS STRING) AS subject_id,
            l.itemid,
            l.charttime,
            l.valuenum
        FROM mimiciv_hosp.labevents l
        WHERE l.subject_id IN (SELECT subject_id FROM hf_cohort)
          AND l.valuenum IS NOT NULL
          AND l.itemid IN ({_itemid_sql})
    """)
    labevents_raw["charttime"] = pd.to_datetime(labevents_raw["charttime"])
    labevents_raw.to_parquet(_path, index=False)
print(
    f"  Lab events: {len(labevents_raw):,} rows, {labevents_raw['subject_id'].nunique():,} patients"
)
print(f"  Unique itemids: {labevents_raw['itemid'].nunique():,}")

# ── 7. Discharge notes ────────────────────────────────────────────────────────
# One discharge summary per admission (mimiciv_note.discharge).
# Fetching only discharge notes keeps the table small; script 01_5 will run
# MedSpaCy CUI extraction on this table.
# Note: discharge notes live in a separate BigQuery dataset (mimic-iv-note).
_path = out / "data" / "notes_raw.parquet"
if _path.exists():
    print("\nPulling discharge notes... [cached]")
    notes_raw = pd.read_parquet(_path)
else:
    print("\nPulling discharge notes...")
    # mimiciv_note lives in a separate BigQuery dataset with no mimiciv_hosp tables,
    # so the HF_SUBJECTS_CTE won't resolve there. Use batched subject_id IN (...)
    # clauses instead, exactly as MS script 05 does.
    set_dataset("mimic-iv-note")
    BATCH_SIZE = 400  # ~3,600 tokens per query, well under the m4 limit
    subject_ids = hf_patients_df["subject_id"].tolist()
    batches = [
        subject_ids[i : i + BATCH_SIZE] for i in range(0, len(subject_ids), BATCH_SIZE)
    ]
    print(f"  Batches: {len(batches)} × {BATCH_SIZE} IDs")
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
        print(f"  Batch {i}/{len(batches)}: {len(chunk):,} notes", flush=True)
    notes_raw = pd.concat(chunks, ignore_index=True)
    notes_raw["charttime"] = pd.to_datetime(notes_raw["charttime"])
    notes_raw.to_parquet(_path, index=False)
    set_dataset("mimic-iv")
print(
    f"  Discharge notes: {len(notes_raw):,} rows, {notes_raw['subject_id'].nunique():,} patients"
)

print("\nDone. Raw tables saved to data/")
print(f"  hf_patients.parquet      : {len(hf_patients_df):,} patients")
print(f"  admissions.parquet       : {admissions['subject_id'].nunique():,} patients")
print(
    f"  diagnoses_raw.parquet    : {diagnoses_raw['subject_id'].nunique():,} patients"
)
print(
    f"  prescriptions_raw.parquet: {prescriptions_raw['subject_id'].nunique():,} patients"
)
print(
    f"  procedures_raw.parquet   : {procedures_raw['subject_id'].nunique():,} patients"
)
print(
    f"  labevents_raw.parquet    : {labevents_raw['subject_id'].nunique():,} patients"
)
print(f"  notes_raw.parquet        : {notes_raw['subject_id'].nunique():,} patients")
