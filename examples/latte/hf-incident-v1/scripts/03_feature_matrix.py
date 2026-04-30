"""03 — Build observation log, apply washup exclusion, and prepare MAP inputs.

Study: hf-incident-v1
Reads : data/{diagnoses,prescriptions,procedures,labevents,admissions}_raw.parquet
        data/cui_obs.parquet (NLP events)
Writes: data/obs_log_full.parquet   — full obs_log before washup (for suitability script)
        data/obs_log.parquet        — washup-filtered obs_log (MAP + LATTE input)
        data/silver_labels.parquet  — T_silver per patient (first HF period)
        data/cohort_ids.parquet     — final patient list with washup metadata
        data/mat_df.parquet / note_df.parquet / once_features_meta.json

Washup rule: keep patients with >= 2 distinct two-month periods containing
any event BEFORE their first HF anchor code period (T_hf).

Run:
    cd output/hf-incident-v1
    uv run python scripts/03_feature_matrix.py
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from map import preprocess_map
from preprocessing.nlp import get_once_features
from preprocessing.structured import build_obs_log, rollup_itemid_to_loinc

REPO_ROOT = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent.parent
MAPPING_ROOT = REPO_ROOT / "mapping_dicts"

BASELINE_DATE = "2100-01-01"
PERIOD_DAYS = 60  # 2-month windows
MAIN_PHECODE = "428.1"
ANCHOR = f"PheCode:{MAIN_PHECODE}"

# ── 1. ONCE features ──────────────────────────────────────────────────────────
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
print(f"  Codified: {len(once_features['codified_list'])} features")
print(f"  NLP CUIs: {len(once_features['nlp_target_cuis'])}")
assert ANCHOR in once_features["codified_list"], (
    f"Anchor {ANCHOR} missing from ONCE file"
)

# ── 2. Load raw tables ────────────────────────────────────────────────────────
print("\nLoading raw tables...")
diagnoses_raw = pd.read_parquet(out / "data" / "diagnoses_raw.parquet")
prescriptions_raw = pd.read_parquet(out / "data" / "prescriptions_raw.parquet")
procedures_raw = pd.read_parquet(out / "data" / "procedures_raw.parquet")
admissions_df = pd.read_parquet(out / "data" / "admissions.parquet")
labevents_raw_path = out / "data" / "labevents_raw.parquet"
labevents_raw = (
    pd.read_parquet(labevents_raw_path) if labevents_raw_path.exists() else None
)

if labevents_raw is not None:
    labevents_loinc = rollup_itemid_to_loinc(
        labevents_raw,
        itemid_column="itemid",
        mapping_file=str(MAPPING_ROOT / "d_labitems_to_loinc.csv"),
    )
    print(f"  labevents (LOINC): {len(labevents_loinc):,} rows")
else:
    labevents_loinc = None

# ── 3. Build full observation log ─────────────────────────────────────────────
print("\nBuilding full observation log...")
obs_full = build_obs_log(
    icd_df=diagnoses_raw,
    icd_col="icd_code",
    icd_date_col="admittime",
    drug_df=prescriptions_raw,
    drug_ndc_col="ndc",
    drug_date_col="starttime",
    drug_col="drug",
    cpt_df=procedures_raw,
    cpt_col="hcpcs_cd",
    cpt_date_col="chartdate",
    notes_df=None,
    lab_df=labevents_loinc,
    lab_loinc_col="loinc_code",
    lab_date_col="charttime",
    lab_value_col="valuenum",
    icd_mapping_file=str(MAPPING_ROOT / "Phecode_map_v1_2_icd9_icd10cm.csv"),
    cpt_mapping_file=str(MAPPING_ROOT / "CCS_Services_Procedures_v2025-1_052425.csv"),
    ndc_mapping_file=str(MAPPING_ROOT / "ndc_to_rxnorm_ingredient.csv"),
    drug_name_mapping_file=str(MAPPING_ROOT / "drug_name_to_rxnorm_ingredient.csv"),
)
print(
    f"  Full obs_log: {len(obs_full):,} rows, {obs_full['subject_id'].nunique():,} patients"
)

# Append NLP CUI events
cui_obs_path = out / "data" / "cui_obs.parquet"
if cui_obs_path.exists():
    cui_obs = pd.read_parquet(cui_obs_path)
    obs_full = pd.concat([obs_full, cui_obs], ignore_index=True)
    print(f"  After NLP: {len(obs_full):,} rows total")
else:
    print("  WARNING: cui_obs.parquet not found — running without NLP features")

obs_full.to_parquet(out / "data" / "obs_log_full.parquet", index=False)

# ── 4. Compute period numbers and apply washup filter ─────────────────────────
print("\nApplying washup filter (>= 2 two-month periods before first HF code)...")
base_dt = pd.to_datetime(BASELINE_DATE)
obs_full["_dt"] = pd.to_datetime(obs_full["datetime"], errors="coerce")
obs_full["T"] = np.floor((obs_full["_dt"] - base_dt).dt.days / PERIOD_DAYS).astype(
    "Int64"
)

# Silver label: first period containing the HF anchor for each patient
anchor_rows = obs_full[obs_full["event"] == ANCHOR].dropna(subset=["T"])
first_hf = anchor_rows.groupby("subject_id")["T"].min().rename("T_silver")

# Periods before first HF (any event type)
pre_hf = obs_full.dropna(subset=["T"]).copy()
pre_hf = pre_hf.merge(first_hf, on="subject_id", how="inner")
pre_hf = pre_hf[pre_hf["T"] < pre_hf["T_silver"]]
pre_periods = pre_hf.groupby("subject_id")["T"].nunique().rename("n_pre_periods")

# Washup criterion
washup_df = (
    first_hf.to_frame().join(pre_periods, how="left").fillna({"n_pre_periods": 0})
)
washup_df["n_pre_periods"] = washup_df["n_pre_periods"].astype(int)
washup_df["passes_washup"] = washup_df["n_pre_periods"] >= 2

n_pass = washup_df["passes_washup"].sum()
n_fail = (~washup_df["passes_washup"]).sum()
print(f"  Pass washup (>= 2 pre-HF periods): {n_pass:,}")
print(f"  Fail washup (< 2 pre-HF periods) : {n_fail:,}")

passing_ids = washup_df[washup_df["passes_washup"]].index.tolist()

# Save silver labels and cohort IDs for script 08
silver_df = washup_df[washup_df["passes_washup"]][
    ["T_silver", "n_pre_periods"]
].reset_index()
silver_df.columns = ["subject_id", "T_silver", "n_pre_periods"]
silver_df.to_parquet(out / "data" / "silver_labels.parquet", index=False)

pd.DataFrame({"subject_id": passing_ids}).to_parquet(
    out / "data" / "cohort_ids.parquet", index=False
)
print(f"  silver_labels.parquet: {len(silver_df):,} patients")

# ── 5. Filter obs_log to washup-passing patients ──────────────────────────────
obs_log = obs_full[
    obs_full["subject_id"].astype(str).isin([str(s) for s in passing_ids])
].copy()
obs_log = obs_log.drop(columns=["_dt", "T"], errors="ignore")
obs_log.to_parquet(out / "data" / "obs_log.parquet", index=False)
print(
    f"\n  Filtered obs_log: {len(obs_log):,} rows, {obs_log['subject_id'].nunique():,} patients"
)

# ── 6. Preprocess MAP inputs ───────────────────────────────────────────────────
print("\nBuilding MAP inputs (preprocess_map)...")
admissions_filtered = admissions_df[
    admissions_df["subject_id"].astype(str).isin([str(s) for s in passing_ids])
]
mat_df, note_df = preprocess_map(
    obs_log=obs_log,
    admissions_df=admissions_filtered,
    once_features=once_features,
    main_phecode=MAIN_PHECODE,
    min_nonzero=20,
)
print(f"  mat_df : {mat_df.shape[0]:,} patients × {mat_df.shape[1]} features")
print(f"  note_df: {len(note_df):,} patients")
print(f"  Anchor '{ANCHOR}' present: {ANCHOR in mat_df.columns}")
print(f"  Patients with anchor > 0: {(mat_df[ANCHOR] > 0).sum():,}")

mat_df.to_parquet(out / "data" / "mat_df.parquet")
note_df.to_parquet(out / "data" / "note_df.parquet")

# Save feature metadata for LATTE
nlp_cuis_list = list(dict.fromkeys("CUI:" + cui for cui in once_features["nlp_list"]))
feature_codes = once_features["codified_list"] + nlp_cuis_list
(out / "data" / "once_features_meta.json").write_text(
    json.dumps(
        {
            "codified_list": once_features["codified_list"],
            "nlp_cuis_list": nlp_cuis_list,
            "feature_codes": feature_codes,
            "main_phecode": MAIN_PHECODE,
            "anchor": ANCHOR,
            "baseline_date": BASELINE_DATE,
            "period_days": PERIOD_DAYS,
        },
        indent=2,
    )
)

print("\nDone.")
print(
    f"  obs_log_full.parquet   : {obs_full['subject_id'].nunique():,} patients (pre-washup)"
)
print(
    f"  obs_log.parquet        : {obs_log['subject_id'].nunique():,} patients (washup-filtered)"
)
print(f"  mat_df.parquet         : {mat_df.shape}")
print(f"  note_df.parquet        : {note_df.shape}")
print(
    f"  once_features_meta.json: {len(feature_codes)} features ({len(once_features['codified_list'])} codified + {len(nlp_cuis_list)} NLP CUIs)"
)
