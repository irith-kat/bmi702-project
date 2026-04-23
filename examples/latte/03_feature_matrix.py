"""02 — Build HF ONCE feature matrix for MAP (codified + NLP CUI features).

Study: HF_test_run_v1
Reads raw tables from data/ (produced by 01_cohort_definition.py).
NLP CUI events are loaded from cui_obs.parquet (produced by 01_5_notes_nlp.py)
and concatenated into the observation log before MAP preprocessing.
Outputs obs_log, mat_df, note_df.
"""

import glob
import json
from pathlib import Path

import pandas as pd
from map import preprocess_map
from preprocessing.nlp import get_once_features
from preprocessing.structured import build_obs_log, rollup_itemid_to_loinc

REPO_ROOT = Path(__file__).resolve().parents[3]

out = Path(__file__).resolve().parent.parent
MAPPING_ROOT = REPO_ROOT / "mapping_dicts"

# ── 1. Load HF ONCE features ───────────────────────────────────────────────────
print("Loading HF ONCE features...")
codified_files = sorted(glob.glob(str(REPO_ROOT / "input" / "ONCE_*PheCode*.csv")))
narrative_files = sorted(glob.glob(str(REPO_ROOT / "input" / "ONCE_*_C[0-9]*.csv")))

# Pick HF-specific files (428.1 anchor, "heart failure" named)
codified_file = next(
    f for f in codified_files if "428" in f and "heart failure" in f.lower()
)
narrative_file = next(
    f for f in narrative_files if "heart failure" in f.lower() and "C0018802" in f
)

print(f"  Codified : {Path(codified_file).name}")
print(f"  Narrative: {Path(narrative_file).name}")

once_features = get_once_features(codified_file, narrative_file)
print(f"  Codified features : {len(once_features['codified_list'])}")
print(f"  NLP CUI targets   : {len(once_features['nlp_target_cuis'])}")

MAIN_PHECODE = "428.1"
anchor = f"PheCode:{MAIN_PHECODE}"
assert anchor in once_features["codified_list"], (
    f"Anchor {anchor} not found in ONCE codified_list. Check the ONCE file.\n"
    f"Available: {once_features['codified_list'][:10]}"
)
print(f"  Anchor confirmed  : {anchor}")

# ── 2. Load raw tables ─────────────────────────────────────────────────────────
print("\nLoading raw tables...")
diagnoses_raw = pd.read_parquet(out / "data" / "diagnoses_raw.parquet")
prescriptions_raw = pd.read_parquet(out / "data" / "prescriptions_raw.parquet")
procedures_raw = pd.read_parquet(out / "data" / "procedures_raw.parquet")
admissions_df = pd.read_parquet(out / "data" / "admissions.parquet")
labevents_raw_path = out / "data" / "labevents_raw.parquet"
labevents_raw = (
    pd.read_parquet(labevents_raw_path) if labevents_raw_path.exists() else None
)
print(f"  diagnoses    : {len(diagnoses_raw):,} rows")
print(f"  prescriptions: {len(prescriptions_raw):,} rows")
print(f"  procedures   : {len(procedures_raw):,} rows")
print(f"  admissions   : {len(admissions_df):,} rows")
if labevents_raw is not None:
    print(
        f"  labevents    : {len(labevents_raw):,} rows  ({labevents_raw['itemid'].nunique():,} unique itemids)"
    )
    labevents_loinc = rollup_itemid_to_loinc(
        labevents_raw,
        itemid_column="itemid",
        mapping_file=str(MAPPING_ROOT / "d_labitems_to_loinc.csv"),
    )
    print(f"  labevents (LOINC): {len(labevents_loinc):,} rows")
else:
    print("  labevents    : not found (run script 01 to fetch lab data)")
    labevents_loinc = None

# ── 3. Build observation log ───────────────────────────────────────────────────
print("\nBuilding observation log (ICD→PheCode, NDC→RxNorm, CPT→CCS)...")
obs_log = build_obs_log(
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
    f"  Observation log: {len(obs_log):,} rows, {obs_log['subject_id'].nunique():,} patients"
)
print(f"  Event types: {obs_log['event_type'].value_counts().to_dict()}")

# Confirm anchor is present in obs_log
anchor_count = (obs_log["event"] == anchor).sum()
print(f"  Anchor '{anchor}' rows in obs_log: {anchor_count:,}")

obs_log.to_parquet(out / "data" / "obs_log.parquet", index=False)

# ── 4. Append NLP CUI events (if 01_5_notes_nlp.py has been run) ──────────────
cui_obs_path = out / "data" / "cui_obs.parquet"
if cui_obs_path.exists():
    cui_obs = pd.read_parquet(cui_obs_path)
    combined_obs = pd.concat([obs_log, cui_obs], ignore_index=True)
    print(
        f"\nNLP CUI events loaded: {len(cui_obs):,} rows "
        f"({cui_obs['subject_id'].nunique():,} patients) — combined obs_log: {len(combined_obs):,} rows"
    )
else:
    combined_obs = obs_log
    print("\ncui_obs.parquet not found — running MAP on codified features only.")
    print("  Run 01_5_notes_nlp.py first to include NLP CUI features.")

# ── 5. Preprocess MAP inputs ───────────────────────────────────────────────────
print("\nBuilding MAP inputs (preprocess_map)...")
mat_df, note_df = preprocess_map(
    obs_log=combined_obs,
    admissions_df=admissions_df,
    once_features=once_features,
    main_phecode=MAIN_PHECODE,
    min_nonzero=20,  # MAP default; prevents flexmix NaN log-likelihood on sparse features
)
print(f"  mat_df : {mat_df.shape[0]:,} patients × {mat_df.shape[1]} features")
print(
    f"  note_df: {len(note_df):,} patients, note_count range "
    f"[{note_df['note_count'].min()}, {note_df['note_count'].max()}]"
)
print(f"  Anchor column present: {anchor in mat_df.columns}")
print(f"  Patients with anchor > 0: {(mat_df[anchor] > 0).sum():,}")

mat_df.to_parquet(out / "data" / "mat_df.parquet")
note_df.to_parquet(out / "data" / "note_df.parquet")

# Save the ONCE feature list so downstream scripts don't need to reload the files.
# feature_codes = codified + NLP CUI codes (deduplicated, CUI format: "CUI:<cui>").
nlp_cuis_list = list(dict.fromkeys("CUI:" + cui for cui in once_features["nlp_list"]))
feature_codes = once_features["codified_list"] + nlp_cuis_list
(out / "data").joinpath("once_features_meta.json").write_text(
    json.dumps(
        {
            "codified_list": once_features["codified_list"],
            "nlp_cuis_list": nlp_cuis_list,
            "feature_codes": feature_codes,
            "main_phecode": MAIN_PHECODE,
            "anchor": anchor,
        },
        indent=2,
    )
)

print("\nDone. Feature matrix saved.")
print(f"  obs_log.parquet         : {len(obs_log):,} rows")
print(f"  mat_df.parquet          : {mat_df.shape}")
print(f"  note_df.parquet         : {note_df.shape}")
print(
    f"  once_features_meta.json : {len(feature_codes)} total features "
    f"({len(once_features['codified_list'])} codified + {len(nlp_cuis_list)} NLP CUIs)"
)
