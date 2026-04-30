"""05 — Gold label generation for HF incident timing using Gemini.

Study: hf-incident-v1
Target: incident Heart Failure — FIRST admission where HF was definitively present
        and was NOT pre-existing at prior admissions.

Pipeline:
  1. Load map_results → map_prefilter → 120 MAP cases for Gemini labeling
  2. Load discharge notes from notes_raw.parquet
  3. Run Gemini labeling with HF_DISEASE_CONFIG (incident mode)
  4. Parse results → labels_to_latte() → per-visit (subject_id, T, Y) DataFrame
  5. Save gold_labels.parquet, unlabeled_pool.parquet, map_pools.parquet

Run:
    cd output/hf-incident-v1
    uv run python scripts/05_gold_labels.py
"""

import logging
from pathlib import Path

import pandas as pd
from latte.gemini import (
    get_cached_subject_ids,
    parse_gemini_results,
    run_gemini_labeling,
)
from latte.labeler_utils import (
    HF_DISEASE_CONFIG,
    build_result_record,
    build_system_instruction,
    labels_to_latte,
    map_prefilter,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

out = Path(__file__).resolve().parent.parent

BASELINE_DATE = "2100-01-01"
MONTH_WINDOW = 2  # 2-month periods
N_CASES = 120  # MAP cases to send to Gemini

CACHE_JSONL = str(out / "data" / "gemini_hf_incident_cache.jsonl")
PROJECT_ID = "just-duality-438820-n4"
LOCATION = "global"
MODEL = "publishers/google/models/gemini-3.1-flash-lite-preview"

# ── 1. MAP prefilter ──────────────────────────────────────────────────────────
print("Loading map_results...")
map_results = pd.read_parquet(out / "data" / "map_results.parquet")
print(
    f"  {len(map_results):,} patients | cases: {(map_results['phenotype'] == 1).sum():,}"
)

notes_raw = pd.read_parquet(out / "data" / "notes_raw.parquet", columns=["subject_id"])
valid_sids = set(notes_raw["subject_id"].astype(str).unique())
print(f"  {len(valid_sids):,} patients with discharge notes")

cached_sids = get_cached_subject_ids(CACHE_JSONL)
print(f"  {len(cached_sids)} already in Gemini cache")

print(f"\nmap_prefilter (n_cases={N_CASES})...")
pools = map_prefilter(
    map_results=map_results,
    n_cases=N_CASES,
    n_controls=0,
    seed=42,
    valid_sids=valid_sids,
    preferred_sids=cached_sids,
)
cases_pool = pools["cases_pool"]
unlabeled_pool = pools["unlabeled_pool"]
print(f"  Gold patients  : {len(cases_pool)}")
print(f"  Unlabeled pool : {len(unlabeled_pool):,}")

pd.DataFrame(
    {
        "subject_id": cases_pool + unlabeled_pool,
        "pool": ["gold_case"] * len(cases_pool) + ["unlabeled"] * len(unlabeled_pool),
    }
).to_parquet(out / "data" / "map_pools.parquet", index=False)

# ── 2. Load notes ─────────────────────────────────────────────────────────────
print(f"\nLoading notes for {len(cases_pool)} gold patients...")
notes_all = pd.read_parquet(out / "data" / "notes_raw.parquet")
notes_df = notes_all[notes_all["subject_id"].astype(str).isin(cases_pool)].copy()
notes_df = notes_df.sort_values(["subject_id", "charttime"]).reset_index(drop=True)

gold_sids_with_notes = [
    s for s in cases_pool if s in set(notes_df["subject_id"].astype(str))
]
missing = [s for s in cases_pool if s not in set(notes_df["subject_id"].astype(str))]
print(f"  Notes available: {len(gold_sids_with_notes)} / {len(cases_pool)}")
if missing:
    logger.warning("No notes for %d patients: %s", len(missing), missing[:3])

# ── 3. Run Gemini (incident mode) ─────────────────────────────────────────────
print(f"\nRunning Gemini incident labeling (model={MODEL})...")
n_new = run_gemini_labeling(
    notes_df=notes_df,
    subject_ids=gold_sids_with_notes,
    cache_jsonl=CACHE_JSONL,
    config=HF_DISEASE_CONFIG,
    model_name=MODEL,
    project_id=PROJECT_ID,
    location=LOCATION,
    max_notes_per_patient=60,
    retry_delay_seconds=5.0,
    record_builder=build_result_record,
    system_instruction_builder=build_system_instruction,
)
print(f"  Newly labeled: {n_new}")

# ── 4. Parse results ──────────────────────────────────────────────────────────
print("\nParsing Gemini results...")
gemini_results = parse_gemini_results(
    cache_jsonl=CACHE_JSONL,
    baseline_date=BASELINE_DATE,
    month_window=MONTH_WINDOW,
)
requested = set(str(s) for s in gold_sids_with_notes)
gemini_results = gemini_results[
    gemini_results["subject_id"].astype(str).isin(requested)
].reset_index(drop=True)

n_cases_labeled = (gemini_results["label"] == 1).sum()
n_controls_labeled = (gemini_results["label"] == 0).sum()
n_errors = gemini_results["parse_error"].sum()
print(f"  Patients parsed : {len(gemini_results)}")
print(f"  HF confirmed (label=1): {n_cases_labeled}")
print(f"  Not HF  (label=0)     : {n_controls_labeled}")
print(f"  Parse errors           : {n_errors}")

valid = gemini_results[~gemini_results["parse_error"] & (gemini_results["label"] != -1)]
t_vals = valid["incident_T"].dropna()
print(
    f"  Incident T range: [{t_vals.min():.0f}, {t_vals.max():.0f}]  (n={len(t_vals)})"
)

gemini_results.to_parquet(out / "data" / "gemini_incident_results.parquet", index=False)

# ── 5. Convert to LATTE per-visit labels ──────────────────────────────────────
print("\nConverting to LATTE per-visit labels...")
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
gold_labels = labels_to_latte(
    results_df=gemini_results,
    obs_log=obs_log,
    baseline_date=BASELINE_DATE,
    month_window=MONTH_WINDOW,
)
print(
    f"  gold_labels: {len(gold_labels):,} rows, {gold_labels['subject_id'].nunique()} patients"
)
print(f"  Y dist: {gold_labels['Y'].value_counts().to_dict()}")

gold_labels.to_parquet(out / "data" / "gold_labels.parquet", index=False)
pd.DataFrame({"subject_id": unlabeled_pool}).to_parquet(
    out / "data" / "unlabeled_pool.parquet", index=False
)

print("\nDone.")
print(f"  gemini_incident_results.parquet : {len(gemini_results)} patients")
print(f"  gold_labels.parquet             : {len(gold_labels):,} rows")
print(f"  unlabeled_pool.parquet          : {len(unlabeled_pool):,} patients")
