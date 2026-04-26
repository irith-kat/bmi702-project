"""04 — Gold label generation for HF decompensation events using Gemini.

Study: HF_test_run_v1
Target outcome: Acute HF decompensation (recurring binary outcome per 3-month window)
  Y=1 at any visit window containing an admission with IV diuresis + volume overload
  Y=0 at all other windows

Pipeline:
  1. Load map_results.parquet → map_prefilter → sample 30 MAP cases for labeling
     (controls not needed: MAP already handles whether; LATTE handles when/activity)
  2. Fetch discharge notes from BigQuery for the 30 selected patients
  3. Run Gemini labeling with HF_DECOMP_DISEASE_CONFIG — asks Gemini to identify
     EVERY decompensation admission per patient (cache: gemini_hf_decomp_cache.jsonl)
  4. Parse results → recurring_labels_to_latte() → per-visit (subject_id, T, Y) DataFrame
  5. Save gold_labels.parquet + unlabeled_pool.parquet + map_pools.parquet

Silver label strategy (informs LATTE key_codes in script 05):
  BNP / NT-proBNP ordered in a window is the decompensation silver proxy.
  Clinically: BNP is reactive, not routine — it's ordered when the care team suspects
  decompensation.  Analogous to MRI being ordered when MS relapse is suspected.

Run:
  uv run python output/HF_test_run_v1/scripts/04_gold_labels.py
"""

import logging
from pathlib import Path

import pandas as pd

from latte.gemini import (
    get_cached_subject_ids,
    parse_gemini_recurring_results,
    run_gemini_labeling,
)
from latte.labeler_utils import (
    HF_DECOMP_DISEASE_CONFIG,
    build_result_record_recurring,
    build_system_instruction_recurring,
    recurring_labels_to_latte,
    map_prefilter,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

out = Path(__file__).resolve().parent.parent

# ── Configuration ──────────────────────────────────────────────────────────────
BASELINE_DATE = "2100-01-01"  # study-wide anchor; MIMIC dates are shifted to ~2100-2200
MONTH_WINDOW = 3
N_CASES = 120  # MAP cases to send to Gemini for decompensation labeling
# (controls not needed — MAP already answered whether)

CACHE_JSONL = str(
    out / "data" / "gemini_hf_decomp_cache.jsonl"
)  # separate from incident cache
PROJECT_ID = "just-duality-438820-n4"
LOCATION = "global"
MODEL = "publishers/google/models/gemini-3.1-flash-lite-preview"

# ── 1. MAP prefilter — cases only ─────────────────────────────────────────────
print("Loading map_results...")
map_results = pd.read_parquet(out / "data" / "map_results.parquet")
print(
    f"  map_results: {len(map_results):,} rows, {map_results['patient_id'].nunique():,} patients"
)
print(f"  phenotype=1 (MAP cases): {(map_results['phenotype'] == 1).sum():,}")

# Pre-check which patients have discharge notes so map_prefilter only samples
# from patients we can actually label — avoids silent skips downstream.
print("\nChecking discharge note availability from notes_raw.parquet...")
notes_raw = pd.read_parquet(out / "data" / "notes_raw.parquet", columns=["subject_id"])
valid_sids = set(notes_raw["subject_id"].astype(str).unique())
print(f"  {len(valid_sids):,} patients have at least one discharge note")

cached_sids = get_cached_subject_ids(CACHE_JSONL)
print(f"  {len(cached_sids)} patients already in Gemini cache — will prioritise these")

print(f"\nRunning map_prefilter (cases only, n_cases={N_CASES})...")
pools = map_prefilter(
    map_results=map_results,
    n_cases=N_CASES,
    n_controls=0,  # MAP already handles whether; we only need confirmed cases for LATTE
    seed=42,
    valid_sids=valid_sids,
    preferred_sids=cached_sids,
)

cases_pool = pools["cases_pool"]
unlabeled_pool = pools["unlabeled_pool"]

print(f"  Gold label patients: {len(cases_pool)} MAP cases")
print(f"  Unlabeled pool: {len(unlabeled_pool):,} patients")

pd.DataFrame(
    {
        "subject_id": cases_pool + unlabeled_pool,
        "pool": ["gold_case"] * len(cases_pool) + ["unlabeled"] * len(unlabeled_pool),
    }
).to_parquet(out / "data" / "map_pools.parquet", index=False)
print("  map_pools.parquet saved")

# ── 2. Load discharge notes from notes_raw.parquet ────────────────────────────
print(
    f"\nLoading discharge notes for {len(cases_pool)} patients from notes_raw.parquet..."
)
notes_all = pd.read_parquet(out / "data" / "notes_raw.parquet")
notes_df = notes_all[notes_all["subject_id"].astype(str).isin(cases_pool)].copy()
notes_df = notes_df.sort_values(["subject_id", "charttime"]).reset_index(drop=True)

patients_with_notes = notes_df["subject_id"].nunique()
print(
    f"  Fetched {len(notes_df):,} notes for {patients_with_notes} patients "
    f"(median {notes_df.groupby('subject_id').size().median():.1f} notes/patient)"
)

patients_with_notes_set = set(notes_df["subject_id"].unique())
gold_sids_with_notes = [s for s in cases_pool if s in patients_with_notes_set]
missing = [s for s in cases_pool if s not in patients_with_notes_set]
if missing:
    logger.warning(
        "%d patients had no discharge notes and will be skipped: %s",
        len(missing),
        missing[:5],
    )

# ── 3. Run Gemini labeling (decompensation mode) ───────────────────────────────
print(f"\nRunning Gemini decompensation labeling (cache={CACHE_JSONL})...")
print(f"  Model  : {MODEL}")
print(f"  Config : {HF_DECOMP_DISEASE_CONFIG.name}")
print("  Mode   : recurring events (all decompensation admissions per patient)")

n_newly_labeled = run_gemini_labeling(
    notes_df=notes_df,
    subject_ids=gold_sids_with_notes,
    cache_jsonl=CACHE_JSONL,
    config=HF_DECOMP_DISEASE_CONFIG,
    model_name=MODEL,
    project_id=PROJECT_ID,
    location=LOCATION,
    max_notes_per_patient=60,
    retry_delay_seconds=5.0,
    record_builder=build_result_record_recurring,
    system_instruction_builder=build_system_instruction_recurring,
)
print(f"  Newly labeled this run: {n_newly_labeled}")

# ── 4. Parse Gemini results ────────────────────────────────────────────────────
print("\nParsing Gemini decompensation results...")
gemini_results = parse_gemini_recurring_results(
    cache_jsonl=CACHE_JSONL,
    baseline_date=BASELINE_DATE,
    month_window=MONTH_WINDOW,
)

# Filter to requested patients (cache may have stale entries)
requested_sids = set(str(s) for s in gold_sids_with_notes)
gemini_results = gemini_results[
    gemini_results["subject_id"].astype(str).isin(requested_sids)
].reset_index(drop=True)

n_with_events = (gemini_results["event_Ts"].apply(len) > 0).sum()
n_total_events = gemini_results["event_Ts"].apply(len).sum()
print(f"  Patients parsed  : {len(gemini_results)}")
print(f"  With ≥1 decompensation event : {n_with_events}")
print(f"  Total decompensation events  : {n_total_events}")
print(f"  Parse errors : {gemini_results['parse_error'].sum()}")

gemini_results.to_parquet(out / "data" / "gemini_decomp_results.parquet", index=False)

# ── 5. Load obs_log and convert to LATTE per-visit labels ─────────────────────
print("\nLoading obs_log for recurring_labels_to_latte...")
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
print(f"  obs_log: {len(obs_log):,} rows")

print("\nConverting to LATTE per-visit labels (recurring_labels_to_latte)...")
gold_labels = recurring_labels_to_latte(
    results_df=gemini_results,
    obs_log=obs_log,
    baseline_date=BASELINE_DATE,
    month_window=MONTH_WINDOW,
)
print(
    f"  gold_labels: {len(gold_labels):,} rows, {gold_labels['subject_id'].nunique()} patients"
)
print(f"  Y distribution: {gold_labels['Y'].value_counts().to_dict()}")
print(f"  T range: [{gold_labels['T'].min()}, {gold_labels['T'].max()}]")

gold_labels.to_parquet(out / "data" / "gold_labels.parquet", index=False)

pd.DataFrame({"subject_id": unlabeled_pool}).to_parquet(
    out / "data" / "unlabeled_pool.parquet", index=False
)

print("\nDone.")
print(f"  gemini_decomp_results.parquet : {len(gemini_results)} patients")
print(f"  gold_labels.parquet           : {len(gold_labels):,} rows (per-visit labels)")
print(f"  unlabeled_pool.parquet        : {len(unlabeled_pool):,} patients")
print("  map_pools.parquet             : pool assignment for all MAP patients")
print("  gemini_hf_decomp_cache.jsonl  : idempotent cache (add patients → re-run)")
