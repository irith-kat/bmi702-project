"""05 — Run LATTE to predict incident HF onset timing.

Study: HF_test_run_v1
Pipeline:
  1. Load obs_log + gold_labels (from scripts 02 and 04)
  2. format_latte_input → train.csv, test.csv, unlabeled.csv
  3. build_cooccurrence_embeddings → embedding.csv
  4. run_latte → per-patient incident probability predictions
  5. Save predictions.parquet

Run:
  uv run python output/HF_test_run_v1/scripts/05_latte_phenotyping.py
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from latte.embeddings import build_cooccurrence_embeddings
from latte.labeler_utils import HF_DECOMP_DISEASE_CONFIG
from latte.latte import format_latte_input, run_latte

REPO_ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

out = Path(__file__).resolve().parent.parent
LATTE_DIR = str(REPO_ROOT / "src" / "LATTE-main" / "LATTE-main")

# ── Configuration ──────────────────────────────────────────────────────────────
BASELINE_DATE = "2100-01-01"
MONTH_WINDOW = 3
EMBEDDING_DIM = 50  # clamped to matrix rank; small for test run (full: 50)
EPOCHS = 35  # optimal
EPOCH_SILVER = 8  # optimal
LAYERS_INCIDENT = "80"  # Run 9 proved single GRU layer best for small label set
TRAIN_FRAC = 0.8
MAX_UNLABELED = 10_000  # cap unlabeled pool to avoid OOM in TF

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading obs_log and gold_labels...")
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
gold_labels = pd.read_parquet(out / "data" / "gold_labels.parquet")
unlabeled_pool_df = pd.read_parquet(out / "data" / "unlabeled_pool.parquet")

unlabeled_ids = unlabeled_pool_df["subject_id"].astype(str).tolist()
if len(unlabeled_ids) > MAX_UNLABELED:
    import random

    random.seed(42)
    unlabeled_ids = random.sample(unlabeled_ids, MAX_UNLABELED)
    print(
        f"  Unlabeled pool capped at {MAX_UNLABELED} (was {len(unlabeled_pool_df):,})"
    )

print(
    f"  obs_log     : {len(obs_log):,} rows, {obs_log['subject_id'].nunique():,} patients"
)
print(
    f"  gold_labels : {len(gold_labels):,} rows, {gold_labels['subject_id'].nunique()} patients"
)
print(
    f"  unlabeled   : {len(unlabeled_ids):,} patients (MAP cases without gold labels)"
)

# ── 2. Load feature codes from ONCE metadata ───────────────────────────────────
print("\nLoading ONCE feature codes...")
meta = json.loads((out / "data" / "once_features_meta.json").read_text())
feature_codes = meta.get("feature_codes", meta["codified_list"])
anchor = meta["anchor"]

print(f"  Feature codes: {len(feature_codes)}")
print(f"  Anchor: {anchor}")

# Key codes for LATTE silver label initialisation (decompensation silver proxy).
# BNP / NT-proBNP are the anchor: ordered reactively when decompensation is suspected
# (analogous to MRI orders being used as the MS relapse silver label in the paper).
# These codes must be present as columns in the LATTE CSVs (i.e., in feature_codes).
raw_key_codes = HF_DECOMP_DISEASE_CONFIG.key_codes  # ["LOINC:33762-6", "ShortName:BNP"]
key_codes_present = [c for c in raw_key_codes if c in feature_codes]
if not key_codes_present:
    # Fallback: use the HF anchor PheCode if BNP codes not in ONCE feature list
    key_codes_present = [anchor]
    logger.warning(
        "None of HF_DECOMP_DISEASE_CONFIG.key_codes (BNP/NT-proBNP) found in ONCE features. "
        "Falling back to anchor code: %s",
        anchor,
    )
key_codes_str = ",".join(key_codes_present)
print(f"  Key codes for LATTE: {key_codes_str}")

# ── 3. Format LATTE input ──────────────────────────────────────────────────────
print("\nFormatting LATTE input (format_latte_input)...")
train_df, test_df, unlabeled_df = format_latte_input(
    obs_log=obs_log,
    gold_labels=gold_labels,
    feature_codes=feature_codes,
    baseline_date=BASELINE_DATE,
    month_window=MONTH_WINDOW,
    unlabeled_ids=unlabeled_ids,
    train_frac=TRAIN_FRAC,
    seed=42,
)
print(f"  train_df    : {len(train_df):,} rows, {train_df['ID'].nunique()} patients")
print(f"  test_df     : {len(test_df):,} rows, {test_df['ID'].nunique()} patients")
print(
    f"  unlabeled_df: {len(unlabeled_df):,} rows, {unlabeled_df['ID'].nunique()} patients"
)
print(f"  Y dist (train): {train_df['Y'].value_counts().to_dict()}")

feature_col_start = 3
feature_col_end = 3 + len(feature_codes)

# ── 4. Build co-occurrence embeddings ──────────────────────────────────────────
print("\nBuilding co-occurrence embeddings (build_cooccurrence_embeddings)...")
embedding_df = build_cooccurrence_embeddings(
    obs_log=obs_log,
    feature_codes=feature_codes,
    n_components=EMBEDDING_DIM,
)
print(f"  embedding_df: {embedding_df.shape}  (components × codes)")

# ── 5. Write LATTE input files ─────────────────────────────────────────────────
latte_data_dir = str(out / "data" / "latte_input") + os.sep
latte_results_dir = str(out / "data" / "latte_results") + os.sep
os.makedirs(latte_data_dir, exist_ok=True)
os.makedirs(latte_results_dir, exist_ok=True)

train_df.to_csv(latte_data_dir + "train.csv", index=False)
test_df.to_csv(latte_data_dir + "test.csv", index=False)
unlabeled_df.to_csv(latte_data_dir + "unlabeled.csv", index=False)
embedding_df.to_csv(latte_data_dir + "embedding.csv")  # index=True — LATTE expects it

print(f"  LATTE input files written to: {latte_data_dir}")
print(f"    train.csv     : {len(train_df):,} rows")
print(f"    test.csv      : {len(test_df):,} rows")
print(f"    unlabeled.csv : {len(unlabeled_df):,} rows")
print(f"    embedding.csv : {embedding_df.shape}")

# ── 6. Run LATTE ──────────────────────────────────────────────────────────────
print("\nRunning LATTE (this calls a_train_final.py via subprocess)...")
print(f"  latte_dir       : {LATTE_DIR}")
print(f"  epochs          : {EPOCHS}  (silver: {EPOCH_SILVER})")
print(f"  embedding_dim   : {EMBEDDING_DIM}")
print(f"  layers_incident : {LAYERS_INCIDENT}")
print(f"  key_codes       : {key_codes_str}")

predictions_df = run_latte(
    latte_dir=LATTE_DIR,
    data_dir=latte_data_dir,
    embedding_file=latte_data_dir + "embedding.csv",
    key_codes=key_codes_str,
    feature_col_start=feature_col_start,
    feature_col_end=feature_col_end,
    save_dir=latte_results_dir,
    results_filename="results_hf_test.csv",
    epochs=EPOCHS,
    epoch_silver=EPOCH_SILVER,
    embedding_dim=EMBEDDING_DIM,
    layers_incident=LAYERS_INCIDENT,
    weight_prevalence=0.2,
    weight_unlabel=0.025,
    weight_contrastive=0.1,
    weight_smooth=0.1,
    weight_additional=0.1,
    flag_train_augment=1,
    month_window=MONTH_WINDOW,
    max_visits=25,
)

print(
    f"\n  predictions_df: {len(predictions_df):,} rows, "
    f"{predictions_df['subject_id'].nunique()} patients"
)
print(
    f"  incident_probability range: "
    f"[{predictions_df['incident_probability'].min():.3f}, "
    f"{predictions_df['incident_probability'].max():.3f}]"
)
print(f"  Columns: {list(predictions_df.columns)}")
print("\nSample predictions (first 8 rows):")
print(predictions_df.head(8).to_string(index=False))

# ── 7. Save ────────────────────────────────────────────────────────────────────
predictions_df.to_parquet(out / "data" / "latte_predictions.parquet", index=False)

# Summary
labeled_preds = predictions_df[predictions_df["Y_true"] != -1]
if len(labeled_preds) > 0:
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(
            labeled_preds["Y_true"], labeled_preds["incident_probability"]
        )
        print(f"\n  AUC on labeled patients: {auc:.3f}")
    except Exception as e:
        print(f"\n  AUC could not be computed: {e}")

print("\nDone. Saved latte_predictions.parquet")
print(f"  Output: {out / 'data' / 'latte_predictions.parquet'}")
