"""06 — Run LATTE to predict incident HF onset timing.

Study: hf-incident-v1
Period: 2-month windows (MONTH_WINDOW=2)
Mode  : incident (first occurrence)

Pipeline:
  1. Load obs_log + gold_labels + unlabeled_pool
  2. format_latte_input → train/test/unlabeled CSVs
  3. build_cooccurrence_embeddings → embedding.csv
  4. run_latte → per-patient per-period incident probability
  5. Save latte_predictions.parquet

Run:
    cd output/hf-incident-v1
    uv run python scripts/06_latte_phenotyping.py
"""

import json
import logging
import os
import random
from pathlib import Path

import pandas as pd
from latte.embeddings import build_cooccurrence_embeddings
from latte.labeler_utils import HF_DISEASE_CONFIG
from latte.latte import format_latte_input, run_latte
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[4]
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
out = Path(__file__).resolve().parent.parent
LATTE_DIR = str(REPO_ROOT / "src" / "LATTE-main")

BASELINE_DATE = "2100-01-01"
MONTH_WINDOW = 2
EMBEDDING_DIM = 50
EPOCHS = 35
EPOCH_SILVER = 8
LAYERS_INCIDENT = "80"  # single GRU layer — best for small label sets
TRAIN_FRAC = 0.8
MAX_UNLABELED = 10_000

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
gold_labels = pd.read_parquet(out / "data" / "gold_labels.parquet")
unlabeled_pool_df = pd.read_parquet(out / "data" / "unlabeled_pool.parquet")

unlabeled_ids = unlabeled_pool_df["subject_id"].astype(str).tolist()
if len(unlabeled_ids) > MAX_UNLABELED:
    random.seed(42)
    unlabeled_ids = random.sample(unlabeled_ids, MAX_UNLABELED)
    print(f"  Unlabeled capped at {MAX_UNLABELED}")

print(
    f"  obs_log     : {len(obs_log):,} rows, {obs_log['subject_id'].nunique():,} patients"
)
print(
    f"  gold_labels : {len(gold_labels):,} rows, {gold_labels['subject_id'].nunique()} patients"
)
print(f"  unlabeled   : {len(unlabeled_ids):,} patients")

# ── 2. Feature codes ──────────────────────────────────────────────────────────
meta = json.loads((out / "data" / "once_features_meta.json").read_text())
feature_codes = meta.get("feature_codes", meta["codified_list"])
anchor = meta["anchor"]
print(f"\nFeature codes: {len(feature_codes)}  |  anchor: {anchor}")

# Key codes for LATTE silver label (HF PheCode presence)
raw_key_codes = HF_DISEASE_CONFIG.key_codes
key_codes_present = [c for c in raw_key_codes if c in feature_codes]
if not key_codes_present:
    key_codes_present = [anchor]
key_codes_str = ",".join(key_codes_present)
print(f"  Key codes: {key_codes_str}")

feature_col_start = 3
feature_col_end = 3 + len(feature_codes)

# ── 3. Format LATTE input ─────────────────────────────────────────────────────
print("\nFormatting LATTE input...")
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
print(f"  train_df     : {len(train_df):,} rows, {train_df['ID'].nunique()} patients")
print(f"  test_df      : {len(test_df):,} rows, {test_df['ID'].nunique()} patients")
print(
    f"  unlabeled_df : {len(unlabeled_df):,} rows, {unlabeled_df['ID'].nunique()} patients"
)
print(f"  Y dist (train): {train_df['Y'].value_counts().to_dict()}")

# ── 4. Co-occurrence embeddings ───────────────────────────────────────────────
print("\nBuilding co-occurrence embeddings...")
embedding_df = build_cooccurrence_embeddings(
    obs_log=obs_log,
    feature_codes=feature_codes,
    n_components=EMBEDDING_DIM,
)
print(f"  embedding_df: {embedding_df.shape}")

# ── 5. Write LATTE input files ────────────────────────────────────────────────
latte_data_dir = str(out / "data" / "latte_input") + os.sep
latte_results_dir = str(out / "data" / "latte_results") + os.sep
os.makedirs(latte_data_dir, exist_ok=True)
os.makedirs(latte_results_dir, exist_ok=True)

train_df.to_csv(latte_data_dir + "train.csv", index=False)
test_df.to_csv(latte_data_dir + "test.csv", index=False)
unlabeled_df.to_csv(latte_data_dir + "unlabeled.csv", index=False)
embedding_df.to_csv(latte_data_dir + "embedding.csv")
print(f"  LATTE files written to {latte_data_dir}")

# ── 6. Run LATTE ──────────────────────────────────────────────────────────────
print("\nRunning LATTE (2-month incident HF)...")
print(
    f"  epochs={EPOCHS}, epoch_silver={EPOCH_SILVER}, embedding_dim={EMBEDDING_DIM}, layers={LAYERS_INCIDENT}"
)

predictions_df = run_latte(
    latte_dir=LATTE_DIR,
    data_dir=latte_data_dir,
    embedding_file=latte_data_dir + "embedding.csv",
    key_codes=key_codes_str,
    feature_col_start=feature_col_start,
    feature_col_end=feature_col_end,
    save_dir=latte_results_dir,
    results_filename="results_hf_incident.csv",
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
    max_visits=15,
)
print(
    f"\n  Predictions: {len(predictions_df):,} rows, {predictions_df['subject_id'].nunique()} patients"
)
print(
    f"  Probability range: [{predictions_df['incident_probability'].min():.3f}, "
    f"{predictions_df['incident_probability'].max():.3f}]"
)

# AUC on labeled hold-out
labeled_preds = predictions_df[predictions_df["Y_true"] != -1]
if len(labeled_preds) > 0 and labeled_preds["Y_true"].nunique() == 2:
    auc = roc_auc_score(labeled_preds["Y_true"], labeled_preds["incident_probability"])
    print(f"\n  Hold-out AUC (labeled patients): {auc:.3f}")
else:
    print(
        f"\n  Could not compute AUC (Y_true unique values: {labeled_preds['Y_true'].unique()})"
    )

predictions_df.to_parquet(out / "data" / "latte_predictions.parquet", index=False)
print("\nSaved latte_predictions.parquet")
