"""07 — 5-fold stratified cross-validation for LATTE incident HF prediction.

Study: HF_test_run_v1
Uses the same obs_log, gold_labels, and unlabeled_pool as script 06.
Replaces the single 80/20 split with 5-fold stratified CV to get a stable
AUC estimate over the full labeled set (120 patients).

Each fold:
  - Train: ~96 patients  (~80%)
  - Test:  ~24 patients  (~20%)
  - Unlabeled pool: same 10k patients for all folds

Results are written to:
  data/cv_results/fold_{k}/   — per-fold LATTE checkpoints + predictions
  data/cv_results/cv_summary.csv — per-fold AUC + best-epoch info

Run:
  uv run python output/HF_test_run_v1/scripts/07_latte_cv.py
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from latte.embeddings import build_cooccurrence_embeddings
from latte.labeler_utils import HF_DECOMP_DISEASE_CONFIG
from latte.latte import format_latte_input, run_latte
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

out = Path(__file__).resolve().parent.parent
LATTE_DIR = str(REPO_ROOT / "src" / "LATTE-main" / "LATTE-main")

# ── Configuration (validated in 10-run tuning experiment) ─────────────────────
BASELINE_DATE = "2100-01-01"
MONTH_WINDOW = 3
EMBEDDING_DIM = 50
EPOCHS = 35
EPOCH_SILVER = 8
LAYERS_INCIDENT = "80"  # single GRU layer — best for small label sets
N_FOLDS = 5
TRAIN_FRAC = 0.8  # ignored (CV uses explicit fold splits)
MAX_UNLABELED = 10_000
SEED = 42

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
gold_labels = pd.read_parquet(out / "data" / "gold_labels.parquet")
unlabeled_pool_df = pd.read_parquet(out / "data" / "unlabeled_pool.parquet")

unlabeled_ids = unlabeled_pool_df["subject_id"].astype(str).tolist()
if len(unlabeled_ids) > MAX_UNLABELED:
    import random

    random.seed(SEED)
    unlabeled_ids = random.sample(unlabeled_ids, MAX_UNLABELED)

print(
    f"  obs_log     : {len(obs_log):,} rows, {obs_log['subject_id'].nunique():,} patients"
)
print(
    f"  gold_labels : {len(gold_labels):,} rows, {gold_labels['subject_id'].nunique()} patients"
)
print(f"  unlabeled   : {len(unlabeled_ids):,} patients")

# ── 2. Load feature codes ──────────────────────────────────────────────────────
meta = json.loads((out / "data" / "once_features_meta.json").read_text())
feature_codes = meta.get("feature_codes", meta["codified_list"])
anchor = meta["anchor"]
print(f"\nFeature codes: {len(feature_codes)}  |  anchor: {anchor}")

raw_key_codes = HF_DECOMP_DISEASE_CONFIG.key_codes
key_codes_present = [c for c in raw_key_codes if c in feature_codes] or [anchor]
key_codes_str = ",".join(key_codes_present)
print(f"Key codes for LATTE: {key_codes_str}")

feature_col_start = 3
feature_col_end = 3 + len(feature_codes)

# ── 3. Build co-occurrence embeddings (shared across all folds) ────────────────
print("\nBuilding co-occurrence embeddings (shared across folds)...")
embedding_df = build_cooccurrence_embeddings(
    obs_log=obs_log,
    feature_codes=feature_codes,
    n_components=EMBEDDING_DIM,
)
print(f"  embedding_df: {embedding_df.shape}")

# ── 4. Set up CV patient splits ────────────────────────────────────────────────
patients = np.array(gold_labels["subject_id"].astype(str).unique().tolist())
patient_y = gold_labels.groupby("subject_id")["Y"].max().reindex(patients).to_numpy()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(patients, patient_y))

print(f"\n{N_FOLDS}-fold stratified CV on {len(patients)} labeled patients")
print(f"  Case rate: {patient_y.mean():.1%}")
for k, (tr, te) in enumerate(splits):
    n_pos_train = patient_y[tr].sum()
    n_pos_test = patient_y[te].sum()
    print(
        f"  Fold {k + 1}: train={len(tr)} ({n_pos_train} cases), "
        f"test={len(te)} ({n_pos_test} cases)"
    )

# ── 5. Per-fold LATTE training ─────────────────────────────────────────────────
cv_results = []

for k, (train_idx, test_idx) in enumerate(splits):
    fold_num = k + 1
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_num}/{N_FOLDS}")
    print(f"{'=' * 60}")

    train_pats = patients[train_idx].tolist()
    test_pats = patients[test_idx].tolist()

    # Fold-specific directories
    fold_data_dir = (
        str(out / "data" / "cv_results" / f"fold_{fold_num}" / "latte_input") + os.sep
    )
    fold_results_dir = (
        str(out / "data" / "cv_results" / f"fold_{fold_num}" / "latte_results") + os.sep
    )
    os.makedirs(fold_data_dir, exist_ok=True)
    os.makedirs(fold_results_dir, exist_ok=True)

    # Format LATTE input using explicit fold split
    train_df, test_df, unlabeled_df = format_latte_input(
        obs_log=obs_log,
        gold_labels=gold_labels,
        feature_codes=feature_codes,
        baseline_date=BASELINE_DATE,
        month_window=MONTH_WINDOW,
        unlabeled_ids=unlabeled_ids,
        train_patients=train_pats,
        test_patients=test_pats,
    )
    print(
        f"  train_df    : {len(train_df):,} rows, {train_df['ID'].nunique()} patients"
    )
    print(f"  test_df     : {len(test_df):,} rows, {test_df['ID'].nunique()} patients")
    print(f"  unlabeled_df: {len(unlabeled_df):,} rows")
    print(f"  Y dist (train): {train_df['Y'].value_counts().to_dict()}")

    # Write fold input files (embedding is shared)
    train_df.to_csv(fold_data_dir + "train.csv", index=False)
    test_df.to_csv(fold_data_dir + "test.csv", index=False)
    unlabeled_df.to_csv(fold_data_dir + "unlabeled.csv", index=False)
    embedding_df.to_csv(fold_data_dir + "embedding.csv")

    results_filename = f"results_fold{fold_num}.csv"

    predictions_df = run_latte(
        latte_dir=LATTE_DIR,
        data_dir=fold_data_dir,
        embedding_file=fold_data_dir + "embedding.csv",
        key_codes=key_codes_str,
        feature_col_start=feature_col_start,
        feature_col_end=feature_col_end,
        save_dir=fold_results_dir,
        results_filename=results_filename,
        epochs=EPOCHS,
        epoch_silver=EPOCH_SILVER,
        embedding_dim=EMBEDDING_DIM,
        layers_incident=LAYERS_INCIDENT,
        weight_prevalence=0.2,
        weight_unlabel=0.015,
        weight_contrastive=0.1,
        weight_smooth=0.1,
        weight_additional=0.1,
        flag_train_augment=1,
        month_window=MONTH_WINDOW,
        max_visits=25,
    )

    # Compute fold AUC
    labeled_preds = predictions_df[predictions_df["Y_true"] != -1]
    fold_auc = float("nan")
    if len(labeled_preds) > 0 and labeled_preds["Y_true"].nunique() == 2:
        from sklearn.metrics import roc_auc_score

        fold_auc = roc_auc_score(
            labeled_preds["Y_true"], labeled_preds["incident_probability"]
        )

    print(f"\n  Fold {fold_num} AUC: {fold_auc:.4f}")
    print(
        f"  Probability range: [{predictions_df['incident_probability'].min():.3f}, "
        f"{predictions_df['incident_probability'].max():.3f}]"
    )

    predictions_df.to_parquet(
        out / "data" / "cv_results" / f"fold_{fold_num}" / "predictions.parquet",
        index=False,
    )

    cv_results.append(
        {
            "fold": fold_num,
            "n_train": len(train_pats),
            "n_test": len(test_pats),
            "n_cases_train": int(patient_y[train_idx].sum()),
            "n_cases_test": int(patient_y[test_idx].sum()),
            "auc": fold_auc,
        }
    )

# ── 6. Aggregate and save ──────────────────────────────────────────────────────
cv_df = pd.DataFrame(cv_results)
aucs = cv_df["auc"].dropna()

print(f"\n{'=' * 60}")
print("CROSS-VALIDATION RESULTS")
print(f"{'=' * 60}")
print(cv_df.to_string(index=False))
print(f"\nMean AUC : {aucs.mean():.4f}")
print(f"Std  AUC : {aucs.std():.4f}")
print(f"Min  AUC : {aucs.min():.4f}")
print(f"Max  AUC : {aucs.max():.4f}")

cv_summary_path = out / "data" / "cv_results" / "cv_summary.csv"
os.makedirs(cv_summary_path.parent, exist_ok=True)
cv_df.to_csv(cv_summary_path, index=False)
print(f"\nSaved: {cv_summary_path}")
print(f"       mean={aucs.mean():.4f} ± {aucs.std():.4f}")
