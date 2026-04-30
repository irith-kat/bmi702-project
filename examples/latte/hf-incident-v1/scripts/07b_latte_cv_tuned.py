"""07b — LATTE CV tuning run: sharper weight_smooth, more epochs.

Study: hf-incident-v1
Tuning hypothesis: reducing weight_smooth (0.1 → 0.04) allows sharper probability
transitions, better localising the incident period. Also increasing epochs 35 → 45.

Results saved to data/cv_results_tuned/ for comparison with baseline 07_latte_cv.py.

Run:
    cd output/hf-incident-v1
    uv run python scripts/07b_latte_cv_tuned.py
"""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from latte.embeddings import build_cooccurrence_embeddings
from latte.labeler_utils import HF_DISEASE_CONFIG
from latte.latte import format_latte_input, run_latte
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[4]
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
out = Path(__file__).resolve().parent.parent
LATTE_DIR = str(REPO_ROOT / "src" / "LATTE-main")

BASELINE_DATE = "2100-01-01"
MONTH_WINDOW = 2
EMBEDDING_DIM = 50
EPOCHS = 45  # ↑ from 35 — allows more training for short sequences
EPOCH_SILVER = 8
LAYERS_INCIDENT = "80"
N_FOLDS = 5
MAX_UNLABELED = 10_000
SEED = 42

WEIGHT_SMOOTH = 0.04  # ↓ from 0.1 — sharper transitions to improve onset localisation
WEIGHT_PREVALENCE = 0.2
WEIGHT_UNLABEL = 0.015
WEIGHT_CONTRASTIVE = 0.12  # ↑ slightly — stronger case/control separation

print(
    f"Tuning run: epochs={EPOCHS}, weight_smooth={WEIGHT_SMOOTH}, weight_contrastive={WEIGHT_CONTRASTIVE}"
)

# ── Load data ─────────────────────────────────────────────────────────────────
obs_log = pd.read_parquet(out / "data" / "obs_log.parquet")
gold_labels = pd.read_parquet(out / "data" / "gold_labels.parquet")
unlabeled_pool_df = pd.read_parquet(out / "data" / "unlabeled_pool.parquet")

unlabeled_ids = unlabeled_pool_df["subject_id"].astype(str).tolist()
if len(unlabeled_ids) > MAX_UNLABELED:
    random.seed(SEED)
    unlabeled_ids = random.sample(unlabeled_ids, MAX_UNLABELED)

meta = json.loads((out / "data" / "once_features_meta.json").read_text())
feature_codes = meta.get("feature_codes", meta["codified_list"])
anchor = meta["anchor"]
raw_key_codes = HF_DISEASE_CONFIG.key_codes
key_codes_present = [c for c in raw_key_codes if c in feature_codes] or [anchor]
key_codes_str = ",".join(key_codes_present)
feature_col_start = 3
feature_col_end = 3 + len(feature_codes)

embedding_df = build_cooccurrence_embeddings(
    obs_log=obs_log, feature_codes=feature_codes, n_components=EMBEDDING_DIM
)
print(f"Embeddings: {embedding_df.shape}")

patients = np.array(gold_labels["subject_id"].astype(str).unique().tolist())
patient_y = gold_labels.groupby("subject_id")["Y"].max().reindex(patients).to_numpy()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(patients, patient_y))

cv_results = []
for k, (train_idx, test_idx) in enumerate(splits):
    fold_num = k + 1
    print(f"\n{'=' * 60}\nFOLD {fold_num}/{N_FOLDS}\n{'=' * 60}")

    train_pats = patients[train_idx].tolist()
    test_pats = patients[test_idx].tolist()

    fold_data_dir = (
        str(out / "data" / "cv_results_tuned" / f"fold_{fold_num}" / "latte_input")
        + os.sep
    )
    fold_results_dir = (
        str(out / "data" / "cv_results_tuned" / f"fold_{fold_num}" / "latte_results")
        + os.sep
    )
    os.makedirs(fold_data_dir, exist_ok=True)
    os.makedirs(fold_results_dir, exist_ok=True)

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

    train_df.to_csv(fold_data_dir + "train.csv", index=False)
    test_df.to_csv(fold_data_dir + "test.csv", index=False)
    unlabeled_df.to_csv(fold_data_dir + "unlabeled.csv", index=False)
    embedding_df.to_csv(fold_data_dir + "embedding.csv")

    preds = run_latte(
        latte_dir=LATTE_DIR,
        data_dir=fold_data_dir,
        embedding_file=fold_data_dir + "embedding.csv",
        key_codes=key_codes_str,
        feature_col_start=feature_col_start,
        feature_col_end=feature_col_end,
        save_dir=fold_results_dir,
        results_filename=f"results_tuned_fold{fold_num}.csv",
        epochs=EPOCHS,
        epoch_silver=EPOCH_SILVER,
        embedding_dim=EMBEDDING_DIM,
        layers_incident=LAYERS_INCIDENT,
        weight_prevalence=WEIGHT_PREVALENCE,
        weight_unlabel=WEIGHT_UNLABEL,
        weight_contrastive=WEIGHT_CONTRASTIVE,
        weight_smooth=WEIGHT_SMOOTH,
        weight_additional=0.1,
        flag_train_augment=1,
        month_window=MONTH_WINDOW,
        max_visits=15,
    )

    labeled_preds = preds[preds["Y_true"] != -1]
    fold_auc = float("nan")
    if len(labeled_preds) > 0 and labeled_preds["Y_true"].nunique() == 2:
        fold_auc = roc_auc_score(
            labeled_preds["Y_true"], labeled_preds["incident_probability"]
        )
    print(f"\n  Fold {fold_num} AUC: {fold_auc:.4f}")

    preds.to_parquet(
        out / "data" / "cv_results_tuned" / f"fold_{fold_num}" / "predictions.parquet",
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

cv_df = pd.DataFrame(cv_results)
aucs = cv_df["auc"].dropna()
print(f"\n{'=' * 60}\nTUNED CV RESULTS\n{'=' * 60}")
print(cv_df.to_string(index=False))
print(
    f"\nMean AUC: {aucs.mean():.4f} ± {aucs.std():.4f}  (baseline was 0.7759 ± 0.0854)"
)
print(f"Min/Max: {aucs.min():.4f} / {aucs.max():.4f}")

cv_path = out / "data" / "cv_results_tuned" / "cv_summary_tuned.csv"
os.makedirs(cv_path.parent, exist_ok=True)
cv_df.to_csv(cv_path, index=False)
print(f"\nSaved: {cv_path}")
