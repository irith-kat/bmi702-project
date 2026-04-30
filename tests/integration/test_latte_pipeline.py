"""
Integration test for Steps 7–9 of the LATTE pipeline using synthetic data.

Generates mock obs_log, gold labels, and silver pool that mimic the real
MIMIC pipeline output, then runs:

  Step 7: format_latte_input  → train.csv, test.csv, unlabeled.csv
  Step 8: build_cooccurrence_embeddings → embedding.csv
  Step 9: run_latte → predictions_df

Run with:
  uv run python tests/integration/test_latte_pipeline.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from latte.latte import format_latte_input, run_latte
from latte.embeddings import build_cooccurrence_embeddings

# ── Reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Configuration (matches the notebook) ──────────────────────────────────────
BASELINE_DATE = "2100-01-01"
MONTH_WINDOW = 3
N_CASES = 40  # labeled cases
N_CONTROLS = 40  # labeled controls
N_UNLABELED = 200  # mid-tier silver pool
N_CODES = 30  # feature codes (small for speed)
N_EPOCHS = 5  # keep test fast; real runs use 50
N_EPOCH_SILVER = 2

LATTE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "LATTE-main")

FEATURE_CODES = (
    [f"PheCode:{400 + i}" for i in range(10)]
    + [f"RXNORM:{1000 + i}" for i in range(10)]
    + [f"CUI:C{str(2000 + i).zfill(7)}" for i in range(10)]
)
KEY_CODES = ["PheCode:400", "PheCode:401"]


# ─────────────────────────────────────────────────────────────────────────────
# Mock data generators
# ─────────────────────────────────────────────────────────────────────────────


def make_obs_log(case_ids, control_ids, unlabeled_ids):
    """
    Build a synthetic long-format observation log.

    Cases have elevated rates of KEY_CODES at and after their incident T.
    Controls and unlabeled patients have background rates only.
    """
    base_dt = pd.Timestamp(BASELINE_DATE)
    records = []

    all_ids = list(case_ids) + list(control_ids) + list(unlabeled_ids)
    is_case = set(case_ids)

    for sid in all_ids:
        # Each patient gets 4–12 visits spread over 36 months
        n_visits = RNG.integers(4, 13)
        # Visit offsets in days from baseline
        visit_offsets = sorted(RNG.integers(0, 36 * 30, size=n_visits).tolist())

        # Cases: incident at a random visit (2nd onwards)
        if sid in is_case and n_visits >= 2:
            incident_visit_idx = RNG.integers(1, n_visits)
        else:
            incident_visit_idx = None

        for vi, offset in enumerate(visit_offsets):
            visit_dt = base_dt + pd.Timedelta(days=int(offset))
            is_after_incident = (
                incident_visit_idx is not None and vi >= incident_visit_idx
            )

            # Each visit: 3–8 random background codes
            n_codes = RNG.integers(3, 9)
            chosen = RNG.choice(FEATURE_CODES, size=n_codes, replace=False)
            for code in chosen:
                records.append(
                    {
                        "subject_id": sid,
                        "event_type": code.split(":")[0],
                        "event": code,
                        "value": 1,
                        "datetime": visit_dt,
                    }
                )

            # Cases after incident get extra key code hits (disease signal)
            if is_after_incident:
                for kc in KEY_CODES:
                    records.append(
                        {
                            "subject_id": sid,
                            "event_type": kc.split(":")[0],
                            "event": kc,
                            "value": 1,
                            "datetime": visit_dt,
                        }
                    )

    obs_log = pd.DataFrame(records)
    obs_log["datetime"] = pd.to_datetime(obs_log["datetime"])
    obs_log["subject_id"] = obs_log["subject_id"].astype(str)
    return obs_log


def make_gold_labels(case_ids, control_ids, obs_log):
    """
    Build latte_labels (subject_id, T, Y) matching the labeler output format.

    Uses the same T formula as labels_to_latte().
    """
    base_dt = pd.Timestamp(BASELINE_DATE)

    obs = obs_log.copy()
    obs["T"] = np.floor(
        (obs["datetime"] - base_dt).dt.days / (30.44 * MONTH_WINDOW)
    ).astype(int)

    records = []
    all_labeled = list(case_ids) + list(control_ids)
    case_set = set(case_ids)

    for sid in all_labeled:
        sid_str = str(sid)
        patient_obs = obs[obs["subject_id"] == sid_str]
        if patient_obs.empty:
            continue
        t_values = sorted(patient_obs["T"].unique())

        if sid in case_set:
            # Incident at roughly the midpoint
            incident_T = t_values[max(1, len(t_values) // 2)]
            for t in t_values:
                records.append(
                    {"subject_id": sid_str, "T": t, "Y": int(t >= incident_T)}
                )
        else:
            for t in t_values:
                records.append({"subject_id": sid_str, "T": t, "Y": 0})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────


def run_test():
    print("=" * 70)
    print("LATTE pipeline integration test")
    print("=" * 70)

    # Patient ID pools — must be numeric strings because LATTE does int(patient_id)
    case_ids = [str(10000 + i) for i in range(N_CASES)]
    control_ids = [str(20000 + i) for i in range(N_CONTROLS)]
    unlabeled_ids = [str(30000 + i) for i in range(N_UNLABELED)]

    # ── Step 1: generate mock data ─────────────────────────────────────────
    print("\n[mock] Building obs_log …")
    obs_log = make_obs_log(case_ids, control_ids, unlabeled_ids)
    print(
        f"  obs_log: {len(obs_log):,} rows, {obs_log['subject_id'].nunique()} patients"
    )
    assert set(["subject_id", "event_type", "event", "value", "datetime"]).issubset(
        obs_log.columns
    ), "obs_log missing required columns"

    print("\n[mock] Building gold labels …")
    gold_labels = make_gold_labels(case_ids, control_ids, obs_log)
    print(
        f"  gold_labels: {len(gold_labels)} rows, "
        f"{gold_labels['subject_id'].nunique()} patients, "
        f"case fraction: {gold_labels.groupby('subject_id')['Y'].max().mean():.2f}"
    )
    assert set(["subject_id", "T", "Y"]).issubset(gold_labels.columns)

    # ── Step 7: format_latte_input ─────────────────────────────────────────
    print("\n[Step 7] format_latte_input …")
    train_df, test_df, unlabeled_df = format_latte_input(
        obs_log=obs_log,
        gold_labels=gold_labels,
        feature_codes=FEATURE_CODES,
        baseline_date=BASELINE_DATE,
        month_window=MONTH_WINDOW,
        unlabeled_ids=unlabeled_ids,
        train_frac=0.8,
        seed=42,
    )

    print(f"  train_df:     {len(train_df)} rows, {train_df['ID'].nunique()} patients")
    print(f"  test_df:      {len(test_df)} rows, {test_df['ID'].nunique()} patients")
    print(
        f"  unlabeled_df: {len(unlabeled_df)} rows, {unlabeled_df['ID'].nunique()} patients"
    )

    # Validate CSV format
    expected_cols = ["ID", "Y", "T"] + FEATURE_CODES
    assert list(train_df.columns) == expected_cols, "train_df column mismatch"
    assert list(test_df.columns) == expected_cols
    assert list(unlabeled_df.columns) == expected_cols

    assert set(train_df["Y"].unique()).issubset({0, 1}), "train Y must be 0/1"
    assert set(test_df["Y"].unique()).issubset({0, 1}), "test Y must be 0/1"
    assert set(unlabeled_df["Y"].unique()) == {-1}, "unlabeled Y must be -1"

    # No overlap between train and test patients
    assert len(set(train_df["ID"]) & set(test_df["ID"])) == 0, "train/test overlap!"

    code_cols = FEATURE_CODES
    assert train_df[code_cols].isin([0, 1]).all().all(), "feature values must be binary"

    print("  ✓ format and constraints OK")

    # ── Step 8: build_cooccurrence_embeddings ──────────────────────────────
    print("\n[Step 8] build_cooccurrence_embeddings …")
    embedding_df = build_cooccurrence_embeddings(
        obs_log=obs_log,
        feature_codes=FEATURE_CODES,
        n_components=10,  # small for speed; real runs use 50
    )
    print(f"  embedding_df: {embedding_df.shape}  (components × codes)")
    assert embedding_df.shape[1] == len(FEATURE_CODES), "wrong number of code columns"
    assert list(embedding_df.columns) == FEATURE_CODES
    print("  ✓ shape and columns OK")

    # ── Step 9: run_latte ─────────────────────────────────────────────────
    print("\n[Step 9] run_latte …")
    with tempfile.TemporaryDirectory(prefix="latte_test_") as tmpdir:
        data_dir = tmpdir + os.sep
        save_dir = os.path.join(tmpdir, "results") + os.sep
        os.makedirs(save_dir, exist_ok=True)

        # Write CSVs
        train_df.to_csv(data_dir + "train.csv", index=False)
        test_df.to_csv(data_dir + "test.csv", index=False)
        unlabeled_df.to_csv(data_dir + "unlabeled.csv", index=False)
        embedding_df.to_csv(data_dir + "embedding.csv")  # index=True expected

        print(f"  Data written to {data_dir}")

        feature_col_start = 3
        feature_col_end = 3 + len(FEATURE_CODES)

        predictions_df = run_latte(
            latte_dir=os.path.abspath(LATTE_DIR),
            data_dir=data_dir,
            embedding_file=data_dir + "embedding.csv",
            key_codes=",".join(KEY_CODES),
            feature_col_start=feature_col_start,
            feature_col_end=feature_col_end,
            save_dir=save_dir,
            results_filename="results_test.csv",
            epochs=N_EPOCHS,
            epoch_silver=N_EPOCH_SILVER,
            embedding_dim=10,
            layers_incident="20",  # small network for speed
            weight_prevalence=0.2,
            weight_unlabel=0.2,
            weight_contrastive=0.1,
            weight_smooth=0.1,
            month_window=MONTH_WINDOW,
            max_visits=20,
        )

        print(
            f"\n  predictions_df: {len(predictions_df)} rows, "
            f"{predictions_df['subject_id'].nunique()} patients"
        )
        print(f"  columns: {list(predictions_df.columns)}")
        print(
            f"  incident_probability range: "
            f"[{predictions_df['incident_probability'].min():.3f}, "
            f"{predictions_df['incident_probability'].max():.3f}]"
        )
        print(predictions_df.head(8).to_string(index=False))

        assert set(
            ["subject_id", "visit_T", "incident_probability", "Y_true"]
        ).issubset(predictions_df.columns), "predictions_df missing columns"
        assert len(predictions_df) > 0, "predictions_df is empty"
        assert predictions_df["incident_probability"].between(0, 1).all(), (
            "probabilities out of [0,1]"
        )

        print("\n  ✓ predictions shape and range OK")

    print("\n" + "=" * 70)
    print("ALL STEPS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
