"""
latte.py
--------
LATTE pipeline utilities: input formatting, training wrapper, and evaluation.

Three independently callable functions:

  format_latte_input()   — convert obs_log + gold labels → LATTE CSV triplets
  run_latte()            — subprocess wrapper around a_train_final.py
  compute_abcgain()      — ABCgain evaluation metric (area between cumulative
                           incidence curves vs. a rule-based baseline)
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 7: format obs_log → LATTE CSV triplets
# ---------------------------------------------------------------------------


def format_latte_input(
    obs_log: pd.DataFrame,
    gold_labels: pd.DataFrame,
    feature_codes: list[str],
    baseline_date: str,
    month_window: int = 3,
    unlabeled_ids: list[str] | None = None,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert a long-format observation log and gold labels into the three CSV
    files that LATTE's ``get_data_from_csv`` expects.

    Parameters
    ----------
    obs_log : pd.DataFrame
        Long-format observation log with columns:
        ``subject_id, event_type, event, value, datetime``.
        One row per code occurrence per patient visit.
    gold_labels : pd.DataFrame
        Per-visit gold labels with columns ``subject_id, T, Y``.
        Produced by ``labels_to_latte()`` / ``medgemma_labels_to_latte()``.
        Y must be 0 (control) or 1 (case) for labeled patients.
    feature_codes : list[str]
        ONCE-selected code strings (e.g. ``["PheCode:428", "RXNORM:4603"]``).
        These become the feature columns in the output CSVs.
        The key codes used for LATTE's silver label initialisation (e.g.
        ``HF_DISEASE_CONFIG.key_codes``) must be present in this list.
    baseline_date : str
        Study-wide anchor date for computing time-window index T.
        Must match the value used when producing ``gold_labels``.
        Example: ``"2100-01-01"``.
    month_window : int
        Width of each time window in months (LATTE paper default: 3).
    unlabeled_ids : list[str] | None
        Patient IDs whose rows should go into ``unlabeled.csv`` with Y = -1.
        Typically the ``unlabeled_pool`` returned by ``silver_prefilter()``.
        If ``None``, an empty unlabeled DataFrame is returned.
    train_frac : float
        Fraction of gold-labeled patients to use for training (default 0.8).
        The complement goes to the test set.
        Stratified by Y label to preserve case/control balance.
    seed : int
        Random seed for the stratified train/test split.

    Returns
    -------
    train_df, test_df, unlabeled_df : pd.DataFrame
        Each DataFrame has columns ``ID, Y, T, <code_1>, …, <code_N>``
        matching LATTE's expected CSV format.
        - ``ID``  : patient identifier (str)
        - ``Y``   : 0/1 for labeled patients, -1 for unlabeled
        - ``T``   : time-window index (int)
        - codes   : binary indicators (1 = code appeared in that window)

        ``colums_min = 3`` and ``colums_max = 3 + len(feature_codes)`` are
        the correct ``--colums_min`` / ``--colums_max`` values to pass to
        ``run_latte()``.

    Notes
    -----
    T is computed identically to ``labels_to_latte`` in ``labeler_utils.py``::

        T = floor((event_datetime - baseline_date).days / (30.44 * month_window))
    """
    base_dt = pd.to_datetime(baseline_date)

    # ------------------------------------------------------------------
    # 1. Compute T for every row of obs_log
    # ------------------------------------------------------------------
    obs = obs_log.copy()
    obs["_dt"] = pd.to_datetime(obs["datetime"], errors="coerce")
    obs["T"] = np.floor((obs["_dt"] - base_dt).dt.days / (30.44 * month_window)).astype(
        "Int64"
    )
    obs["subject_id"] = obs["subject_id"].astype(str)

    # ------------------------------------------------------------------
    # 2. Filter to feature_codes only, then pivot to wide binary format
    # ------------------------------------------------------------------
    obs_feat = obs[obs["event"].isin(feature_codes)].copy()

    if obs_feat.empty:
        raise ValueError(
            "No obs_log rows matched any feature_codes. "
            "Verify that the event column values match the code format."
        )

    # Build (subject_id, T, code) → binary indicator
    wide = (
        obs_feat.groupby(["subject_id", "T", "event"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=feature_codes, fill_value=0)
        .clip(upper=1)  # binary indicator: appeared or not
        .astype(int)
        .reset_index()
    )
    # wide columns: subject_id, T, <code_1>, ..., <code_N>

    # ------------------------------------------------------------------
    # 3. Build labeled portion (train + test)
    # ------------------------------------------------------------------
    gold = gold_labels[gold_labels["Y"].isin([0, 1])].copy()
    gold["subject_id"] = gold["subject_id"].astype(str)
    gold["T"] = gold["T"].astype("Int64")

    labeled_wide = wide[wide["subject_id"].isin(gold["subject_id"].unique())].copy()
    labeled_wide = labeled_wide.merge(
        gold[["subject_id", "T", "Y"]], on=["subject_id", "T"], how="inner"
    )

    if labeled_wide.empty:
        raise ValueError(
            "No (subject_id, T) pairs were shared between obs_log and "
            "gold_labels. Check that baseline_date and month_window match."
        )

    # Stratified train/test split on unique labeled patients
    labeled_patients = np.array(gold["subject_id"].unique().tolist())
    patient_labels = gold.groupby("subject_id")["Y"].max().reindex(labeled_patients)

    train_patients, test_patients = train_test_split(
        labeled_patients,
        train_size=train_frac,
        stratify=patient_labels.to_numpy(),
        random_state=seed,
    )

    train_df = labeled_wide[labeled_wide["subject_id"].isin(train_patients)].copy()
    test_df = labeled_wide[labeled_wide["subject_id"].isin(test_patients)].copy()

    # ------------------------------------------------------------------
    # 4. Build unlabeled portion
    # ------------------------------------------------------------------
    if unlabeled_ids:
        unlabeled_ids_str = [str(s) for s in unlabeled_ids]
        unlabeled_wide = wide[wide["subject_id"].isin(unlabeled_ids_str)].copy()
        unlabeled_wide["Y"] = -1
    else:
        unlabeled_wide = pd.DataFrame(columns=wide.columns.tolist() + ["Y"])

    # ------------------------------------------------------------------
    # 5. Reorder columns to LATTE format: ID, Y, T, <codes...>
    # ------------------------------------------------------------------
    def _to_latte_format(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.rename(columns={"subject_id": "ID"})
        df["T"] = df["T"].astype(int)
        cols = ["ID", "Y", "T"] + feature_codes
        # Add any missing code columns with 0
        for c in feature_codes:
            if c not in df.columns:
                df[c] = 0
        return df[cols].reset_index(drop=True)

    train_out = _to_latte_format(train_df)
    test_out = _to_latte_format(test_df)
    unlabeled_out = _to_latte_format(unlabeled_wide)

    logger.info(
        "LATTE input formatted — train: %d rows (%d patients), "
        "test: %d rows (%d patients), unlabeled: %d rows (%d patients), "
        "feature codes: %d.",
        len(train_out),
        train_out["ID"].nunique(),
        len(test_out),
        test_out["ID"].nunique(),
        len(unlabeled_out),
        unlabeled_out["ID"].nunique(),
        len(feature_codes),
    )

    return train_out, test_out, unlabeled_out


# ---------------------------------------------------------------------------
# Step 9: run LATTE training via subprocess
# ---------------------------------------------------------------------------


def run_latte(
    latte_dir: str,
    data_dir: str,
    embedding_file: str,
    key_codes: str,
    feature_col_start: int,
    feature_col_end: int,
    save_dir: str,
    results_filename: str = "results_latte.csv",
    epochs: int = 50,
    epoch_silver: int = 8,
    embedding_dim: int = 50,
    layers_incident: str = "80,80",
    weight_prevalence: float = 0.2,
    weight_unlabel: float = 0.2,
    weight_contrastive: float = 0.1,
    weight_smooth: float = 0.1,
    month_window: int = 3,
    max_visits: int = 115,
    python_executable: str | None = None,
) -> pd.DataFrame:
    """
    Run LATTE training by calling ``a_train_final.py`` in a subprocess and
    return the per-patient, per-visit incident predictions.

    The function uses ``flag_cross_dataset=1`` so that the pre-split
    ``train.csv``, ``test.csv``, and ``unlabeled.csv`` in ``data_dir`` are
    used as-is (no internal re-splitting).

    Parameters
    ----------
    latte_dir : str
        Path to the ``src/LATTE-main/LATTE-main/`` directory.
        ``a_train_final.py`` must be present there and all its relative
        imports (``a_utilize_semi``, ``a_semi_model_final``) must resolve.
    data_dir : str
        Directory containing ``train.csv``, ``test.csv``, and
        ``unlabeled.csv``.  The embedding file must also live here (see
        ``embedding_file``).
    embedding_file : str
        Full path to the embedding CSV produced by
        ``build_cooccurrence_embeddings()``.  Must reside inside ``data_dir``;
        only the basename is passed to LATTE (which prepends ``data_dir``
        internally).
    key_codes : str
        Comma-separated key code names used by LATTE's silver label
        initialisation.  Example: ``"PheCode:428,PheCode:428.1"``.
        Each code must be a column in the CSV files.
    feature_col_start : int
        Zero-based column index of the first feature column in the CSVs.
        For CSVs formatted by ``format_latte_input``, this is always ``3``
        (after ``ID``, ``Y``, ``T``).
    feature_col_end : int
        Zero-based column index one past the last feature column.
        For ``N`` feature codes: ``feature_col_end = feature_col_start + N``.
    save_dir : str
        Directory where LATTE writes its prediction CSVs and model files.
    results_filename : str
        Base filename for prediction output (e.g. ``"results_latte.csv"``).
        LATTE prefixes this with epoch information.
    epochs : int
        Total training epochs (LATTE paper uses 50).
    epoch_silver : int
        Pre-training epochs using silver labels (LATTE paper uses 8–10).
    embedding_dim : int
        Dimension of code embeddings; must match the number of rows in the
        embedding CSV (i.e., ``n_components`` from
        ``build_cooccurrence_embeddings``).
    layers_incident : str
        GRU layer sizes as a comma-separated string.
        ``"80"`` → one GRU layer of 80 units;
        ``"80,80"`` → two stacked GRU layers.
    weight_prevalence : float
        Weight on the EVER/NEVER prevalence loss.
    weight_unlabel : float
        Weight on the unlabeled/weak-supervision loss.
    weight_contrastive : float
        Weight on the contrastive representation learning loss.
    weight_smooth : float
        Weight on the temporal smoothness loss.
    month_window : int
        Time-window width in months; must match the value used in
        ``format_latte_input``.
    max_visits : int
        Maximum visit sequence length (longer sequences are truncated).
    python_executable : str | None
        Path to the Python interpreter to use.  Defaults to
        ``sys.executable`` (the current interpreter).  Override this if
        TensorFlow 2.3 is installed in a separate virtualenv.

    Returns
    -------
    pd.DataFrame
        One row per (patient, time-window) with columns:
        ``subject_id``, ``visit_T``, ``incident_probability``, ``Y_true``.

    Raises
    ------
    FileNotFoundError
        If no prediction CSV is found in ``save_dir`` after training.
    subprocess.CalledProcessError
        If ``a_train_final.py`` exits with a non-zero status.

    Notes
    -----
    TensorFlow 2.3 is required by the LATTE source code.
    If your current environment does not have TF installed, pass the path
    to a Python executable that does via ``python_executable``.
    """
    # Normalise paths
    data_dir = data_dir.rstrip(os.sep) + os.sep
    save_dir = save_dir.rstrip(os.sep) + os.sep
    os.makedirs(save_dir, exist_ok=True)

    # embedding_file must be inside data_dir; pass only the basename to LATTE
    # because get_data_from_csv loads it as mdir + embedding_filename.
    emb_basename = os.path.basename(embedding_file)
    if os.path.abspath(os.path.dirname(embedding_file)) != os.path.abspath(
        data_dir.rstrip(os.sep)
    ):
        raise ValueError(
            f"embedding_file must reside in data_dir.\n"
            f"  embedding_file: {embedding_file}\n"
            f"  data_dir:       {data_dir}"
        )

    exe = python_executable or sys.executable

    cmd = [
        exe,
        "a_train_final.py",
        "--train_directory",
        data_dir,
        "--train_filename",
        "train.csv",
        "--test_directory",
        data_dir,
        "--test_filename",
        "test.csv",
        "--unlabel_filename",
        "unlabeled.csv",
        "--embedding_filename",
        emb_basename,
        "--key_code",
        key_codes,
        "--flag_cross_dataset",
        "1",
        "--save_directory",
        save_dir,
        "--results_filename",
        results_filename,
        "--colums_min",
        str(feature_col_start),
        "--colums_max",
        str(feature_col_end),
        "--epochs",
        str(epochs),
        "--epoch_silver",
        str(epoch_silver),
        "--embedding_dim",
        str(embedding_dim),
        "--layers_incident",
        layers_incident,
        "--weight_prevalence",
        str(weight_prevalence),
        "--weight_unlabel",
        str(weight_unlabel),
        "--weight_constrastive",
        str(weight_contrastive),
        "--weight_smooth",
        str(weight_smooth),
        "--month_window",
        str(month_window),
        "--max_visits",
        str(max_visits),
    ]

    logger.info("Launching LATTE: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=latte_dir, check=True)

    # ------------------------------------------------------------------
    # Parse output: LATTE saves "Incident_epoch{N}_1__results_latte.csv"
    # at the final epoch (epoch_num = epochs - 1).
    # ------------------------------------------------------------------
    pattern = os.path.join(save_dir, "Incident_epoch*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No LATTE prediction files found matching {pattern!r}. "
            "Check save_dir and results_filename."
        )

    # Sort by epoch number embedded in the filename
    def _epoch_num(path: str) -> int:
        basename = os.path.basename(path)
        try:
            return int(basename.split("epoch")[1].split("_")[0])
        except (IndexError, ValueError):
            return -1

    output_path = max(candidates, key=_epoch_num)
    logger.info("Reading LATTE predictions from %s", output_path)

    pred_df = pd.read_csv(output_path, index_col=0)
    pred_df = pred_df.rename(
        columns={
            "Patient_num": "subject_id",
            "Date": "visit_T",
            "Y_pred": "incident_probability",
            "Y_label:": "Y_true",
        }
    )
    pred_df["subject_id"] = pred_df["subject_id"].astype(str)

    return pred_df[
        ["subject_id", "visit_T", "incident_probability", "Y_true"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 10 (evaluation): ABCgain metric
# ---------------------------------------------------------------------------


def compute_abcgain(
    latte_predictions: pd.DataFrame,
    baseline_labels: pd.DataFrame,
    baseline_date: str,
    month_window: int = 3,
    threshold: float = 0.5,
    time_col: str = "visit_T",
    pred_col: str = "incident_probability",
    subject_col: str = "subject_id",
) -> float:
    """
    Compute ABCgain: the fractional area between LATTE's cumulative incidence
    curve and a rule-based baseline curve.

    A value of 0.4 means LATTE detects 40% more of the true incident cases
    earlier than the rule-based approach across all time windows.

    Parameters
    ----------
    latte_predictions : pd.DataFrame
        Must have columns ``[subject_id, visit_T, incident_probability, Y_true]``
        — the output of ``run_latte()``.
    baseline_labels : pd.DataFrame
        Rule-based baseline with columns
        ``[subject_id, rule_label, rule_incident_admittime]``
        — the output of ``validate_against_principal_dx()``.
        ``rule_incident_admittime`` is used to place baseline detections on
        the same T axis as ``latte_predictions``.
    baseline_date : str
        The same study-wide anchor date used in ``format_latte_input`` and
        ``labels_to_latte``.  Needed to convert ``rule_incident_admittime``
        to a T index.
    month_window : int
        Time-window width in months.  Must match the training configuration.
    threshold : float
        Classification threshold applied to ``incident_probability``
        to determine whether LATTE has "detected" a case at a given window.
    time_col : str
        Column name for time window in ``latte_predictions``.
    pred_col : str
        Column name for predicted probability in ``latte_predictions``.
    subject_col : str
        Column name for patient IDs (must be consistent across both DataFrames).

    Returns
    -------
    float
        ABCgain in [0, 1].  Positive values indicate LATTE detects cases
        earlier than the baseline; negative values indicate the baseline is
        better.
    """
    base_dt = pd.to_datetime(baseline_date)

    # ------------------------------------------------------------------
    # Identify true cases and get the sorted unique T axis
    # ------------------------------------------------------------------
    cases_pred = latte_predictions[latte_predictions["Y_true"] == 1].copy()
    all_T = np.sort(latte_predictions[time_col].unique())

    if cases_pred.empty:
        logger.warning("No true cases found in latte_predictions; ABCgain = 0.")
        return 0.0

    # ------------------------------------------------------------------
    # LATTE cumulative detection: first T where prob >= threshold
    # ------------------------------------------------------------------
    above_thresh = cases_pred[cases_pred[pred_col] >= threshold]
    latte_first_T = above_thresh.groupby(subject_col)[time_col].min().rename("latte_T")

    # Patients never detected by LATTE get T = +inf
    all_case_ids = cases_pred[subject_col].unique()
    latte_first_T = latte_first_T.reindex(all_case_ids, fill_value=np.inf)

    # ------------------------------------------------------------------
    # Baseline cumulative detection: convert rule_incident_admittime → T
    # ------------------------------------------------------------------
    bl = baseline_labels.copy()
    bl[subject_col] = bl[subject_col].astype(str)

    if "rule_incident_admittime" in bl.columns:
        bl["_admittime"] = pd.to_datetime(
            bl["rule_incident_admittime"], errors="coerce"
        )
        bl["baseline_T"] = np.floor(
            (bl["_admittime"] - base_dt).dt.days / (30.44 * month_window)
        )
    else:
        bl["baseline_T"] = np.inf

    # Patients with rule_label=0 are never detected by rule
    bl.loc[bl["rule_label"] == 0, "baseline_T"] = np.inf

    baseline_first_T = bl.set_index(subject_col)["baseline_T"].reindex(
        all_case_ids, fill_value=np.inf
    )

    # ------------------------------------------------------------------
    # Build cumulative incidence curves over sorted T
    # ------------------------------------------------------------------
    n_cases = len(all_case_ids)

    def _cumulative_detected(
        first_T_series: pd.Series, t_axis: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [(first_T_series <= t).sum() / n_cases for t in t_axis], dtype=float
        )

    latte_curve = _cumulative_detected(latte_first_T, all_T)
    baseline_curve = _cumulative_detected(baseline_first_T, all_T)

    # ABCgain = area between curves / max possible area
    # Max possible: perfect early detection (all at T=all_T[0]) vs. baseline
    perfect_curve = np.ones(len(all_T))
    max_area = np.trapz(perfect_curve - baseline_curve, all_T)

    if max_area == 0:
        logger.warning("Baseline already detects all cases; ABCgain undefined.")
        return 0.0

    gained_area = np.trapz(latte_curve - baseline_curve, all_T)
    abcgain = gained_area / max_area

    logger.info(
        "ABCgain: %.3f  (LATTE area=%.2f, baseline area=%.2f, max gain=%.2f).",
        abcgain,
        np.trapz(latte_curve, all_T),
        np.trapz(baseline_curve, all_T),
        max_area,
    )

    return float(abcgain)
