import os
import subprocess
import sys
import tempfile

import pandas as pd

# Allow imports from the sibling preprocessing package when running scripts directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "map_runner.R")


# preprocess_map
# --------------
# Transform an observation log into the two inputs required by run_map():
# a wide patient × feature count matrix (mat_df) and a per-patient note
# count denominator (note_df).
#
# This function is the bridge between the algorithm-agnostic observation log
# produced by build_obs_log() (preprocessing/preprocessing.py) and the MAP
# algorithm. It applies MAP-specific constraints:
#   - Only ONCE-selected features are retained (codified + NLP).
#   - Patients with zero counts across all ONCE features are excluded
#     (they cannot be phenotyped).
#   - Features observed in fewer than min_nonzero patients are dropped to
#     prevent MAP's flexmix EM from returning "Log-likelihood: NA".
#   - The anchor feature (main_phecode) is always retained as the first
#     column regardless of sparsity.
#
# Args:
#   obs_log       (pd.DataFrame) : Output of build_obs_log(). Must have columns
#                                  [subject_col, "event_type", "event", "value", "datetime"].
#                                  event values use prefixed format:
#                                  "PheCode:714.1", "RXNORM:1049630", "CUI:C0003873"
#   admissions_df (pd.DataFrame) : MIMIC-IV admissions table used to build a
#                                  per-patient note count proxy. Must contain
#                                  subject_col. Used when MIMIC-IV-Note is unavailable;
#                                  admission count ≈ discharge note count.
#   once_features (dict)         : Output of get_once_features() (once.py). Keys used:
#                                  - "codified_list": list of prefixed feature strings
#                                    e.g. ["PheCode:714.1", "CCS:3", "RXNORM:1049630"]
#                                  - "nlp_target_cuis": list of {"term":..., "cui":...}
#                                    e.g. [{"term": "rheumatoid arthritis", "cui": "C0003873"}]
#   main_phecode  (str)          : Anchor PheCode (without prefix) for the MAP model.
#                                  Example: "714.1"
#                                  Must be present in once_features["codified_list"]
#                                  as "PheCode:714.1".
#   subject_col   (str)          : Patient ID column name shared across obs_log and
#                                  admissions_df. Default: "subject_id"
#   min_nonzero   (int)          : Minimum number of patients with a non-zero count
#                                  required to keep a feature. Default: 20.
#                                  Prevents MAP's flexmix EM from failing on sparse features.
#
# Returns:
#   tuple:
#     mat_df  (pd.DataFrame) : Wide feature count matrix.
#                              Index = subject_id; columns = prefixed event strings.
#                              First column is always the anchor (e.g. "PheCode:714.1").
#                              Example shape: (4500 patients, 38 features)
#     note_df (pd.DataFrame) : Per-patient note count (Poisson exposure denominator).
#                              Index = subject_id; single column "note_count" (int >= 1).
#                              Patients absent from admissions_df receive count 1.
def preprocess_map(
    obs_log: pd.DataFrame,
    admissions_df: pd.DataFrame,
    once_features: dict,
    main_phecode: str,
    subject_col: str = "subject_id",
    min_nonzero: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_event = f"PheCode:{main_phecode}"

    # Build the complete set of ONCE target events across all modalities
    codified_events = set(once_features.get("codified_list", []))
    nlp_events = {
        f"CUI:{item['cui']}" for item in once_features.get("nlp_target_cuis", [])
    }
    target_events = codified_events | nlp_events

    if main_event not in target_events:
        raise ValueError(
            f"Anchor '{main_event}' not found in once_features['codified_list']. "
            f"Ensure it appears as 'PheCode:{main_phecode}'."
        )

    # Filter the observation log to ONCE features and pivot to wide count matrix
    filtered = obs_log[obs_log["event"].isin(target_events)]
    mat = filtered.groupby([subject_col, "event"]).size().unstack(fill_value=0)

    # Place anchor column first; keep only columns that exist in the data
    available = [c for c in mat.columns if c in target_events]
    ordered = [main_event] + [c for c in available if c != main_event]
    mat = mat[ordered]

    # Study population: patients with at least one non-zero ONCE feature count
    mat = mat[mat.sum(axis=1) > 0]

    # Drop features below the sparsity threshold (anchor is always exempt)
    nonzero_counts = (mat > 0).sum()
    sparse = nonzero_counts[
        (nonzero_counts < min_nonzero) & (nonzero_counts.index != main_event)
    ].index
    if len(sparse):
        mat = mat.drop(columns=sparse)

    # Build note count proxy from admission counts (MAP's Poisson exposure denominator).
    # Patients absent from admissions_df receive 1 (MAP requires non-zero denominators).
    note_df = (
        admissions_df.groupby(subject_col)
        .size()
        .to_frame("note_count")
        .reindex(mat.index)
        .fillna(1)
        .clip(lower=1)
        .astype(int)
    )

    return mat, note_df


# run_map
# -------
# Run the MAP (Multimodal Automated Phenotyping) algorithm on pre-built inputs.
# Shells out to the MAP R package via map_runner.R. The R script renames
# main_icd_col to "ICD" internally as required by the MAP package.
#
# Args:
#   mat_df       (pd.DataFrame) : Wide feature count matrix from preprocess_map().
#                                 Index = subject_id; columns = prefixed event strings.
#                                 Example: output of preprocess_map()[0]
#   note_df      (pd.DataFrame) : Per-patient note count from preprocess_map().
#                                 Index = subject_id; column = "note_count".
#                                 Example: output of preprocess_map()[1]
#   main_icd_col (str)          : Column name in mat_df to use as MAP's anchor feature.
#                                 Must match a column in mat_df exactly.
#                                 Example: "PheCode:714.1"
#                                 Pass f"PheCode:{main_phecode}" to match preprocess_map().
#
# Returns:
#   pd.DataFrame : Results table with columns:
#                  - "patient_id" (int | str) : patient identifier
#                  - "score"      (float)      : MAP posterior probability score
#                  - "phenotype"  (int)        : binary label — 1 = case, 0 = control
#                                               (1 when score >= MAP's prevalence cutoff)
def run_map(
    mat_df: pd.DataFrame, note_df: pd.DataFrame, main_icd_col: str
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        mat_path = os.path.join(tmp, "mat.csv")
        note_path = os.path.join(tmp, "note.csv")
        out_path = os.path.join(tmp, "results.csv")

        mat_df.to_csv(mat_path, index=True)
        note_df.to_csv(note_path, index=True)

        result = subprocess.run(
            ["Rscript", _R_SCRIPT, mat_path, note_path, out_path, main_icd_col],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MAP R script failed:\n{result.stderr}")

        return pd.read_csv(out_path)
