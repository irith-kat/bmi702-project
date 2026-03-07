"""
komap_skill.py

Single function that runs the full KOMAP pipeline (gen_cov_input + KOMAP_corrupt)
by delegating to komap_runner.R via subprocess.
"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

_R_SCRIPT = Path(__file__).parent / "komap_runner.R"


def run_komap(
    ehr_data: pd.DataFrame,
    main_surrogates: str,
    target_code: str,
    nm_disease: str,
    nm_corrupt_code: str,
    output_dir: str,
    target_cui: str = None,
    nm_corrupt_cui: str = None,
    codify_feature: list[str] = None,
    nlp_feature: list[str] = None,
) -> dict:
    """
    Run the full KOMAP unsupervised phenotyping pipeline on EHR data.

    This function bridges to the R implementation of both gen_cov_input
    and KOMAP_corrupt by calling the komap_runner.R script.

    Calls gen_cov_input to build privacy-preserving covariance matrices,
    then KOMAP_corrupt to estimate regression coefficients and predict
    disease probabilities for all patients in ehr_data. Patient-level
    prediction data (dat_part) is derived automatically from ehr_data
    inside the R script as a wide log-count matrix.

    ICD/RxNorm/CPT codes in ehr_data should already be rolled up to
    PheCode/ingredient/CCS level (via mapping.py) before calling this.

    Args:
        ehr_data (pd.DataFrame): Long-format EHR data with columns
            patient_num, days_since_admission, concept_type, concept_code.
        main_surrogates (str): Main surrogate code from ONCE
            (e.g., 'PheCode:250').
        target_code (str): Primary ICD/PheCode for the target disease
            (e.g., 'PheCode:250.2').
        nm_disease (str): Short label for the disease used in output tables
            (e.g., 'T2DM').
        nm_corrupt_code (str): Column name for the corrupted ICD surrogate
            (e.g., 'corrupt_mainICD').
        output_dir (str): Directory where all output CSVs will be written.
            Created automatically if it does not exist. Results are saved
            with constant filenames so they can be reliably referenced
            after the call (e.g., '{output_dir}/pred_prob.csv').
        target_cui (str): Main NLP/CUI surrogate from ONCE
            (e.g., 'C0011849'). Optional.
        nm_corrupt_cui (str): Column name for the corrupted NLP surrogate
            (e.g., 'corrupt_mainNLP'). Optional.
        codify_feature (list[str]): High-confidence codified features from
            ONCE (once.get_once_features()['codified_list']). Optional.
        nlp_feature (list[str]): High-confidence CUI features from ONCE
            (once.get_once_features()['nlp_list']). Optional.

    Returns:
        dict: Paths to output CSVs keyed by output type:
            - 'train_cov': training covariance matrix
            - 'valid_cov': validation covariance matrix
            - 'coefficients': estimated regression coefficients (long format)
            - 'pred_score': raw linear scores per patient
            - 'pred_prob': disease probabilities per patient
            - 'pred_cluster': cluster assignments per patient
    """
    os.makedirs(output_dir, exist_ok=True)

    # gen_cov_input renames columns by position and expects exactly 3:
    # patient_num, days_since_admission, code — concept_type is not used.
    ehr_path = os.path.join(output_dir, "ehr.csv")
    ehr_data[["patient_num", "days_since_admission", "concept_code"]].to_csv(
        ehr_path, index=False
    )

    payload = {
        "ehr_path": ehr_path,
        "main_surrogates": main_surrogates,
        "output_dir": output_dir,
        "target_code": target_code,
        "target_cui": target_cui,
        "nm_disease": nm_disease,
        "nm_corrupt_code": nm_corrupt_code,
        "nm_corrupt_cui": nm_corrupt_cui,
        "codify_feature": codify_feature,
        "nlp_feature": nlp_feature,
    }

    proc = subprocess.run(
        ["Rscript", str(_R_SCRIPT), json.dumps(payload)],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"komap_runner.R failed:\n{proc.stderr}")

    last_line = proc.stdout.strip().splitlines()[-1]
    paths = json.loads(last_line)

    return {key.replace("_path", ""): val for key, val in paths.items()}
