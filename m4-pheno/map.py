import subprocess
import tempfile
import os
import pandas as pd

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "map_runner.R")


def run_map(
    mat_df: pd.DataFrame, note_df: pd.DataFrame, main_icd_col: str
) -> pd.DataFrame:
    """
    Run the MAP (Multimodal Automated Phenotyping) algorithm on EHR data.

    Args:
        mat_df: Wide-format feature matrix (patients x features).
            Index = patient_id; Columns = features (e.g., PheCodes, CUIs)
        note_df: DataFrame with patient note counts. Index = patient_id, column = 'note_count'
        main_icd_col: Column name of main ICD/PheCode feature (anchor for MAP)

    Returns:
        DataFrame with columns ['patient_id', 'score', 'phenotype']
    """
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
