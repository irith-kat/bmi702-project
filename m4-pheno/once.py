import pandas as pd
import os


def get_once_features(codified_file: str, narrative_file: str):
    """
    Automate the discovery of 'Codified' (PheCodes) and 'Narrative' (CUIs) features
    by reading specific ONCE output files.

    Args:
        codified_file (str): Path to the ONCE PheCode CSV file.
        narrative_file (str): Path to the ONCE CUI/Narrative CSV file.

    Returns:
        dict: A dictionary containing:
            - 'codified': Full DataFrame of PheCodes/CCS features.
            - 'narrative': Full DataFrame of CUIs.
            - 'codified_list': List of 'Variable' strings for high-confidence features.
            - 'nlp_list': List of 'CUI' strings.
    """
    # 1. Read files using explicit paths
    if not os.path.exists(codified_file):
        raise FileNotFoundError(f"Codified file not found: {codified_file}")
    if not os.path.exists(narrative_file):
        raise FileNotFoundError(f"Narrative file not found: {narrative_file}")

    codified_df = pd.read_csv(codified_file)
    # Narrative files use '|' as separator based on inspection
    narrative_df = pd.read_csv(narrative_file, sep="|")

    # 2. Extract specific lists for downstream models (e.g., KOMAP)
    # Filter for 'phenotyping_features' == 'true' (string or bool)
    mask = codified_df["phenotyping_features"].astype(str).str.lower() == "true"
    codified_list = codified_df[mask]["Variable"].tolist()

    # Extract CUIs and build the target_cuis format expected by note_ner.extract_cui_features.
    # The narrative file must have a 'CUI' column and a 'Term' column (the text literal
    # that MedSpaCy will search for in clinical notes).
    nlp_list = []
    nlp_target_cuis = []
    if "CUI" in narrative_df.columns:
        nlp_list = narrative_df["CUI"].tolist()
        if "Term" in narrative_df.columns:
            nlp_target_cuis = [
                {"term": row["Term"], "cui": row["CUI"]}
                for _, row in narrative_df[["Term", "CUI"]].dropna().iterrows()
            ]

    return {
        "codified": codified_df,
        "narrative": narrative_df,
        "codified_list": codified_list,
        "nlp_list": nlp_list,
        # Ready-to-use input for note_ner.extract_cui_features
        "nlp_target_cuis": nlp_target_cuis,
    }
