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
    # Auto-detect separator by peeking at the first line.
    # ONCE outputs vary: '|' for STR|CUI exports, ',' for others.
    # Column names are normalized to uppercase so 'term'/'Term'/'STR' all resolve to 'TERM'.
    with open(narrative_file, "r") as _f:
        _first_line = _f.readline()
    if "|" in _first_line:
        _sep = "|"
    elif "\t" in _first_line:
        _sep = "\t"
    else:
        _sep = ","
    narrative_df = pd.read_csv(narrative_file, sep=_sep)
    narrative_df.columns = [c.upper() for c in narrative_df.columns]
    # Some ONCE exports label the term column 'STR' instead of 'TERM'
    if "TERM" not in narrative_df.columns and "STR" in narrative_df.columns:
        narrative_df = narrative_df.rename(columns={"STR": "TERM"})

    # 2. Extract specific lists for downstream models (e.g., KOMAP)
    # Filter for 'phenotyping_features' == 'true' (string or bool)
    mask = codified_df["phenotyping_features"].astype(str).str.lower() == "true"
    codified_list = codified_df[mask]["Variable"].tolist()

    # Extract CUIs and build the target_cuis format expected by note_ner.extract_cui_features.
    # The narrative file must have a 'CUI' column and a 'TERM' column (the text literal
    # that MedSpaCy will search for in clinical notes).
    nlp_list = []
    nlp_target_cuis = []
    if "CUI" in narrative_df.columns:
        nlp_list = narrative_df["CUI"].tolist()
        if "TERM" in narrative_df.columns:
            nlp_target_cuis = [
                {"term": row["TERM"], "cui": row["CUI"]}
                for _, row in narrative_df[["TERM", "CUI"]].dropna().iterrows()
            ]

    return {
        "codified": codified_df,
        "narrative": narrative_df,
        "codified_list": codified_list,
        "nlp_list": nlp_list,
        # Ready-to-use input for note_ner.extract_cui_features
        "nlp_target_cuis": nlp_target_cuis,
    }


# Known ONCE variable prefixes and their phenotyping modality
_MODALITY_PREFIXES = {
    "PheCode:": "phecode",
    "RXNORM:": "rxnorm",
    "LNC:": "loinc",
    "CCS:": "ccs",
    "ShortName:": "shortname",
}


def parse_once_by_modality(once_features: dict) -> dict[str, list[str]]:
    """
    Split the ONCE codified feature list by code modality.

    ONCE Variable strings carry a prefix that identifies the vocabulary:
    - 'PheCode:714.1'  → ICD diagnoses rolled up to PheCodes
    - 'RXNORM:614391'  → Prescriptions / medications (RxNorm ingredient)
    - 'LNC:4548-4'     → Lab tests (LOINC code)
    - 'CCS:3'          → Procedures (AHRQ CCS category)
    - 'ShortName:MCV'  → Lab tests identified by short name (no LOINC)

    Args:
        once_features: Return value of get_once_features() — must have 'codified_list'.

    Returns:
        dict with keys 'phecode', 'rxnorm', 'loinc', 'ccs', 'shortname', 'other'.
        Values are lists of code strings with the prefix stripped, e.g.
        'PheCode:714.1' → once['phecode'] = ['714.1', ...].
    """
    by_modality: dict[str, list[str]] = {
        "phecode": [],
        "rxnorm": [],
        "loinc": [],
        "ccs": [],
        "shortname": [],
        "other": [],
    }
    for feat in once_features.get("codified_list", []):
        matched = False
        for prefix, key in _MODALITY_PREFIXES.items():
            if feat.startswith(prefix):
                by_modality[key].append(feat[len(prefix) :])
                matched = True
                break
        if not matched:
            by_modality["other"].append(feat)
    return by_modality
