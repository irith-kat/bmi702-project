import pandas as pd
from note_ner import extract_cui_features, aggregate_features
from once import get_once_features
from rollup import rollup_icd_to_phecode


def prepare_map_inputs(
    ehr_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    icd_col: str,
    include_nlp: bool = False,
    once_codified_file: str | None = None,
    once_narrative_file: str | None = None,
):
    """
    Prepare matrices for MAP input: feature counts (mat_df) and note counts (note_df)

    Args:
        ehr_df: EHR table with columns for ICD, CPT, RxNorm, etc.
        notes_df: Table of patient notes (id + text)
        icd_col: Column in ehr_df with ICD codes to roll up to PheCode
        include_nlp: Whether to include NLP features (CUIs)
        once_codified_file: ONCE codified file path
        once_narrative_file: ONCE narrative file path
    Returns:
        mat_df, note_df
    """
    # 1. Codified features
    phecode_df = rollup_icd_to_phecode(ehr_df, icd_col)
    mat_df = phecode_df.pivot_table(
        index="subject_id", columns="Phecode", aggfunc="size", fill_value=0
    )

    # 2. Note counts
    note_df = notes_df.groupby("subject_id").size().to_frame("note_count")

    # 3. NLP features
    target_cuis = None
    if include_nlp and once_narrative_file:
        once_features = get_once_features(once_codified_file, once_narrative_file)
        target_cuis = once_features["nlp_target_cuis"]

        cui_df = extract_cui_features(
            notes_df,
            text_column="note_text",
            id_column="subject_id",
            target_cuis=target_cuis,
        )
        cui_wide = aggregate_features(
            cui_df, id_column="subject_id", feature_column="cui"
        )
        mat_df = mat_df.join(cui_wide, how="outer").fillna(0)

    return mat_df, note_df
