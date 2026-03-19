import pandas as pd
from note_ner import extract_cui_features, aggregate_features
from once import get_once_features
from rollup import rollup_icd_to_phecode


def build_map_feature_matrix(
    phecode_df: pd.DataFrame,
    once_phecodes: list[str],
    main_phecode: str,
    min_nonzero: int = 20,
) -> pd.DataFrame:
    """
    Build the MAP feature matrix from PheCode-rolled EHR data and ONCE feature list.

    Replicates the notebook steps: full PheCode pivot → restrict to ONCE features →
    filter study population → drop sparse features.

    Args:
        phecode_df: Output of rollup_icd_to_phecode — must have 'subject_id' and 'Phecode'.
        once_phecodes: PheCode strings without 'PheCode:' prefix from ONCE codified features.
        main_phecode: Anchor PheCode (e.g. '714.1'). Always retained as first column.
        min_nonzero: Drop features with fewer non-zero patients than this. Default 20.
            MAP's flexmix EM requires enough non-zero observations per feature to fit
            two components; features below this threshold cause "Log-likelihood: NA".

    Returns:
        pd.DataFrame: mat_df with index=subject_id, columns=PheCodes.
            - Restricted to ONCE features present in the data.
            - Patients with zero counts across all ONCE features are excluded.
            - Sparse features (< min_nonzero non-zero patients) are dropped.

    Raises:
        ValueError: If main_phecode is not in once_phecodes.
    """
    if main_phecode not in once_phecodes:
        raise ValueError(f"main_phecode '{main_phecode}' not found in once_phecodes.")

    mapped = phecode_df.dropna(subset=["Phecode"])
    all_wide = mapped.pivot_table(
        index="subject_id", columns="Phecode", aggfunc="size", fill_value=0
    )

    # Restrict to ONCE PheCodes present in the data; anchor always first
    available = [c for c in once_phecodes if c in all_wide.columns]
    feature_cols = [main_phecode] + [c for c in available if c != main_phecode]
    mat = all_wide[feature_cols].copy()

    # Study population: patients with at least one ONCE feature count
    mat = mat[mat.sum(axis=1) > 0]

    # Drop sparse features, but always keep the anchor
    nonzero = (mat > 0).sum()
    sparse = nonzero[(nonzero < min_nonzero) & (nonzero.index != main_phecode)].index
    if len(sparse):
        mat = mat.drop(columns=sparse)

    return mat


def build_note_proxy(admissions_df: pd.DataFrame, study_index) -> pd.DataFrame:
    """
    Build a note_count DataFrame from admission counts as a proxy for discharge notes.

    Use this when MIMIC-IV-Note is unavailable. MAP requires a per-patient note count
    as the exposure denominator in its Poisson model. Admission count is the best
    available proxy (1 admission ≈ 1 discharge note).

    Args:
        admissions_df: DataFrame with a 'subject_id' column (from admissions.csv.gz).
        study_index: Index of the study population (typically mat_df.index).

    Returns:
        pd.DataFrame with index=subject_id, column='note_count'.
            Patients absent from admissions_df receive count 1 (not 0) because
            MAP requires non-zero denominators in its Poisson exposure term.
    """
    return (
        admissions_df.groupby("subject_id")
        .size()
        .to_frame("note_count")
        .reindex(study_index)
        .fillna(1)
        .clip(lower=1)
        .astype(int)
    )


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
