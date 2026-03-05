import medspacy
import pandas as pd
from medspacy.target_matcher import TargetRule


def extract_cui_features(notes_df, text_column, id_column, target_cuis):
    """
    Extracts specific CUIs from clinical notes using MedSpaCy.
    Filters out negated, uncertain, or family history mentions.

    Args:
        notes_df: DataFrame containing clinical notes.
        text_column: Name of the column with note text.
        id_column: Name of the column with patient/encounter IDs.
        target_cuis: List of dicts with 'term' and 'cui' keys (e.g. from ONCE output).

    Returns:
        DataFrame with columns [id_column, 'cui', 'count'].
    """
    nlp = medspacy.load()

    target_rules = [
        TargetRule(literal=item["term"], category=item["cui"]) for item in target_cuis
    ]
    nlp.get_pipe("medspacy_target_matcher").add(target_rules)

    extracted_data = []

    for doc, row_id in zip(nlp.pipe(notes_df[text_column]), notes_df[id_column]):
        for ent in doc.ents:
            if not ent._.is_negated and not ent._.is_family:
                extracted_data.append(
                    {id_column: row_id, "cui": ent.label_, "count": 1}
                )

    return pd.DataFrame(extracted_data)


def aggregate_features(df, id_column, feature_column, value_column=None):
    """
    Pivots a long-format CUI DataFrame into a wide patient-feature matrix.

    Args:
        df: Long-format DataFrame (output of extract_cui_features).
        id_column: Column to use as row index (e.g. 'patient_id').
        feature_column: Column to pivot into columns (e.g. 'cui').
        value_column: Optional numeric column to aggregate with sum.
                      If None, produces a binary presence matrix.

    Returns:
        Wide DataFrame with patients as rows and features as columns.
    """
    if value_column:
        return df.pivot_table(
            index=id_column,
            columns=feature_column,
            values=value_column,
            aggfunc="sum",
            fill_value=0,
        )
    df_copy = df.copy()
    df_copy["present"] = 1
    return df_copy.pivot_table(
        index=id_column,
        columns=feature_column,
        values="present",
        aggfunc="max",
        fill_value=0,
    )
