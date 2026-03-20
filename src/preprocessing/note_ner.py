import medspacy
import pandas as pd
from medspacy.target_matcher import TargetRule
from tqdm.auto import tqdm

from loguru import logger as _loguru_logger

_loguru_logger.disable("PyRuSH")


def extract_cui_features(
    notes_df,
    text_column,
    id_column,
    target_cuis,
    date_column=None,
    max_note_chars=None,
    n_process=1,
    batch_size=128,
):
    """
    Extract CUI mentions from clinical notes using MedSpaCy.
    Negated and family-history mentions are excluded by MedSpaCy's context rules.

    Args:
        notes_df       : DataFrame with one row per note.
        text_column    : Column of raw note text.
        id_column      : Patient/encounter ID column.
        target_cuis    : List of {"term": ..., "cui": ...} dicts.
                         Use get_once_features()["nlp_target_cuis"].
        date_column    : Optional note date column; adds "datetime" to output.
        max_note_chars : Truncate each note to this many characters before NLP.
                         None = no truncation. Use e.g. 10_000 for a good
                         speed/recall tradeoff on MIMIC-IV discharge notes.
        n_process      : Worker processes for nlp.pipe(). Keep at 1 in Jupyter
                         notebooks (multiprocessing can deadlock there).
        batch_size     : Texts per spaCy batch. Larger = better CPU utilisation.

    Returns:
        DataFrame with columns [id_column, "cui", "count"] and optionally "datetime".
    """
    cols = [id_column, "cui", "count"] + (["datetime"] if date_column else [])

    if notes_df.empty:
        return pd.DataFrame(columns=cols)

    nlp = medspacy.load()
    nlp.get_pipe("medspacy_target_matcher").add(
        [TargetRule(literal=item["term"], category=item["cui"]) for item in target_cuis]
    )

    texts = notes_df[text_column].astype(str)
    if max_note_chars is not None:
        texts = texts.str[:max_note_chars]

    ids = notes_df[id_column]
    dates = notes_df[date_column] if date_column is not None else [None] * len(notes_df)

    extracted_data = []
    pipe = tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
        total=len(notes_df),
        desc="  MedSpaCy",
        unit="note",
        dynamic_ncols=True,
    )
    for doc, row_id, row_date in zip(pipe, ids, dates):
        for ent in doc.ents:
            if not ent._.is_negated and not ent._.is_family:
                record = {id_column: row_id, "cui": ent.label_, "count": 1}
                if date_column is not None:
                    record["datetime"] = row_date
                extracted_data.append(record)

    if not extracted_data:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(extracted_data)
