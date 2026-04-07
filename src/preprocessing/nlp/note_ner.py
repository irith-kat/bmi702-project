import math
import multiprocessing as mp
import pandas as pd
from tqdm.auto import tqdm
import logging
from loguru import logger as _loguru_logger

_loguru_logger.disable("PyRuSH")
logging.getLogger("PyRuSH").setLevel(logging.ERROR)


def _process_chunk(args):
    """
    Multiprocessing worker for CUI extraction.

    Each subprocess loads its own medspacy model from scratch, which sidesteps
    the TargetRule msgpack serialization error that occurs when spaCy's own
    n_process > 1 tries to IPC Doc objects containing custom extensions.

    Args are bundled as a tuple so Pool.imap can be used directly.
    """
    (
        chunk_records,
        text_column,
        id_column,
        target_cuis,
        date_column,
        max_note_chars,
        batch_size,
        chunk_idx,
    ) = args

    import medspacy
    from medspacy.target_matcher import TargetRule
    import logging
    from loguru import logger as _lgr

    _lgr.disable("PyRuSH")
    logging.getLogger("PyRuSH").setLevel(logging.ERROR)
    nlp = medspacy.load()
    nlp.get_pipe("medspacy_target_matcher").add(
        [TargetRule(literal=item["term"], category=item["cui"]) for item in target_cuis]
    )

    texts = [
        str(t)[:max_note_chars] if max_note_chars else str(t)
        for t in chunk_records[text_column]
    ]
    ids = chunk_records[id_column].tolist()
    dates = (
        chunk_records[date_column].tolist()
        if date_column is not None
        else [None] * len(chunk_records)
    )

    extracted = []
    pipe = tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=1),
        total=len(texts),
        desc=f"  Worker {chunk_idx}",
        unit="note",
        position=chunk_idx,
        leave=True,
        dynamic_ncols=True,
    )
    for doc, row_id, row_date in zip(pipe, ids, dates):
        for ent in doc.ents:
            if not ent._.is_negated and not ent._.is_family:
                record = {id_column: row_id, "cui": ent.label_, "count": 1}
                if date_column is not None:
                    record["datetime"] = row_date
                extracted.append(record)

    return extracted


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
        n_process      : Worker processes. n_process=1 uses spaCy's single-process
                         pipe. n_process > 1 spawns Python subprocesses that each
                         load their own medspacy model — avoids the TargetRule
                         serialization crash in spaCy's built-in multiprocessing.
                         Keep at 1 in Jupyter notebooks (fork can deadlock there).
        batch_size     : Texts per spaCy batch. Larger = better CPU utilisation.

    Returns:
        DataFrame with columns [id_column, "cui", "count"] and optionally "datetime".
    """
    import medspacy
    from medspacy.target_matcher import TargetRule

    cols = [id_column, "cui", "count"] + (["datetime"] if date_column else [])

    if notes_df.empty:
        return pd.DataFrame(columns=cols)

    # ── Multi-process path ────────────────────────────────────────────────────
    # Split notes into n_process chunks; each subprocess loads its own model.
    # We pass plain DataFrames and dicts (pickle-safe), never spaCy objects.
    if n_process > 1:
        n_workers = min(n_process, len(notes_df))
        chunk_size = math.ceil(len(notes_df) / n_workers)
        chunks = [
            notes_df.iloc[i : i + chunk_size].copy()
            for i in range(0, len(notes_df), chunk_size)
        ]

        args_list = [
            (
                chunk,
                text_column,
                id_column,
                target_cuis,
                date_column,
                max_note_chars,
                batch_size,
                idx,
            )
            for idx, chunk in enumerate(chunks)
        ]

        print(f"  Parallel NLP: {len(chunks)} workers × ~{chunk_size} notes")
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(chunks)) as pool:
            results = list(pool.imap(_process_chunk, args_list))

        extracted_data = [rec for chunk_result in results for rec in chunk_result]

        if not extracted_data:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(extracted_data)

    # ── Single-process path ───────────────────────────────────────────────────
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
        nlp.pipe(texts, batch_size=batch_size, n_process=1),
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


def aggregate_features(cui_df, id_column="subject_id"):
    """Pivot long CUI mentions to a wide patient × CUI count matrix."""
    if cui_df.empty:
        return pd.DataFrame()
    return cui_df.pivot_table(
        index=id_column,
        columns="cui",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
