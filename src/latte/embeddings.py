"""
embeddings.py
-------------
Code embedding generation for the LATTE phenotyping pipeline.

Builds a code-level embedding matrix via truncated SVD on a log1p-transformed
patient × code co-occurrence matrix.  The output matches the format expected by
LATTE's get_data_from_csv (one row per SVD component, one column per code).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


def build_cooccurrence_embeddings(
    obs_log: pd.DataFrame,
    feature_codes: list[str],
    n_components: int = 50,
    subject_col: str = "subject_id",
    event_col: str = "event",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build code embeddings via truncated SVD on a log1p-transformed
    patient × code co-occurrence matrix.

    Each code receives an embedding vector of length ``n_components`` derived
    from the shared latent structure of which patients co-use codes together.
    This is conceptually identical to KOMAP's embedding step.

    Parameters
    ----------
    obs_log : pd.DataFrame
        Long-format observation log with at least ``subject_col`` and
        ``event_col`` columns.  One row per code occurrence per patient visit.
    feature_codes : list[str]
        The ONCE-selected code strings to embed (e.g. ``["PheCode:428",
        "RXNORM:4603"]``).  Codes absent from ``obs_log`` will receive an all-
        zero embedding column.
    n_components : int
        Number of SVD components (embedding dimensions).  Clamped to
        ``min(n_components, n_patients - 1, n_codes - 1)``.
    subject_col : str
        Column name for patient identifiers in ``obs_log``.
    event_col : str
        Column name for event/code strings in ``obs_log``.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_components, len(feature_codes))``.
        - columns = feature code strings
        - index   = 0, 1, …, n_components - 1  (integer component index)

        Write this to ``embedding.csv`` with ``df.to_csv(path)``; the resulting
        file matches the format LATTE's ``get_data_from_csv`` expects.
    """
    # Filter to requested codes only
    obs_filtered = obs_log[obs_log[event_col].isin(feature_codes)].copy()
    if obs_filtered.empty:
        raise ValueError(
            "No rows in obs_log match any of the requested feature_codes. "
            "Check that event column values match the code format."
        )

    # Build patient × code count matrix; ensure all feature_codes are present
    matrix = (
        obs_filtered.groupby([subject_col, event_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=feature_codes, fill_value=0)
    )

    n_patients, n_codes = matrix.shape
    logger.info("Co-occurrence matrix: %d patients × %d codes.", n_patients, n_codes)

    # log1p stabilises rare-code variance (same transform used in KOMAP)
    X = np.log1p(matrix.values.astype(float))

    # Clamp n_components to what the matrix can support
    actual_components = min(n_components, n_patients - 1, n_codes - 1)
    if actual_components < n_components:
        logger.warning(
            "Requested n_components=%d, but matrix only supports %d. Using %d.",
            n_components,
            actual_components,
            actual_components,
        )

    svd = TruncatedSVD(n_components=actual_components, random_state=random_state)
    svd.fit(X)  # shape of X: (n_patients, n_codes)

    # svd.components_ shape: (n_components, n_codes)
    embedding_df = pd.DataFrame(
        svd.components_,
        columns=feature_codes,
    )

    explained = svd.explained_variance_ratio_.sum()
    logger.info(
        "SVD complete: %d components explain %.1f%% of variance.",
        actual_components,
        100 * explained,
    )

    return embedding_df
