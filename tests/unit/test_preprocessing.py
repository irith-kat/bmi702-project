import pandas as pd
import pytest

from preprocessing import build_map_feature_matrix, build_note_proxy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _phecode_df(rows):
    """Return a minimal phecode_df (output of rollup_icd_to_phecode)."""
    return pd.DataFrame(rows, columns=["subject_id", "icd_code", "Phecode"])


def _admissions_df(rows):
    return pd.DataFrame(rows, columns=["subject_id", "hadm_id"])


# ---------------------------------------------------------------------------
# build_map_feature_matrix
# ---------------------------------------------------------------------------


def test_restricts_to_once_features():
    df = _phecode_df(
        [
            (1, "M059", "714.1"),
            (1, "I10", "401.1"),  # not in ONCE list
            (2, "M059", "714.1"),
        ]
    )
    mat = build_map_feature_matrix(df, once_phecodes=["714.1"], main_phecode="714.1")
    assert "401.1" not in mat.columns
    assert "714.1" in mat.columns


def test_study_population_excludes_zero_patients():
    # Patient 3 has only non-ONCE features
    df = _phecode_df(
        [
            (1, "M059", "714.1"),
            (3, "I10", "401.1"),  # 401.1 not in ONCE list
        ]
    )
    mat = build_map_feature_matrix(df, once_phecodes=["714.1"], main_phecode="714.1")
    assert 3 not in mat.index
    assert 1 in mat.index


def test_anchor_always_first_column():
    df = _phecode_df(
        [(i, "code", phecode) for i in range(30) for phecode in ["714.1", "714.2"]]
    )
    mat = build_map_feature_matrix(
        df, once_phecodes=["714.2", "714.1"], main_phecode="714.1"
    )
    assert mat.columns[0] == "714.1"


def test_sparse_features_dropped():
    # 714.2 appears in only 5 patients — below default min_nonzero=20
    rows = [(i, "M059", "714.1") for i in range(30)]
    rows += [(i, "M060", "714.2") for i in range(5)]
    df = _phecode_df(rows)
    mat = build_map_feature_matrix(
        df, once_phecodes=["714.1", "714.2"], main_phecode="714.1"
    )
    assert "714.2" not in mat.columns
    assert "714.1" in mat.columns  # anchor always kept


def test_anchor_kept_even_if_sparse():
    # anchor has only 10 non-zero patients — still kept
    rows = [(i, "M059", "714.1") for i in range(10)]
    rows += [(i, "I10", "401.1") for i in range(30)]  # non-ONCE
    df = _phecode_df(rows)
    mat = build_map_feature_matrix(
        df, once_phecodes=["714.1"], main_phecode="714.1", min_nonzero=20
    )
    assert "714.1" in mat.columns


def test_unmapped_rows_ignored():
    df = _phecode_df([(1, "???", None), (2, "M059", "714.1")])
    mat = build_map_feature_matrix(df, once_phecodes=["714.1"], main_phecode="714.1")
    assert 1 not in mat.index
    assert 2 in mat.index


def test_count_values_are_per_patient_occurrence():
    # Patient 1 has 3 occurrences of 714.1
    rows = [(1, "M059", "714.1")] * 3 + [(2, "M059", "714.1")]
    df = _phecode_df(rows)
    mat = build_map_feature_matrix(df, once_phecodes=["714.1"], main_phecode="714.1")
    assert mat.loc[1, "714.1"] == 3
    assert mat.loc[2, "714.1"] == 1


def test_raises_if_main_phecode_not_in_once_list():
    df = _phecode_df([(1, "M059", "714.1")])
    with pytest.raises(ValueError, match="main_phecode"):
        build_map_feature_matrix(df, once_phecodes=["714.2"], main_phecode="714.1")


def test_custom_min_nonzero():
    rows = [(i, "M059", "714.1") for i in range(30)]
    rows += [(i, "I10", "714.2") for i in range(15)]
    df = _phecode_df(rows)
    # With min_nonzero=10, 714.2 (15 patients) should be kept
    mat = build_map_feature_matrix(
        df, once_phecodes=["714.1", "714.2"], main_phecode="714.1", min_nonzero=10
    )
    assert "714.2" in mat.columns


# ---------------------------------------------------------------------------
# build_note_proxy
# ---------------------------------------------------------------------------


def test_note_proxy_sums_admissions():
    admissions = _admissions_df([(1, 101), (1, 102), (2, 201)])
    study_index = pd.Index([1, 2], name="subject_id")
    note_df = build_note_proxy(admissions, study_index)
    assert note_df.loc[1, "note_count"] == 2
    assert note_df.loc[2, "note_count"] == 1


def test_note_proxy_fills_missing_with_one():
    admissions = _admissions_df([(1, 101)])
    study_index = pd.Index([1, 2], name="subject_id")
    note_df = build_note_proxy(admissions, study_index)
    assert note_df.loc[2, "note_count"] == 1


def test_note_proxy_aligned_to_study_index():
    admissions = _admissions_df([(1, 101), (99, 999)])  # 99 not in study pop
    study_index = pd.Index([1, 2], name="subject_id")
    note_df = build_note_proxy(admissions, study_index)
    assert set(note_df.index) == {1, 2}
    assert 99 not in note_df.index


def test_note_proxy_clips_to_minimum_one():
    # Reindex with a patient that has no admissions → fillna(1) ensures min=1
    admissions = _admissions_df([])
    study_index = pd.Index([1, 2], name="subject_id")
    note_df = build_note_proxy(admissions, study_index)
    assert (note_df["note_count"] >= 1).all()


def test_note_proxy_returns_integer_dtype():
    admissions = _admissions_df([(1, 101)])
    note_df = build_note_proxy(admissions, pd.Index([1]))
    assert note_df["note_count"].dtype == int


def test_note_proxy_single_column_named_note_count():
    admissions = _admissions_df([(1, 101)])
    note_df = build_note_proxy(admissions, pd.Index([1]))
    assert list(note_df.columns) == ["note_count"]
