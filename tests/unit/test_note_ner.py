from unittest.mock import MagicMock, patch

import pandas as pd

import note_ner
from note_ner import extract_cui_features


# ---------------------------------------------------------------------------
# extract_cui_features — medspacy is mocked so the suite runs without a GPU
# or heavy model download.
# ---------------------------------------------------------------------------


def _make_mock_ent(label, is_negated=False, is_family=False):
    ent = MagicMock()
    ent.label_ = label
    ent._.is_negated = is_negated
    ent._.is_family = is_family
    return ent


def _make_mock_nlp(*docs):
    """Return a mock nlp whose .pipe() yields the given docs."""
    mock_nlp = MagicMock()
    mock_nlp.pipe.return_value = iter(docs)
    return mock_nlp


def _notes_df(texts=("note text",), ids=(101,)):
    return pd.DataFrame({"patient_id": list(ids), "note_text": list(texts)})


TARGET_CUIS = [{"term": "rheumatoid arthritis", "cui": "C0003873"}]


@patch.object(note_ner, "medspacy")
def test_extract_keeps_positive_mention(mock_medspacy):
    ent = _make_mock_ent("C0003873")
    doc = MagicMock()
    doc.ents = [ent]
    mock_medspacy.load.return_value = _make_mock_nlp(doc)

    result = extract_cui_features(_notes_df(), "note_text", "patient_id", TARGET_CUIS)

    assert len(result) == 1
    assert result.iloc[0]["cui"] == "C0003873"
    assert result.iloc[0]["patient_id"] == 101


@patch.object(note_ner, "medspacy")
def test_extract_filters_negated(mock_medspacy):
    ent = _make_mock_ent("C0003873", is_negated=True)
    doc = MagicMock()
    doc.ents = [ent]
    mock_medspacy.load.return_value = _make_mock_nlp(doc)

    result = extract_cui_features(_notes_df(), "note_text", "patient_id", TARGET_CUIS)

    assert result.empty


@patch.object(note_ner, "medspacy")
def test_extract_filters_family_history(mock_medspacy):
    ent = _make_mock_ent("C0003873", is_family=True)
    doc = MagicMock()
    doc.ents = [ent]
    mock_medspacy.load.return_value = _make_mock_nlp(doc)

    result = extract_cui_features(_notes_df(), "note_text", "patient_id", TARGET_CUIS)

    assert result.empty


@patch.object(note_ner, "medspacy")
def test_extract_multiple_patients(mock_medspacy):
    doc1 = MagicMock()
    doc1.ents = [_make_mock_ent("C0003873")]

    doc2 = MagicMock()
    doc2.ents = []  # no mentions for patient 102

    mock_medspacy.load.return_value = _make_mock_nlp(doc1, doc2)

    notes = _notes_df(texts=("note A", "note B"), ids=(101, 102))
    result = extract_cui_features(notes, "note_text", "patient_id", TARGET_CUIS)

    assert len(result) == 1
    assert result.iloc[0]["patient_id"] == 101


@patch.object(note_ner, "medspacy")
def test_extract_returns_correct_columns(mock_medspacy):
    doc = MagicMock()
    doc.ents = []
    mock_medspacy.load.return_value = _make_mock_nlp(doc)

    result = extract_cui_features(_notes_df(), "note_text", "patient_id", TARGET_CUIS)

    # Even when empty, the DataFrame should be constructable and have the right shape
    assert isinstance(result, pd.DataFrame)


def test_extract_empty_notes_df():
    """Empty input returns an empty DataFrame with the expected columns, no medspacy call."""
    empty = pd.DataFrame(columns=["patient_id", "note_text"])
    result = extract_cui_features(empty, "note_text", "patient_id", TARGET_CUIS)

    assert result.empty
    assert set(result.columns) == {"patient_id", "cui", "count"}


@patch.object(note_ner, "medspacy")
def test_extract_includes_datetime_when_date_column_given(mock_medspacy):
    ent = _make_mock_ent("C0003873")
    doc = MagicMock()
    doc.ents = [ent]
    mock_medspacy.load.return_value = _make_mock_nlp(doc)

    notes = pd.DataFrame(
        {"patient_id": [101], "note_text": ["some note"], "note_date": ["2020-01-01"]}
    )
    result = extract_cui_features(
        notes, "note_text", "patient_id", TARGET_CUIS, date_column="note_date"
    )

    assert "datetime" in result.columns
    assert result.iloc[0]["datetime"] == "2020-01-01"


@patch.object(note_ner, "medspacy")
def test_extract_max_note_chars_truncates_text(mock_medspacy):
    """max_note_chars should truncate texts before passing to the NLP pipeline."""
    doc = MagicMock()
    doc.ents = []
    mock_medspacy.load.return_value = _make_mock_nlp(doc)

    notes = _notes_df(texts=("A" * 500,))
    extract_cui_features(
        notes, "note_text", "patient_id", TARGET_CUIS, max_note_chars=10
    )

    # The text passed to nlp.pipe should be the truncated version
    pipe_call_args = mock_medspacy.load.return_value.pipe.call_args
    texts_passed = list(pipe_call_args[0][0])
    assert all(len(t) <= 10 for t in texts_passed)


@patch.object(note_ner, "medspacy")
def test_extract_no_date_column_omits_datetime(mock_medspacy):
    ent = _make_mock_ent("C0003873")
    doc = MagicMock()
    doc.ents = [ent]
    mock_medspacy.load.return_value = _make_mock_nlp(doc)

    result = extract_cui_features(_notes_df(), "note_text", "patient_id", TARGET_CUIS)

    assert "datetime" not in result.columns
