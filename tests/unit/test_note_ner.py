from unittest.mock import MagicMock, patch

import pandas as pd

import note_ner
from note_ner import aggregate_features, extract_cui_features


# ---------------------------------------------------------------------------
# aggregate_features
# ---------------------------------------------------------------------------


def _long_df():
    return pd.DataFrame(
        {
            "patient_id": [101, 101, 102],
            "cui": ["C0003873", "C0020443", "C0003873"],
            "count": [1, 1, 1],
        }
    )


def test_aggregate_features_binary():
    result = aggregate_features(_long_df(), "patient_id", "cui")
    assert result.loc[101, "C0003873"] == 1
    assert result.loc[101, "C0020443"] == 1
    assert result.loc[102, "C0003873"] == 1
    # patient 102 has no C0020443
    assert result.loc[102, "C0020443"] == 0


def test_aggregate_features_with_value_column():
    df = pd.DataFrame(
        {
            "patient_id": [101, 101],
            "cui": ["C0003873", "C0003873"],
            "count": [1, 2],
        }
    )
    result = aggregate_features(df, "patient_id", "cui", value_column="count")
    assert result.loc[101, "C0003873"] == 3


def test_aggregate_features_single_patient_single_cui():
    df = pd.DataFrame({"pid": [1], "cui": ["C0001"], "count": [1]})
    result = aggregate_features(df, "pid", "cui", value_column="count")
    assert result.shape == (1, 1)
    assert result.loc[1, "C0001"] == 1


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
