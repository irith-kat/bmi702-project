from pathlib import Path

import pandas as pd
import pytest

from once import get_once_features, parse_once_by_modality


# ---------------------------------------------------------------------------
# Helpers: write minimal CSV fixtures to a tmp_path
# ---------------------------------------------------------------------------


def _write_codified(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "codified.csv"
    df.to_csv(p, index=False)
    return p


def _write_narrative(tmp_path: Path, rows: list[dict], sep="|") -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "narrative.csv"
    df.to_csv(p, sep=sep, index=False)
    return p


CODIFIED_ROWS = [
    {"Variable": "PheCode:714.1", "phenotyping_features": "true"},
    {"Variable": "PheCode:401", "phenotyping_features": "false"},
    {"Variable": "PheCode:250", "phenotyping_features": "True"},  # case-insensitive
]

NARRATIVE_ROWS = [
    {"CUI": "C0003873", "Term": "rheumatoid arthritis"},
    {"CUI": "C0020443", "Term": "hypertension"},
]


# ---------------------------------------------------------------------------
# get_once_features — return structure
# ---------------------------------------------------------------------------


def test_returns_expected_keys(tmp_path):
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    nar = _write_narrative(tmp_path, NARRATIVE_ROWS)
    result = get_once_features(str(cod), str(nar))
    assert set(result.keys()) == {
        "codified",
        "narrative",
        "codified_list",
        "nlp_list",
        "nlp_target_cuis",
    }


def test_codified_list_filters_phenotyping_features(tmp_path):
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    nar = _write_narrative(tmp_path, NARRATIVE_ROWS)
    result = get_once_features(str(cod), str(nar))
    assert set(result["codified_list"]) == {"PheCode:714.1", "PheCode:250"}
    assert "PheCode:401" not in result["codified_list"]


def test_nlp_list_contains_all_cuis(tmp_path):
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    nar = _write_narrative(tmp_path, NARRATIVE_ROWS)
    result = get_once_features(str(cod), str(nar))
    assert result["nlp_list"] == ["C0003873", "C0020443"]


def test_nlp_target_cuis_format(tmp_path):
    """nlp_target_cuis must be a list of {'term': str, 'cui': str} dicts —
    the format expected by note_ner.extract_cui_features."""
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    nar = _write_narrative(tmp_path, NARRATIVE_ROWS)
    result = get_once_features(str(cod), str(nar))

    target_cuis = result["nlp_target_cuis"]
    assert len(target_cuis) == 2
    for entry in target_cuis:
        assert "term" in entry and "cui" in entry

    cuis = {e["cui"] for e in target_cuis}
    terms = {e["term"] for e in target_cuis}
    assert cuis == {"C0003873", "C0020443"}
    assert terms == {"rheumatoid arthritis", "hypertension"}


def test_nlp_target_cuis_missing_term_column(tmp_path):
    """If the narrative file has no 'Term' column, nlp_target_cuis should be
    empty rather than crashing."""
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    # Narrative without Term column
    nar_rows = [{"CUI": "C0003873"}, {"CUI": "C0020443"}]
    nar = _write_narrative(tmp_path, nar_rows)
    result = get_once_features(str(cod), str(nar))
    assert result["nlp_target_cuis"] == []
    assert result["nlp_list"] == ["C0003873", "C0020443"]


def test_missing_codified_file_raises(tmp_path):
    nar = _write_narrative(tmp_path, NARRATIVE_ROWS)
    with pytest.raises(FileNotFoundError):
        get_once_features(str(tmp_path / "missing.csv"), str(nar))


def test_missing_narrative_file_raises(tmp_path):
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    with pytest.raises(FileNotFoundError):
        get_once_features(str(cod), str(tmp_path / "missing.csv"))


# ---------------------------------------------------------------------------
# Column normalization: comma-sep files and STR-column ONCE exports
# ---------------------------------------------------------------------------


def test_comma_separated_narrative_parsed(tmp_path):
    """Narrative files with comma separator (some ONCE downloads) must be parsed."""
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    # Write as comma-separated with lowercase column names
    nar_rows = [{"cui": "C0003873", "term": "rheumatoid arthritis"}]
    nar = _write_narrative(tmp_path, nar_rows, sep=",")
    result = get_once_features(str(cod), str(nar))
    assert result["nlp_list"] == ["C0003873"]
    assert len(result["nlp_target_cuis"]) == 1
    assert result["nlp_target_cuis"][0] == {
        "term": "rheumatoid arthritis",
        "cui": "C0003873",
    }


def test_str_column_mapped_to_term(tmp_path):
    """ONCE STR|CUI exports (e.g. hemorrhoids) should be handled via STR→TERM rename."""
    cod = _write_codified(tmp_path, CODIFIED_ROWS)
    nar_rows = [{"STR": "Hemorrhoids", "CUI": "C0019112"}]
    nar = _write_narrative(tmp_path, nar_rows, sep="|")
    result = get_once_features(str(cod), str(nar))
    assert result["nlp_list"] == ["C0019112"]
    assert len(result["nlp_target_cuis"]) == 1
    assert result["nlp_target_cuis"][0] == {"term": "Hemorrhoids", "cui": "C0019112"}


# ---------------------------------------------------------------------------
# parse_once_by_modality
# ---------------------------------------------------------------------------

_MIXED_FEATURES = {
    "codified_list": [
        "PheCode:714.1",
        "PheCode:401",
        "RXNORM:614391",
        "RXNORM:1049630",
        "LNC:4548-4",
        "CCS:3",
        "ShortName:MCV",
        "UnknownPrefix:XYZ",
    ]
}


def test_parse_splits_phecodes():
    result = parse_once_by_modality(_MIXED_FEATURES)
    assert result["phecode"] == ["714.1", "401"]


def test_parse_splits_rxnorm():
    result = parse_once_by_modality(_MIXED_FEATURES)
    assert result["rxnorm"] == ["614391", "1049630"]


def test_parse_splits_loinc():
    result = parse_once_by_modality(_MIXED_FEATURES)
    assert result["loinc"] == ["4548-4"]


def test_parse_splits_ccs():
    result = parse_once_by_modality(_MIXED_FEATURES)
    assert result["ccs"] == ["3"]


def test_parse_splits_shortname():
    result = parse_once_by_modality(_MIXED_FEATURES)
    assert result["shortname"] == ["MCV"]


def test_parse_unknown_prefix_goes_to_other():
    result = parse_once_by_modality(_MIXED_FEATURES)
    assert result["other"] == ["UnknownPrefix:XYZ"]


def test_parse_empty_codified_list():
    result = parse_once_by_modality({"codified_list": []})
    assert all(len(v) == 0 for v in result.values())


def test_parse_phecode_only_features():
    features = {"codified_list": ["PheCode:455", "PheCode:578.8"]}
    result = parse_once_by_modality(features)
    assert result["phecode"] == ["455", "578.8"]
    assert result["rxnorm"] == []
    assert result["loinc"] == []
