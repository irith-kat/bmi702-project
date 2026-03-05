from pathlib import Path

import pandas as pd
import pytest

from mapping import (
    rollup_cpt_to_ccs,
    rollup_icd_to_phecode,
    rollup_rxnorm_to_ingredient,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _icd_mapping_file(tmp_path: Path) -> Path:
    """Minimal Phecode mapping CSV with the columns rollup_icd_to_phecode expects."""
    df = pd.DataFrame(
        {
            "ICD": ["714.0", "401.1", "250.00"],
            "Phecode": ["714.1", "401", "250.2"],
            "PhecodeString": [
                "Rheumatoid arthritis",
                "Hypertension",
                "Type 2 diabetes",
            ],
        }
    )
    p = tmp_path / "Phecode_map_v1_2_icd9_icd10cm.csv"
    df.to_csv(p, index=False)
    return p


def _ccs_mapping_file(tmp_path: Path) -> Path:
    """Minimal CCS mapping CSV — headers already lowercase so the normalisation is a no-op."""
    df = pd.DataFrame(
        {
            "start_code": ["00100", "10021", "99201"],
            "end_code": ["01999", "10022", "99499"],
            "ccs_category": ["CCS1", "CCS2", "CCS3"],
            "ccs_category_description": ["Anesthesia", "Biopsy", "E&M visit"],
        }
    )
    p = tmp_path / "CCS_Services_Procedures_v2025-1.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# rollup_icd_to_phecode
# ---------------------------------------------------------------------------


def test_icd_maps_known_code(tmp_path):
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["714.0", "401.1"]})
    result = rollup_icd_to_phecode(df, "icd_code", str(p))

    assert result.loc[0, "Phecode"] == "714.1"
    assert result.loc[0, "PhecodeString"] == "Rheumatoid arthritis"
    assert result.loc[1, "Phecode"] == "401"


def test_icd_left_join_keeps_unmatched_row(tmp_path):
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["714.0", "999.99"]})
    result = rollup_icd_to_phecode(df, "icd_code", str(p))

    assert len(result) == 2
    assert pd.isna(result.loc[1, "Phecode"])


def test_icd_preserves_original_row_count(tmp_path):
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["714.0", "401.1", "250.00", "000.00"]})
    result = rollup_icd_to_phecode(df, "icd_code", str(p))

    assert len(result) == len(df)


def test_icd_drops_duplicate_icd_column_when_renamed(tmp_path):
    """When icd_column != 'ICD', the raw 'ICD' merge key should be dropped."""
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"diagnosis_code": ["714.0"]})
    result = rollup_icd_to_phecode(df, "diagnosis_code", str(p))

    assert "ICD" not in result.columns
    assert "diagnosis_code" in result.columns


def test_icd_keeps_icd_column_when_named_icd(tmp_path):
    """When icd_column is already 'ICD', no column should be dropped."""
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"ICD": ["714.0"]})
    result = rollup_icd_to_phecode(df, "ICD", str(p))

    assert "ICD" in result.columns


def test_icd_missing_file_raises(tmp_path):
    df = pd.DataFrame({"icd_code": ["714.0"]})
    with pytest.raises(FileNotFoundError):
        rollup_icd_to_phecode(df, "icd_code", str(tmp_path / "missing.csv"))


# ---------------------------------------------------------------------------
# rollup_rxnorm_to_ingredient  (stub — returns df unchanged)
# ---------------------------------------------------------------------------


def test_rxnorm_returns_dataframe_unchanged():
    df = pd.DataFrame({"rxnorm": ["856917", "308460"], "dose": [10, 20]})
    result = rollup_rxnorm_to_ingredient(df, "rxnorm")

    pd.testing.assert_frame_equal(result, df)


def test_rxnorm_preserves_all_columns():
    df = pd.DataFrame({"rxnorm": ["856917"], "patient_id": [101], "qty": [30]})
    result = rollup_rxnorm_to_ingredient(df, "rxnorm")

    assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# rollup_cpt_to_ccs
# ---------------------------------------------------------------------------


def test_cpt_matches_code_in_range(tmp_path):
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["00500"]})
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert result.loc[0, "ccs_category"] == "CCS1"
    assert result.loc[0, "ccs_description"] == "Anesthesia"


def test_cpt_no_match_returns_none(tmp_path):
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["50000"]})  # falls between ranges
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert pd.isna(result.loc[0, "ccs_category"])
    assert pd.isna(result.loc[0, "ccs_description"])


def test_cpt_pads_short_code_with_zeros(tmp_path):
    """A code like '500' should be zero-padded to '00500' and match."""
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["500"]})
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert result.loc[0, "ccs_category"] == "CCS1"


def test_cpt_uppercases_input(tmp_path):
    """Lowercase input should be uppercased before range matching."""
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["00500"]})
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert result.loc[0, "ccs_category"] == "CCS1"


def test_cpt_code_longer_than_5_chars_returns_none(tmp_path):
    """Codes that are not exactly 5 characters after processing yield no match."""
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["123456"]})  # 6 chars
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert pd.isna(result.loc[0, "ccs_category"])


def test_cpt_preserves_original_columns(tmp_path):
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["00500"], "patient_id": [101]})
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert "patient_id" in result.columns
    assert result.loc[0, "patient_id"] == 101


def test_cpt_missing_file_raises(tmp_path):
    df = pd.DataFrame({"cpt": ["00500"]})
    with pytest.raises(FileNotFoundError):
        rollup_cpt_to_ccs(df, "cpt", str(tmp_path / "missing.csv"))


def test_cpt_multiple_codes_mixed_matches(tmp_path):
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["00500", "10021", "50000", "99213"]})
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert result.loc[0, "ccs_category"] == "CCS1"  # Anesthesia range
    assert result.loc[1, "ccs_category"] == "CCS2"  # Biopsy range
    assert pd.isna(result.loc[2, "ccs_category"])  # no match
    assert result.loc[3, "ccs_category"] == "CCS3"  # E&M range
