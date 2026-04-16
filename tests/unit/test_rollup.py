from pathlib import Path

import pandas as pd
import pytest

from rollup import (
    rollup_cpt_to_ccs,
    rollup_icd_to_phecode,
    rollup_ndc_to_ingredient,
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
    """Minimal CCS mapping CSV in AHRQ format: metadata title row + combined Code Range column."""
    p = tmp_path / "CCS_Services_Procedures_v2025-1.csv"
    p.write_text(
        '"CLINCAL CLASSIFICATIONS SOFTWARE (CCS) FOR SERVICES AND PROCEDURES, v2025.1"\n'
        "Code Range,CCS,CCS Label\n"
        "'00100-01999',1,Anesthesia\n"
        "'10021-10022',2,Biopsy\n"
        "'99201-99499',3,E&M visit\n"
    )
    return p


def _ndc_mapping_file(tmp_path: Path) -> Path:
    """Minimal NDC→RxNorm ingredient mapping CSV."""
    df = pd.DataFrame(
        {
            "ndc": ["00071015523", "00006007154"],
            "ingredient_id": ["321988", "723"],
            "ingredient_name": ["lisinopril", "aspirin"],
        }
    )
    p = tmp_path / "ndc_to_rxnorm_ingredient.csv"
    df.to_csv(p, index=False)
    return p


def _drug_name_mapping_file(tmp_path: Path) -> Path:
    """Minimal drug-name fallback mapping CSV."""
    df = pd.DataFrame(
        {
            "drug_name": ["metformin", "atorvastatin"],
            "ingredient_id": ["6809", "83367"],
            "ingredient_name": ["metformin", "atorvastatin"],
        }
    )
    p = tmp_path / "drug_name_to_rxnorm_ingredient.csv"
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


def test_icd_has_dots_true_skips_dot_insertion(tmp_path):
    """has_dots=True: pre-dotted codes (e.g. TriNetX/OMOP) match without modification."""
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["714.0"]})
    result = rollup_icd_to_phecode(df, "icd_code", str(p), has_dots=True)

    assert result.loc[0, "Phecode"] == "714.1"


def test_icd_has_dots_false_inserts_dot(tmp_path):
    """has_dots=False: codes without dots (MIMIC-IV raw format) get dot inserted."""
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["7140"]})  # no dot, as stored in MIMIC-IV
    result = rollup_icd_to_phecode(df, "icd_code", str(p), has_dots=False)

    assert result.loc[0, "Phecode"] == "714.1"


def test_icd_has_dots_none_autodetects_dotted(tmp_path):
    """has_dots=None: auto-detects dots present, skips insertion."""
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["714.0"]})
    result = rollup_icd_to_phecode(df, "icd_code", str(p), has_dots=None)

    assert result.loc[0, "Phecode"] == "714.1"


def test_icd_has_dots_none_autodetects_no_dot(tmp_path):
    """has_dots=None: auto-detects no dot present, inserts dot before matching."""
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["7140"]})  # no dot
    result = rollup_icd_to_phecode(df, "icd_code", str(p), has_dots=None)

    assert result.loc[0, "Phecode"] == "714.1"


def test_icd_has_dots_true_does_not_double_dot(tmp_path):
    """has_dots=True should not insert a second dot if one is already present."""
    p = _icd_mapping_file(tmp_path)
    df = pd.DataFrame({"icd_code": ["714.0"]})
    result = rollup_icd_to_phecode(df, "icd_code", str(p), has_dots=True)

    assert result.loc[0, "icd_code"] == "714.0"


# ---------------------------------------------------------------------------
# rollup_ndc_to_ingredient
# ---------------------------------------------------------------------------


def test_ndc_maps_known_code(tmp_path):
    ndc_file = _ndc_mapping_file(tmp_path)
    df = pd.DataFrame({"ndc": ["00071015523", "00006007154"]})
    result = rollup_ndc_to_ingredient(df, "ndc", ndc_mapping_file=str(ndc_file))

    assert result.loc[0, "rxnorm_ingredient_id"] == "321988"
    assert result.loc[0, "rxnorm_ingredient_name"] == "lisinopril"
    assert result.loc[1, "rxnorm_ingredient_id"] == "723"
    assert result.loc[1, "rxnorm_ingredient_name"] == "aspirin"


def test_ndc_unmatched_code_returns_nan(tmp_path):
    ndc_file = _ndc_mapping_file(tmp_path)
    df = pd.DataFrame({"ndc": ["99999999999"]})
    result = rollup_ndc_to_ingredient(df, "ndc", ndc_mapping_file=str(ndc_file))

    assert pd.isna(result.loc[0, "rxnorm_ingredient_id"])


def test_ndc_drug_name_fallback(tmp_path):
    ndc_file = _ndc_mapping_file(tmp_path)
    name_file = _drug_name_mapping_file(tmp_path)
    df = pd.DataFrame({"ndc": ["99999999999"], "drug": ["Metformin"]})
    result = rollup_ndc_to_ingredient(
        df,
        "ndc",
        drug_column="drug",
        ndc_mapping_file=str(ndc_file),
        drug_name_mapping_file=str(name_file),
    )

    assert result.loc[0, "rxnorm_ingredient_id"] == "6809"
    assert result.loc[0, "rxnorm_ingredient_name"] == "metformin"


def test_ndc_missing_file_raises(tmp_path):
    df = pd.DataFrame({"ndc": ["00071015523"]})
    with pytest.raises(FileNotFoundError):
        rollup_ndc_to_ingredient(
            df, "ndc", ndc_mapping_file=str(tmp_path / "missing.csv")
        )


def test_ndc_10digit_normalised_to_11digit(tmp_path):
    """10-digit NDC (missing leading zero) should be padded to 11 digits and match."""
    ndc_file = _ndc_mapping_file(tmp_path)
    df = pd.DataFrame(
        {"ndc": ["0071015523"]}
    )  # 10 digits; mapping key is "00071015523"
    result = rollup_ndc_to_ingredient(df, "ndc", ndc_mapping_file=str(ndc_file))

    assert result.loc[0, "rxnorm_ingredient_id"] == "321988"
    assert result.loc[0, "rxnorm_ingredient_name"] == "lisinopril"


def test_ndc_hyphenated_normalised(tmp_path):
    """Hyphenated NDC should have hyphens stripped and match the 11-digit key."""
    ndc_file = _ndc_mapping_file(tmp_path)
    df = pd.DataFrame({"ndc": ["00071-0155-23"]})  # hyphens; maps to "00071015523"
    result = rollup_ndc_to_ingredient(df, "ndc", ndc_mapping_file=str(ndc_file))

    assert result.loc[0, "rxnorm_ingredient_name"] == "lisinopril"


def test_ndc_normalisation_does_not_mutate_caller(tmp_path):
    """The caller's DataFrame should not be modified by NDC normalisation."""
    ndc_file = _ndc_mapping_file(tmp_path)
    df = pd.DataFrame({"ndc": ["0071015523"]})
    original_value = df.loc[0, "ndc"]
    rollup_ndc_to_ingredient(df, "ndc", ndc_mapping_file=str(ndc_file))

    assert df.loc[0, "ndc"] == original_value


def test_ndc_preserves_original_columns(tmp_path):
    ndc_file = _ndc_mapping_file(tmp_path)
    df = pd.DataFrame({"ndc": ["00071015523"], "patient_id": [42]})
    result = rollup_ndc_to_ingredient(df, "ndc", ndc_mapping_file=str(ndc_file))

    assert "patient_id" in result.columns
    assert result.loc[0, "patient_id"] == 42


# ---------------------------------------------------------------------------
# rollup_cpt_to_ccs
# ---------------------------------------------------------------------------


def test_cpt_matches_code_in_range(tmp_path):
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["00500"]})
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert result.loc[0, "ccs_category"] == 1
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

    assert result.loc[0, "ccs_category"] == 1


def test_cpt_uppercases_input(tmp_path):
    """Lowercase input should be uppercased before range matching."""
    p = _ccs_mapping_file(tmp_path)
    df = pd.DataFrame({"cpt": ["00500"]})
    result = rollup_cpt_to_ccs(df, "cpt", str(p))

    assert result.loc[0, "ccs_category"] == 1


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

    assert result.loc[0, "ccs_category"] == 1  # Anesthesia range
    assert result.loc[1, "ccs_category"] == 2  # Biopsy range
    assert pd.isna(result.loc[2, "ccs_category"])  # no match
    assert result.loc[3, "ccs_category"] == 3  # E&M range
