from vocab import get_code_definition


def test_icd_def():
    """ICD10CM code E11.9 should resolve to its human-readable name."""
    result = get_code_definition("E11.9", "ICD10CM")
    assert result == "Type 2 diabetes mellitus without complications"


def test_rxnorm_def():
    """RXNORM code 105585 should resolve to its human-readable name."""
    result = get_code_definition("105585", "RXNORM")
    assert result == "methotrexate 2.5 MG Oral Tablet"


def test_loinc_def():
    """LOINC code 4548-4 should resolve to its human-readable name."""
    result = get_code_definition("4548-4", "LNC")
    assert result == "Hemoglobin A1c/Hemoglobin.total in Blood"
