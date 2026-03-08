"""
Integration test: MIMIC-IV-like EHR → prepare_map_inputs → run_map.

diagnoses_icd: (subject_id, hadm_id, seq_num, icd_code, icd_version)
discharge:     (subject_id, hadm_id, note_id, text)
"""

import pandas as pd
import pytest
from unittest.mock import patch
import preprocessing
from rollup import rollup_icd_to_phecode
from map import run_map

RA_IDS = [10001, 10002, 10003, 10004, 10005, 10006]
NON_RA_IDS = [20001, 20002, 20003, 20004, 20005, 20006]

# RA patients have 7-12 M059 codes across many admissions (needed for MAP
# mixture models to converge); non-RA patients have only HTN/DM codes.
_DIAGNOSES = [
    # RA patients
    (10001, 1001, 1, "M059", 10),
    (10001, 1001, 2, "I10", 10),
    (10001, 1002, 1, "M059", 10),
    (10001, 1003, 1, "M059", 10),
    (10001, 1004, 1, "M059", 10),
    (10001, 1004, 2, "M060", 10),
    (10001, 1005, 1, "M059", 10),
    (10001, 1006, 1, "M059", 10),
    (10001, 1007, 1, "M059", 10),
    (10001, 1007, 2, "I10", 10),
    (10001, 1008, 1, "M059", 10),
    (10001, 1009, 1, "M059", 10),
    (10001, 1010, 1, "M059", 10),
    (10001, 1010, 2, "M060", 10),
    (10002, 1011, 1, "M059", 10),
    (10002, 1011, 2, "M060", 10),
    (10002, 1012, 1, "M059", 10),
    (10002, 1013, 1, "M059", 10),
    (10002, 1014, 1, "M059", 10),
    (10002, 1014, 2, "E119", 10),
    (10002, 1015, 1, "M059", 10),
    (10002, 1016, 1, "M059", 10),
    (10002, 1017, 1, "M059", 10),
    (10002, 1018, 1, "M059", 10),
    (10002, 1019, 1, "M059", 10),
    (10002, 1019, 2, "M060", 10),
    (10003, 1020, 1, "M059", 10),
    (10003, 1021, 1, "M059", 10),
    (10003, 1022, 1, "M059", 10),
    (10003, 1022, 2, "I10", 10),
    (10003, 1023, 1, "M059", 10),
    (10003, 1024, 1, "M059", 10),
    (10003, 1025, 1, "M059", 10),
    (10003, 1026, 1, "M059", 10),
    (10003, 1027, 1, "M059", 10),
    (10004, 1030, 1, "M059", 10),
    (10004, 1031, 1, "M059", 10),
    (10004, 1032, 1, "M059", 10),
    (10004, 1032, 2, "E119", 10),
    (10004, 1033, 1, "M059", 10),
    (10004, 1034, 1, "M059", 10),
    (10004, 1035, 1, "M059", 10),
    (10004, 1035, 2, "M060", 10),
    (10004, 1036, 1, "M059", 10),
    (10004, 1037, 1, "M059", 10),
    (10005, 1040, 1, "M059", 10),
    (10005, 1040, 2, "M060", 10),
    (10005, 1041, 1, "M059", 10),
    (10005, 1042, 1, "M059", 10),
    (10005, 1043, 1, "M059", 10),
    (10005, 1044, 1, "M059", 10),
    (10005, 1045, 1, "M059", 10),
    (10005, 1045, 2, "I10", 10),
    (10005, 1046, 1, "M059", 10),
    (10005, 1047, 1, "M059", 10),
    (10005, 1048, 1, "M059", 10),
    (10005, 1049, 1, "M059", 10),
    (10006, 1050, 1, "M059", 10),
    (10006, 1051, 1, "M059", 10),
    (10006, 1052, 1, "M059", 10),
    (10006, 1052, 2, "E119", 10),
    (10006, 1053, 1, "M059", 10),
    (10006, 1054, 1, "M059", 10),
    (10006, 1055, 1, "M059", 10),
    (10006, 1056, 1, "M059", 10),
    # Non-RA patients (20003 has one stray M059 — miscoding)
    (20001, 2001, 1, "I10", 10),
    (20001, 2001, 2, "E119", 10),
    (20001, 2002, 1, "I10", 10),
    (20002, 2003, 1, "I10", 10),
    (20002, 2004, 1, "I10", 10),
    (20003, 2005, 1, "M059", 10),
    (20003, 2005, 2, "I10", 10),
    (20004, 2006, 1, "E119", 10),
    (20004, 2007, 1, "I10", 10),
    (20005, 2008, 1, "I10", 10),
    (20005, 2009, 1, "E119", 10),
    (20006, 2010, 1, "E119", 10),
    (20006, 2010, 2, "I10", 10),
]

_NOTES = [
    # RA — many notes per patient
    (10001, 1001, "n01", "Rheumatoid arthritis. Methotrexate continued."),
    (10001, 1002, "n02", "RA flare. ESR elevated. Rheumatology follow-up."),
    (10001, 1003, "n03", "RA management. Joint swelling bilateral."),
    (10001, 1004, "n04", "Stable RA. No new erosions."),
    (10001, 1005, "n05", "RA follow-up. DMARD adjusted."),
    (10001, 1006, "n06", "Seropositive RA. RF and anti-CCP positive."),
    (10002, 1011, "n07", "RA active disease. DAS28 elevated."),
    (10002, 1012, "n08", "RA follow-up. Morning stiffness >1hr."),
    (10002, 1013, "n09", "Biologic therapy for RA considered."),
    (10002, 1014, "n10", "RA with DM2. Complex medication management."),
    (10002, 1015, "n11", "RA stable on current regimen."),
    (10003, 1020, "n12", "RA diagnosis confirmed. Starting hydroxychloroquine."),
    (10003, 1021, "n13", "RA follow-up. Good response to MTX."),
    (10003, 1022, "n14", "RA management. Mild synovitis."),
    (10003, 1023, "n15", "RA stable. No constitutional symptoms."),
    (10004, 1030, "n16", "Seropositive RA. Rituximab infusion today."),
    (10004, 1031, "n17", "Post-infusion monitoring. RA ongoing."),
    (10004, 1032, "n18", "RA with DM. Endocrine co-management."),
    (10004, 1033, "n19", "RA follow-up. Joint count improving."),
    (10005, 1040, "n20", "Rheumatoid arthritis. Abatacept started."),
    (10005, 1041, "n21", "RA flare. Prednisone taper prescribed."),
    (10005, 1042, "n22", "RA follow-up. CRP normalizing."),
    (10005, 1043, "n23", "RA stable. Continued DMARD."),
    (10005, 1044, "n24", "RA management. Physical therapy referral."),
    (10006, 1050, "n25", "RA confirmed. Leflunomide initiated."),
    (10006, 1051, "n26", "RA follow-up. Tolerating medication well."),
    (10006, 1052, "n27", "RA management. Hand grip improving."),
    # Non-RA — few notes
    (20001, 2001, "n28", "Hypertension follow-up. BP controlled."),
    (20002, 2003, "n29", "HTN. Dietary counseling."),
    (20003, 2005, "n30", "Joint pain. Likely osteoarthritis. No inflammatory markers."),
    (20004, 2006, "n31", "Type 2 diabetes. Metformin continued."),
    (20005, 2008, "n32", "HTN visit. No acute complaints."),
    (20006, 2010, "n33", "DM2 follow-up. Foot exam normal."),
]


@pytest.fixture(scope="module")
def phecode_mapping(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mapping")
    pd.DataFrame(
        {
            "ICD": ["M059", "M060", "I10", "E119"],
            "Phecode": ["714.1", "714.1", "401.1", "250.2"],
            "PhecodeString": [
                "Rheumatoid arthritis",
                "Rheumatoid arthritis",
                "Hypertension",
                "Type 2 diabetes",
            ],
        }
    ).to_csv(tmp / "Phecode_map_v1_2_icd9_icd10cm.csv", index=False)
    return str(tmp / "Phecode_map_v1_2_icd9_icd10cm.csv")


@pytest.fixture(scope="module")
def ehr_df():
    return pd.DataFrame(
        _DIAGNOSES,
        columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )


@pytest.fixture(scope="module")
def notes_df():
    return pd.DataFrame(_NOTES, columns=["subject_id", "hadm_id", "note_id", "text"])


@pytest.fixture(scope="module")
def pipeline_outputs(ehr_df, notes_df, phecode_mapping):
    # patch rollup_icd_to_phecode to use the test mapping file
    with patch.object(
        preprocessing,
        "rollup_icd_to_phecode",
        side_effect=lambda df, col: rollup_icd_to_phecode(df, col, phecode_mapping),
    ):
        mat_df, note_df = preprocessing.prepare_map_inputs(
            ehr_df, notes_df, icd_col="icd_code"
        )

    results = run_map(mat_df, note_df, main_icd_col="714.1")
    return mat_df, note_df, results


def test_mat_has_ra_phecode(pipeline_outputs):
    mat_df, _, _ = pipeline_outputs
    assert "714.1" in mat_df.columns


def test_note_df_covers_all_patients(pipeline_outputs):
    _, note_df, _ = pipeline_outputs
    assert set(RA_IDS + NON_RA_IDS).issubset(set(note_df.index))


def test_map_output_schema(pipeline_outputs):
    _, _, results = pipeline_outputs
    assert set(results.columns) == {"patient_id", "score", "phenotype"}
    assert len(results) == len(RA_IDS) + len(NON_RA_IDS)
    assert not results.isnull().any().any()


def test_ra_patients_score_higher(pipeline_outputs):
    _, _, results = pipeline_outputs
    scored = results.set_index("patient_id")
    assert scored.loc[RA_IDS, "score"].mean() > scored.loc[NON_RA_IDS, "score"].mean()


def test_ra_patients_mostly_phenotype_positive(pipeline_outputs):
    _, _, results = pipeline_outputs
    scored = results.set_index("patient_id")
    assert scored.loc[RA_IDS, "phenotype"].mean() >= 0.5
