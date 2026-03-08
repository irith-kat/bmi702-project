import pytest
import pandas as pd
import numpy as np
from map import run_map

RA_IDS = [10001, 10002, 10003, 10004]
NON_RA_IDS = [20001, 20002, 20003, 20004]
ALL_IDS = RA_IDS + NON_RA_IDS


@pytest.fixture(scope="module")
def synthetic_data():
    np.random.seed(42)

    mat_df = pd.DataFrame(
        np.vstack(
            [
                np.random.poisson(5, size=(len(RA_IDS), 3)),
                np.random.poisson(2, size=(len(NON_RA_IDS), 3)),
            ]
        ),
        index=pd.Index(ALL_IDS, name="subject_id"),
        columns=["PheCode:714.1", "C0003873", "PheCode:714.0"],
    )

    note_df = pd.DataFrame(
        np.random.poisson([40] * len(RA_IDS) + [15] * len(NON_RA_IDS)),
        index=pd.Index(ALL_IDS, name="subject_id"),
        columns=["note_count"],
    )

    return mat_df, note_df


@pytest.fixture(scope="module")
def results(synthetic_data):
    mat_df, note_df = synthetic_data
    return run_map(mat_df, note_df, main_icd_col="PheCode:714.1")


def test_output_schema(results, synthetic_data):
    mat_df, _ = synthetic_data

    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == {"patient_id", "score", "phenotype"}
    assert len(results) == len(mat_df)
    assert not results.isnull().any().any()
    assert pd.api.types.is_float_dtype(results["score"])
    assert pd.api.types.is_integer_dtype(results["phenotype"])


def test_score_range(results):
    assert results["score"].between(0.0, 1.0).all()


def test_binary_labels(results):
    assert set(results["phenotype"]).issubset({0, 1})


def test_score_separation(results):
    scored = results.set_index("patient_id")
    assert scored.loc[RA_IDS, "score"].mean() > scored.loc[NON_RA_IDS, "score"].mean()


def test_class_separation(results):
    assert results["phenotype"].nunique() == 2
