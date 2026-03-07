import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from komap import run_komap

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="Rscript not available"
)


@pytest.fixture
def fake_ehr():
    """
    Minimal fake EHR matching KOMAP's fake_ehr format.
    Codes are already rolled up (ICD → PheCode) since our pipeline
    does rollup via mapping.py before calling run_komap.
    """
    rng = np.random.default_rng(42)
    codes = ["PheCode:250", "PheCode:401", "LAB-LOINC:1742-6", "LAB-LOINC:2532-0"]
    rows = [
        {
            "patient_num": pid,
            "days_since_admission": int(day),
            "concept_type": code.split(":")[0],
            "concept_code": code,
        }
        for pid in range(1, 31)
        for code in codes
        if rng.random() > 0.3
        for day in rng.choice(30, rng.integers(1, 4), replace=False)
    ]
    return pd.DataFrame(rows)


def test_komap_out(fake_ehr, tmp_path):
    paths = run_komap(
        ehr_data=fake_ehr,
        main_surrogates="PheCode:250",
        target_code="PheCode:250",
        nm_disease="T2DM",
        nm_corrupt_code="corrupt_PheCode:250",
        output_dir=str(tmp_path / "komap_out"),
    )

    for key in (
        "train_cov",
        "valid_cov",
        "coefficients",
        "pred_score",
        "pred_prob",
        "pred_cluster",
    ):
        assert key in paths, f"Missing output: {key}"
        assert Path(paths[key]).exists(), f"File not written: {paths[key]}"

    n_patients = fake_ehr["patient_num"].nunique()

    pred_score = pd.read_csv(paths["pred_score"])
    pred_prob = pd.read_csv(paths["pred_prob"])
    pred_cluster = pd.read_csv(paths["pred_cluster"])
    coefficients = pd.read_csv(paths["coefficients"])

    # All prediction frames have one row per patient
    for df in (pred_score, pred_prob, pred_cluster):
        assert "patient_num" in df.columns
        assert len(df) == n_patients

    # KOMAP produces at least one method column beyond patient_num
    method_cols = [c for c in pred_prob.columns if c != "patient_num"]
    assert len(method_cols) >= 1

    # Probabilities are in [0, 1]
    assert ((pred_prob[method_cols] >= 0) & (pred_prob[method_cols] <= 1)).all().all()

    # Cluster labels are exactly "disease" or "no disease"
    assert set(pred_cluster[method_cols].stack().unique()) <= {"disease", "no disease"}

    # Coefficients table has expected columns and at least one row
    assert {"disease", "method", "feat", "coeff"}.issubset(coefficients.columns)
    assert len(coefficients) > 0

    # Covariance matrices: square, symmetric, non-negative diagonal, required features present
    for key in ("train_cov", "valid_cov"):
        cov = pd.read_csv(paths[key], index_col=0)
        assert cov.shape[0] == cov.shape[1], f"{key} is not square"
        assert "utl" in cov.index, f"{key} missing utl row"
        assert "corrupt_PheCode:250" in cov.index, f"{key} missing surrogate row"
        assert (np.diag(cov.values) >= 0).all(), (
            f"{key} has negative diagonal (variance)"
        )
        np.testing.assert_allclose(
            cov.values, cov.values.T, rtol=1e-5, err_msg=f"{key} is not symmetric"
        )
