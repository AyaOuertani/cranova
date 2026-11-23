import numpy as np
import pytest

from evaluate_test import PRIMARY
from reconstruction_engine import build_bundle, predict_case


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return build_bundle(tmp_path_factory.mktemp("models") / "bundle.joblib")


@pytest.fixture
def typical_case(bundle):
    return {code: float(bundle["ranges"][code].mean()) for code in PRIMARY}


def test_prediction_has_66_distances_inside_their_ranges(bundle, typical_case):
    result = predict_case(bundle, typical_case, "F", "EUR")
    table = result["distances"]
    assert len(table) == 66
    assert (table["low_90_mm"] < table["predicted_mm"]).all() and (table["predicted_mm"] < table["high_90_mm"]).all()
    assert result["landmarks"].shape == (12, 3) and (result["radius_90_mm"] > 0).all()


def test_no_warning_for_a_typical_case_and_a_warning_outside_the_range(bundle, typical_case):
    assert predict_case(bundle, typical_case, "M", "ASN")["warnings"] == []
    typical_case["GOL"] = 250.0
    warnings = predict_case(bundle, typical_case, "M", "ASN")["warnings"]
    assert len(warnings) == 1 and warnings[0].startswith("GOL")


def test_unknown_ancestry_is_refused(bundle, typical_case):
    with pytest.raises(ValueError):
        predict_case(bundle, typical_case, "F", "HIS")


def test_wider_ranges_for_the_group_with_larger_errors(bundle):
    assert np.mean(bundle["half_width"]["AFR_M"]) > np.mean(bundle["half_width"]["ASN_F"])