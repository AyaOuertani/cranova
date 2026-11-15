"""Tests that the raw data and the cleaning behave as documented. Run: python -m pytest -q"""
import pandas as pd
import pytest

from build_clean_data import PRIMARY_CRANIAL, UNDOCUMENTED_TD, build_subject_registry, clean_tissue_depth
from load_data import ID_FIXES, load_cranial, load_facial, load_tissue_depth


@pytest.fixture(scope="module")
def data():
    return load_cranial(), load_facial("k"), load_facial("r"), load_tissue_depth()


def test_subject_counts(data):
    cranial, facial_k, facial_r, tissue = data
    assert (len(cranial), len(facial_k), len(facial_r), len(tissue)) == (287, 388, 388, 388)


def test_no_duplicate_ids(data):
    for table in data:
        assert not table.index.duplicated().any()


def test_id_typo_is_fixed(data):
    tissue = data[3]
    assert "asn012" in tissue.index and "ans012" not in tissue.index and "ans012" in ID_FIXES


def test_no_summary_rows_in_tissue_depth(data):
    assert all(i[:3] in {"afr", "asn", "cau", "his"} for i in data[3].index)


def test_facial_versions_share_subjects_and_distances(data):
    _, facial_k, facial_r, _ = data
    assert set(facial_k.index) == set(facial_r.index)
    assert list(facial_k.columns) == list(facial_r.columns)
    assert facial_k.shape[1] == 67  # sex + 66 distances


def test_matching_counts(data):
    cranial, facial_k, _, tissue = data
    registry = build_subject_registry(cranial, facial_k, tissue)
    assert (registry["status"] == "modelling set").sum() == 274
    assert (registry["status"] == "modelling set")[registry["in_tissue_depth"]].sum() == 271


def test_sex_agrees_between_cranial_and_facial(data):
    cranial, facial_k, _, _ = data
    both = sorted(set(cranial.index) & set(facial_k.index))
    assert (cranial.loc[both, "sex"] == facial_k.loc[both, "sex"]).all()


def test_obbl_missing_only_for_asnm(data):
    cranial = data[0]
    missing = cranial[cranial["OBBL"].isna()]
    assert len(missing) == 47 and set(missing["ancestry"]) == {"ASN"} and set(missing["sex"]) == {"M"}
    assert cranial[PRIMARY_CRANIAL].notna().all().all()


def test_split_is_frozen_and_clean(data):
    split = pd.read_csv("splits/split_v1.csv", index_col="id")
    cranial, facial_k, _, _ = data
    assert set(split.index) == set(cranial.index) & set(facial_k.index)
    assert (split["role"] == "test").sum() == 55 and (split["role"] == "dev").sum() == 219


def test_tissue_depth_cleaning(data):
    cleaned, log = clean_tissue_depth(data[3])
    assert not set(UNDOCUMENTED_TD) & set(cleaned.columns) and cleaned.shape[1] == 36
    assert pd.isna(cleaned.loc["his021", "mmb R"])      # entry error removed
    assert not pd.isna(cleaned.loc["asn029", "sc L"])   # extreme but plausible: kept
    assert list(log["id"]) == ["his021"]