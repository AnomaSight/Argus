"""Checks configs/datasets/*.yaml against the split rules from task 0.4 and contract C1."""

import math
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "datasets"
DATASETS = ["mvtec_ad", "isp_ad"]
ROLES = ["support", "calib", "eval_normal"]
DELTA = 0.05
MIN_EVAL_NORMALS = 100


def load(name):
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def min_calibration_size(alpha, delta=DELTA):
    # Same formula as contract C4 (P1.2 implements it in argus.calibration.sample_size).
    return math.ceil(math.log(delta) / math.log(1 - alpha))


def categories(name):
    return list(load(name)["categories"].items())


@pytest.mark.parametrize("name", DATASETS)
def test_top_level_keys(name):
    cfg = load(name)
    assert cfg["dataset"] == name
    assert cfg["root"] == name
    assert cfg["preprocess"] in {"patchcore", "deng"}
    assert cfg["categories"]


def test_seeds_are_three_distinct_and_shared():
    seeds = {name: load(name)["seeds"] for name in DATASETS}
    for s in seeds.values():
        assert len(s) == 3 and len(set(s)) == 3
        assert all(isinstance(x, int) for x in s)
    assert seeds["mvtec_ad"] == seeds["isp_ad"]


@pytest.mark.parametrize("name", DATASETS)
def test_split_counts_are_positive_ints(name):
    for cat, c in categories(name):
        assert list(c["split"]) == ROLES, cat
        for role in ROLES:
            n = c["split"][role]
            assert isinstance(n, int) and n > 0, (cat, role)


@pytest.mark.parametrize("name", DATASETS)
def test_calib_supports_5_percent_target(name):
    for cat, c in categories(name):
        assert c["split"]["calib"] >= min_calibration_size(0.05), cat


@pytest.mark.parametrize(("cat", "c"), categories("mvtec_ad"))
def test_mvtec_split_uses_exactly_the_training_images(cat, c):
    assert sum(c["split"].values()) == c["expected_counts"]["train_good"]


@pytest.mark.parametrize(("cat", "c"), categories("mvtec_ad"))
def test_mvtec_has_enough_evaluation_normals(cat, c):
    # FPR uses held-out train goods plus the original test goods (C1).
    assert c["split"]["eval_normal"] + c["expected_counts"]["test_good"] >= MIN_EVAL_NORMALS


@pytest.mark.parametrize(("cat", "c"), categories("isp_ad"))
def test_isp_ad_matches_deng_and_fits_available_normals(cat, c):
    assert c["split"] == {"support": 300, "calib": 150, "eval_normal": 200}
    assert c["split"]["eval_normal"] >= MIN_EVAL_NORMALS
    # C1: the normal pool is train/good + test/good (LSM_2 has only 401 train goods).
    good = c["expected_counts"]["train_good"] + c["expected_counts"]["test_good"]
    assert sum(c["split"].values()) <= good


@pytest.mark.parametrize("name", DATASETS)
def test_expected_counts_are_ints(name):
    for cat, c in categories(name):
        assert c["expected_counts"], cat
        for field, n in c["expected_counts"].items():
            assert isinstance(n, int) and not isinstance(n, bool) and n >= 0, (cat, field)


def test_min_calibration_size_reference_values():
    assert [min_calibration_size(a) for a in (0.10, 0.05, 0.02)] == [29, 59, 149]
