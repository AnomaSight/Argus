import math

import pytest

from argus.calibration.sample_size import (
    min_alpha,
    min_calibration_size,
    planning_table,
    wilks_rank,
)


@pytest.mark.parametrize(
    "alpha, expected",
    [(0.10, 29), (0.05, 59), (0.02, 149), (0.01, 299), (0.005, 598), (0.001, 2995)],
)
def test_min_calibration_size_matches_c4(alpha, expected):
    assert min_calibration_size(alpha, 0.05) == expected


def test_min_alpha_matches_c4():
    assert min_alpha(150, 0.05) == pytest.approx(0.01977, abs=5e-6)


@pytest.mark.parametrize("alpha", [0.10, 0.05, 0.02, 0.01, 0.005])
def test_closed_form_agrees_with_binomial_rank(alpha):
    # min_calibration_size is exactly where wilks_rank starts to succeed.
    m = min_calibration_size(alpha, 0.05)
    assert wilks_rank(m, alpha, 0.05) is not None
    assert wilks_rank(m - 1, alpha, 0.05) is None


def test_min_alpha_is_the_supported_boundary():
    m = 150
    a = min_alpha(m, 0.05)
    assert wilks_rank(m, a * 1.001, 0.05) is not None
    assert wilks_rank(m, a * 0.999, 0.05) is None


def test_planning_table_deng_table_6():
    rows = planning_table([0.10, 0.05, 0.02, 0.01], m=150)
    got = [(r["rank"], r["supported"]) for r in rows]
    assert got == [(142, True), (148, True), (150, True), (None, False)]
    assert rows[0]["achieved_confidence"] == pytest.approx(0.9693, abs=5e-5)
    assert rows[3]["min_m"] == 299


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, math.nan])
def test_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        min_calibration_size(bad)
    with pytest.raises(ValueError):
        min_calibration_size(0.05, bad)
