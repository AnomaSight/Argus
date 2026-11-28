import math

import numpy as np
import pytest
from scipy.optimize import brentq

from argus.evaluation.operating_point import (
    operating_point,
    recall_by_defect_type,
    wilson_interval,
)


def test_hand_made_counts():
    scores = [0.1, 0.5, 0.9, 1.0, 2.0, 0.4, 1.5, 3.0]
    labels = [0, 0, 0, 0, 0, 1, 1, 1]
    op = operating_point(scores, labels, tau=0.9)
    # Strict rule: 0.9 is not flagged. Normals above: 1.0, 2.0. Defects above: 1.5, 3.0.
    assert (op.n_normal, op.n_defect, op.fp, op.tp) == (5, 3, 2, 2)
    assert op.fpr == pytest.approx(2 / 5)
    assert op.recall == pytest.approx(2 / 3)


@pytest.mark.parametrize("x, n", [(1, 10), (7, 131), (63, 200), (3, 41)])
def test_wilson_matches_its_definition(x, n):
    # The Wilson interval is the set of p with |x/n - p| <= z * sqrt(p(1 - p) / n).
    # Solve that boundary numerically and compare, independently of the closed form.
    z, ph = 1.959963984540054, x / n

    def f(p):
        return (ph - p) ** 2 - z * z * p * (1 - p) / n

    lo, hi = wilson_interval(x, n)
    assert lo == pytest.approx(brentq(f, 1e-12, ph), abs=1e-9)
    assert hi == pytest.approx(brentq(f, ph, 1 - 1e-12), abs=1e-9)


def test_wilson_edges():
    # 0 of n has upper z²/(n+z²); n of n has lower n/(n+z²).
    z2 = 1.959963984540054**2
    assert wilson_interval(0, 10) == (0.0, pytest.approx(z2 / (10 + z2), abs=1e-12))
    assert wilson_interval(10, 10)[0] == pytest.approx(10 / (10 + z2), abs=1e-12)


def test_wilson_is_symmetric():
    lo, hi = wilson_interval(7, 131)
    lo2, hi2 = wilson_interval(131 - 7, 131)
    assert lo == pytest.approx(1 - hi2, abs=1e-12)
    assert hi == pytest.approx(1 - lo2, abs=1e-12)


def test_one_class_only_gives_nan_for_the_other():
    op = operating_point([0.1, 0.2], [0, 0], tau=0.15)
    assert op.fpr == 0.5 and math.isnan(op.recall) and math.isnan(op.recall_ci[0])


@pytest.mark.parametrize(
    "scores, labels",
    [([], []), ([1.0], [2]), ([np.nan], [0]), ([1.0, 2.0], [0]), ([[1.0]], [[0]])],
)
def test_rejects_bad_input(scores, labels):
    with pytest.raises(ValueError):
        operating_point(scores, labels, 0.5)


def test_recall_by_defect_type():
    out = recall_by_defect_type(
        [5, 1, 5, 5, 0], [1, 1, 1, 1, 0], ["crack", "crack", "cut", "cut", "good"], tau=2
    )
    assert out["crack"][0] == 0.5 and out["crack"][2] == 2
    assert out["cut"][0] == 1.0
    assert "good" not in out
