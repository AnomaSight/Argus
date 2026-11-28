import numpy as np
import pytest

from argus.memory.aso import aso
from argus.memory.greedy_coreset import greedy_coreset


@pytest.mark.parametrize("p", [0.25, 0.5, 1.0])
def test_synthetic_curve_gives_known_optimum(p):
    # d(k) = (k10 / k)^p, so d~ = x^-p with x = k / k10. Minimising x^2 + x^(-2p) gives
    # x* = p^(1 / (2p + 2)).
    k10 = 1000
    k = np.arange(1, k10 + 1)
    r = aso((k10 / k) ** p, n_total=10 * k10)
    expected = p ** (1 / (2 * p + 2)) * k10
    assert abs(r.k - expected) <= 1
    assert r.fraction == pytest.approx(r.k / (10 * k10))


def test_without_normalisation_it_collapses_towards_zero():
    # Real covering radii are small feature distances (here ~3 to ~17) while k runs to
    # thousands, so raw k dominates and the rule picks a tiny prefix (the ASO paper's point).
    k10 = 1000
    k = np.arange(1, k10 + 1)
    d = 3 * (k10 / k) ** 0.25
    raw, norm = aso(d, n_total=10 * k10, normalise=False), aso(d, n_total=10 * k10)
    assert raw.k / k10 < 0.02
    assert norm.k / k10 > 0.5


def test_on_a_real_greedy_curve_picks_inside_the_range():
    x = np.random.default_rng(0).standard_normal((3000, 64), dtype=np.float32)
    r = aso(greedy_coreset(x, 300, seed=1).d, n_total=3000)
    assert 1 < r.k < 300 and r.dist[r.k - 1] == r.dist.min()


def test_save_curve(tmp_path):
    r = aso(np.linspace(2, 1, 10), n_total=100)
    lines = r.save_curve(tmp_path / "curve.csv").read_text().splitlines()
    assert lines[0] == "k,k_norm,d_norm,dist,chosen" and len(lines) == 11
    assert sum(line.endswith(",1") for line in lines[1:]) == 1


@pytest.mark.parametrize("bad", [[], [1.0, np.nan], [1.0, 0.0]])
def test_rejects_bad_curve(bad):
    with pytest.raises(ValueError):
        aso(bad, n_total=10)
