"""How many calibration scores a false-positive target needs (contract C4, task P1.2).

The Wilks rule picks the k-th smallest of m calibration scores as tau. Rank k gives
confidence P(Binomial(m, 1 - alpha) <= k - 1) that the true FPR is at most alpha. The best
any m can do is k = m, which is where the closed forms below come from.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def _check_unit(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")


def min_calibration_size(alpha: float, delta: float = 0.05) -> int:
    """Smallest m for which rank k = m reaches confidence 1 - delta."""
    _check_unit("alpha", alpha)
    _check_unit("delta", delta)
    return math.ceil(math.log(delta) / math.log(1 - alpha))


def min_alpha(m: int, delta: float = 0.05) -> float:
    """Smallest FPR target that m calibration scores can support at confidence 1 - delta."""
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    _check_unit("delta", delta)
    return 1 - delta ** (1 / m)


def wilks_rank(m: int, alpha: float, delta: float = 0.05) -> tuple[int, float] | None:
    """Smallest 1-based rank k with P(Binomial(m, 1 - alpha) <= k - 1) >= 1 - delta.

    Returns (k, achieved_confidence), or None when even k = m falls short.
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    _check_unit("alpha", alpha)
    _check_unit("delta", delta)
    k = np.arange(1, m + 1)
    confidence = binom.cdf(k - 1, m, 1 - alpha)
    ok = np.flatnonzero(confidence >= 1 - delta)
    if ok.size == 0:
        return None
    i = int(ok[0])
    return int(k[i]), float(confidence[i])


def planning_table(alphas: list[float], m: int, delta: float = 0.05) -> list[dict]:
    """One row per target, in Deng's Table 6 layout."""
    rows = []
    for alpha in alphas:
        hit = wilks_rank(m, alpha, delta)
        rows.append(
            {
                "alpha": alpha,
                "rank": hit[0] if hit else None,
                "achieved_confidence": hit[1] if hit else None,
                "min_m": min_calibration_size(alpha, delta),
                "supported": hit is not None,
            }
        )
    return rows
