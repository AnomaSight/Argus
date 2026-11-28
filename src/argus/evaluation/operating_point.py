"""FPR and recall at a cut-off, with Wilson intervals (contract C4, task P1.3).

An image is flagged when score > tau, strictly. FPR is counted on label-0 rows and recall on
label-1 rows; which rows to pass is the caller's job (C1's role table).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


@dataclass(frozen=True)
class OperatingPoint:
    fpr: float
    fpr_ci: tuple[float, float]
    recall: float
    recall_ci: tuple[float, float]
    n_normal: int
    n_defect: int
    fp: int
    tp: int


def wilson_interval(x: int, n: int, ci_level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for x successes out of n. (nan, nan) when n = 0."""
    if n == 0:
        return (math.nan, math.nan)
    z = float(norm.ppf(1 - (1 - ci_level) / 2))
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _inputs(scores: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1 or s.shape != y.shape:
        raise ValueError("scores and labels must be 1-D and the same length")
    if not np.all(np.isfinite(s)):
        raise ValueError("every score must be finite")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be 0 or 1")
    return s, y.astype(np.int8)


def operating_point(
    scores: ArrayLike, labels: ArrayLike, tau: float, ci_level: float = 0.95
) -> OperatingPoint:
    s, y = _inputs(scores, labels)
    flagged = s > tau
    n_normal = int(np.sum(y == 0))
    n_defect = int(np.sum(y == 1))
    if n_normal == 0 and n_defect == 0:
        raise ValueError("no rows to evaluate")
    fp = int(np.sum(flagged & (y == 0)))
    tp = int(np.sum(flagged & (y == 1)))
    return OperatingPoint(
        fpr=fp / n_normal if n_normal else math.nan,
        fpr_ci=wilson_interval(fp, n_normal, ci_level),
        recall=tp / n_defect if n_defect else math.nan,
        recall_ci=wilson_interval(tp, n_defect, ci_level),
        n_normal=n_normal,
        n_defect=n_defect,
        fp=fp,
        tp=tp,
    )


def recall_by_defect_type(
    scores: ArrayLike,
    labels: ArrayLike,
    defect_types: ArrayLike,
    tau: float,
    ci_level: float = 0.95,
) -> dict[str, tuple[float, tuple[float, float], int]]:
    """Recall per defect type over label-1 rows: {type: (recall, ci, n)}."""
    s, y = _inputs(scores, labels)
    types = np.asarray(defect_types)
    out = {}
    for t in sorted(set(types[y == 1])):
        rows = (y == 1) & (types == t)
        n, tp = int(rows.sum()), int(np.sum(s[rows] > tau))
        out[str(t)] = (tp / n, wilson_interval(tp, n, ci_level), n)
    return out
