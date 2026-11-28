"""Threshold-free ranking metric (task P1.4): image AUROC, for comparison with the papers.

AUROC is the chance a random defect image scores above a random good one. It can stay flat
while FPR or recall at the calibrated cut-off moves, which is part of what this study shows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score


def image_auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """AUROC of scores against labels (1 = defect). Ties count as half, as in scikit-learn."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError("scores and labels must be 1-D and the same length")
    if not np.all(np.isfinite(s)):
        raise ValueError("every score must be finite")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be 0 or 1")
    if not (np.any(y == 0) and np.any(y == 1)):
        raise ValueError("AUROC needs at least one good and one defect image")
    return float(roc_auc_score(y, s))
