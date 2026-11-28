"""Localisation metrics from heatmaps and defect masks (task P2.1): pixel AUROC and PRO.

Heatmaps and masks must already be in the same frame (C2: masks go through the run's
preset with nearest resampling). Good images count too: their masks are all False.

PRO follows the MVTec AD reference (Bergmann et al.): each connected defect region
(8-connectivity) weighs the same, the per-region overlap curve is plotted against pixel
FPR, integrated up to `fpr_limit` and divided by it, so a perfect map scores 1.0.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def _stack(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    # ponytail: holds every test pixel (float16 + bool, 3 bytes each, ~17 MB for a category
    # of 110 images at 224 x 224); bin scores into a histogram if datasets get much bigger.
    maps, masks = [], []
    for heatmap, mask in pairs:
        h, m = np.asarray(heatmap), np.asarray(mask, dtype=bool)
        if h.ndim != 2 or h.shape != m.shape:
            raise ValueError(f"heatmap {h.shape} and mask {m.shape} must be 2-D and equal")
        if not np.all(np.isfinite(h)):
            raise ValueError("every heatmap value must be finite")
        maps.append(h.astype(np.float16, copy=False))
        masks.append(m)
    if not maps:
        raise ValueError("no heatmaps given")
    return np.stack(maps), np.stack(masks)


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """AUROC over every pixel of every image (1 = defect pixel)."""
    y = masks.ravel()
    if y.all() or not y.any():
        raise ValueError("pixel AUROC needs both defect and good pixels")
    return float(roc_auc_score(y, maps.ravel().astype(np.float32)))


def pro_curve(maps: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pixel FPR, mean per-region overlap) at every distinct threshold, from (0, 0)."""
    labels = np.zeros(masks.shape, dtype=np.int64)
    n_regions = 0
    for i, m in enumerate(masks):
        lab, n = ndimage.label(m, structure=EIGHT_CONNECTED)
        labels[i] = np.where(lab > 0, lab + n_regions, 0)
        n_regions += n
    n_ok = int((~masks).sum())
    if n_regions == 0 or n_ok == 0:
        raise ValueError("PRO needs at least one defect region and one good pixel")

    labels = labels.ravel()
    sizes = np.bincount(labels)
    # Each pixel's step in FPR and in mean region overlap once the threshold passes it.
    fp_step = np.where(labels == 0, 1.0 / n_ok, 0.0)
    pro_step = np.where(labels > 0, 1.0 / (n_regions * sizes[labels]), 0.0)

    scores = maps.ravel()
    order = np.argsort(scores, kind="stable")[::-1]
    fpr, pro = np.cumsum(fp_step[order]), np.cumsum(pro_step[order])
    s = scores[order]
    last_of_tie = np.r_[s[1:] != s[:-1], True]  # flag score >= t: take whole tie groups
    fpr = np.clip(np.r_[0.0, fpr[last_of_tie]], 0.0, 1.0)
    pro = np.clip(np.r_[0.0, pro[last_of_tie]], 0.0, 1.0)
    return fpr, pro


def area_to(fpr: np.ndarray, y: np.ndarray, fpr_limit: float) -> float:
    """Area under y(fpr) from 0 to fpr_limit, interpolating the end point, over fpr_limit."""
    if not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit must be in (0, 1]")
    i = np.searchsorted(fpr, fpr_limit, side="right")
    x = np.r_[fpr[:i], fpr_limit]
    y = np.r_[y[:i], np.interp(fpr_limit, fpr, y)]
    return float(np.trapezoid(y, x) / fpr_limit)


def pro_score(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3) -> float:
    return area_to(*pro_curve(maps, masks), fpr_limit)


def pixel_metrics(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]], fpr_limit: float = 0.3
) -> tuple[float, float]:
    """(pixel AUROC, PRO up to fpr_limit) over (heatmap, bool mask) pairs."""
    maps, masks = _stack(pairs)
    return pixel_auroc(maps, masks), pro_score(maps, masks, fpr_limit)
