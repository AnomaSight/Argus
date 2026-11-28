import numpy as np
import pytest
from scipy import ndimage

from argus.evaluation.pixel_metrics import (
    EIGHT_CONNECTED,
    area_to,
    pixel_metrics,
    pro_curve,
)


def _masks(rng, n=6, size=32):
    """Mix of good images and images with one or two square defects of different sizes."""
    out = np.zeros((n, size, size), dtype=bool)
    for i in range(2, n):
        for _ in range(1 + i % 2):
            y, x, s = rng.integers(0, size - 10), rng.integers(0, size - 10), rng.integers(2, 10)
            out[i, y : y + s, x : x + s] = True
    return out


def test_perfect_heatmap_scores_one():
    masks = _masks(np.random.default_rng(0))
    auroc, pro = pixel_metrics(zip(masks.astype(np.float32), masks, strict=True))
    assert auroc == 1.0 and pro == pytest.approx(1.0)


def test_random_heatmap_auroc_near_half():
    rng = np.random.default_rng(1)
    masks = _masks(rng, n=20, size=64)
    auroc, pro = pixel_metrics((rng.random(m.shape), m) for m in masks)
    assert auroc == pytest.approx(0.5, abs=0.02)
    assert pro == pytest.approx(0.15, abs=0.03)  # random map: PRO ~ FPR, area/0.3 ~ 0.15


def _brute_force_pro(maps, masks, fpr_limit):
    """Reference definition, threshold by threshold: flag >= t, overlap averaged over regions."""
    regions = []
    for h, m in zip(maps, masks, strict=True):
        lab, n = ndimage.label(m, structure=EIGHT_CONNECTED)
        regions += [h[lab == k] for k in range(1, n + 1)]
    good = maps[~masks]
    xs, ys = [0.0], [0.0]
    for t in np.unique(maps)[::-1]:
        xs.append(np.mean(good >= t))
        ys.append(np.mean([np.mean(r >= t) for r in regions]))
    return area_to(np.array(xs), np.array(ys), fpr_limit)


@pytest.mark.parametrize("limit", [0.3, 0.05, 1.0])
def test_pro_matches_threshold_sweep(limit):
    rng = np.random.default_rng(2)
    masks = _masks(rng, n=5, size=12)
    # Coarse values give ties; defects get a weak bump so the curve is not trivial.
    maps = np.round(rng.random(masks.shape) + 0.6 * masks, 1).astype(np.float16)
    _, pro = pixel_metrics(zip(maps, masks, strict=True), fpr_limit=limit)
    assert pro == pytest.approx(_brute_force_pro(maps, masks, limit), abs=1e-9)


def test_small_regions_weigh_as_much_as_large_ones():
    masks = np.zeros((1, 20, 20), dtype=bool)
    masks[0, :10, :10] = True  # 100-pixel region, found
    masks[0, 18, 18] = True  # 1-pixel region, missed
    maps = masks.astype(np.float32)
    maps[0, 18, 18] = 0.0
    fpr, pro = pro_curve(maps, masks)
    assert pro[1] == pytest.approx(0.5)  # first threshold catches only one of two regions


def test_rejects_mismatched_shapes_and_missing_classes():
    with pytest.raises(ValueError, match="equal"):
        pixel_metrics([(np.zeros((4, 4)), np.zeros((4, 5), dtype=bool))])
    with pytest.raises(ValueError, match="both"):
        pixel_metrics([(np.zeros((4, 4)), np.zeros((4, 4), dtype=bool))])
