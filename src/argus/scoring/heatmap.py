"""Heatmaps from patch scores (task A2.4), as in the official PatchCore code.

Reshape to the patch grid, bilinearly upsample to the input size (align_corners=False), then
Gaussian-smooth with sigma = 4. The smoothing is why heatmap.max() is usually below the image
score (C2).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

SIGMA = 4


def heatmaps(
    patch_scores: torch.Tensor, grid: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    """patch_scores (B, h*w) -> float32 (B, H, W)."""
    x = patch_scores.reshape(-1, 1, *grid).float()
    up = F.interpolate(x, size=size, mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    return np.stack([gaussian_filter(m, sigma=SIGMA) for m in up]).astype(np.float32)
