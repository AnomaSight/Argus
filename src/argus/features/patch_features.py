"""Locally aware patch features, as in the official PatchCore code (task A1.3).

For each of layer2 and layer3: unfold 3 x 3 neighbourhoods (stride 1, padding 1). layer3's
patches are bilinearly resampled onto layer2's grid. Each layer's C*3*3 vector is pooled to
1024 dims (adaptive average pooling), and the two layers are pooled together to 1024.
Output: (B, H*W, 1024) with H x W = layer2's grid (28 x 28 at 224, 32 x 32 at 256).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from argus.features.backbone import Backbone

PATCH = 3
DIM = 1024  # official repo default for both pretrain_embed_dimension and target dimension


def _patchify(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """(B, C, H, W) -> (B, H*W, C, 3, 3) and the (H, W) grid."""
    b, c, h, w = x.shape
    u = F.unfold(x, kernel_size=PATCH, stride=1, padding=(PATCH - 1) // 2)
    return u.reshape(b, c, PATCH, PATCH, h * w).permute(0, 4, 1, 2, 3), (h, w)


def _resample_to(p: torch.Tensor, grid: tuple[int, int], ref: tuple[int, int]) -> torch.Tensor:
    """Bilinearly move (B, h*w, C, 3, 3) patches from grid (h, w) onto ref (H, W)."""
    b, _, c, k, _ = p.shape
    x = p.reshape(b, *grid, c, k, k).permute(0, 3, 4, 5, 1, 2)  # (B, C, 3, 3, h, w)
    shape = x.shape
    x = F.interpolate(x.reshape(-1, 1, *grid), size=ref, mode="bilinear", align_corners=False)
    x = x.reshape(*shape[:-2], *ref).permute(0, 4, 5, 1, 2, 3)  # (B, H, W, C, 3, 3)
    return x.reshape(b, ref[0] * ref[1], c, k, k)


def _pool(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Adaptive average pooling of each row's flattened vector to `dim` values."""
    return F.adaptive_avg_pool1d(x.reshape(len(x), 1, -1), dim).squeeze(1)


@torch.inference_mode()
def patch_features(
    backbone: Backbone, images: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, int]]:
    """images: normalised (B, 3, H, W). Returns float32 (B, h*w, 1024) and the (h, w) grid."""
    l2, l3 = backbone(images)
    p2, grid = _patchify(l2)
    p3, grid3 = _patchify(l3)
    p3 = _resample_to(p3, grid3, grid)
    b, n = p2.shape[:2]
    per_layer = [_pool(p.reshape(b * n, -1), DIM) for p in (p2, p3)]  # each (B*N, 1024)
    stacked = torch.stack(per_layer, dim=1)  # (B*N, 2, 1024)
    return _pool(stacked, DIM).reshape(b, n, DIM).float(), grid
