"""Locally aware patch features, exactly as in the official PatchCore code (task A1.3).

Reference: amazon-science/patchcore-inspection at commit
``fcaa92f124fb1ad74a7acf56726decd4b27cbcad``, ``src/patchcore/patchcore.py``
(``PatchCore._embed``, ``PatchMaker.patchify``) and ``src/patchcore/common.py``
(``Preprocessing``, ``MeanMapper``, ``Aggregator``), with the WRN-50 MVTec AD settings of
``sample_training.sh``: ``-le layer2 -le layer3 --pretrain_embed_dimension 1024
--target_embed_dimension 1024 --patchsize 3``. ``tests/_patchcore_reference.py`` transcribes
that code, and the tests require ``torch.equal`` with it on CPU.

Steps, for each of layer2 (512 channels) and layer3 (1024 channels):

1. Unfold 3 x 3 neighbourhoods (stride 1, padding 1): one (C, 3, 3) patch per position.
2. layer3's patches are bilinearly resampled (``align_corners=False``) onto layer2's grid,
   each of the C*3*3 entries as its own channel.
3. Each patch's flattened C*3*3 vector is adaptive-average-pooled to 1024 values (4608 -> 1024
   for layer2, 9216 -> 1024 for layer3), then the two 1024-vectors are stacked and pooled
   together, 2048 -> 1024.

The issue text describes this as "3 x 3 local average aggregation". The official code does not
apply an average pool over the 3 x 3 window; it pools over the flattened C*3*3 vector as above,
which mixes neighbouring channels and positions. We follow the code, since the task is to
match the official extractor.

Output: float32 (B, H*W, 1024) with H x W = layer2's grid (28 x 28 at 224, 32 x 32 at 256).

Apple MPS cannot adaptive-pool when the input length is not a multiple of the output length
(layer2's 4608 -> 1024). There, and only there, the pooling is done as a matrix product with
PyTorch's own adaptive-pooling bins (``_pool_matrix``); CPU and CUDA run the official
``F.adaptive_avg_pool1d``.

Memory: the largest intermediates are layer3's unfolded and resampled patches, 9216 x 784
floats (29 MB) per image at 224 (9216 x 1024, 38 MB, at 256), and a few copies exist at
once. Measured for one batch-16 call (scripts/check_patch_features.py, fresh process, Mac):
about 1.5 GB at 224 and 2.0 GB at 256 on CPU (peak RSS growth); the MPS allocator reserves
about 2 GB. The weights up to layer3 add 95 MB (24.9 M float32 parameters).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from argus.features.backbone import Backbone

PATCH = 3
DIM = 1024  # official repo default for both pretrain_embed_dimension and target dimension

_POOL_MATRICES: dict[tuple, torch.Tensor] = {}


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


def _pool_matrix(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(length, dim) matrix P with ``v @ P == adaptive_avg_pool1d(v, dim)``.

    Output i averages inputs start_i <= j < end_i, with start_i = floor(i * length / dim) and
    end_i = ceil((i + 1) * length / dim): PyTorch's adaptive-pooling bins. Cached per
    (length, dim, device, dtype).
    """
    key = (length, dim, torch.device(device), dtype)
    if key not in _POOL_MATRICES:
        i = torch.arange(dim)
        start = (i * length) // dim
        end = ((i + 1) * length + dim - 1) // dim
        j = torch.arange(length)[:, None]
        inside = (j >= start) & (j < end)
        p = inside.double() / (end - start).double()
        _POOL_MATRICES[key] = p.to(device=device, dtype=dtype)
    return _POOL_MATRICES[key]


def _pool(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Adaptive average pooling of each row's flattened vector to `dim` values."""
    v = x.reshape(len(x), -1)
    length = v.shape[1]
    if v.device.type == "mps" and length % dim:
        return v @ _pool_matrix(length, dim, v.device, v.dtype)
    return F.adaptive_avg_pool1d(v.unsqueeze(1), dim).squeeze(1)


def _check_input(backbone: Backbone, images: torch.Tensor) -> None:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
        shape = tuple(images.shape) if isinstance(images, torch.Tensor) else type(images)
        raise ValueError(f"images must be a (B, 3, H, W) tensor, got {shape}")
    if images.dtype != torch.float32:
        raise ValueError(
            f"images must be float32 (normalised, as from image_transform), got {images.dtype}"
        )
    want = next(backbone.parameters()).device
    if images.device != want:
        raise ValueError(
            f"images are on {images.device} but the backbone is on {want}; "
            f"move them with images.to({str(want)!r})"
        )


@torch.inference_mode()
def patch_features(
    backbone: Backbone, images: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Official PatchCore patch embedding of a batch.

    images: normalised float32 (B, 3, H, W) on the backbone's device (ValueError otherwise).
    Returns float32 (B, h*w, 1024) on that device, and the (h, w) grid.

    The result is an inference tensor: reading it, ``.cpu()`` and ``.numpy()`` are fine, but
    an in-place change outside ``torch.inference_mode`` raises. Clone it first if you need
    that; it is not cloned here, to save memory.
    """
    _check_input(backbone, images)
    return _embed(*backbone(images))


def _embed(l2: torch.Tensor, l3: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Steps 1-3 of the module docstring on the layer2 and layer3 outputs."""
    p2, grid = _patchify(l2)
    p3, grid3 = _patchify(l3)
    p3 = _resample_to(p3, grid3, grid)
    b, n = p2.shape[:2]
    per_layer = [_pool(p.reshape(b * n, -1), DIM) for p in (p2, p3)]  # each (B*N, 1024)
    stacked = torch.stack(per_layer, dim=1)  # (B*N, 2, 1024)
    return _pool(stacked, DIM).reshape(b, n, DIM).float(), grid
