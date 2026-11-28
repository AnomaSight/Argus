"""Preprocessing presets shared by both halves of the code (A1.2; used by C2 and C3).

This is the only place that knows preset geometry:

- ``patchcore``: shorter side resized to 256, centre-crop 224 (input 224 x 224).
- ``deng``: resized to exactly 256 x 256, no crop (input 256 x 256).

Image, view and mask transforms are all built from one geometry step, so they cannot drift
apart: images use bilinear resampling with antialiasing, masks use nearest. Images are always
converted to RGB first (MVTec AD ``screw`` and ISP-AD ``ASM`` are greyscale).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Preset:
    """``resize``: int = shorter side (aspect kept), tuple = exact (H, W).
    ``crop``: centre-crop side, or None."""

    resize: int | tuple[int, int]
    crop: int | None


PRESETS = {
    "patchcore": Preset(resize=256, crop=224),
    "deng": Preset(resize=(256, 256), crop=None),
}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; valid presets: {sorted(PRESETS)}") from None


def input_size(preset: str) -> tuple[int, int]:
    """(H, W) of the model input, the heatmap and the transformed mask."""
    p = get_preset(preset)
    if p.crop is not None:
        return (p.crop, p.crop)
    if isinstance(p.resize, int):
        raise ValueError(f"preset {preset!r} has no fixed input size")
    return tuple(p.resize)


def _geometry(
    preset: str, interpolation: InterpolationMode
) -> Callable[[Image.Image], Image.Image]:
    p = get_preset(preset)
    size = [p.resize] if isinstance(p.resize, int) else list(p.resize)

    def apply(img: Image.Image) -> Image.Image:
        out = F.resize(img, size, interpolation=interpolation, antialias=True)
        if p.crop is not None:
            out = F.center_crop(out, [p.crop, p.crop])
        return out

    return apply


def view_transform(preset: str) -> Callable[[Image.Image], Image.Image]:
    """RGB, resized and cropped like the model input, not normalised (C3 ``input_view``)."""
    geom = _geometry(preset, InterpolationMode.BILINEAR)
    return lambda img: geom(img.convert("RGB"))


def image_transform(preset: str) -> Callable[[Image.Image], torch.Tensor]:
    """RGB -> resize -> crop -> float tensor (3, H, W) in [0, 1] -> ImageNet normalisation."""
    view = view_transform(preset)

    def apply(img: Image.Image) -> torch.Tensor:
        t = F.to_tensor(view(img))
        return F.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

    return apply


def mask_transform(preset: str) -> Callable[[Image.Image], np.ndarray]:
    """Nearest resize -> same crop -> bool array (H, W), True where the mask value is > 0 (C2)."""
    geom = _geometry(preset, InterpolationMode.NEAREST)

    def apply(mask: Image.Image) -> np.ndarray:
        if mask.mode not in ("L", "1", "I", "I;16", "F"):
            mask = mask.convert("L")
        return np.asarray(geom(mask)) > 0

    return apply


def load_image(data_dir: Path, path: str) -> Image.Image:
    """Open the C1 ``path`` under ``data_dir`` as RGB; the file is closed (no lazy handle)."""
    with Image.open(Path(data_dir) / path) as img:
        return img.convert("RGB")


def load_mask(data_dir: Path, path: str) -> Image.Image:
    """Open a mask without colour conversion; the file is closed."""
    with Image.open(Path(data_dir) / path) as img:
        img.load()
        return img.copy()
