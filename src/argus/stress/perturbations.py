"""Deterministic image changes that mimic factory drift, for the stress tests (task P2.4).

Each is a pure `PIL.Image -> PIL.Image` function of (image, severity): severity 0 returns the
image unchanged, 1.. picks the parameter from configs/experiments/stress.yaml. None of them
draws random numbers, so the same input always gives the same output. Size and mode are kept.
"""

from __future__ import annotations

import io
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance, ImageFilter

from argus.utils.config import REPO_ROOT

STRESS_CONFIG = REPO_ROOT / "configs" / "experiments" / "stress.yaml"


def _brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def _contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def _blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def _jpeg(img: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert(img.mode)


PERTURBATIONS = {"brightness": _brightness, "contrast": _contrast, "blur": _blur, "jpeg": _jpeg}


def load_levels(path: str | Path = STRESS_CONFIG) -> dict[str, list[float]]:
    levels = yaml.safe_load(Path(path).read_text())
    unknown = set(levels) - set(PERTURBATIONS)
    if unknown:
        raise ValueError(f"unknown perturbations in {path}: {sorted(unknown)}")
    return levels


def perturb(
    img: Image.Image, name: str, severity: int, levels: dict[str, list[float]] | None = None
) -> Image.Image:
    """Apply perturbation `name` at `severity` (0 = unchanged copy)."""
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; valid: {sorted(PERTURBATIONS)}")
    params = (levels or load_levels())[name]
    if not 0 <= severity <= len(params):
        raise ValueError(f"{name} severity must be 0..{len(params)}, got {severity}")
    if severity == 0:
        return img.copy()
    return PERTURBATIONS[name](img, params[severity - 1])
