"""The public detector API (contract C3, task A2.5).

    det = argus.load_detector("artifacts/banks/mvtec_ad/hazelnut/m10/seed20260801")
    score, heatmap = det.score(image)

Scores go through the same functions as the pipeline (transforms, patch_features,
score_patches, heatmaps), so they match the run's C2 scores (C3 rule 1).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from argus.data.transforms import image_transform, view_transform
from argus.features.backbone import Backbone, device_name, resolve_device
from argus.features.patch_features import patch_features
from argus.memory.bank import read_bank_json
from argus.scoring.heatmap import heatmaps
from argus.scoring.patchcore import score_patches

BACKBONE_WEIGHTS: str | None = "IMAGENET1K_V1"  # tests set None to avoid the download


class Detector(Protocol):
    @property
    def info(self) -> dict: ...

    def score(self, image: Image.Image) -> tuple[float, np.ndarray]: ...

    def score_batch(
        self, images: list[Image.Image], batch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def input_view(self, image: Image.Image) -> Image.Image: ...


@cache
def _backbone(weights: str | None, device: str) -> Backbone:
    """One backbone per process and device, shared by every loaded bank."""
    return Backbone(weights).to(device)


class PatchCoreDetector:
    """A memory bank plus the shared backbone. Not thread-safe: one call at a time (C3 rule 3)."""

    def __init__(self, folder: Path, device: torch.device):
        info = read_bank_json(folder)
        self._device = device
        self._info = {**info, "device": device_name(device)}
        self._size = tuple(info["input_size"])
        self._bank = torch.from_numpy(np.load(Path(folder) / "features.npy")).to(
            device, torch.float32
        )
        self._net = _backbone(BACKBONE_WEIGHTS, str(device))
        self._transform = image_transform(info["preprocess"])
        self._view = view_transform(info["preprocess"])

    @property
    def info(self) -> dict:
        return dict(self._info)

    def input_view(self, image: Image.Image) -> Image.Image:
        return self._view(image)

    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        scores, maps = self.score_batch([image], batch_size=1)
        return float(scores[0]), maps[0]

    def score_batch(
        self, images: list[Image.Image], batch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray]:
        if not images:
            return np.zeros(0, np.float32), np.zeros((0, *self._size), np.float32)
        all_scores, all_maps = [], []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            x = torch.stack([self._transform(im) for im in chunk]).to(self._device)
            feats, grid = patch_features(self._net, x)
            image_scores, patch_scores = score_patches(feats, self._bank)
            all_scores.append(image_scores.cpu().numpy())
            all_maps.append(heatmaps(patch_scores, grid, self._size))
        return np.concatenate(all_scores).astype(np.float32), np.concatenate(all_maps)


def load_detector(path: str | Path, device: str = "auto") -> Detector:
    """Load a saved bank folder (C3). FileNotFoundError if it or its bank.json is missing,
    ValueError for an unsupported schema_version."""
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"no bank folder at {folder}")
    return PatchCoreDetector(folder, resolve_device(device))
