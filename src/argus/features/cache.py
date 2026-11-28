"""Per-image patch-feature cache (task A1.4).

Features depend only on the image, the preprocessing preset and the backbone config, so they
are computed once per image and shared by every seed, split and memory setting. Stored as
float16 at artifacts/features/{preset}-{hash}/{image_id}.npy (~1.6 MB per 224 image).
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from argus.data.transforms import image_transform, load_image
from argus.features.backbone import Backbone, resolve_device
from argus.features.patch_features import DIM, PATCH, patch_features

log = logging.getLogger(__name__)
BACKBONE_CONFIG = f"wide_resnet50_2/IMAGENET1K_V1/layer2+layer3/patch{PATCH}/dim{DIM}"
Extractor = Callable[[list[Image.Image]], np.ndarray]  # images -> (B, P, D)


def cache_key(preset: str, backbone_config: str = BACKBONE_CONFIG) -> str:
    return f"{preset}-{hashlib.sha256(f'{preset}|{backbone_config}'.encode()).hexdigest()[:10]}"


def backbone_extractor(
    preset: str, device: str = "auto", weights: str | None = "IMAGENET1K_V1"
) -> Extractor:
    dev = resolve_device(device)
    net = Backbone(weights).to(dev)
    transform = image_transform(preset)

    def extract(images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([transform(im) for im in images]).to(dev)
        feats, _ = patch_features(net, x)
        return feats.cpu().numpy()

    return extract


class FeatureCache:
    def __init__(
        self,
        preset: str,
        data_dir: Path,
        artifacts: Path,
        extractor: Extractor | None = None,
        batch_size: int = 16,
        backbone_config: str = BACKBONE_CONFIG,
    ):
        self.preset = preset
        self.data_dir = Path(data_dir)
        self.root = Path(artifacts) / "features" / cache_key(preset, backbone_config)
        self._extractor = extractor
        self.batch_size = batch_size
        self.hits = self.misses = 0

    @property
    def extractor(self) -> Extractor:
        if self._extractor is None:  # only build the CNN if something is missing
            self._extractor = backbone_extractor(self.preset)
        return self._extractor

    def path(self, image_id: str) -> Path:
        return self.root / f"{image_id}.npy"

    def get_features(self, image_ids: Sequence[str]) -> np.ndarray:
        """(len(image_ids), patches, dim) float16, computing only what isn't cached yet."""
        missing = [i for i in dict.fromkeys(image_ids) if not self.path(i).exists()]
        self.misses += len(missing)
        self.hits += len(set(image_ids)) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            images = [load_image(self.data_dir, f"{i}.png") for i in batch]
            for image_id, f in zip(batch, self.extractor(images), strict=True):
                self._write(image_id, f.astype(np.float16))
        if missing:
            log.info(
                "feature cache %s: %d hits, %d computed", self.root.name, self.hits, self.misses
            )
        return np.stack([np.load(self.path(i)) for i in image_ids])

    def _write(self, image_id: str, features: np.ndarray) -> None:
        path = self.path(image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, features)
        os.replace(tmp, path)  # atomic: an interrupted run never leaves a half-written file

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*.npy"))
