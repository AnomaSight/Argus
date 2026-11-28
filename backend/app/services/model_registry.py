"""The memory banks the API can serve. Until A2.5 lands, a StubDetector stands in for each.

StubDetector has the same four members as C3's Detector, so P3.2 swaps it for
argus.load_detector() without touching the routes. Its scores are not results: they come
from local contrast in the image, only so the API returns contract-shaped data.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from app.schemas import ModelInfo

SEED = 20260801
# setting -> (coreset_size, bank_mb, stub tau offset). Sizes follow C2's sample arithmetic.
SETTINGS = {
    "m100": (156_800, 612.5, 0.0),
    "m10": (15_680, 61.25, 0.05),
    "m1": (1_568, 6.13, 0.15),
    "aso": (4_210, 16.45, 0.08),
}
HEATMAP_RANGE = (1.5, 3.0)


class StubDetector:
    """C3-shaped stand-in: deterministic score and blob-shaped heatmap from local contrast."""

    def __init__(self, info: dict):
        self._info = info
        self.lock = threading.Lock()  # C3 rule 3: one call at a time per detector

    @property
    def info(self) -> dict:
        return dict(self._info)

    def input_view(self, image: Image.Image) -> Image.Image:
        """`patchcore` preset: RGB, short side to 256, centre crop 224."""
        img = image.convert("RGB")
        scale = 256 / min(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        left, top = (img.width - 224) // 2, (img.height - 224) // 2
        return img.crop((left, top, left + 224, top + 224))

    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        g = np.asarray(self.input_view(image).convert("L"), dtype=np.float32) / 255
        local = np.abs(g - gaussian_filter(g, 6))
        patch = 1.4 + 8 * gaussian_filter(local, 2)
        return float(patch.max()), gaussian_filter(patch, 4).astype(np.float32)

    def score_batch(self, images, batch_size: int = 16):
        if not images:
            return np.zeros(0, np.float32), np.zeros((0, 224, 224), np.float32)
        pairs = [self.score(im) for im in images]
        return np.array([p[0] for p in pairs], np.float32), np.stack([p[1] for p in pairs])


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, tuple[ModelInfo, StubDetector]] = {}
        for c, category in enumerate(("hazelnut", "screw")):
            for setting, (size, mb, offset) in SETTINGS.items():
                model_id = f"mvtec_ad/{category}/{setting}/seed{SEED}"
                info = {
                    "dataset": "mvtec_ad",
                    "category": category,
                    "setting": setting,
                    "seed": SEED,
                    "config_hash": "stub",
                    "preprocess": "patchcore",
                    "input_size": [224, 224],
                    "coreset_size": size,
                    "bank_mb": mb,
                    "device": "cpu",
                }
                meta = ModelInfo(
                    model_id=model_id,
                    dataset="mvtec_ad",
                    category=category,
                    setting=setting,
                    seed=SEED,
                    coreset_size=size,
                    bank_mb=mb,
                    tau=2.2 + 0.3 * c + offset,
                    alpha=0.05,
                    delta=0.05,
                    review_band=None,
                )
                self._models[model_id] = (meta, StubDetector(info))

    def list(self) -> list[ModelInfo]:
        return [self._models[k][0] for k in sorted(self._models)]

    def get(self, model_id: str) -> tuple[ModelInfo, StubDetector]:
        """Raises KeyError for an unknown id. Ids are only ever dict keys, never paths."""
        return self._models[model_id]


registry = ModelRegistry()
