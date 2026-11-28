import numpy as np
import torch
from PIL import Image

from argus.features.cache import FeatureCache, backbone_extractor, cache_key


def _images(root, n):
    ids = []
    for i in range(n):
        image_id = f"mvtec_ad/hazelnut/train/good/{i:03d}"
        (root / image_id).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (i * 20, 100, 50)).save(root / f"{image_id}.png")
        ids.append(image_id)
    return ids


class CountingExtractor:
    def __init__(self):
        self.images_seen = 0

    def __call__(self, images):
        self.images_seen += len(images)
        return np.stack([np.full((4, 8), np.asarray(im).mean(), np.float32) for im in images])


def test_second_run_does_no_cnn_passes(tmp_path):
    ids = _images(tmp_path / "data", 5)
    ex = CountingExtractor()
    cache = FeatureCache("patchcore", tmp_path / "data", tmp_path / "art", ex, batch_size=2)
    first = cache.get_features(ids)
    assert ex.images_seen == 5 and (cache.hits, cache.misses) == (0, 5)
    again = FeatureCache("patchcore", tmp_path / "data", tmp_path / "art", ex).get_features(ids)
    assert ex.images_seen == 5  # nothing recomputed
    assert first.dtype == np.float16 and np.array_equal(first, again)


def test_only_missing_images_are_computed(tmp_path):
    ids = _images(tmp_path / "data", 4)
    ex = CountingExtractor()
    cache = FeatureCache("patchcore", tmp_path / "data", tmp_path / "art", ex)
    cache.get_features(ids[:2])
    cache.get_features(ids + ids[:1])  # duplicates allowed, returned in order
    assert ex.images_seen == 4 and (cache.hits, cache.misses) == (2, 4)


def test_presets_and_backbones_get_separate_folders():
    assert cache_key("patchcore") != cache_key("deng")
    assert cache_key("patchcore") != cache_key("patchcore", "other-backbone")


def test_cached_features_equal_fresh_ones_within_float16(tmp_path):
    torch.manual_seed(0)
    ids = _images(tmp_path / "data", 2)
    extract = backbone_extractor("patchcore", device="cpu", weights=None)
    fresh = extract([Image.open(tmp_path / "data" / f"{i}.png") for i in ids])
    cached = FeatureCache("patchcore", tmp_path / "data", tmp_path / "art", extract).get_features(
        ids
    )
    assert cached.shape == (2, 784, 1024)
    assert np.allclose(cached.astype(np.float32), fresh, rtol=1e-3, atol=1e-3)
