import numpy as np
import pytest
import torch
from PIL import Image

from argus.data.transforms import image_transform
from argus.features.backbone import Backbone
from argus.features.patch_features import patch_features
from argus.scoring.heatmap import heatmaps
from argus.scoring.patchcore import score_patches


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    return Backbone(weights=None)


def _texture(seed, size=256):
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 12, (size // 8, size // 8, 3))
    img = np.kron(base, np.ones((8, 8, 1)))  # blocky texture, like a surface
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def _features(backbone, arrays):
    t = image_transform("patchcore")
    x = torch.stack([t(Image.fromarray(a)) for a in arrays])
    return patch_features(backbone, x)


def test_image_identical_to_support_scores_zero(backbone):
    support = [_texture(s) for s in range(3)]
    feats, _ = _features(backbone, support)
    bank = feats.reshape(-1, feats.shape[-1])
    image, patch = score_patches(feats[1:2], bank)
    assert image.item() < 1e-3 and patch.max().item() < 1e-3


def test_pasted_defect_is_hottest_inside_the_paste(backbone):
    support = [_texture(s) for s in range(4)]
    feats, grid = _features(backbone, support)
    bank = feats.reshape(-1, feats.shape[-1])
    test = _texture(10)
    # 224 crop of 256: paste at input-frame rows/cols 130..170 -> raw 146..186.
    test[146:186, 66:106] = (250, 30, 30)
    t_feats, _ = _features(backbone, [test])
    image, patch = score_patches(t_feats, bank)
    hm = heatmaps(patch, grid, (224, 224))[0]
    r, c = np.unravel_index(np.argmax(hm), hm.shape)
    assert 120 <= r <= 180 and 40 <= c <= 100, (r, c)
    assert hm.shape == (224, 224) and hm.dtype == np.float32
    assert hm.max() <= image.item() + 1e-5  # smoothing keeps the heatmap below the score


def test_score_is_max_patch_distance_and_eq7_shrinks_it():
    torch.manual_seed(1)
    bank = torch.randn(50, 8)
    feats = torch.randn(2, 6, 8)
    image, patch = score_patches(feats, bank)
    assert torch.allclose(image, patch.max(dim=1).values)
    reweighted, _ = score_patches(feats, bank, reweight_b=5)
    assert torch.all(reweighted <= image) and torch.all(reweighted >= 0)


def test_heatmap_upsamples_a_constant_to_the_same_constant():
    hm = heatmaps(torch.full((1, 784), 2.5), (28, 28), (224, 224))
    assert np.allclose(hm, 2.5, atol=1e-5)
