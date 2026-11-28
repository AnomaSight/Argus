"""Patch feature shapes and determinism. Random weights, so no download and no GPU needed."""

import pytest
import torch
from PIL import Image

from argus.data.transforms import image_transform
from argus.features.backbone import Backbone, resolve_device
from argus.features.patch_features import _patchify, _resample_to, patch_features


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    return Backbone(weights=None)


@pytest.mark.parametrize("preset, grid", [("patchcore", (28, 28)), ("deng", (32, 32))])
def test_shapes_for_both_presets(backbone, preset, grid):
    img = Image.new("RGB", (300, 260), (90, 120, 150))
    x = torch.stack([image_transform(preset)(img)] * 2)
    feats, g = patch_features(backbone, x)
    assert g == grid
    assert feats.shape == (2, grid[0] * grid[1], 1024) and feats.dtype == torch.float32


def test_same_image_same_features(backbone):
    x = torch.rand(1, 3, 224, 224)
    a, _ = patch_features(backbone, x)
    b, _ = patch_features(backbone, x.clone())
    assert torch.equal(a, b)


def test_batch_does_not_change_features(backbone):
    x = torch.rand(3, 3, 224, 224)
    batched, _ = patch_features(backbone, x)
    single, _ = patch_features(backbone, x[1:2])
    assert torch.allclose(batched[1], single[0], atol=1e-5)


def test_patchify_gathers_the_3x3_neighbourhood():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    p, grid = _patchify(x)
    assert grid == (4, 4) and p.shape == (1, 16, 1, 3, 3)
    # Patch at (1, 1) is the 3x3 block centred on value 5.
    assert p[0, 5, 0].tolist() == [[0, 1, 2], [4, 5, 6], [8, 9, 10]]


def test_resample_is_identity_on_same_grid():
    p = torch.rand(2, 9, 4, 3, 3)
    assert torch.allclose(_resample_to(p, (3, 3), (3, 3)), p, atol=1e-6)


def test_device_auto_honours_env(monkeypatch):
    monkeypatch.setenv("ARGUS_DEVICE", "cpu")
    assert resolve_device("auto").type == "cpu"
