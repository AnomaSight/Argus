"""A1.2: preprocessing presets shared by C2 (mask alignment) and C3 (score / input_view)."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from argus.data import transforms as T

PRESETS = ["patchcore", "deng"]


def test_input_size():
    assert T.input_size("patchcore") == (224, 224)
    assert T.input_size("deng") == (256, 256)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("mode", ["RGB", "L"])  # screw and ASM are greyscale
@pytest.mark.parametrize("size", [(1024, 1024), (256, 256), (900, 700)])
def test_output_shapes(preset, mode, size):
    img = Image.new(mode, size, 100)
    h, w = T.input_size(preset)
    t = T.image_transform(preset)(img)
    assert t.shape == (3, h, w) and t.dtype == torch.float32
    view = T.view_transform(preset)(img)
    assert view.mode == "RGB" and view.size == (w, h)  # PIL size is (W, H)
    mask = T.mask_transform(preset)(Image.new("L", size, 255))
    assert mask.shape == (h, w) and mask.dtype == bool and mask.all()


@pytest.mark.parametrize("preset", PRESETS)
def test_normalisation_on_constant_image(preset):
    value = 128
    t = T.image_transform(preset)(Image.new("RGB", (512, 512), (value,) * 3))
    expected = [(value / 255 - m) / s for m, s in zip(T.IMAGENET_MEAN, T.IMAGENET_STD, strict=True)]
    assert torch.allclose(t.mean(dim=(1, 2)), torch.tensor(expected), atol=1e-5)
    assert torch.allclose(t.std(dim=(1, 2)), torch.zeros(3), atol=1e-5)


@pytest.mark.parametrize("preset", PRESETS)
def test_view_is_not_normalised(preset):
    view = T.view_transform(preset)(Image.new("L", (1024, 1024), 77))
    assert isinstance(view, Image.Image)
    assert np.unique(np.asarray(view)).tolist() == [77]


@pytest.mark.parametrize("preset", PRESETS)
def test_mask_keeps_only_original_classes(preset):
    mask = Image.new("L", (1024, 1024), 0)
    ImageDraw.Draw(mask).ellipse((100, 200, 700, 800), fill=255)
    out = T.mask_transform(preset)(mask)
    assert out.dtype == bool and set(np.unique(out).tolist()) == {False, True}
    # Nearest resampling never invents values: the 8-bit result is only 0 or 255.
    geom = T._geometry(preset, T.InterpolationMode.NEAREST)
    assert set(np.unique(np.asarray(geom(mask))).tolist()) == {0, 255}


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("size", [(1024, 1024), (1024, 768)])
def test_image_and_mask_stay_aligned(preset, size):
    # A white square drawn at the same place in an image and its mask must land at the same
    # place after preprocessing, including the centre crop of `patchcore`.
    box = (300, 250, 560, 610)  # partly outside the 224 crop's centre, not grid-aligned
    img = Image.new("L", size, 0)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(img).rectangle(box, fill=255)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    bright = np.asarray(T.view_transform(preset)(img))[..., 0] > 127
    m = T.mask_transform(preset)(mask)
    iou = (bright & m).sum() / (bright | m).sum()
    assert iou > 0.95, iou
    ys, xs = np.nonzero(m)
    by, bx = np.nonzero(bright)
    assert abs(ys.mean() - by.mean()) < 0.5 and abs(xs.mean() - bx.mean()) < 0.5


def test_patchcore_crop_is_centred():
    # Shorter side 1024 -> 256 is a factor of 4, and the 224 crop starts 16 px in, i.e. 64 px
    # in the original. A mask filling exactly that window must fill the whole crop; an
    # off-centre crop would include unfilled border pixels.
    mask = Image.new("L", (1024, 1024), 0)
    ImageDraw.Draw(mask).rectangle((64, 64, 1024 - 65, 1024 - 65), fill=255)
    out = T.mask_transform("patchcore")(mask)
    assert out.all()


def test_unknown_preset_lists_valid_names():
    for fn in (T.input_size, T.image_transform, T.view_transform, T.mask_transform):
        with pytest.raises(ValueError, match=r"unknown preset 'imagenet'.*deng.*patchcore"):
            fn("imagenet")


def test_load_image_converts_to_rgb_and_closes_file(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (32, 16), 9).save(path)
    img = T.load_image(tmp_path, "grey.png")
    assert img.mode == "RGB" and img.size == (32, 16)
    assert getattr(img, "fp", None) is None
    path.unlink()  # still usable after the file is gone
    assert np.asarray(img).shape == (16, 32, 3)


def test_load_mask_keeps_mode(tmp_path):
    Image.new("L", (8, 8), 255).save(tmp_path / "m.png")
    m = T.load_mask(tmp_path, "m.png")
    assert m.mode == "L" and T.mask_transform("deng")(m).all()
