"""C3 detector on a tiny bank built from fake images (random backbone weights, CPU)."""

import numpy as np
import pytest
import torch
from PIL import Image

import argus
from argus import detector as det_module
from argus.data.transforms import image_transform
from argus.features.backbone import Backbone
from argus.features.patch_features import patch_features
from argus.memory.bank import build_banks, save_banks
from argus.scoring.patchcore import score_patches

C3_KEYS = {
    "dataset",
    "category",
    "setting",
    "seed",
    "config_hash",
    "preprocess",
    "input_size",
    "coreset_size",
    "bank_mb",
    "device",
}


def _img(seed, defect=False):
    rng = np.random.default_rng(seed)
    a = np.kron(rng.normal(128, 12, (32, 32, 3)), np.ones((8, 8, 1)))
    if defect:
        a[146:186, 66:106] = (250, 30, 30)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@pytest.fixture(scope="module")
def bank_dir(tmp_path_factory):
    old, det_module.BACKBONE_WEIGHTS = det_module.BACKBONE_WEIGHTS, None
    torch.manual_seed(0)
    net = det_module._backbone(None, "cpu")
    t = image_transform("patchcore")
    feats, _ = patch_features(net, torch.stack([t(_img(s)) for s in range(4)]))
    support = feats.reshape(-1, feats.shape[-1]).numpy()
    banks, coreset = build_banks(
        support, dataset="mvtec_ad", category="hazelnut", seed=1, preprocess="patchcore"
    )
    yield save_banks(banks, coreset, tmp_path_factory.mktemp("art"))
    det_module.BACKBONE_WEIGHTS = old


def test_c3_example_shape(bank_dir):
    det = argus.load_detector(bank_dir["m10"], device="cpu")
    assert set(det.info) == C3_KEYS and det.info["setting"] == "m10" and det.info["device"] == "cpu"
    score, heatmap = det.score(_img(10, defect=True))
    assert isinstance(score, float) and heatmap.shape == (224, 224) and heatmap.dtype == np.float32
    assert det.input_view(_img(10)).size == (224, 224)


def test_batch_matches_single_and_batch_size(bank_dir):
    det = argus.load_detector(bank_dir["m100"], device="cpu")
    images = [_img(11), _img(12, defect=True), _img(13)]
    s4, h4 = det.score_batch(images, batch_size=4)
    s1, h1 = det.score_batch(images, batch_size=1)
    assert s4.shape == (3,) and h4.shape == (3, 224, 224) and s4.dtype == np.float32
    assert np.allclose(s4, s1, rtol=1e-4, atol=1e-5) and np.allclose(h4, h1, atol=1e-4)
    assert det.score(images[1])[0] == pytest.approx(float(s1[1]), rel=1e-4)
    empty_s, empty_h = det.score_batch([])
    assert empty_s.shape == (0,) and empty_h.shape == (0, 224, 224)


def test_matches_the_pipeline_path(bank_dir):
    # C3 rule 1: the same code path as the pipeline gives the same scores.
    det = argus.load_detector(bank_dir["m10"], device="cpu")
    img = _img(14, defect=True)
    net = det_module._backbone(None, "cpu")
    feats, _ = patch_features(net, image_transform("patchcore")(img)[None])
    bank = torch.from_numpy(np.load(bank_dir["m10"] / "features.npy")).float()
    expected, _ = score_patches(feats, bank)
    assert det.score(img)[0] == pytest.approx(expected.item(), rel=1e-4, abs=1e-5)


def test_defect_scores_higher_and_banks_share_the_backbone(bank_dir):
    a = argus.load_detector(bank_dir["m100"], device="cpu")
    b = argus.load_detector(bank_dir["m1"], device="cpu")
    assert a._net is b._net
    assert a.score(_img(20, defect=True))[0] > a.score(_img(20))[0]


def test_missing_or_bad_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        argus.load_detector(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        argus.load_detector(tmp_path)
    (tmp_path / "bank.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError):
        argus.load_detector(tmp_path)


def test_import_argus_stays_light():
    assert "Backbone" not in dir(argus) and callable(argus.load_detector)
    assert isinstance(Backbone, type)
