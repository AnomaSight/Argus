"""Patch features (A1.3): shapes, determinism, parity with the official PatchCore code, devices.

Random weights, so no download and no GPU needed. The MPS test runs only where MPS exists.
"""

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import wide_resnet50_2

from argus.data.transforms import image_transform
from argus.features import backbone as backbone_mod
from argus.features import patch_features as pf
from argus.features.backbone import Backbone, resolve_device
from argus.features.patch_features import _patchify, _resample_to, patch_features
from tests import _patchcore_reference as ref

HAS_MPS = torch.backends.mps.is_available()
OTHER_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if HAS_MPS else None


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    return Backbone(weights=None)


@pytest.fixture(scope="module")
def hooked():
    """A full random-weight torchvision WRN-50 and a Backbone holding the same weights."""
    torch.manual_seed(1)
    net = wide_resnet50_2(weights=None).eval()
    names = {"conv1": "stem.0", "bn1": "stem.1", "layer1": "stem.4"}
    state = {}
    for key, value in net.state_dict().items():
        top, _, rest = key.partition(".")
        if top in ("layer4", "fc"):
            continue
        state[f"{names.get(top, top)}.{rest}"] = value
    bb = Backbone(weights=None)
    bb.load_state_dict(state, strict=True)
    return net, bb


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


# --- Parity with the official code (tests/_patchcore_reference.py, commit fcaa92f) ---------


def _official(l2, l3):
    feats, shapes = ref.embed_layers([l2, l3])
    b = l2.shape[0]
    return feats.reshape(b, -1, feats.shape[-1]), tuple(shapes[0])


@pytest.mark.parametrize("grids", [(28, 14), (32, 16)])
def test_parity_on_synthetic_layers(grids):
    g2, g3 = grids
    gen = torch.Generator().manual_seed(g2)
    l2 = torch.randn(2, 512, g2, g2, generator=gen)
    l3 = torch.randn(2, 1024, g3, g3, generator=gen)
    ours, grid = pf._embed(l2, l3)
    theirs, their_grid = _official(l2, l3)
    assert grid == their_grid == (g2, g2)
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("size", [224, 256])
def test_parity_on_backbone_layers(backbone, size):
    x = torch.rand(2, 3, size, size, generator=torch.Generator().manual_seed(size))
    with torch.no_grad():
        l2, l3 = backbone(x)
    ours, _ = patch_features(backbone, x)
    assert torch.equal(ours, _official(l2, l3)[0])


def test_backbone_equals_hooked_full_network(hooked):
    # The official code runs the full torchvision model with forward hooks on layer2 and
    # layer3 and stops there; dropping layer4 / avgpool / fc and the hooks changes nothing.
    net, bb = hooked
    x = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(3))
    outs = ref.NetworkFeatureAggregator(net, ["layer2", "layer3"])(x)
    l2, l3 = bb(x)
    assert torch.equal(l2, outs["layer2"]) and torch.equal(l3, outs["layer3"])
    ours, grid = patch_features(bb, x)
    theirs, shapes = ref.official_embed(net, x)
    assert grid == tuple(shapes[0]) == (28, 28)
    assert torch.equal(ours, theirs.reshape(ours.shape))


# --- The MPS pooling fallback -------------------------------------------------------------


@pytest.mark.parametrize("length", [4608, 9216, 2048, 1000, 7])
@pytest.mark.parametrize("dim", [1024, 3])
def test_pool_matrix_matches_adaptive_pool(length, dim):
    p = pf._pool_matrix(length, dim, torch.device("cpu"), torch.float32)
    assert p.shape == (length, dim)
    assert torch.allclose(p.sum(0), torch.ones(dim), atol=1e-6)
    v = torch.randn(5, length, generator=torch.Generator().manual_seed(length + dim))
    expected = F.adaptive_avg_pool1d(v.unsqueeze(1), dim).squeeze(1)
    assert torch.allclose(v @ p, expected, atol=1e-6)


def test_pool_on_cpu_is_the_official_operation():
    v = torch.randn(4, 512, 3, 3)
    expected = F.adaptive_avg_pool1d(v.reshape(4, 1, -1), 1024).squeeze(1)
    assert torch.equal(pf._pool(v, 1024), expected)


@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_mps_runs_and_matches_cpu(backbone, record_property):
    import copy

    x = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(4))
    cpu, _ = patch_features(backbone, x)
    net = copy.deepcopy(backbone).to("mps")
    mps, grid = patch_features(net, x.to("mps"))
    assert grid == (28, 28) and mps.device.type == "mps"
    mps = mps.cpu()
    cos = F.cosine_similarity(mps.reshape(-1, 1024), cpu.reshape(-1, 1024), dim=1)
    max_abs = (mps - cpu).abs().max().item()
    record_property("mps_min_cosine", cos.min().item())
    record_property("mps_max_abs_diff", max_abs)
    print(f"MPS vs CPU: min cosine {cos.min().item():.8f}, max |diff| {max_abs:.3g}")
    assert cos.min() > 0.9999


# --- Devices ------------------------------------------------------------------------------


def _fake(monkeypatch, cuda=False, mps=False, count=0):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_cuda_unavailable_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ARGUS_DEVICE", raising=False)
    _fake(monkeypatch, mps=True)
    with pytest.raises(ValueError, match=r"device='cuda'.*CUDA is not available.*mps, cpu"):
        resolve_device("cuda")


def test_env_cuda_unavailable_names_the_variable(monkeypatch):
    monkeypatch.setenv("ARGUS_DEVICE", "cuda")
    _fake(monkeypatch)
    with pytest.raises(ValueError, match=r"ARGUS_DEVICE='cuda'.*not available.*unset ARGUS"):
        resolve_device("auto")


def test_cuda_index_out_of_range(monkeypatch):
    monkeypatch.delenv("ARGUS_DEVICE", raising=False)
    _fake(monkeypatch, cuda=True, count=1)
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(ValueError, match=r"device='cuda:5'.*only 1 CUDA device"):
        resolve_device("cuda:5")


def test_mps_unavailable_and_unknown_names(monkeypatch):
    monkeypatch.delenv("ARGUS_DEVICE", raising=False)
    _fake(monkeypatch)
    with pytest.raises(ValueError, match="MPS .* not available"):
        resolve_device("mps")
    for bad in ("gpu", "cuda0", "tpu", ""):
        with pytest.raises(ValueError, match="is not a device"):
            resolve_device(bad)


@pytest.mark.parametrize(
    "cuda, mps, expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.delenv("ARGUS_DEVICE", raising=False)
    _fake(monkeypatch, cuda=cuda, mps=mps, count=int(cuda))
    assert resolve_device("auto") == torch.device(expected)
    monkeypatch.setenv("ARGUS_DEVICE", "auto")
    assert resolve_device("auto") == torch.device(expected)


def test_explicit_device_ignores_env(monkeypatch):
    monkeypatch.setenv("ARGUS_DEVICE", "cuda")
    _fake(monkeypatch)
    assert resolve_device("cpu") == torch.device("cpu")


# --- Frozen backbone ----------------------------------------------------------------------


def test_backbone_cannot_leave_eval_mode(backbone):
    buffers = {k: v.clone() for k, v in backbone.named_buffers()}
    backbone.train()
    nn.Sequential(backbone).train()  # a parent switching to train mode
    assert not backbone.training
    bns = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and not any(m.training for m in backbone.modules())
    with torch.no_grad():
        backbone(torch.rand(2, 3, 64, 64))
    assert all(torch.equal(v, buffers[k]) for k, v in backbone.named_buffers())  # BN stats kept
    assert not any(p.requires_grad for p in backbone.parameters())
    for name in ("layer4", "avgpool", "fc"):
        assert not hasattr(backbone, name)
    assert not any(k.startswith(("layer4", "fc")) for k in backbone.state_dict())


@pytest.mark.parametrize("weights", ["IMAGENET1K_V2", "DEFAULT", "imagenet"])
def test_only_patchcore_weights_are_accepted(weights):
    with pytest.raises(ValueError, match="IMAGENET1K_V1"):
        Backbone(weights=weights)


# --- Input validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "images, match",
    [
        (torch.rand(3, 224, 224), r"\(B, 3, H, W\).*\(3, 224, 224\)"),
        (torch.rand(1, 1, 224, 224), r"\(1, 1, 224, 224\)"),
        (torch.zeros(1, 3, 224, 224, dtype=torch.uint8), "float32.*uint8"),
        (torch.rand(1, 3, 224, 224, dtype=torch.float64), "float32.*float64"),
    ],
)
def test_bad_input_is_rejected(backbone, images, match):
    with pytest.raises(ValueError, match=match):
        patch_features(backbone, images)


@pytest.mark.skipif(OTHER_DEVICE is None, reason="needs a second device")
def test_wrong_device_is_rejected(backbone):
    with pytest.raises(ValueError, match=r"images are on .* but the backbone is on cpu"):
        patch_features(backbone, torch.rand(1, 3, 224, 224, device=OTHER_DEVICE))


def test_available_devices_ends_with_cpu():
    assert backbone_mod.available_devices()[-1] == "cpu"
