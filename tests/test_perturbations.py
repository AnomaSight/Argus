import numpy as np
import pytest
from PIL import Image

from argus.stress.perturbations import PERTURBATIONS, load_levels, perturb

LEVELS = load_levels()
CASES = [(n, s) for n in PERTURBATIONS for s in range(1, len(LEVELS[n]) + 1)]


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_severity_zero_is_identity(img, name):
    assert np.array_equal(np.asarray(perturb(img, name, 0)), np.asarray(img))


@pytest.mark.parametrize(("name", "severity"), CASES)
def test_deterministic_and_keeps_size_and_mode(img, name, severity):
    a, b = perturb(img, name, severity), perturb(img, name, severity)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert a.size == img.size and a.mode == img.mode
    assert not np.array_equal(np.asarray(a), np.asarray(img))


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_change_grows_with_severity(img, name):
    diffs = [
        np.abs(np.asarray(perturb(img, name, s), float) - np.asarray(img, float)).mean()
        for s in range(1, len(LEVELS[name]) + 1)
    ]
    assert diffs == sorted(diffs)


def test_greyscale_stays_greyscale(img):
    assert perturb(img.convert("L"), "jpeg", 3).mode == "L"


def test_rejects_bad_name_and_severity(img):
    with pytest.raises(ValueError, match="unknown"):
        perturb(img, "rain", 1)
    with pytest.raises(ValueError, match="severity"):
        perturb(img, "blur", 99)
