import numpy as np
import pytest

from argus.evaluation.image_metrics import image_auroc


def _pairwise_auroc(scores, labels):
    # Direct definition: share of (defect, good) pairs ranked correctly, ties as half.
    s, y = np.asarray(scores, float), np.asarray(labels)
    d, g = s[y == 1][:, None], s[y == 0][None, :]
    return float(np.mean((d > g) + 0.5 * (d == g)))


def test_worked_example_eight_of_nine():
    # 3 good, 3 defects; only the pair (defect 0.4, good 0.5) is ranked the wrong way round.
    scores = [0.1, 0.3, 0.5, 0.4, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    assert image_auroc(scores, labels) == pytest.approx(8 / 9)


def test_matches_pair_count_with_ties():
    rng = np.random.default_rng(1)
    scores = np.round(rng.normal(size=300), 1)  # rounding creates ties
    labels = (rng.random(300) < 0.3).astype(int)
    scores[labels == 1] += 0.7
    assert image_auroc(scores, labels) == pytest.approx(_pairwise_auroc(scores, labels), abs=1e-12)


@pytest.mark.parametrize(
    "scores, labels",
    [([1.0, 2.0], [0, 0]), ([1.0, np.nan], [0, 1]), ([1.0, 2.0], [0, 2]), ([1.0], [0, 1])],
)
def test_rejects_bad_input(scores, labels):
    with pytest.raises(ValueError):
        image_auroc(scores, labels)
