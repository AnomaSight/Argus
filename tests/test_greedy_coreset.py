import numpy as np
import pytest

from argus.memory.greedy_coreset import greedy_coreset


def test_d_is_non_increasing_and_order_unique():
    x = np.random.default_rng(0).standard_normal((2000, 256), dtype=np.float32)
    c = greedy_coreset(x, 200, seed=1)
    assert len(c.order) == 200 and len(set(c.order.tolist())) == 200
    assert np.all(np.diff(c.d) <= 1e-6)
    assert np.all(np.diff(c.seconds) >= 0) and c.setup_s(200) == c.seconds[-1]


def test_every_separated_cluster_is_covered():
    # Five far-apart 2-D clusters (cf. PatchCore Fig. 3): five picks land in five clusters.
    rng = np.random.default_rng(2)
    centres = np.array([[0, 0], [50, 0], [0, 50], [50, 50], [25, 100]], np.float32)
    x = np.concatenate([c + rng.normal(0, 1, (200, 2)) for c in centres]).astype(np.float32)
    c = greedy_coreset(x, 5, seed=3)
    assert sorted(set(c.order // 200)) == [0, 1, 2, 3, 4]


def test_same_seed_same_order_different_seed_different_start():
    x = np.random.default_rng(4).standard_normal((500, 64), dtype=np.float32)
    a, b, other = (greedy_coreset(x, 50, seed=s) for s in (7, 7, 8))
    assert np.array_equal(a.order, b.order) and np.array_equal(a.d, b.d)
    assert a.order[0] != other.order[0]


def test_prefix_property():
    # A 1% coreset is exactly the start of the 10% run.
    x = np.random.default_rng(5).standard_normal((1000, 32), dtype=np.float32)
    assert np.array_equal(
        greedy_coreset(x, 10, seed=9).order, greedy_coreset(x, 100, seed=9).order[:10]
    )


def test_rejects_bad_k():
    with pytest.raises(ValueError):
        greedy_coreset(np.zeros((3, 2), np.float32), 4, seed=0)
