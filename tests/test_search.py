import numpy as np
import pytest
import torch

from argus.memory.search import knn


def _brute(q, b, k):
    d = np.sqrt(((q[:, None, :].astype(np.float64) - b[None, :, :]) ** 2).sum(-1))
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, idx, 1), idx


@pytest.mark.parametrize("k, qc, bc", [(1, 4096, 65536), (1, 7, 13), (3, 5, 11), (5, 64, 2)])
def test_matches_brute_force(k, qc, bc):
    rng = np.random.default_rng(k + qc)
    bank = rng.standard_normal((300, 32), dtype=np.float32)
    queries = rng.standard_normal((40, 32), dtype=np.float32)
    d, i = knn(queries, bank, k=k, query_chunk=qc, bank_chunk=bc)
    bd, bi = _brute(queries, bank, k)
    assert np.array_equal(i.numpy(), bi)
    assert np.allclose(d.numpy(), bd, atol=1e-4)


def test_query_in_bank_finds_itself_at_zero():
    bank = torch.randn(50, 8)
    d, i = knn(bank[[3, 17]], bank, bank_chunk=7)
    assert i[:, 0].tolist() == [3, 17] and torch.all(d[:, 0] < 1e-3)


def test_rejects_bad_shapes_and_k():
    with pytest.raises(ValueError):
        knn(np.zeros((2, 3)), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        knn(np.zeros((2, 3)), np.zeros((4, 3)), k=5)


def test_small_distances_are_exact_despite_large_norms():
    # Large-norm vectors make cdist's matrix-multiply path inexact near zero.
    bank = torch.randn(5000, 1024) * 0.3 + 5
    d, i = knn(bank[:100], bank)
    assert torch.equal(i[:, 0], torch.arange(100)) and d.max().item() == 0.0
