"""Exact k-nearest-neighbour search against a memory bank (task A1.5).

Chunked over both queries and bank, keeping a running top-k, so memory stays bounded at
query_chunk x bank_chunk distances whatever the bank size. The same code runs on GPU and CPU.
"""

from __future__ import annotations

import time

import numpy as np
import torch

QUERY_CHUNK = 4096
BANK_CHUNK = 65536


def _as_tensor(x, device: torch.device | None) -> torch.Tensor:
    t = torch.as_tensor(x)
    return t.to(device=device or t.device, dtype=torch.float32)


@torch.inference_mode()
def knn(
    queries,
    bank,
    k: int = 1,
    query_chunk: int = QUERY_CHUNK,
    bank_chunk: int = BANK_CHUNK,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euclidean distances (Q, k) and bank indices (Q, k) of each query's k nearest rows.

    queries (Q, D) and bank (N, D) may be numpy arrays or tensors; results are on `device`
    (default: the bank's device), nearest first.
    """
    device = torch.device(device) if device else None
    b = _as_tensor(bank, device)
    q = _as_tensor(queries, b.device)
    if q.ndim != 2 or b.ndim != 2 or q.shape[1] != b.shape[1]:
        raise ValueError(
            f"queries {tuple(q.shape)} and bank {tuple(b.shape)} must be (Q, D), (N, D)"
        )
    if not 1 <= k <= len(b):
        raise ValueError(f"k must be between 1 and the bank size {len(b)}, got {k}")
    out_d = torch.empty(len(q), k, device=b.device)
    out_i = torch.empty(len(q), k, dtype=torch.long, device=b.device)
    for qs in range(0, len(q), query_chunk):
        qc = q[qs : qs + query_chunk]
        best_d = torch.full((len(qc), k), float("inf"), device=b.device)
        best_i = torch.zeros((len(qc), k), dtype=torch.long, device=b.device)
        for bs in range(0, len(b), bank_chunk):
            d = torch.cdist(qc, b[bs : bs + bank_chunk])
            kk = min(k, d.shape[1])
            cd, ci = torch.topk(d, kk, dim=1, largest=False)
            merged_d = torch.cat([best_d, cd], dim=1)
            merged_i = torch.cat([best_i, ci + bs], dim=1)
            best_d, pos = torch.topk(merged_d, k, dim=1, largest=False)
            best_i = torch.gather(merged_i, 1, pos)
        # cdist's matrix-multiply path loses precision when distances are small next to the
        # vectors' norms (a patch's distance to itself can come out ~0.1). Rank with it, then
        # recompute the chosen neighbours' distances exactly.
        exact = torch.linalg.vector_norm(qc[:, None, :] - b[best_i], dim=-1)
        out_d[qs : qs + len(qc)], out_i[qs : qs + len(qc)] = exact, best_i
    return out_d, out_i


def benchmark(
    n_bank: int = 156_800, n_queries: int = 784, dim: int = 1024, device: str = "cpu"
) -> float:
    """Queries per second for one 224 x 224 image (784 patches) against a hazelnut-sized bank."""
    rng = np.random.default_rng(0)
    bank = torch.from_numpy(rng.standard_normal((n_bank, dim), dtype=np.float32)).to(device)
    q = torch.from_numpy(rng.standard_normal((n_queries, dim), dtype=np.float32)).to(device)
    knn(q[:8], bank)  # warm-up
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    knn(q, bank)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return n_queries / (time.perf_counter() - start)


if __name__ == "__main__":
    import sys

    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(f"{benchmark(device=dev):,.0f} queries/s on {dev}, bank 156,800 x 1024")
