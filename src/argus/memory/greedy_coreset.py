"""Greedy coreset that records its selection order and d(k) (task A2.1).

PatchCore's greedy k-center selection (Algorithm 1) on a random 128-dim projection
(Johnson-Lindenstrauss), used for selection only. Because points are added one at a time,
the 1% and ASO coresets are prefixes of the 10% run's order: one run gives every setting.

d[k-1] is the covering radius of the first k points: the largest distance from any point to
its nearest selected point. It is non-increasing, and ASO (A2.2) reads it as its curve.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)
PROJECTION_DIM = 128
TIMING_EVERY = 256  # steps between synchronised timestamps


@dataclass(frozen=True)
class Coreset:
    order: np.ndarray  # (K,) int64 indices into the features, in selection order
    d: np.ndarray  # (K,) float32 covering radius after 1..K points
    seconds: np.ndarray  # (K,) float64 elapsed selection time after 1..K points

    def setup_s(self, k: int) -> float:
        """Seconds of greedy selection needed to reach k points (C2's setup_s)."""
        return float(self.seconds[k - 1])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def greedy_coreset(features, k: int, seed: int, device: str | torch.device = "cpu") -> Coreset:
    """Select k of the N rows of features (N, D) greedily, starting from a seeded random row."""
    device = torch.device(device)
    x = torch.as_tensor(features).to(device=device, dtype=torch.float32)
    n = len(x)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    gen = torch.Generator().manual_seed(seed)
    if x.shape[1] > PROJECTION_DIM:
        proj = torch.randn(x.shape[1], PROJECTION_DIM, generator=gen) / PROJECTION_DIM**0.5
        x = x @ proj.to(device)
    idx = int(torch.randint(n, (1,), generator=gen))

    order = torch.empty(k, dtype=torch.long, device=device)
    d = torch.empty(k, dtype=torch.float32, device=device)
    min_d = torch.full((n,), float("inf"), device=device)
    stamps: list[tuple[int, float]] = []
    _sync(device)
    start = time.perf_counter()
    for j in range(k):
        order[j] = idx
        min_d = torch.minimum(min_d, torch.linalg.vector_norm(x - x[idx], dim=1))
        d[j], nxt = torch.max(min_d, dim=0)
        idx = int(nxt)
        if (j + 1) % TIMING_EVERY == 0 or j + 1 == k:
            _sync(device)
            stamps.append((j + 1, time.perf_counter() - start))
    steps, secs = zip(*stamps, strict=True)
    seconds = np.interp(np.arange(1, k + 1), (0, *steps), (0.0, *secs))
    log.info("greedy coreset: %d of %d points in %.1f s on %s", k, n, seconds[-1], device)
    return Coreset(order.cpu().numpy(), d.cpu().numpy(), seconds)
