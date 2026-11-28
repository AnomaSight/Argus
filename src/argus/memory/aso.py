"""ASO: automatic memory size from the greedy curve (task A2.2, ASO paper Eq. 5-8).

Both axes are normalised by the 10% run's end point: k~ = k / k10, d~ = d(k) / d(k10). The
chosen size k* minimises the distance sqrt(k~^2 + d~^2) to the ideal corner (small and
well covered). Without normalisation k dwarfs d, and the rule collapses to k = 1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AsoResult:
    k: int  # chosen coreset size
    fraction: float  # k / N (share of all support patches)
    k_norm: np.ndarray  # (K,) k / k10 for k = 1..K
    d_norm: np.ndarray  # (K,) d(k) / d(k10)
    dist: np.ndarray  # (K,) sqrt(k_norm^2 + d_norm^2)

    def save_curve(self, path: str | Path) -> Path:
        """CSV of the normalised curve, for the write-up figure (P4.2)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            out = csv.writer(f, lineterminator="\n")
            out.writerow(["k", "k_norm", "d_norm", "dist", "chosen"])
            for i, row in enumerate(zip(self.k_norm, self.d_norm, self.dist, strict=True)):
                out.writerow([i + 1, *(f"{v:.6g}" for v in row), int(i + 1 == self.k)])
        return path


def aso(d: np.ndarray, n_total: int, normalise: bool = True) -> AsoResult:
    """d: covering radii for k = 1..k10 (Coreset.d). n_total: all support patches."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or len(d) == 0 or not np.all(np.isfinite(d)) or d[-1] <= 0:
        raise ValueError("d must be a non-empty 1-D array of finite values with d[-1] > 0")
    k = np.arange(1, len(d) + 1, dtype=np.float64)
    k_norm, d_norm = (k / len(d), d / d[-1]) if normalise else (k, d)
    dist = np.hypot(k_norm, d_norm)
    best = int(np.argmin(dist)) + 1
    return AsoResult(best, best / n_total, k_norm, d_norm, dist)
