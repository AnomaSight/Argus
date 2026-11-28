"""Distribution-free cut-off with a false-positive guarantee (contract C4, task P1.1).

fit_threshold picks tau as the k-th smallest calibration score (Wilks, as in Deng). If the
calibration images and new good images are independent draws from the same conditions,
then with probability at least 1 - delta the true FPR of `score > tau` is at most alpha.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from argus.calibration.sample_size import min_calibration_size, wilks_rank

SCHEMA_VERSION = 1


class CalibrationSizeError(ValueError):
    """Too few calibration scores for the requested alpha and delta."""

    def __init__(self, m: int, m_required: int, alpha: float, delta: float):
        self.m = m
        self.m_required = m_required
        super().__init__(
            f"need at least {m_required} calibration scores for alpha={alpha:g}, "
            f"delta={delta:g}; got {m}"
        )


@dataclass(frozen=True)
class Threshold:
    tau: float
    rank: int  # 1-based position of tau in the sorted calib scores
    m: int  # number of calib scores
    alpha: float  # FPR target
    delta: float  # 1 - confidence
    achieved_confidence: float  # >= 1 - delta


def fit_threshold(calib_scores: ArrayLike, alpha: float, delta: float = 0.05) -> Threshold:
    """Fit tau on calibration scores only. Images are flagged when score > tau."""
    scores = np.asarray(calib_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"calib_scores must be 1-D, got shape {scores.shape}")
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("calib_scores must be non-empty and all finite")
    m = scores.size
    hit = wilks_rank(m, alpha, delta)  # validates alpha and delta
    if hit is None:
        raise CalibrationSizeError(m, min_calibration_size(alpha, delta), alpha, delta)
    rank, confidence = hit
    tau = float(np.sort(scores)[rank - 1])
    return Threshold(tau, rank, m, alpha, delta, confidence)


def _threshold_path(run_dir: Path, alpha: float, delta: float) -> Path:
    return run_dir / "thresholds" / f"alpha{alpha:g}_delta{delta:g}.json"


def _run_identity(run_dir: Path) -> tuple[str, str]:
    """(config_hash from run.json, SHA-256 of scores.parquet): what a threshold is tied to."""
    config_hash = json.loads((run_dir / "run.json").read_text())["config_hash"]
    digest = hashlib.sha256((run_dir / "scores.parquet").read_bytes()).hexdigest()
    return config_hash, digest


def save_threshold(threshold: Threshold, run_dir: str | Path) -> Path:
    """Write run_dir/thresholds/alpha{alpha:g}_delta{delta:g}.json and return its path."""
    run_dir = Path(run_dir)
    config_hash, digest = _run_identity(run_dir)
    path = _threshold_path(run_dir, threshold.alpha, threshold.delta)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": SCHEMA_VERSION, **asdict(threshold)}
    body |= {"config_hash": config_hash, "scores_sha256": digest}
    path.write_text(json.dumps(body, indent=2) + "\n")
    return path


def load_threshold(run_dir: str | Path, alpha: float, delta: float = 0.05) -> Threshold:
    """Read a saved threshold. This is the test-set lock: call it before reading eval rows.

    Raises FileNotFoundError if it was never fitted, and ValueError if the run's scores or
    config changed since (the threshold is stale and must be refitted on calib rows).
    """
    run_dir = Path(run_dir)
    path = _threshold_path(run_dir, alpha, delta)
    if not path.exists():
        raise FileNotFoundError(f"no threshold fitted for alpha={alpha:g}, delta={delta:g}: {path}")
    body = json.loads(path.read_text())
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported threshold schema_version in {path}")
    config_hash, digest = _run_identity(run_dir)
    if body["config_hash"] != config_hash or body["scores_sha256"] != digest:
        raise ValueError(f"stale threshold, the run changed since it was fitted: {path}")
    return Threshold(**{f: body[f] for f in Threshold.__dataclass_fields__})
