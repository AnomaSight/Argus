import json

import numpy as np
import pytest

from argus.calibration.tolerance import (
    CalibrationSizeError,
    fit_threshold,
    load_threshold,
    save_threshold,
)


@pytest.mark.parametrize(
    "alpha, rank, confidence",
    [(0.10, 142, 0.9693), (0.05, 148, 0.9818), (0.02, 150, 0.9517)],
)
def test_deng_table_6_for_m_150(alpha, rank, confidence):
    scores = np.random.default_rng(0).normal(size=150)
    t = fit_threshold(scores, alpha)
    assert (t.rank, t.m) == (rank, 150)
    assert t.achieved_confidence == pytest.approx(confidence, abs=5e-5)
    assert t.tau == np.sort(scores)[rank - 1]


def test_alpha_1_percent_needs_more_than_150():
    with pytest.raises(CalibrationSizeError) as e:
        fit_threshold(np.arange(150.0), 0.01)
    assert (e.value.m, e.value.m_required) == (150, 299)


def test_c4_size_error_message():
    with pytest.raises(CalibrationSizeError, match="need at least 59 .* got 40") as e:
        fit_threshold(np.arange(40.0), 0.05)
    assert e.value.m_required == 59


def test_ties_keep_strict_rule():
    # All-equal scores: tau is that value, and score > tau flags none of them.
    scores = np.full(100, 2.0)
    t = fit_threshold(scores, 0.05)
    assert t.tau == 2.0
    assert not np.any(scores > t.tau)


@pytest.mark.parametrize("bad", [[], [1.0, np.nan], [[1.0, 2.0]]])
def test_rejects_bad_scores(bad):
    with pytest.raises(ValueError):
        fit_threshold(bad, 0.05)


def test_monte_carlo_guarantee_holds():
    # 10,000 calibrations of m = 100 from Uniform(0, 1), where the true FPR of tau is 1 - tau.
    # The guarantee says FPR <= alpha in at least 95% of them (here ~96.3% expected).
    rng = np.random.default_rng(20260801)
    m, alpha = 100, 0.05
    sims = np.sort(rng.random((10_000, m)), axis=1)
    rank = fit_threshold(sims[0], alpha).rank
    true_fpr = 1 - sims[:, rank - 1]
    assert np.mean(true_fpr <= alpha) >= 0.95


def _fake_run(tmp_path, config_hash="abc123"):
    (tmp_path / "run.json").write_text(json.dumps({"config_hash": config_hash}))
    (tmp_path / "scores.parquet").write_bytes(b"not really parquet, only hashed")
    return tmp_path


def test_save_and_load_round_trip(tmp_path):
    run = _fake_run(tmp_path)
    t = fit_threshold(np.arange(100.0), 0.05)
    path = save_threshold(t, run)
    assert path.name == "alpha0.05_delta0.05.json"
    assert load_threshold(run, 0.05) == t
    assert (
        save_threshold(fit_threshold(np.arange(100.0), 0.1), run).name == "alpha0.1_delta0.05.json"
    )


def test_load_refuses_missing_and_stale(tmp_path):
    run = _fake_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_threshold(run, 0.05)
    save_threshold(fit_threshold(np.arange(100.0), 0.05), run)
    (run / "scores.parquet").write_bytes(b"rescored")
    with pytest.raises(ValueError, match="stale"):
        load_threshold(run, 0.05)
    _fake_run(tmp_path, config_hash="other")
    with pytest.raises(ValueError, match="stale"):
        load_threshold(run, 0.05)
