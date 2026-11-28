import math

import pandas as pd
import pytest

from argus.cli import main
from argus.evaluation.fake_scores import write_fake_run
from argus.evaluation.report import build_report, calibrate, evaluate, load_targets

SMALL = dict(n_support=10, n_calib=60, n_eval_normal=20, n_test_good=6, n_test_defect=8)


def test_four_row_table_with_pixel_metrics(tmp_path):
    art, data = tmp_path / "artifacts", tmp_path / "data"
    for setting in ("m100", "m10", "m1", "aso"):
        write_fake_run(art, setting=setting, data_dir=data, **SMALL)
    csv = build_report(art, "mvtec_ad", "hazelnut", 0.05, 0.05, data_dir=data)

    t = pd.read_csv(csv)
    assert list(t.setting) == ["m100", "m10", "m1", "aso"]
    assert (t.seeds == 1).all()
    assert t[["fpr", "recall", "image_auroc", "pixel_auroc", "pro"]].notna().all().all()
    assert (t.pixel_auroc > 0.9).all() and (t.pro > 0.5).all()  # fake blobs sit on the masks
    assert (t.fpr_ci_low <= t.fpr).all() and (t.fpr <= t.fpr_ci_high).all()
    assert t.bank_mb.iloc[0] > t.bank_mb.iloc[1] > t.bank_mb.iloc[2]
    assert "| m100 |" in csv.with_suffix(".md").read_text()
    assert len(pd.read_csv(csv.with_name(csv.stem + "_seeds.csv"))) == 4


def test_seeds_aggregate_to_mean_and_range(fake_run):
    for seed in (1, 2, 3):
        fake_run(setting="m10", seed=seed)
    art = fake_run(setting="m10", seed=1).parents[4]
    t = pd.read_csv(build_report(art, "mvtec_ad", "hazelnut", 0.1, 0.05))
    seeds = pd.read_csv(art / "results/mvtec_ad/hazelnut/memory_sweep_alpha0.1_seeds.csv")
    row = t.iloc[0]
    assert row.seeds == 3 and row.fpr == pytest.approx(seeds.fpr.mean())
    assert row.fpr_min == seeds.fpr.min() and row.fpr_ci_high == seeds.fpr_ci_high.max()
    assert math.isnan(row.pixel_auroc)  # no data_dir: pixel metrics left empty


def test_evaluate_refuses_without_threshold(fake_run):
    run = fake_run()
    with pytest.raises(FileNotFoundError, match="no threshold"):
        evaluate(run, 0.05, 0.05)
    calibrate(run, 0.05, 0.05)
    assert evaluate(run, 0.05, 0.05)["setting"] == "m100"


def test_too_few_calibration_images_skips_with_warning(fake_run, capsys):
    run = fake_run(n_calib=40)
    assert build_report(run.parents[4], "mvtec_ad", "hazelnut", 0.05, 0.05) is None
    assert "need at least 59" in capsys.readouterr().err


def test_refuses_invalid_run(fake_run):
    run = fake_run()
    (run / "run.json").unlink()
    with pytest.raises(ValueError):
        build_report(run.parents[4], "mvtec_ad", "hazelnut", 0.05, 0.05)


def test_targets_config():
    assert load_targets() == ([0.10, 0.05, 0.02], 0.05)


def test_cli_writes_a_table_per_target(fake_run, capsys):
    art = fake_run().parents[4]
    main(
        ["report", "--dataset", "mvtec_ad", "--category", "hazelnut", "--artifacts", str(art)]
        + ["--skip-pixel"]
    )
    out = art / "results/mvtec_ad/hazelnut"
    assert (out / "memory_sweep_alpha0.1.csv").exists()
    assert (out / "memory_sweep_alpha0.05.md").exists()
    assert not (out / "memory_sweep_alpha0.02.csv").exists()  # 100 calib < 149
