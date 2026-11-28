import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from argus.cli import main


def test_writes_c2_folder(fake_run):
    run = fake_run(setting="m10")
    df = pd.read_parquet(run / "scores.parquet")
    info = json.loads((run / "run.json").read_text())
    assert list(df.columns) == [
        "image_id",
        "role",
        "label",
        "defect_type",
        "score",
        "heatmap_path",
        "mask_path",
    ]
    types = {f.name: str(f.type) for f in pq.read_schema(run / "scores.parquet")}
    assert types["label"] == "int8" and types["score"] == "float"
    assert len(df) == 100 + 91 + 40 + 70 and df.image_id.is_unique
    assert set(df.role) == {"calib", "eval_normal", "test"}
    assert (df.label == 1).sum() == 70 and set(df[df.label == 1].role) == {"test"}
    assert (df.label == 0).eq(df.defect_type == "good").all()
    assert (df[df.label == 0].mask_path == "").all()
    assert info["setting"] == "m10" and info["input_size"] == [16, 16]
    h = np.load(run / df.heatmap_path.iloc[-1])
    assert h.dtype == np.float16 and h.shape == (16, 16)


def test_same_seed_same_scores(fake_run, tmp_path):
    a = pd.read_parquet(fake_run(seed=1) / "scores.parquet")
    b = pd.read_parquet(fake_run(seed=1) / "scores.parquet")
    assert a.equals(b)


def test_good_scores_drift_up_as_memory_shrinks(fake_run):
    def mean_good(setting):
        df = pd.read_parquet(fake_run(setting=setting) / "scores.parquet")
        return df[df.label == 0].score.mean()

    assert mean_good("m1") > mean_good("m100")


def test_cli_writes_every_setting_and_seed(tmp_path, capsys):
    main(["fake-scores", "--artifacts", str(tmp_path), "--settings", "m100", "m1", "--seeds", "7"])
    assert (tmp_path / "scores/mvtec_ad/hazelnut/m1/seed7/run.json").exists()
    assert (tmp_path / "splits/mvtec_ad/hazelnut/seed7.csv").exists()
