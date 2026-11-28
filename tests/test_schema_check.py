import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from argus.cli import main
from argus.evaluation.fake_scores import SCORES_SCHEMA
from argus.evaluation.schema_check import SchemaError, assert_valid, check_run


def _rewrite(run, change, keep_schema=True):
    df = pd.read_parquet(run / "scores.parquet")
    df = change(df)
    table = pa.Table.from_pandas(
        df, schema=SCORES_SCHEMA if keep_schema else None, preserve_index=False
    )
    # Keep NaN as a float NaN, not a null (from_pandas would turn it into a null).
    score = pa.array(df.score.to_numpy(), type=table.schema.field("score").type)
    table = table.set_column(table.schema.get_field_index("score"), "score", score)
    pq.write_table(table, run / "scores.parquet")


def _edit_info(run, **kw):
    info = json.loads((run / "run.json").read_text())
    info.update(kw)
    for k in [k for k, v in kw.items() if v is None]:
        del info[k]
    (run / "run.json").write_text(json.dumps(info))


def _null_mask(run):
    t = pq.read_table(run / "scores.parquet")
    masks = t.column("mask_path").to_pylist()
    masks[0] = None
    return t.set_column(
        t.schema.get_field_index("mask_path"), "mask_path", pa.array(masks, pa.string())
    )


@pytest.mark.parametrize("setting", ["m100", "m10", "m1", "aso"])
def test_fake_runs_pass(fake_run, setting):
    assert check_run(fake_run(setting=setting)) == []


def _set(col, i, value):
    def change(df):
        df.loc[i, col] = value
        return df

    return change


CORRUPTIONS = {
    "nan score": (lambda r: _rewrite(r, _set("score", 3, np.nan)), "NaN or infinite"),
    "duplicate id": (
        lambda r: _rewrite(
            r, lambda df: df.assign(image_id=df.image_id.where(df.index != 1, df.image_id[0]))
        ),
        "not unique",
    ),
    "bad role": (lambda r: _rewrite(r, _set("role", 0, "support")), "not allowed"),
    "defect outside test": (
        lambda r: _rewrite(
            r, lambda df: df.assign(label=df.label.where(df.index != 0, np.int8(1)))
        ),
        "outside role 'test'",
    ),
    "null value": (
        lambda r: pq.write_table(_null_mask(r), r / "scores.parquet"),
        "has nulls",
    ),
    "label vs type": (lambda r: _rewrite(r, _set("defect_type", 0, "crack")), "defect_type 'good'"),
    "mask on good": (
        lambda r: _rewrite(r, _set("mask_path", 0, "x.png")),
        "mask_path must be empty",
    ),
    "wrong type": (
        lambda r: _rewrite(
            r, lambda df: df.assign(label=df.label.astype("int64")), keep_schema=False
        ),
        "has type int64",
    ),
    "missing column": (
        lambda r: pq.write_table(
            pq.read_table(r / "scores.parquet").drop(["mask_path"]), r / "scores.parquet"
        ),
        "columns are",
    ),
    "unsorted": (
        lambda r: _rewrite(r, lambda df: df.iloc[::-1].reset_index(drop=True)),
        "not sorted",
    ),
    "missing heatmap": (
        lambda r: next((r / "heatmaps").rglob("*.npy")).unlink(),
        "heatmap missing",
    ),
    "heatmap dtype": (
        lambda r: [
            np.save(p, np.zeros((16, 16), np.float32)) for p in (r / "heatmaps").rglob("*.npy")
        ],
        "expected float16",
    ),
    "run.json key": (lambda r: _edit_info(r, bank_mb=None), "missing key 'bank_mb'"),
    "schema version": (lambda r: _edit_info(r, schema_version=2), "schema_version is 2"),
    "folder mismatch": (lambda r: _edit_info(r, setting="m1"), "does not match folder name 'm100'"),
    "manifest mismatch": (
        lambda r: _rewrite(r, lambda df: df[df.index != 5].reset_index(drop=True)),
        "1 missing, 0 extra",
    ),
}


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_each_corruption_has_its_own_message(fake_run, name):
    run = fake_run(setting="m100")
    corrupt, message = CORRUPTIONS[name]
    corrupt(run)
    problems = check_run(run)
    assert any(message in p for p in problems), problems
    with pytest.raises(SchemaError):
        assert_valid(run)


def test_cli_exit_codes(fake_run, capsys):
    run = fake_run()
    assert main(["check-scores", str(run)]) == 0
    _edit_info(run, schema_version=3)
    assert main(["check-scores", str(run)]) == 1
    assert "schema_version" in capsys.readouterr().out
