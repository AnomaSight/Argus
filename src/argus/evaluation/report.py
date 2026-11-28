"""Results tables from C2 run folders (task P2.2).

For each run (setting x seed) of a dataset/category: check the folder (P2.3), fit tau on
`calib` rows only and save it (C4), then, behind the test-set lock, read `eval_normal` and
`test` rows and measure FPR, recall, image AUROC, pixel AUROC and PRO. Seeds are summarised
per setting as mean and min-max; the FPR and recall CIs are the envelope of the per-seed
Wilson intervals (lowest low, highest high), since seeds share one test set.

Writes artifacts/results/{dataset}/{category}/memory_sweep_alpha{a:g}{.csv,.md,_seeds.csv}.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from argus.calibration.tolerance import (
    CalibrationSizeError,
    fit_threshold,
    load_threshold,
    save_threshold,
)
from argus.data.transforms import load_mask, mask_transform
from argus.evaluation.image_metrics import image_auroc
from argus.evaluation.operating_point import operating_point
from argus.evaluation.pixel_metrics import pixel_metrics
from argus.evaluation.schema_check import assert_valid
from argus.utils.config import REPO_ROOT

SETTING_ORDER = ("m100", "m10", "m1", "aso")
METRICS = ("fpr", "recall", "image_auroc", "pixel_auroc", "pro")  # mean, min, max
COSTS = ("bank_mb", "ms_per_image", "setup_s")  # mean only
TARGETS_CONFIG = REPO_ROOT / "configs" / "experiments" / "fpr_targets.yaml"


def load_targets(path: Path = TARGETS_CONFIG) -> tuple[list[float], float]:
    cfg = yaml.safe_load(Path(path).read_text())
    return [float(a) for a in cfg["targets"]], float(cfg["delta"])


def calibrate(run_dir: Path, alpha: float, delta: float) -> None:
    """Fit tau on calib rows (the only rows read) and save it. Raises CalibrationSizeError."""
    calib = pq.read_table(
        run_dir / "scores.parquet", columns=["score"], filters=[("role", "==", "calib")]
    )
    save_threshold(fit_threshold(calib.column("score").to_numpy(), alpha, delta), run_dir)


def _pixel_pairs(run_dir: Path, test: pd.DataFrame, data_dir: Path, preset: str):
    to_mask = mask_transform(preset)
    for row in test.itertuples():
        if row.label == 1 and not row.mask_path:
            continue  # a defect without a mask cannot be scored per pixel
        heatmap = np.load(run_dir / row.heatmap_path)
        mask = (
            to_mask(load_mask(data_dir, row.mask_path))
            if row.label == 1
            else np.zeros(heatmap.shape, dtype=bool)
        )
        yield heatmap, mask


def evaluate(run_dir: Path, alpha: float, delta: float, data_dir: Path | None = None) -> dict:
    """Metrics for one run at one target. Raises unless `calibrate` saved a current tau."""
    tau = load_threshold(run_dir, alpha, delta).tau  # test-set lock: before any eval row
    df = pq.read_table(
        run_dir / "scores.parquet", filters=[("role", "in", ["eval_normal", "test"])]
    ).to_pandas()
    info = json.loads((run_dir / "run.json").read_text())
    op = operating_point(df.score, df.label, tau)
    test = df[df.role == "test"]
    pixel_auroc = pro = math.nan
    if data_dir is not None:
        pixel_auroc, pro = pixel_metrics(_pixel_pairs(run_dir, test, data_dir, info["preprocess"]))
    return {
        "setting": info["setting"],
        "seed": info["seed"],
        "tau": tau,
        "fpr": op.fpr,
        "fpr_ci_low": op.fpr_ci[0],
        "fpr_ci_high": op.fpr_ci[1],
        "recall": op.recall,
        "recall_ci_low": op.recall_ci[0],
        "recall_ci_high": op.recall_ci[1],
        "image_auroc": image_auroc(test.score, test.label),
        "pixel_auroc": pixel_auroc,
        "pro": pro,
        **{k: info[k] for k in COSTS},
    }


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    """One row per setting: mean/min/max of each metric, CI envelope, mean costs."""
    rows = []
    for setting, g in per_seed.groupby("setting", sort=False):
        row = {"setting": setting, "seeds": len(g)}
        for m in METRICS:
            row |= {m: g[m].mean(), f"{m}_min": g[m].min(), f"{m}_max": g[m].max()}
            if m in ("fpr", "recall"):
                row |= {
                    f"{m}_ci_low": g[f"{m}_ci_low"].min(),
                    f"{m}_ci_high": g[f"{m}_ci_high"].max(),
                }
        row |= {k: g[k].mean() for k in COSTS}
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(SETTING_ORDER)}
    return out.sort_values("setting", key=lambda s: s.map(order)).reset_index(drop=True)


def _cell(row, m: str, ci: bool = False) -> str:
    if math.isnan(row[m]):
        return "n/a"
    text = f"{row[m]:.3f}"
    if row["seeds"] > 1:
        text += f" ({row[f'{m}_min']:.3f}–{row[f'{m}_max']:.3f})"
    if ci:
        text += f"<br>CI {row[f'{m}_ci_low']:.3f}–{row[f'{m}_ci_high']:.3f}"
    return text


def to_markdown(summary: pd.DataFrame, title: str) -> str:
    cols = ["Setting", "FPR", "Recall", "Image AUROC", "Pixel AUROC", "PRO"]
    cols += ["Bank MB", "ms/image", "Setup s"]
    lines = [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in summary.iterrows():
        cells = [
            f"{r.setting}",
            _cell(r, "fpr", ci=True),
            _cell(r, "recall", ci=True),
            _cell(r, "image_auroc"),
            _cell(r, "pixel_auroc"),
            _cell(r, "pro"),
            f"{r.bank_mb:.2f}",
            f"{r.ms_per_image:.1f}",
            f"{r.setup_s:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Mean over seeds, (min–max) when there is more than one; CI is the 95% Wilson",
        "interval envelope across seeds. Flagged means score > tau.",
        "",
    ]
    return "\n".join(lines)


def find_runs(artifacts: Path, dataset: str, category: str) -> list[Path]:
    return sorted((artifacts / "scores" / dataset / category).glob("*/seed*/"))


def build_report(
    artifacts: Path,
    dataset: str,
    category: str,
    alpha: float,
    delta: float,
    data_dir: Path | None = None,
) -> Path | None:
    """Write the table for one target; return the CSV path, or None if every run was skipped."""
    runs = find_runs(artifacts, dataset, category)
    if not runs:
        raise FileNotFoundError(f"no runs under {artifacts / 'scores' / dataset / category}")
    results, skipped = [], []
    for run in runs:
        assert_valid(run)
        try:
            calibrate(run, alpha, delta)
        except CalibrationSizeError as e:
            skipped.append(e)
            continue
        results.append(evaluate(run, alpha, delta, data_dir))
    if skipped:
        print(
            f"warning: alpha={alpha:g}: skipped {len(skipped)} of {len(runs)} runs: {skipped[0]}",
            file=sys.stderr,
        )
    if not results:
        return None
    order = {s: i for i, s in enumerate(SETTING_ORDER)}
    per_seed = pd.DataFrame(results).sort_values(
        ["setting", "seed"], key=lambda c: c.map(order) if c.name == "setting" else c
    )
    out = artifacts / "results" / dataset / category
    out.mkdir(parents=True, exist_ok=True)
    stem = f"memory_sweep_alpha{alpha:g}"
    summary = summarise(per_seed)
    summary.to_csv(out / f"{stem}.csv", index=False)
    per_seed.to_csv(out / f"{stem}_seeds.csv", index=False)
    title = f"{dataset} / {category}, FPR target {alpha:g} (delta {delta:g})"
    (out / f"{stem}.md").write_text(to_markdown(summary, title))
    return out / f"{stem}.csv"
