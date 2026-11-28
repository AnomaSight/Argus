"""Fake C2 run folders for tests and for building the analysis before real runs exist (P1.5).

Writes a C1 split manifest and a matching C2 run folder (scores.parquet, run.json, float16
heatmaps). Nothing here is a result: scores are drawn from normal distributions, with good
images shifted upwards as memory shrinks so the report has something to show.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLE_ORDER = ("support", "calib", "eval_normal", "test")
# How far good images drift up, in units of their spread, as memory shrinks.
SETTING_SHIFT = {"m100": 0.0, "m10": 0.1, "aso": 0.15, "m1": 0.4}
SETTING_FRACTION = {"m100": 1.0, "m10": 0.1, "aso": 0.03, "m1": 0.01}
DEFECT_TYPES = ("crack", "cut", "hole", "print")
PATCHES_PER_IMAGE = 784  # 28 x 28 feature map at 224 x 224
SCORES_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("role", pa.string()),
        ("label", pa.int8()),
        ("defect_type", pa.string()),
        ("score", pa.float32()),
        ("heatmap_path", pa.string()),
        ("mask_path", pa.string()),
    ]
)


def _manifest_rows(
    dataset, category, n_support, n_calib, n_eval_normal, n_test_good, n_test_defect
):
    base = f"{dataset}/{category}"
    rows = []
    goods = n_support + n_calib + n_eval_normal
    roles = ["support"] * n_support + ["calib"] * n_calib + ["eval_normal"] * n_eval_normal
    for i, role in enumerate(roles[:goods]):
        rows.append((f"{base}/train/good/{i:03d}", role, 0, "good", ""))
    for i in range(n_test_good):
        rows.append((f"{base}/test/good/{i:03d}", "test", 0, "good", ""))
    for i in range(n_test_defect):
        t = DEFECT_TYPES[i % len(DEFECT_TYPES)]
        rows.append(
            (f"{base}/test/{t}/{i:03d}", "test", 1, t, f"{base}/ground_truth/{t}/{i:03d}_mask.png")
        )
    rows.sort(key=lambda r: (ROLE_ORDER.index(r[1]), r[0]))
    return rows


def _heatmap(rng, size, score, defect):
    """Smooth float16 map; defects get one hot blob. Max stays below the image score."""
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = rng.normal(0, 0.05, size).astype(np.float32) + 0.6 * score
    if defect:
        cy, cx, r = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w, 0.12 * min(h, w)
        base += 0.35 * score * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    return np.minimum(base, 0.95 * score).astype(np.float16)


def write_fake_run(
    artifacts: str | Path,
    *,
    dataset: str = "mvtec_ad",
    category: str = "hazelnut",
    setting: str = "m100",
    seed: int = 20260801,
    n_support: int = 200,
    n_calib: int = 100,
    n_eval_normal: int = 91,
    n_test_good: int = 40,
    n_test_defect: int = 70,
    input_size: tuple[int, int] = (224, 224),
    normal_mean: float = 2.0,
    normal_sd: float = 0.25,
    defect_shift: float = 4.0,
) -> Path:
    """Write splits/... and scores/... under `artifacts`; return the run folder."""
    if setting not in SETTING_SHIFT:
        raise ValueError(f"setting must be one of {sorted(SETTING_SHIFT)}, got {setting!r}")
    artifacts = Path(artifacts)
    rng = np.random.default_rng([seed, list(SETTING_SHIFT).index(setting)])
    rows = _manifest_rows(
        dataset, category, n_support, n_calib, n_eval_normal, n_test_good, n_test_defect
    )

    manifest_rel = f"splits/{dataset}/{category}/seed{seed}.csv"
    manifest = artifacts / manifest_rel
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["image_id", "path", "role", "label", "defect_type", "mask_path"])
        for image_id, role, label, dtype, mask in rows:
            out.writerow([image_id, f"{image_id}.png", role, label, dtype, mask])

    run = artifacts / "scores" / dataset / category / setting / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    scored = [r for r in rows if r[1] != "support"]
    shift = SETTING_SHIFT[setting] * normal_sd
    cols: dict[str, list] = {name: [] for name in SCORES_SCHEMA.names}
    for image_id, role, label, dtype, mask in scored:
        mean = normal_mean + shift + (defect_shift * normal_sd if label else 0.0)
        score = float(np.float32(abs(rng.normal(mean, normal_sd))))
        rel = f"heatmaps/{image_id}.npy"
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(run / rel, _heatmap(rng, input_size, score, label == 1))
        for name, value in zip(
            SCORES_SCHEMA.names, (image_id, role, label, dtype, score, rel, mask), strict=True
        ):
            cols[name].append(value)
    pq.write_table(pa.table(cols, schema=SCORES_SCHEMA), run / "scores.parquet")

    fraction = SETTING_FRACTION[setting]
    coreset = round(n_support * PATCHES_PER_IMAGE * fraction)
    params = f"{dataset}/{category}/{setting}/{seed}/{input_size}"
    info = {
        "schema_version": 1,
        "config_hash": hashlib.sha256(params.encode()).hexdigest()[:12],
        "dataset": dataset,
        "category": category,
        "setting": setting,
        "seed": seed,
        "split_manifest": manifest_rel,
        "preprocess": "patchcore",
        "input_size": list(input_size),
        "coreset_size": coreset,
        "coreset_fraction": fraction,
        "bank_mb": round(coreset * 1024 * 4 / 2**20, 2),
        "setup_s": 0.0 if setting == "m100" else round(400 * fraction, 1),
        "ms_per_image": round(40 + 30 * fraction, 1),
        "device": "cpu",
        "git_commit": "fake",
    }
    (run / "run.json").write_text(json.dumps(info, indent=2) + "\n")
    return run
