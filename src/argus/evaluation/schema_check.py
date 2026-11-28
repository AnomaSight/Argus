"""C2 score-folder checker (task P2.3). report.py refuses any folder that fails it.

check_run returns every problem found as a specific message; an empty list means valid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = {
    "image_id": "string",
    "role": "string",
    "label": "int8",
    "defect_type": "string",
    "score": "float",  # pyarrow's name for float32
    "heatmap_path": "string",
    "mask_path": "string",
}
RUN_KEYS = (
    "schema_version config_hash dataset category setting seed split_manifest preprocess "
    "input_size coreset_size coreset_fraction bank_mb setup_s ms_per_image device git_commit"
).split()
ROLE_ORDER = ("calib", "eval_normal", "test")
COPIED = ["role", "label", "defect_type", "mask_path"]


class SchemaError(ValueError):
    def __init__(self, run_dir: Path, problems: list[str]):
        self.problems = problems
        super().__init__(f"{run_dir} fails the C2 check:\n  " + "\n  ".join(problems))


def _artifacts_dir(run_dir: Path, stress: bool) -> Path:
    if os.environ.get("ARGUS_ARTIFACTS_DIR"):
        return Path(os.environ["ARGUS_ARTIFACTS_DIR"])
    # .../scores/{dataset}/{category}/{setting}/seed{n}, plus {perturbation}/{severity} for stress
    return run_dir.parents[6 if stress else 4]


def check_run(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    problems: list[str] = []
    stress = "scores_stress" in run_dir.parts

    # Rule 7: run.json
    try:
        info = json.loads((run_dir / "run.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return [f"run.json unreadable: {e}"]
    keys = RUN_KEYS + (["perturbation", "severity"] if stress else [])
    problems += [f"run.json is missing key {k!r}" for k in keys if k not in info]
    if info.get("schema_version") != 1:
        problems.append(f"run.json schema_version is {info.get('schema_version')!r}, expected 1")
    want = dict(zip(("dataset", "category", "setting"), run_dir.parts[-4:-1], strict=True))
    want["seed"] = run_dir.name.removeprefix("seed")
    for k, v in want.items():
        if k in info and str(info[k]) != v:
            problems.append(f"run.json {k}={info[k]!r} does not match folder name {v!r}")

    # Rule 1: columns, types, no nulls
    try:
        table = pq.read_table(run_dir / "scores.parquet")
    except (FileNotFoundError, OSError) as e:
        return problems + [f"scores.parquet unreadable: {e}"]
    df = table.to_pandas()
    types = {f.name: str(f.type) for f in table.schema}
    if list(types) != list(COLUMNS):
        problems.append(f"scores.parquet columns are {list(types)}, expected {list(COLUMNS)}")
    for col, t in COLUMNS.items():
        if col in types and types[col] not in (t, "large_string" if t == "string" else t):
            problems.append(f"column {col!r} has type {types[col]}, expected {t}")
    # Count true nulls in Arrow; pandas would also count NaN scores, which rule 5 reports.
    for col in table.column_names:
        if table.column(col).null_count:
            problems.append(f"column {col!r} has nulls")
    if problems or df.empty:
        return problems + (["scores.parquet has no rows"] if df.empty else [])

    # Rule 2: ids and roles
    if not df.image_id.is_unique:
        dupes = df.image_id[df.image_id.duplicated()].unique()[:3]
        problems.append(f"image_id not unique, e.g. {list(dupes)}")
    allowed = {"eval_normal", "test"} if stress else set(ROLE_ORDER)
    if bad := sorted(set(df.role) - allowed):
        problems.append(f"role values {bad} not allowed here (allowed: {sorted(allowed)})")
    elif not df.sort_values(["role", "image_id"], key=_role_key).index.equals(df.index):
        problems.append("rows are not sorted by role (calib, eval_normal, test) then image_id")

    # Rule 3: labels consistent with roles, defect types and masks
    if ((df.label == 1) & (df.role != "test")).any():
        problems.append("label 1 (defect) appears outside role 'test'")
    if not df.label.isin([0, 1]).all():
        problems.append("label holds values other than 0 and 1")
    if ((df.label == 0) != (df.defect_type == "good")).any():
        problems.append("label 0 must go with defect_type 'good' and only with it")
    if ((df.label == 0) & (df.mask_path != "")).any():
        problems.append("mask_path must be empty for label 0 rows")

    # Rule 5: finite scores
    if not np.isfinite(df.score.to_numpy()).all():
        problems.append("score has NaN or infinite values")

    # Rule 6: heatmaps
    size = tuple(info.get("input_size", ()))
    for rel in df.heatmap_path:
        path = run_dir / rel
        if not path.exists():
            problems.append(f"heatmap missing: {rel}")
            break
        h = np.load(path, mmap_mode="r")
        if h.dtype != np.float16 or h.shape != size:
            problems.append(f"heatmap {rel} is {h.dtype} {h.shape}, expected float16 {size}")
            break

    # Rule 4: clean runs match the split manifest's non-support rows exactly
    if not stress and "split_manifest" in info:
        manifest = _artifacts_dir(run_dir, stress) / info["split_manifest"]
        if not manifest.exists():
            problems.append(f"split manifest not found: {manifest}")
        else:
            m = pd.read_csv(manifest, dtype=str, keep_default_na=False)
            m = m[m.role != "support"].set_index("image_id")[COPIED]
            got = df.set_index("image_id")[COPIED].astype(str)
            if set(m.index) != set(got.index):
                problems.append(
                    f"rows differ from the manifest: {len(set(m.index) - set(got.index))} missing, "
                    f"{len(set(got.index) - set(m.index))} extra"
                )
            elif not m.loc[got.index].equals(got):
                problems.append("role, label, defect_type or mask_path differ from the manifest")
    return problems


def _role_key(col: pd.Series) -> pd.Series:
    return col.map(ROLE_ORDER.index) if col.name == "role" else col


def assert_valid(run_dir: str | Path) -> None:
    if problems := check_run(run_dir):
        raise SchemaError(Path(run_dir), problems)
