"""Seeded split manifests (contract C1), A1.2.

Algorithm (fixed: changing it silently changes every result):

1. ``verify.verify`` must pass for the category: counts equal ``expected_counts`` before any
   role is assigned.
2. List the images (``mvtec_ad.list_images`` / ``isp_ad.list_images``) and separate the normal
   pool, sorted by ``image_id`` (Python string order), from the fixed test rows.
3. ``order = numpy.random.default_rng(seed).permutation(len(pool))``. The first ``support``
   entries of ``order`` become ``support``, the next ``calib`` become ``calib``, the next
   ``eval_normal`` become ``eval_normal``; the rest is left out of the manifest and unused
   (C1 rule 8; on MVTec AD there is no rest).
4. Every test record gets role ``test``. Rows are sorted by (role order, ``image_id``).

The seed and the sorted pool are the only inputs to the draw: no ``random``, no global numpy
state, no filesystem or ``set`` order. This module does not import torch.
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from argus.data import isp_ad, mvtec_ad, verify
from argus.data.records import ImageRecord

ROLES = ("support", "calib", "eval_normal", "test")
DRAWN_ROLES = ROLES[:3]
COLUMNS = ("image_id", "path", "role", "label", "defect_type", "mask_path")
LISTERS = {"mvtec_ad": mvtec_ad.list_images, "isp_ad": isp_ad.list_images}


class SplitError(ValueError):
    """The data or config does not allow a valid C1 split."""


def list_images(dataset: str, category: str, data_dir: Path, config_path: Path | None = None):
    cfg = verify.load_config(dataset, config_path)
    return LISTERS[dataset](Path(data_dir), category, root=cfg["root"])


def _row(rec: ImageRecord, role: str) -> dict:
    return {
        "image_id": rec.image_id,
        "path": rec.path,
        "role": role,
        "label": rec.label,
        "defect_type": rec.defect_type,
        "mask_path": rec.mask_path,
    }


def _sort_key(row: dict):
    return (ROLES.index(row["role"]), row["image_id"])


def make_split(
    dataset: str,
    category: str,
    seed: int,
    *,
    data_dir: Path,
    config_path: Path | None = None,
) -> list[dict]:
    """Rows of the C1 manifest for ``dataset``/``category``/``seed`` (see module docstring)."""
    cfg = verify.load_config(dataset, config_path)
    if category not in cfg["categories"]:
        raise SplitError(f"{category!r} is not a category in the {dataset} config")
    report = verify.verify(dataset, data_dir, config_path, [category])
    if not report["ok"]:
        raise SplitError(
            f"{dataset}/{category} does not verify, so it cannot be split:\n  "
            + "\n  ".join(report["errors"])
        )
    sizes = cfg["categories"][category]["split"]
    records = list_images(dataset, category, data_dir, config_path)
    pool = sorted((r for r in records if r.pool == "normal"), key=lambda r: r.image_id)
    test = [r for r in records if r.pool == "test"]
    need = sum(sizes[role] for role in DRAWN_ROLES)
    if len(pool) < need:
        raise SplitError(
            f"{dataset}/{category}: the normal pool has {len(pool)} images but the split needs "
            f"{need} ({', '.join(f'{r} {sizes[r]}' for r in DRAWN_ROLES)})"
        )
    order = np.random.default_rng(seed).permutation(len(pool))
    rows, start = [], 0
    for role in DRAWN_ROLES:
        rows += [_row(pool[i], role) for i in order[start : start + sizes[role]]]
        start += sizes[role]
    rows += [_row(r, "test") for r in test]
    return sorted(rows, key=_sort_key)


def manifest_bytes(rows: list[dict]) -> bytes:
    """The exact C1 file content: UTF-8 without BOM, LF endings, header in COLUMNS order."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([int(row[c]) if c == "label" else row[c] for c in COLUMNS])
    return buf.getvalue().encode("utf-8")


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write atomically (temp file in the same folder, then ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(manifest_bytes(rows))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_manifest(path: Path) -> list[dict]:
    """Rows as dicts in file-header order; ``label`` is an int, ``mask_path`` stays ``""``."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def manifest_file(splits_dir: Path, dataset: str, category: str, seed: int) -> Path:
    return Path(splits_dir) / dataset / category / f"seed{seed}.csv"


def manifest_path(artifacts_dir: Path, dataset: str, category: str, seed: int) -> Path:
    """``artifacts/splits/{dataset}/{category}/seed{seed}.csv`` (C1 location)."""
    return manifest_file(Path(artifacts_dir) / "splits", dataset, category, seed)


def role_counts(rows: list[dict]) -> dict:
    counts = Counter(r["role"] for r in rows)
    return {role: counts.get(role, 0) for role in ROLES}


def check_manifest(
    rows: list[dict],
    *,
    dataset: str,
    category: str,
    data_dir: Path,
    config_path: Path | None = None,
) -> list[str]:
    """C1 rules 1-7 for one manifest; returns the violations (empty if valid).

    Also checks that every support / calib / eval_normal row comes from the normal pool.
    Rules 4 and 5 across seeds (identical ``test``, different draws) need several files and
    are checked by ``argus splits --check``.
    """
    errors = []
    data_dir = Path(data_dir)
    cfg = verify.load_config(dataset, config_path)
    if category not in cfg["categories"]:
        return [f"{category!r} is not a category in the {dataset} config"]
    sizes = cfg["categories"][category]["split"]

    for i, row in enumerate(rows):
        if tuple(row) != COLUMNS:
            errors.append(f"row {i + 1}: columns are {list(row)}, expected {list(COLUMNS)}")
            return errors
    bad_roles = sorted({r["role"] for r in rows} - set(ROLES))
    if bad_roles:
        return errors + [f"unknown roles: {bad_roles}"]

    # Rule 1: unique image_id.
    dupes = [k for k, n in Counter(r["image_id"] for r in rows).items() if n > 1]
    if dupes:
        errors.append(f"rule 1: duplicate image_id: {dupes[:5]}")
    for r in rows:
        iid = r["image_id"]
        if r["path"].rsplit(".", 1)[0] != iid:
            errors.append(f"{iid}: image_id is not path without extension ({r['path']})")
        if not isinstance(r["label"], int) or r["label"] not in (0, 1):
            errors.append(f"{iid}: label must be int 0 or 1, got {r['label']!r}")
            continue
        # Rule 2: defects only in test.
        if r["label"] == 1 and r["role"] != "test":
            errors.append(f"rule 2: defect {iid} has role {r['role']}")
        # Rule 3: label 0 iff good; good rows have no mask.
        if (r["label"] == 0) != (r["defect_type"] == "good"):
            errors.append(
                f"rule 3: {iid} has label {r['label']} and defect_type {r['defect_type']}"
            )
        if r["label"] == 0 and r["mask_path"]:
            errors.append(f"rule 3: good image {iid} has a mask_path")
        # Rule 7: files exist.
        if not (data_dir / r["path"]).is_file():
            errors.append(f"rule 7: {r['path']} not found under {data_dir}")
        if r["mask_path"] and not (data_dir / r["mask_path"]).is_file():
            errors.append(f"rule 7: {r['mask_path']} not found under {data_dir}")

    # Rule 5 (sort order).
    if [_sort_key(r) for r in rows] != sorted(_sort_key(r) for r in rows):
        errors.append("rule 5: rows are not sorted by (role, image_id)")

    # Rule 6: role counts.
    counts = role_counts(rows)
    for role in DRAWN_ROLES:
        if counts[role] != sizes[role]:
            errors.append(f"rule 6: {counts[role]} {role} rows, config says {sizes[role]}")

    # Rules 4 and 6: test is exactly the listed test set; drawn rows come from the pool.
    try:
        records = list_images(dataset, category, data_dir, config_path)
    except (OSError, ValueError) as e:
        return errors + [f"cannot list {dataset}/{category}: {e}"]
    expected_test = {
        (_row(r, "test")["image_id"]): _row(r, "test") for r in records if r.pool == "test"
    }
    got_test = {r["image_id"]: r for r in rows if r["role"] == "test"}
    missing = sorted(set(expected_test) - set(got_test))
    extra = sorted(set(got_test) - set(expected_test))
    if missing:
        errors.append(f"rule 4/6: {len(missing)} test images missing from test, e.g. {missing[0]}")
    if extra:
        errors.append(f"rule 4: {len(extra)} test rows are not in the test set, e.g. {extra[0]}")
    for iid in sorted(set(got_test) & set(expected_test)):
        if dict(got_test[iid]) != expected_test[iid]:
            errors.append(f"rule 4: test row {iid} differs from the listed image")
    pool = {r.image_id: _row(r, "") for r in records if r.pool == "normal"}
    for r in rows:
        if r["role"] in DRAWN_ROLES:
            want = pool.get(r["image_id"])
            if want is None:
                errors.append(f"{r['image_id']} has role {r['role']} but is not in the normal pool")
            elif {**want, "role": r["role"]} != dict(r):
                errors.append(f"{r['image_id']}: row differs from the listed image")
    return errors
