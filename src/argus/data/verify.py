"""Count downloaded dataset images and check them against configs/datasets/*.yaml (A1.1).

Usage:
    python -m argus.data.verify --dataset {mvtec_ad,isp_ad} [--data-dir DIR] [--config PATH]
                                [--category NAME ...] [--json PATH] [--list-categories]

Only ``*.png`` files are counted; hidden files (``.DS_Store``, macOS ``._*`` files) and other
extensions are ignored. A ``null`` in ``expected_counts`` means "record, don't enforce".
Standard library plus pyyaml only, so this runs before torch is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from argus.utils import config

DATASETS = ("mvtec_ad", "isp_ad")
SCHEMA_VERSION = 1
REPO_ROOT = config.REPO_ROOT
CATEGORY_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

MVTEC_FIELDS = ("train_good", "test_good", "test_defect")
ISP_FIELDS = ("train_good", "test_good", "point_defects", "area_defects", "total_defects")
ISP_DEFECT_DIRS = {"points": "point_defects", "area": "area_defects"}


def list_pngs(folder: Path) -> list[Path]:
    """``*.png`` files directly in ``folder``, sorted; hidden files (``.DS_Store``, ``._*``)
    and other extensions are skipped. Shared by the counters here and the A1.2 listings."""
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.suffix == ".png" and not p.name.startswith(".") and p.is_file()
    )


def list_subdirs(folder: Path) -> list[Path]:
    """Non-hidden subfolders of ``folder``, sorted."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def count_mvtec(root: Path, category: str) -> dict:
    """Count one MVTec AD category under ``root`` (= ``data/mvtec_ad``).

    Masks are expected at ``ground_truth/{defect_type}/{stem}_mask.png``.
    """
    base = Path(root) / category
    defect_types = {}
    missing_masks = []
    for d in list_subdirs(base / "test"):
        if d.name == "good":
            continue
        images = list_pngs(d)
        defect_types[d.name] = len(images)
        for img in images:
            mask = base / "ground_truth" / d.name / f"{img.stem}_mask.png"
            if not mask.is_file():
                missing_masks.append(f"test/{d.name}/{img.name}")
    return {
        "train_good": len(list_pngs(base / "train" / "good")),
        "test_good": len(list_pngs(base / "test" / "good")),
        "test_defect": sum(defect_types.values()),
        "defect_types": defect_types,
        "missing_masks": missing_masks,
    }


def count_isp_ad(root: Path, modality: str) -> dict:
    """Count one ISP-AD modality under ``root`` (= ``data/isp_ad``).

    Masks share the test image's filename: ``ground_truth/{points,area}/{name}``.
    ``train/good_reduced`` duplicates 500 ``train/good`` images, so its presence is reported.
    """
    base = Path(root) / modality
    counts = {}
    missing_masks = []
    for folder, field in ISP_DEFECT_DIRS.items():
        images = list_pngs(base / "test" / folder)
        counts[field] = len(images)
        for img in images:
            if not (base / "ground_truth" / folder / img.name).is_file():
                missing_masks.append(f"test/{folder}/{img.name}")
    known = {"good", *ISP_DEFECT_DIRS}
    return {
        "train_good": len(list_pngs(base / "train" / "good")),
        "test_good": len(list_pngs(base / "test" / "good")),
        **counts,
        "total_defects": sum(counts.values()),
        "missing_masks": missing_masks,
        "good_reduced_present": (base / "train" / "good_reduced").exists(),
        "unexpected_test_dirs": [
            d.name for d in list_subdirs(base / "test") if d.name not in known
        ],
    }


COUNTERS = {"mvtec_ad": count_mvtec, "isp_ad": count_isp_ad}


def load_config(dataset: str, config_path: Path | None = None) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    path = (
        Path(config_path) if config_path else REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("dataset") != dataset:
        raise ValueError(f"{path}: dataset is {cfg.get('dataset')!r}, expected {dataset!r}")
    for name in cfg.get("categories") or {}:
        if not CATEGORY_NAME.match(str(name)):
            raise ValueError(f"{path}: category name {name!r} must match {CATEGORY_NAME.pattern}")
    return cfg


def _check_category(dataset: str, root: Path, name: str, expected: dict) -> dict:
    folder = root / name
    if not folder.is_dir():
        return {
            "counts": None,
            "expected": expected,
            "errors": [f"{name}: folder {folder} not found"],
        }
    counts = COUNTERS[dataset](root, name)
    errors = []
    for field, want in expected.items():
        if field not in counts or isinstance(counts[field], (list, dict, bool)):
            errors.append(f"{name}: unknown expected_counts field {field!r}")
        elif want is not None and counts[field] != want:
            errors.append(f"{name}: {field} expected {want}, found {counts[field]}")
    for img in counts["missing_masks"]:
        errors.append(f"{name}: defect image {img} has no mask")
    if counts.get("good_reduced_present"):
        errors.append(
            f"{name}: train/good_reduced is present; it holds byte-identical copies of "
            "train/good images (leakage risk). Delete that folder."
        )
    for d in counts.get("unexpected_test_dirs", []):
        errors.append(f"{name}: unexpected folder test/{d}")
    return {"counts": counts, "expected": expected, "errors": errors}


def verify(
    dataset: str,
    data_dir: Path,
    config_path: Path | None = None,
    categories: list[str] | None = None,
) -> dict:
    """Compare the folders under ``data_dir/{root}`` with ``expected_counts``.

    Returns a report dict; ``report["ok"]`` is False if any count differs, a configured
    category folder is missing, or a defect image has no mask.
    """
    cfg = load_config(dataset, config_path)
    root = Path(data_dir) / cfg["root"]
    configured = cfg["categories"]
    names = list(configured) if categories is None else list(categories)
    report = {
        "dataset": dataset,
        "root": str(root),
        "config": str(config_path) if config_path else f"configs/datasets/{dataset}.yaml",
        "categories": {},
    }
    for name in names:
        if name not in configured:
            report["categories"][name] = {
                "counts": None,
                "expected": {},
                "errors": [f"{name}: not a category in the {dataset} config"],
            }
            continue
        expected = dict(configured[name].get("expected_counts") or {})
        report["categories"][name] = _check_category(dataset, root, name, expected)
    report["errors"] = [e for c in report["categories"].values() for e in c["errors"]]
    report["ok"] = not report["errors"]
    return report


def format_report(report: dict) -> str:
    fields = MVTEC_FIELDS if report["dataset"] == "mvtec_ad" else ISP_FIELDS
    header = ["category", *fields, "status"]
    rows, notes, null_used = [], [], False
    for name, c in report["categories"].items():
        counts, expected = c["counts"], c["expected"]
        row = [name]
        for f in fields:
            if counts is None:
                row.append("-")
                continue
            cell = str(counts[f])
            if f in expected and expected[f] is None:
                cell += "*"
                null_used = True
            elif f in expected and expected[f] != counts[f]:
                cell += f" (expected {expected[f]})"
            row.append(cell)
        row.append("ok" if not c["errors"] else "FAIL")
        rows.append(row)
        if counts and counts.get("defect_types"):
            types = ", ".join(f"{k} {v}" for k, v in counts["defect_types"].items())
            notes.append(f"  {name} defect types: {types}")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    lines = [f"{report['dataset']}: {report['root']}"]
    for r in [header, *rows]:
        lines.append("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())
    lines += notes
    if null_used:
        lines.append("  * recorded, not enforced (expected_counts is null)")
    shown = report["errors"][:20]
    lines += [f"FAIL {e}" for e in shown]
    if len(report["errors"]) > len(shown):
        lines.append(f"FAIL ... and {len(report['errors']) - len(shown)} more")
    return "\n".join(lines)


def source_record(report: dict, archives: list[dict], meta: dict, previous: dict | None) -> dict:
    """Build the ``.argus_source.json`` record; keeps earlier archives' entries (MVTec AD
    per-category archives may arrive in separate runs)."""
    redone = {c for a in archives for c in a["categories"]}
    kept = []
    for a in (previous or {}).get("archives", []):
        cats = [c for c in a.get("categories", []) if c not in redone]
        if cats:
            kept.append({**a, "categories": cats})
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": report["dataset"],
        "written_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **meta,
        "archives": kept + archives,
        "counts": {
            name: {k: v for k, v in c["counts"].items() if k != "missing_masks"}
            for name, c in report["categories"].items()
        },
    }


def _parse_meta(items: list[str]) -> dict:
    meta = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--meta expects KEY=VALUE, got {item!r}")
        meta[key] = value
    return meta


def _parse_archive(spec: list[str]) -> dict:
    path, algo, digest, cats = spec
    p = Path(path)
    return {
        "file": p.name,
        "size": p.stat().st_size if p.exists() else None,
        algo: digest,
        "categories": [c for c in cats.split(",") if c],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m argus.data.verify", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data-dir", type=Path, default=None, help="default: $ARGUS_DATA_DIR or data")
    ap.add_argument(
        "--config", type=Path, default=None, help="default: configs/datasets/<dataset>.yaml"
    )
    ap.add_argument("--category", action="append", help="check only this category (repeatable)")
    ap.add_argument(
        "--list-categories", action="store_true", help="print configured names and exit"
    )
    ap.add_argument("--quiet", action="store_true", help="print errors only")
    ap.add_argument(
        "--json", type=Path, help="on success, write the source record (.argus_source.json)"
    )
    ap.add_argument(
        "--meta", action="append", default=[], metavar="KEY=VALUE", help="extra field for --json"
    )
    ap.add_argument(
        "--archive",
        nargs=4,
        action="append",
        default=[],
        metavar=("PATH", "ALGO", "DIGEST", "CATEGORIES"),
        help="archive used in this run, for --json (CATEGORIES comma-separated)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.dataset, args.config)
    if args.list_categories:
        print("\n".join(cfg["categories"]))
        return 0
    data_dir = args.data_dir or config.data_dir()
    report = verify(args.dataset, data_dir, args.config, args.category)
    if args.quiet:
        for e in report["errors"]:
            print(f"FAIL {e}")
    else:
        print(format_report(report))
    if not report["ok"]:
        return 1
    if args.json:
        previous = None
        if args.json.is_file():
            try:
                previous = json.loads(args.json.read_text())
            except ValueError:
                previous = None
        record = source_record(
            report, [_parse_archive(a) for a in args.archive], _parse_meta(args.meta), previous
        )
        tmp = args.json.with_name(args.json.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
