"""The ``argus`` command. Each task adds its own subcommand; A1.2 adds ``splits``; P1.5 adds
``fake-scores``; P2.3 adds ``check-scores``; P2.2 adds ``report``.

argus splits --dataset {mvtec_ad,isp_ad} [--category NAME ...] [--seeds N ...]
             [--data-dir DIR] [--out-dir DIR] [--force] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from argus.data import splits, verify
from argus.utils import config


def _show(path: Path) -> str:
    """Relative to the current directory when inside it, else absolute."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return str(path)
    return str(Path(path).resolve()) if rel.startswith("..") else rel


def _counts(rows) -> str:
    return " ".join(f"{role}={n}" for role, n in splits.role_counts(rows).items())


def _diff_summary(old_rows, new_rows) -> str:
    old = {r["image_id"]: r["role"] for r in old_rows}
    new = {r["image_id"]: r["role"] for r in new_rows}
    moved = sum(1 for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    return f"existing: {_counts(old_rows)}; new: {_counts(new_rows)}; {moved} ids change role"


def _targets(args, cfg):
    cats = args.category or list(cfg["categories"])
    unknown = [c for c in cats if c not in cfg["categories"]]
    if unknown:
        raise SystemExit(
            f"error: not a category in the {args.dataset} config: {', '.join(unknown)}"
        )
    seeds = args.seeds or list(cfg["seeds"])
    for s in seeds:
        if s not in cfg["seeds"]:
            print(
                f"warning: seed {s} is not one of the config's seeds {cfg['seeds']}",
                file=sys.stderr,
            )
    return cats, seeds


def _write(args, cfg, data_dir: Path, out_dir: Path) -> int:
    cats, seeds = _targets(args, cfg)
    tally = {"written": 0, "unchanged": 0, "refused": 0, "failed": 0}
    for cat in cats:
        for seed in seeds:
            path = splits.manifest_file(out_dir, args.dataset, cat, seed)
            try:
                rows = splits.make_split(
                    args.dataset, cat, seed, data_dir=data_dir, config_path=args.config
                )
            except (splits.SplitError, OSError, ValueError) as e:
                print(f"FAIL {args.dataset}/{cat} seed{seed}: {e}", file=sys.stderr)
                tally["failed"] += 1
                continue
            content = splits.manifest_bytes(rows)
            digest = hashlib.sha256(content).hexdigest()[:12]
            if path.is_file() and path.read_bytes() == content:
                status = "unchanged"
            elif path.is_file() and not args.force:
                old = splits.read_manifest(path)
                print(
                    f"REFUSED {_show(path)}: exists with different content "
                    f"({_diff_summary(old, rows)}). Re-splitting invalidates thresholds fitted "
                    "on the old file; pass --force only if you mean it.",
                    file=sys.stderr,
                )
                tally["refused"] += 1
                continue
            else:
                status = "overwritten (--force)" if path.is_file() else "written"
                splits.write_manifest(rows, path)
            tally["unchanged" if status == "unchanged" else "written"] += 1
            print(f"{_show(path)}  {_counts(rows)}  sha256={digest}  {status}")
    print(", ".join(f"{n} {k}" for k, n in tally.items()))
    return 1 if tally["refused"] or tally["failed"] else 0


def _check(args, cfg, data_dir: Path, out_dir: Path) -> int:
    cats, seeds = _targets(args, cfg)
    n_bad = 0
    for cat in cats:
        loaded = {}
        for seed in seeds:
            path = splits.manifest_file(out_dir, args.dataset, cat, seed)
            if not path.is_file():
                print(f"FAIL {_show(path)}: missing")
                n_bad += 1
                continue
            rows = splits.read_manifest(path)
            errors = splits.check_manifest(
                rows, dataset=args.dataset, category=cat, data_dir=data_dir, config_path=args.config
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            print(
                f"{_show(path)}  {_counts(rows)}  sha256={digest}  {'ok' if not errors else 'FAIL'}"
            )
            for e in errors:
                print(f"  FAIL {e}")
            n_bad += bool(errors)
            loaded[seed] = rows
        # Cross-seed: identical test rows (rule 4), different draws (rule 5).
        items = list(loaded.items())
        for i, (s1, r1) in enumerate(items):
            for s2, r2 in items[i + 1 :]:
                t1 = [r for r in r1 if r["role"] == "test"]
                t2 = [r for r in r2 if r["role"] == "test"]
                if t1 != t2:
                    print(f"  FAIL {args.dataset}/{cat}: test rows differ, seed{s1} vs seed{s2}")
                    n_bad += 1
                d1 = [(r["image_id"], r["role"]) for r in r1 if r["role"] != "test"]
                d2 = [(r["image_id"], r["role"]) for r in r2 if r["role"] != "test"]
                if d1 == d2:
                    print(f"  FAIL {args.dataset}/{cat}: seed{s1} and seed{s2} draw the same split")
                    n_bad += 1
    print("all manifests valid" if not n_bad else f"{n_bad} problem(s) found")
    return 1 if n_bad else 0


def cmd_splits(args) -> int:
    cfg = verify.load_config(args.dataset, args.config)
    data_dir = args.data_dir or config.data_dir()
    out_dir = args.out_dir or config.artifacts_dir() / "splits"
    if args.check:
        return _check(args, cfg, data_dir, out_dir)
    return _write(args, cfg, data_dir, out_dir)


def cmd_fake_scores(args) -> int:
    from argus.evaluation.fake_scores import write_fake_run

    artifacts = args.artifacts or config.artifacts_dir()
    for seed in args.seeds:
        for setting in args.settings:
            kw = dict(dataset=args.dataset, category=args.category, setting=setting, seed=seed)
            print(_show(write_fake_run(artifacts, **kw)))
    return 0


def cmd_check_scores(args) -> int:
    from argus.evaluation.schema_check import check_run

    failed = False
    for folder in args.folders:
        problems = check_run(folder)
        failed |= bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {_show(folder)}")
        for p in problems:
            print(f"     {p}")
    return 1 if failed else 0


def cmd_report(args) -> int:
    from argus.evaluation.report import build_report, load_targets

    targets, delta = load_targets()
    alphas = args.alpha or targets
    delta = args.delta or delta
    artifacts = args.artifacts or config.artifacts_dir()
    data_dir = None if args.skip_pixel else (args.data_dir or config.data_dir())
    for alpha in alphas:
        path = build_report(artifacts, args.dataset, args.category, alpha, delta, data_dir)
        print(_show(path) if path else f"alpha={alpha:g}: every run skipped, no table")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="argus", description="Argus command line")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sp = sub.add_parser("splits", help="write or check the C1 split manifests (A1.2)")
    sp.add_argument("--dataset", required=True, choices=verify.DATASETS)
    sp.add_argument("--category", nargs="+", help="default: every category in the config")
    sp.add_argument("--seeds", nargs="+", type=int, help="default: the config's seeds")
    sp.add_argument("--data-dir", type=Path, help="default: $ARGUS_DATA_DIR, else <repo>/data")
    sp.add_argument(
        "--out-dir",
        type=Path,
        help="default: $ARGUS_ARTIFACTS_DIR/splits, else <repo>/artifacts/splits",
    )
    sp.add_argument("--config", type=Path, help="default: configs/datasets/<dataset>.yaml")
    sp.add_argument(
        "--force", action="store_true", help="overwrite manifests whose content changed"
    )
    sp.add_argument(
        "--check", action="store_true", help="validate existing manifests; write nothing"
    )
    sp.set_defaults(func=cmd_splits)

    fk = sub.add_parser(
        "fake-scores", help="write fake C1 split + C2 run folders for testing (P1.5)"
    )
    fk.add_argument("--artifacts", type=Path, help="default: $ARGUS_ARTIFACTS_DIR")
    fk.add_argument("--dataset", default="mvtec_ad")
    fk.add_argument("--category", default="hazelnut")
    fk.add_argument("--settings", nargs="+", default=["m100", "m10", "m1", "aso"])
    fk.add_argument("--seeds", nargs="+", type=int, default=[20260801, 20260802, 20260803])
    fk.set_defaults(func=cmd_fake_scores)

    ck = sub.add_parser("check-scores", help="check C2 run folders; exit 1 on any problem (P2.3)")
    ck.add_argument("folders", nargs="+", type=Path)
    ck.set_defaults(func=cmd_check_scores)

    rp = sub.add_parser("report", help="calibrate runs and write the results tables (P2.2)")
    rp.add_argument("--dataset", required=True)
    rp.add_argument("--category", required=True)
    rp.add_argument(
        "--alpha", nargs="+", type=float, help="default: configs/experiments/fpr_targets.yaml"
    )
    rp.add_argument("--delta", type=float, help="default: from fpr_targets.yaml (0.05)")
    rp.add_argument("--artifacts", type=Path, help="default: $ARGUS_ARTIFACTS_DIR")
    rp.add_argument(
        "--data-dir", type=Path, help="masks for pixel metrics; default $ARGUS_DATA_DIR"
    )
    rp.add_argument("--skip-pixel", action="store_true", help="leave pixel AUROC and PRO empty")
    rp.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
