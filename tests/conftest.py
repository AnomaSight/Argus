"""Shared helpers that build tiny fake dataset trees and configs (never the real data/)."""

from pathlib import Path

import pytest
import yaml

from argus.evaluation.fake_scores import write_fake_run

DEFAULT_SPLIT = {"support": 1, "calib": 1, "eval_normal": 1}


def touch(path: Path, data: bytes = b"\x89PNG fake"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data + path.name.encode())


def make_mvtec(root: Path, cat: str, train=3, good=2, defects=None, masks=True):
    base = root / "mvtec_ad" / cat
    for i in range(train):
        touch(base / "train" / "good" / f"{i:03d}.png")
    for i in range(good):
        touch(base / "test" / "good" / f"{i:03d}.png")
    for dtype, n in (defects or {"crack": 2, "hole": 1}).items():
        for i in range(n):
            touch(base / "test" / dtype / f"{i:03d}.png")
            if masks:
                touch(base / "ground_truth" / dtype / f"{i:03d}_mask.png")
    return base


def make_isp(
    root: Path, mod: str, train=4, good=3, points=2, area=1, suffix="lsm1", good_start=100
):
    """ISP-AD-like modality. ``good_start=0`` makes test/good reuse train/good's filenames,
    as the real archive does (they are different images)."""
    base = root / "isp_ad" / mod
    for i in range(train):
        touch(base / "train" / "good" / f"{i}_good_{suffix}.png", b"train ")
    for i in range(good):
        touch(base / "test" / "good" / f"{good_start + i}_good_{suffix}.png", b"test ")
    for folder, n in (("points", points), ("area", area)):
        for i in range(n):
            name = f"{200 + i}_{folder}_{suffix}.png"
            touch(base / "test" / folder / name)
            touch(base / "ground_truth" / folder / name)
    return base


def write_config(path: Path, dataset: str, cats: dict, splits=None, seeds=None) -> Path:
    """``cats`` maps category -> expected_counts; ``splits`` optionally category -> split."""
    cfg = {
        "dataset": dataset,
        "root": dataset,
        "preprocess": "patchcore" if dataset == "mvtec_ad" else "deng",
        "seeds": seeds or [20260801, 20260802, 20260803],
        "categories": {
            k: {"expected_counts": v, "split": dict((splits or {}).get(k, DEFAULT_SPLIT))}
            for k, v in cats.items()
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


@pytest.fixture
def fake_run(tmp_path):
    """Factory for small fake C2 run folders: fake_run(setting="m10", ...) -> run folder."""

    def make(**kw):
        kw = {"input_size": (16, 16), "n_support": 10, **kw}
        return write_fake_run(tmp_path / "artifacts", **kw)

    return make
