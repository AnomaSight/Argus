"""End-to-end runs of scripts/download_*.sh on tiny fake archives (no network)."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("bash", "tar", "unzip")),
    reason="needs bash, tar and unzip",
)


def run(script, *args, cwd, env=None):
    # Start from a clean slate so the caller's ISP_AD_MD5 / ARGUS_DATA_DIR never leak in.
    full_env = {k: v for k, v in os.environ.items() if k not in ("ISP_AD_MD5", "ARGUS_DATA_DIR")}
    full_env.update({"PYTHON": sys.executable, **(env or {})})
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *map(str, args)],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def mtimes(root: Path) -> dict:
    return {str(p): p.stat().st_mtime_ns for p in sorted(root.rglob("*")) if p.is_file()}


def write_config(path, dataset, cats):
    cfg = {
        "dataset": dataset,
        "root": dataset,
        "preprocess": "patchcore",
        "categories": {
            k: {"expected_counts": v, "split": {"support": 1, "calib": 1, "eval_normal": 1}}
            for k, v in cats.items()
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def png(name: str) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + name.encode()


# --- MVTec AD -------------------------------------------------------------------------


def mvtec_members(cat, train=3, good=2, defects=None):
    out = [f"{cat}/train/good/{i:03d}.png" for i in range(train)]
    out += [f"{cat}/test/good/{i:03d}.png" for i in range(good)]
    for d, n in (defects or {"crack": 2, "hole": 1}).items():
        out += [f"{cat}/test/{d}/{i:03d}.png" for i in range(n)]
        out += [f"{cat}/ground_truth/{d}/{i:03d}_mask.png" for i in range(n)]
    return out


def make_tar_xz(path: Path, members, wrapper=""):
    # Same modes as the real MVTec AD archive: folders r-xr-x---, files r--r-----.
    dirs = sorted({str(Path(wrapper + m).parent) for m in members} - {"."})
    dirs = sorted({str(p) for d in dirs for p in [Path(d), *Path(d).parents]} - {"."})
    with tarfile.open(path, "w:xz") as tf:
        for d in dirs:
            info = tarfile.TarInfo(d)
            info.type, info.mode = tarfile.DIRTYPE, 0o550
            tf.addfile(info)
        for m in members:
            data = png(m) if m.endswith(".png") else b"text"
            info = tarfile.TarInfo(wrapper + m)
            info.size, info.mode = len(data), 0o440
            tf.addfile(info, io.BytesIO(data))
    return path


MVTEC_COUNTS = {"train_good": 3, "test_good": 2, "test_defect": 3}


@pytest.fixture
def mvtec_setup(tmp_path):
    cfg = write_config(
        tmp_path / "mvtec.yaml", "mvtec_ad", {"alpha": MVTEC_COUNTS, "beta": MVTEC_COUNTS}
    )
    work = tmp_path / "elsewhere"  # scripts must not depend on the working directory
    work.mkdir()
    return tmp_path, cfg, work


@pytest.mark.parametrize("wrapper", ["", "mvtec_anomaly_detection/"])
def test_mvtec_extracts_configured_categories_only_and_rerun_skips(mvtec_setup, wrapper):
    tmp, cfg, work = mvtec_setup
    members = mvtec_members("alpha") + mvtec_members("beta") + mvtec_members("gamma")
    members += ["license.txt", "readme.txt"]
    arch = make_tar_xz(tmp / "mvtec_anomaly_detection.tar.xz", members, wrapper)
    data = tmp / "data"

    r = run("download_mvtec_ad.sh", arch, "--data-dir", data, "--config", cfg, cwd=work)
    assert r.returncode == 0, r.stdout + r.stderr
    root = data / "mvtec_ad"
    assert sorted(p.name for p in root.iterdir()) == [
        ".argus_source.json",
        "alpha",
        "beta",
        "license.txt",
    ]
    assert (root / "alpha/ground_truth/crack/001_mask.png").is_file()
    assert "crack 2, hole 1" in r.stdout
    record = json.loads((root / ".argus_source.json").read_text())
    sha = hashlib.sha256(arch.read_bytes()).hexdigest()
    assert record["archives"][0]["sha256"] == sha
    assert record["archives"][0]["categories"] == ["alpha", "beta"]
    assert record["counts"]["beta"]["test_defect"] == 3
    assert not list(data.glob(".argus-tmp-*"))

    before = mtimes(data)
    r2 = run("download_mvtec_ad.sh", arch, "--data-dir", data, "--config", cfg, cwd=work)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert r2.stdout.splitlines() == [
        "skip alpha: already present and verified",
        "skip beta: already present and verified",
    ]
    assert mtimes(data) == before


def test_mvtec_per_category_archives(mvtec_setup):
    tmp, cfg, work = mvtec_setup
    a = make_tar_xz(tmp / "alpha.tar.xz", mvtec_members("alpha"))
    b = make_tar_xz(tmp / "beta.tar.xz", mvtec_members("beta") + ["license.txt"])
    data = tmp / "data"
    r = run("download_mvtec_ad.sh", a, b, "--data-dir", data, "--config", cfg, cwd=work)
    assert r.returncode == 0, r.stdout + r.stderr
    record = json.loads((data / "mvtec_ad/.argus_source.json").read_text())
    assert [x["file"] for x in record["archives"]] == ["alpha.tar.xz", "beta.tar.xz"]
    assert (data / "mvtec_ad/license.txt").is_file()


def test_mvtec_missing_category_in_archive_fails_but_keeps_found_ones(mvtec_setup):
    tmp, cfg, work = mvtec_setup
    a = make_tar_xz(tmp / "alpha.tar.xz", mvtec_members("alpha"))
    data = tmp / "data"
    r = run("download_mvtec_ad.sh", a, "--data-dir", data, "--config", cfg, cwd=work)
    assert r.returncode != 0
    assert "beta" in r.stderr
    assert (data / "mvtec_ad/alpha").is_dir() and not (data / "mvtec_ad/beta").exists()
    # The second archive completes the set.
    b = make_tar_xz(tmp / "beta.tar.xz", mvtec_members("beta"))
    r = run("download_mvtec_ad.sh", b, "--data-dir", data, "--config", cfg, cwd=work)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skip alpha" in r.stdout
    record = json.loads((data / "mvtec_ad/.argus_source.json").read_text())
    assert [x["categories"] for x in record["archives"]] == [["alpha"], ["beta"]]


def test_mvtec_wrong_counts_extract_nothing(mvtec_setup):
    tmp, cfg, work = mvtec_setup
    members = mvtec_members("alpha") + mvtec_members("beta", train=2)
    arch = make_tar_xz(tmp / "m.tar.xz", members)
    data = tmp / "data"
    r = run("download_mvtec_ad.sh", arch, "--data-dir", data, "--config", cfg, cwd=work)
    assert r.returncode != 0
    assert "beta: train_good expected 3, found 2" in r.stderr
    assert not (data / "mvtec_ad").exists() or not any((data / "mvtec_ad").iterdir())
    assert not list(data.glob(".argus-tmp-*"))


def test_mvtec_existing_bad_category_stops_without_deleting(mvtec_setup):
    tmp, cfg, work = mvtec_setup
    arch = make_tar_xz(tmp / "m.tar.xz", mvtec_members("alpha") + mvtec_members("beta"))
    data = tmp / "data"
    bad = data / "mvtec_ad/alpha/train/good/000.png"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"x")
    r = run("download_mvtec_ad.sh", arch, "--data-dir", data, "--config", cfg, cwd=work)
    assert r.returncode != 0
    assert "exists but does not verify" in r.stderr
    assert bad.is_file()
    assert not (data / "mvtec_ad/beta").exists()


def test_mvtec_without_archive_prints_instructions(tmp_path):
    r = run("download_mvtec_ad.sh", cwd=tmp_path)
    assert r.returncode == 2
    assert "mvtec.com" in r.stderr and "CC BY-NC-SA 4.0" in r.stderr


# --- ISP-AD ---------------------------------------------------------------------------

ISP_COUNTS = {"train_good": 4, "test_good": 3, "point_defects": 2, "area_defects": 1}


def isp_members(mod, suffix):
    out = [f"unsupervised/{mod}/train/good/{i}_good_{suffix}.png" for i in range(4)]
    out += [f"unsupervised/{mod}/test/good/{100 + i}_good_{suffix}.png" for i in range(3)]
    for folder, n in (("points", 2), ("area", 1)):
        for i in range(n):
            name = f"{200 + i}_{folder}_{suffix}.png"
            out += [f"unsupervised/{mod}/test/{folder}/{name}"]
            out += [f"unsupervised/{mod}/ground_truth/{folder}/{name}"]
    return out


def make_isp_zip(path: Path) -> Path:
    members = isp_members("LSM_1", "lsm1") + isp_members("LSM_2", "lsm2")
    # Duplicates of train/good images, and other methods' configs: neither may be extracted.
    members += [f"unsupervised/LSM_1/train/good_reduced/{i}_good_lsm1.png" for i in range(2)]
    members += ["unsupervised/configs/LSM_1/patchcore.yaml", "unsupervised/configs/ASM/x.yaml"]
    members += ["unsupervised/license.txt", "unsupervised/changelog.txt"]
    with zipfile.ZipFile(path, "w") as zf:
        for m in members:
            zf.writestr(m, png(m) if m.endswith(".png") else b"text")
    return path


@pytest.fixture
def isp_setup(tmp_path):
    cfg = write_config(tmp_path / "isp.yaml", "isp_ad", {"LSM_1": ISP_COUNTS, "LSM_2": ISP_COUNTS})
    arch = make_isp_zip(tmp_path / "unsupervised.zip")
    md5 = hashlib.md5(arch.read_bytes()).hexdigest()
    work = tmp_path / "elsewhere"
    work.mkdir()
    return tmp_path, cfg, arch, md5, work


def test_isp_extracts_strips_prefix_and_rerun_skips(isp_setup):
    tmp, cfg, arch, md5, work = isp_setup
    data = tmp / "argus_data"
    # Data dir from ARGUS_DATA_DIR this time, not --data-dir.
    env = {"ISP_AD_MD5": md5, "ARGUS_DATA_DIR": str(data)}
    r = run("download_isp_ad.sh", "--archive", arch, "--config", cfg, cwd=work, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ISP_AD_MD5 is set" in r.stdout and "tests only" in r.stdout
    root = data / "isp_ad"
    assert sorted(p.name for p in root.iterdir()) == [
        ".argus_source.json",
        "LSM_1",
        "LSM_2",
        "changelog.txt",
        "license.txt",
    ]
    assert (root / "LSM_1/test/area/200_area_lsm1.png").is_file()
    assert (root / "LSM_1/ground_truth/area/200_area_lsm1.png").is_file()
    assert not (root / "LSM_1/train/good_reduced").exists()
    assert not (root / "configs").exists() and not (root / "unsupervised").exists()
    record = json.loads((root / ".argus_source.json").read_text())
    assert record["zenodo_record"] == "14911043"
    assert record["archives"][0]["md5"] == md5
    assert record["counts"]["LSM_2"]["total_defects"] == 3
    assert not list(data.glob(".argus-tmp-*"))

    before = mtimes(data)
    r2 = run("download_isp_ad.sh", "--archive", arch, "--config", cfg, cwd=work, env=env)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    skips = [line for line in r2.stdout.splitlines() if not line.startswith("note:")]
    assert skips == [
        "skip LSM_1: already present and verified",
        "skip LSM_2: already present and verified",
    ]
    assert mtimes(data) == before


def test_isp_wrong_md5_extracts_nothing(isp_setup):
    tmp, cfg, arch, _, work = isp_setup
    data = tmp / "data"
    env = {"ISP_AD_MD5": "0" * 32}
    r = run(
        "download_isp_ad.sh",
        "--archive",
        arch,
        "--data-dir",
        data,
        "--config",
        cfg,
        cwd=work,
        env=env,
    )
    assert r.returncode != 0
    assert "MD5 mismatch" in r.stderr
    assert not (data / "isp_ad").exists()
    assert not list(data.glob(".argus-tmp-*"))


def test_isp_pinned_md5_rejects_other_archives(isp_setup):
    tmp, cfg, arch, _, work = isp_setup
    data = tmp / "data"
    r = run("download_isp_ad.sh", "--archive", arch, "--data-dir", data, "--config", cfg, cwd=work)
    assert r.returncode != 0
    assert "expected edfb54cb26629b53963fa08bec9e4a6b" in r.stderr
    assert not (data / "isp_ad").exists()
