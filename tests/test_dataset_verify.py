"""argus.data.verify and argus.data.archive on tiny fake dataset trees (no network)."""

import json
from pathlib import Path

import pytest
import yaml

from argus.data import archive, verify


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


def make_isp(root: Path, mod: str, train=4, good=3, points=2, area=1, suffix="lsm1"):
    base = root / "isp_ad" / mod
    for i in range(train):
        touch(base / "train" / "good" / f"{i}_good_{suffix}.png")
    for i in range(good):
        touch(base / "test" / "good" / f"{100 + i}_good_{suffix}.png")
    for folder, n in (("points", points), ("area", area)):
        for i in range(n):
            name = f"{200 + i}_{folder}_{suffix}.png"
            touch(base / "test" / folder / name)
            touch(base / "ground_truth" / folder / name)
    return base


def write_config(path: Path, dataset: str, cats: dict) -> Path:
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


MVTEC_OK = {"train_good": 3, "test_good": 2, "test_defect": 3}
ISP_OK = {"train_good": 4, "test_good": 3, "point_defects": 2, "area_defects": 1}


@pytest.fixture
def mvtec(tmp_path):
    make_mvtec(tmp_path, "nut")
    return tmp_path, write_config(tmp_path / "m.yaml", "mvtec_ad", {"nut": MVTEC_OK})


@pytest.fixture
def isp(tmp_path):
    make_isp(tmp_path, "LSM_1")
    return tmp_path, write_config(tmp_path / "i.yaml", "isp_ad", {"LSM_1": ISP_OK})


def test_count_mvtec(mvtec):
    root, _ = mvtec
    c = verify.count_mvtec(root / "mvtec_ad", "nut")
    assert c["train_good"] == 3 and c["test_good"] == 2 and c["test_defect"] == 3
    assert c["defect_types"] == {"crack": 2, "hole": 1}
    assert c["missing_masks"] == []


def test_count_isp_ad(isp):
    root, _ = isp
    c = verify.count_isp_ad(root / "isp_ad", "LSM_1")
    assert (c["train_good"], c["test_good"]) == (4, 3)
    assert (c["point_defects"], c["area_defects"], c["total_defects"]) == (2, 1, 3)
    assert c["missing_masks"] == [] and not c["good_reduced_present"]


@pytest.mark.parametrize("fixture", ["mvtec", "isp"])
def test_correct_counts_pass(fixture, request):
    root, cfg = request.getfixturevalue(fixture)
    report = verify.verify(yaml.safe_load(cfg.read_text())["dataset"], root, cfg)
    assert report["ok"], report["errors"]


def test_wrong_count_names_field_expected_and_actual(tmp_path):
    make_mvtec(tmp_path, "nut", train=2)
    cfg = write_config(tmp_path / "m.yaml", "mvtec_ad", {"nut": MVTEC_OK})
    report = verify.verify("mvtec_ad", tmp_path, cfg)
    assert not report["ok"]
    assert report["errors"] == ["nut: train_good expected 3, found 2"]


def test_missing_category_folder_fails(tmp_path):
    cfg = write_config(tmp_path / "m.yaml", "mvtec_ad", {"nut": MVTEC_OK})
    report = verify.verify("mvtec_ad", tmp_path, cfg)
    assert not report["ok"] and "not found" in report["errors"][0]


def test_mvtec_defect_without_mask_fails(mvtec):
    root, cfg = mvtec
    (root / "mvtec_ad/nut/ground_truth/crack/001_mask.png").unlink()
    report = verify.verify("mvtec_ad", root, cfg)
    assert report["errors"] == ["nut: defect image test/crack/001.png has no mask"]


def test_isp_defect_without_mask_fails(isp):
    root, cfg = isp
    (root / "isp_ad/LSM_1/ground_truth/area/200_area_lsm1.png").unlink()
    report = verify.verify("isp_ad", root, cfg)
    assert report["errors"] == ["LSM_1: defect image test/area/200_area_lsm1.png has no mask"]


def test_hidden_and_non_png_files_are_ignored(mvtec, isp):
    for root, cfg, base in (
        (*mvtec, mvtec[0] / "mvtec_ad/nut"),
        (*isp, isp[0] / "isp_ad/LSM_1"),
    ):
        for sub in ("train/good", "test/good"):
            touch(base / sub / ".DS_Store")
            touch(base / sub / "._000.png")  # macOS AppleDouble file
            touch(base / sub / "notes.txt")
            touch(base / sub / "image.jpg")
        (base / "test" / ".hidden").mkdir()
        dataset = yaml.safe_load(cfg.read_text())["dataset"]
        report = verify.verify(dataset, root, cfg)
        assert report["ok"], report["errors"]


def test_good_reduced_is_flagged(isp):
    root, cfg = isp
    touch(root / "isp_ad/LSM_1/train/good_reduced/0_good_lsm1.png")
    c = verify.count_isp_ad(root / "isp_ad", "LSM_1")
    assert c["good_reduced_present"]
    report = verify.verify("isp_ad", root, cfg)
    assert not report["ok"]
    assert "good_reduced" in report["errors"][0]


def test_null_expected_counts_are_recorded_not_enforced(tmp_path):
    make_isp(tmp_path, "LSM_1", train=7)
    expected = {**ISP_OK, "train_good": None}
    cfg = write_config(tmp_path / "i.yaml", "isp_ad", {"LSM_1": expected})
    report = verify.verify("isp_ad", tmp_path, cfg)
    assert report["ok"], report["errors"]
    assert report["categories"]["LSM_1"]["counts"]["train_good"] == 7
    assert "7*" in verify.format_report(report)


def test_unknown_expected_field_fails(tmp_path):
    make_isp(tmp_path, "LSM_1")
    cfg = write_config(tmp_path / "i.yaml", "isp_ad", {"LSM_1": {**ISP_OK, "all_defects": 3}})
    report = verify.verify("isp_ad", tmp_path, cfg)
    assert report["errors"] == ["LSM_1: unknown expected_counts field 'all_defects'"]


def test_category_subset_and_unknown_category(mvtec):
    root, cfg = mvtec
    assert verify.verify("mvtec_ad", root, cfg, ["nut"])["ok"]
    report = verify.verify("mvtec_ad", root, cfg, ["bolt"])
    assert not report["ok"] and "not a category" in report["errors"][0]


def test_cli_exit_codes_table_and_json(mvtec, capsys):
    root, cfg = mvtec
    out_json = root / "mvtec_ad" / ".argus_source.json"
    arch = root / "nut.tar.xz"
    arch.write_bytes(b"x")
    args = ["--dataset", "mvtec_ad", "--data-dir", str(root), "--config", str(cfg)]
    rc = verify.main(
        [*args, "--json", str(out_json), "--archive", str(arch), "sha256", "abc", "nut"]
        + ["--meta", "license=CC BY-NC-SA 4.0"]
    )
    assert rc == 0
    table = capsys.readouterr().out
    assert "nut" in table and "crack 2, hole 1" in table
    record = json.loads(out_json.read_text())
    assert record["dataset"] == "mvtec_ad" and record["license"] == "CC BY-NC-SA 4.0"
    assert record["archives"] == [
        {"file": "nut.tar.xz", "size": 1, "sha256": "abc", "categories": ["nut"]}
    ]
    assert record["counts"]["nut"]["test_defect"] == 3

    (root / "mvtec_ad/nut/train/good/000.png").unlink()
    out_json.unlink()
    assert verify.main([*args, "--json", str(out_json)]) == 1
    assert "FAIL nut: train_good expected 3, found 2" in capsys.readouterr().out
    assert not out_json.exists()  # never written for a failing tree


def test_cli_list_categories(mvtec, capsys):
    _, cfg = mvtec
    assert verify.main(["--dataset", "mvtec_ad", "--config", str(cfg), "--list-categories"]) == 0
    assert capsys.readouterr().out.split() == ["nut"]


def test_config_with_wrong_dataset_or_bad_name_is_rejected(tmp_path):
    cfg = write_config(tmp_path / "m.yaml", "mvtec_ad", {"nut": MVTEC_OK})
    with pytest.raises(ValueError):
        verify.load_config("isp_ad", cfg)
    bad = write_config(tmp_path / "b.yaml", "mvtec_ad", {"a b": MVTEC_OK})
    with pytest.raises(ValueError):
        verify.load_config("mvtec_ad", bad)


def test_source_record_keeps_earlier_archives():
    report = {"dataset": "mvtec_ad", "categories": {"a": {"counts": {"x": 1, "missing_masks": []}}}}
    prev = {
        "archives": [
            {"file": "a.tar.xz", "categories": ["a"]},
            {"file": "old.tar.xz", "categories": ["b"]},
        ]
    }
    new = [{"file": "a2.tar.xz", "categories": ["a"]}]
    rec = verify.source_record(report, new, {}, prev)
    assert [a["file"] for a in rec["archives"]] == ["old.tar.xz", "a2.tar.xz"]
    assert rec["counts"] == {"a": {"x": 1}}


def test_real_configs_load():
    for name in verify.DATASETS:
        cfg = verify.load_config(name)
        assert cfg["categories"]


# --- argus.data.archive ---------------------------------------------------------------


def test_file_digest_and_cache(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert archive.file_digest(f, "md5") == "5d41402abc4b2a76b9719d911017c592"
    cache = tmp_path / "a.bin.md5.json"
    assert json.loads(cache.read_text())["md5"] == "5d41402abc4b2a76b9719d911017c592"
    # A cached value is reused while size and mtime match ...
    data = json.loads(cache.read_text())
    cache.write_text(json.dumps({**data, "md5": "cached"}))
    assert archive.file_digest(f, "md5") == "cached"
    # ... and ignored once the file changes.
    f.write_bytes(b"hello world")
    assert archive.file_digest(f, "md5") == "5eb63bbbe01eeed093cb22bb8f5acdc3"
    assert archive.file_digest(f, "sha256", cache=False) == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


LISTING_FLAT = [
    "license.txt\n",
    "readme.txt\n",
    "hazelnut/\n",
    "hazelnut/train/good/000.png\n",
    "hazelnut/ground_truth/crack/000_mask.png\n",
    "screw/test/good/000.png\n",
    "bottle/train/good/000.png\n",
]


def test_plan_tar_without_wrapper():
    plan = archive.plan_tar(LISTING_FLAT, ["hazelnut", "screw"])
    assert plan["categories"] == {"hazelnut": "hazelnut", "screw": "screw"}
    assert plan["license"] == "license.txt" and plan["prefix"] == []


@pytest.mark.parametrize("lead", ["mvtec_anomaly_detection/", "./", "./wrap/"])
def test_plan_tar_with_wrapper(lead):
    listing = [lead + m for m in LISTING_FLAT] + [lead + "bottle/license.txt"]
    plan = archive.plan_tar(listing, ["hazelnut", "screw", "carpet"])
    assert plan["categories"] == {"hazelnut": lead + "hazelnut", "screw": lead + "screw"}
    assert plan["license"] == lead + "license.txt"


def test_plan_tar_ignores_lookalike_folders():
    listing = ["hazelnut/notes/hazelnut.txt\n", "x/screw/readme.txt\n"]
    assert archive.plan_tar(listing, ["hazelnut", "screw"])["categories"] == {}


def test_zenodo_check(tmp_path, capsys):
    rec = {
        "id": 1,
        "metadata": {"version": "1.0"},
        "files": [
            {"key": "big.zip", "size": 5e9, "checksum": "md5:1", "links": {"self": "u1"}},
            {"key": "unsupervised.zip", "size": 10, "checksum": "md5:abc", "links": {"self": "u2"}},
        ],
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rec))
    info = archive.zenodo_check(path, "unsupervised.zip", "abc")
    assert info == {"size": 10, "url": "u2", "version": "1.0"}
    err = capsys.readouterr().err
    assert "big.zip" in err and "<- download" in err
    with pytest.raises(SystemExit, match="record has changed"):
        archive.zenodo_check(path, "unsupervised.zip", "def")
