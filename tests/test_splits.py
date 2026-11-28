"""A1.2: dataset listings, the seeded split generator and contract C1 rules (fake trees only)."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from argus import cli
from argus.data import isp_ad, mvtec_ad, splits
from tests.conftest import make_isp, make_mvtec, touch, write_config

SEED = 20260801
SEEDS = [20260801, 20260802, 20260803]
MVTEC_COUNTS = {"train_good": 10, "test_good": 2, "test_defect": 3}
MVTEC_SPLIT = {"support": 4, "calib": 3, "eval_normal": 3}  # sums to train_good, as in 0.4
ISP_COUNTS = {"train_good": 6, "test_good": 5, "point_defects": 2, "area_defects": 1}
ISP_SPLIT = {"support": 3, "calib": 2, "eval_normal": 2}  # 11 pooled goods, 4 unused


@pytest.fixture
def mvtec(tmp_path):
    data = tmp_path / "data"
    make_mvtec(data, "nut", train=10, good=2)
    cfg = write_config(
        tmp_path / "mvtec.yaml", "mvtec_ad", {"nut": MVTEC_COUNTS}, {"nut": MVTEC_SPLIT}
    )
    return data, cfg


@pytest.fixture
def isp(tmp_path):
    data = tmp_path / "data"
    # good_start=0: test/good reuses train/good's filenames, as in the real archive.
    make_isp(data, "LSM_2", train=6, good=5, suffix="lsm2", good_start=0)
    cfg = write_config(tmp_path / "isp.yaml", "isp_ad", {"LSM_2": ISP_COUNTS}, {"LSM_2": ISP_SPLIT})
    return data, cfg


def split(fixture, seed=SEED):
    data, cfg = fixture
    dataset = "mvtec_ad" if "mvtec" in cfg.name else "isp_ad"
    cat = "nut" if dataset == "mvtec_ad" else "LSM_2"
    return splits.make_split(dataset, cat, seed, data_dir=data, config_path=cfg)


def check(rows, fixture):
    data, cfg = fixture
    dataset = "mvtec_ad" if "mvtec" in cfg.name else "isp_ad"
    cat = "nut" if dataset == "mvtec_ad" else "LSM_2"
    return splits.check_manifest(
        rows, dataset=dataset, category=cat, data_dir=data, config_path=cfg
    )


def by_role(rows, role):
    return [r for r in rows if r["role"] == role]


# --- listings -------------------------------------------------------------------------


def test_list_mvtec(mvtec):
    data, _ = mvtec
    recs = mvtec_ad.list_images(data, "nut")
    assert [r.image_id for r in recs] == sorted(r.image_id for r in recs)
    assert sum(r.pool == "normal" for r in recs) == 10
    crack = next(r for r in recs if r.image_id == "mvtec_ad/nut/test/crack/001")
    assert crack.path == "mvtec_ad/nut/test/crack/001.png"
    assert (crack.label, crack.defect_type, crack.pool) == (1, "crack", "test")
    assert crack.mask_path == "mvtec_ad/nut/ground_truth/crack/001_mask.png"
    good = next(r for r in recs if r.image_id == "mvtec_ad/nut/test/good/000")
    assert (good.label, good.defect_type, good.mask_path, good.pool) == (0, "good", "", "test")


def test_list_mvtec_missing_mask_gives_empty_mask_path(tmp_path):
    make_mvtec(tmp_path, "nut", masks=False)
    recs = mvtec_ad.list_images(tmp_path, "nut")
    assert all(r.mask_path == "" for r in recs)


def test_list_isp_pools_both_good_folders_and_keeps_colliding_names_apart(isp):
    data, _ = isp
    recs = isp_ad.list_images(data, "LSM_2")
    normal = [r.image_id for r in recs if r.pool == "normal"]
    assert len(normal) == 11 and len(set(normal)) == 11
    assert "isp_ad/LSM_2/train/good/0_good_lsm2" in normal
    assert "isp_ad/LSM_2/test/good/0_good_lsm2" in normal
    test = [r for r in recs if r.pool == "test"]
    assert {r.defect_type for r in test} == {"points", "area"} and len(test) == 3
    assert all(r.label == 1 and r.mask_path.startswith("isp_ad/LSM_2/ground_truth/") for r in test)
    area = next(r for r in test if r.defect_type == "area")
    assert area.mask_path == "isp_ad/LSM_2/ground_truth/area/200_area_lsm2.png"


def test_list_isp_refuses_good_reduced_and_unknown_test_folders(isp):
    data, _ = isp
    base = data / "isp_ad/LSM_2"
    touch(base / "train/good_reduced/0_good_lsm2.png")
    with pytest.raises(ValueError, match="good_reduced"):
        isp_ad.list_images(data, "LSM_2")
    shutil.rmtree(base / "train/good_reduced")
    touch(base / "test/scratch/1.png")
    with pytest.raises(ValueError, match="scratch"):
        isp_ad.list_images(data, "LSM_2")


def test_listing_skips_hidden_and_non_png(mvtec):
    data, _ = mvtec
    before = mvtec_ad.list_images(data, "nut")
    base = data / "mvtec_ad/nut"
    touch(base / "train/good/.DS_Store")
    touch(base / "train/good/._000.png")
    touch(base / "test/good/notes.txt")
    (base / "test/.hidden").mkdir()
    assert mvtec_ad.list_images(data, "nut") == before


def test_listing_and_splits_do_not_import_torch():
    code = "import sys, argus.data.splits, argus.cli; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- make_split -----------------------------------------------------------------------


def test_mvtec_split_counts_disjoint_and_covers_train_good(mvtec):
    rows = split(mvtec)
    assert splits.role_counts(rows) == {**MVTEC_SPLIT, "test": 5}
    ids = [r["image_id"] for r in rows]
    assert len(ids) == len(set(ids))  # roles are disjoint
    drawn = {r["image_id"] for r in rows if r["role"] != "test"}
    assert drawn == {f"mvtec_ad/nut/train/good/{i:03d}" for i in range(10)}
    test = by_role(rows, "test")
    assert {r["image_id"].split("/")[3] for r in test} == {"good", "crack", "hole"}
    assert all(r["role"] == "test" for r in rows if r["label"] == 1)
    for r in rows:
        assert (r["label"] == 0) == (r["defect_type"] == "good")
        if r["label"] == 0:
            assert r["mask_path"] == ""
    assert check(rows, mvtec) == []


def test_isp_split_counts_leftovers_absent_and_test_defects_only(isp):
    rows = split(isp)
    assert splits.role_counts(rows) == {**ISP_SPLIT, "test": 3}
    assert all(r["label"] == 1 for r in by_role(rows, "test"))
    pool = {r.image_id for r in isp_ad.list_images(isp[0], "LSM_2") if r.pool == "normal"}
    listed = {r["image_id"] for r in rows}
    assert len(pool - listed) == 4  # unused goods are left out (C1 rule 8)
    assert check(rows, isp) == []


def test_isp_colliding_filenames_can_both_be_drawn(isp):
    a, b = "isp_ad/LSM_2/train/good/0_good_lsm2", "isp_ad/LSM_2/test/good/0_good_lsm2"
    for seed in range(100):
        drawn = {r["image_id"] for r in split(isp, seed) if r["role"] != "test"}
        if a in drawn and b in drawn:
            break
    else:
        pytest.fail("no seed drew both same-named images")
    rows = split(isp, seed)
    assert check(rows, isp) == []


def test_isp_draws_from_both_folders(isp):
    folders = set()
    for seed in SEEDS:
        for r in split(isp, seed):
            if r["role"] != "test":
                folders.add(r["image_id"].split("/")[2])
    assert folders == {"train", "test"}


def test_same_seed_identical_bytes_and_different_seeds_differ(mvtec, isp, tmp_path):
    for fx in (mvtec, isp):
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        splits.write_manifest(split(fx), a)
        splits.write_manifest(split(fx), b)
        assert a.read_bytes() == b.read_bytes()
        draws = []
        for seed in SEEDS:
            rows = split(fx, seed)
            draws.append([(r["image_id"], r["role"]) for r in rows if r["role"] != "test"])
            assert by_role(rows, "test") == by_role(split(fx), "test")  # rule 4
        assert len({tuple(d) for d in draws}) == 3  # rule 5


def test_filesystem_order_does_not_matter(mvtec, isp, monkeypatch):
    expected = [splits.manifest_bytes(split(fx)) for fx in (mvtec, isp)]
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: reversed(list(original(self))))
    assert [splits.manifest_bytes(split(fx)) for fx in (mvtec, isp)] == expected


def test_files_created_in_shuffled_order_give_same_bytes(tmp_path):
    def build(root, order):
        for i in order:
            touch(root / "mvtec_ad/nut/train/good" / f"{i:03d}.png")
        for i in (1, 0):
            touch(root / "mvtec_ad/nut/test/good" / f"{i:03d}.png")
        for d, n in (("hole", 1), ("crack", 2)):
            for i in range(n):
                touch(root / "mvtec_ad/nut/test" / d / f"{i:03d}.png")
                touch(root / "mvtec_ad/nut/ground_truth" / d / f"{i:03d}_mask.png")

    cfg = write_config(
        tmp_path / "mvtec.yaml", "mvtec_ad", {"nut": MVTEC_COUNTS}, {"nut": MVTEC_SPLIT}
    )
    build(tmp_path / "a", range(10))
    build(tmp_path / "b", [7, 2, 9, 0, 5, 3, 8, 1, 6, 4])
    out = [
        splits.manifest_bytes(
            splits.make_split("mvtec_ad", "nut", SEED, data_dir=d, config_path=cfg)
        )
        for d in (tmp_path / "a", tmp_path / "b")
    ]
    assert out[0] == out[1]


def test_golden_manifest(mvtec):
    # If this fails, splits changed: the sampling algorithm, the listing, the CSV format or
    # numpy's default_rng stream. Do not just update the hash; find out why first, because
    # every manifest (and every threshold fitted on one) would change with it.
    content = splits.manifest_bytes(split(mvtec))
    assert hashlib.sha256(content).hexdigest() == GOLDEN_SHA256


GOLDEN_SHA256 = "ccb099ddbb3111b69d9dd717421d205de733c6b27b8926b1ed056dbbe54db7e6"


# --- file format ----------------------------------------------------------------------


def test_manifest_format(mvtec, tmp_path):
    path = tmp_path / "sub" / "seed1.csv"
    rows = split(mvtec)
    splits.write_manifest(rows, path)
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode().split("\n")
    assert lines[0] == "image_id,path,role,label,defect_type,mask_path"
    assert lines[1] == f"{rows[0]['image_id']},{rows[0]['path']},support,0,good,"
    assert [p.name for p in path.parent.iterdir()] == ["seed1.csv"]  # no temp file left
    crack = next(line for line in lines if "/crack/000," in line)
    assert crack.split(",")[3] == "1"


def test_read_manifest_round_trip(isp, tmp_path):
    rows = split(isp)
    path = tmp_path / "m.csv"
    splits.write_manifest(rows, path)
    back = splits.read_manifest(path)
    assert back == rows
    assert all(isinstance(r["label"], int) for r in back)
    assert all(r["mask_path"] == "" for r in back if r["label"] == 0)


def test_manifest_path():
    p = splits.manifest_path(Path("artifacts"), "isp_ad", "LSM_2", 20260801)
    assert p == Path("artifacts/splits/isp_ad/LSM_2/seed20260801.csv")


# --- refusals -------------------------------------------------------------------------


def test_refuses_count_mismatch(tmp_path):
    make_mvtec(tmp_path, "nut", train=9, good=2)
    cfg = write_config(tmp_path / "m.yaml", "mvtec_ad", {"nut": MVTEC_COUNTS}, {"nut": MVTEC_SPLIT})
    with pytest.raises(splits.SplitError, match="train_good expected 10, found 9"):
        splits.make_split("mvtec_ad", "nut", SEED, data_dir=tmp_path, config_path=cfg)


def test_refuses_pool_smaller_than_split(tmp_path):
    make_mvtec(tmp_path, "nut", train=10, good=2)
    big = {"support": 6, "calib": 3, "eval_normal": 3}
    cfg = write_config(tmp_path / "m.yaml", "mvtec_ad", {"nut": MVTEC_COUNTS}, {"nut": big})
    with pytest.raises(splits.SplitError, match="pool has 10 images but the split needs 12"):
        splits.make_split("mvtec_ad", "nut", SEED, data_dir=tmp_path, config_path=cfg)


def test_refuses_good_reduced_and_unknown_test_folder(isp):
    data, _ = isp
    base = data / "isp_ad/LSM_2"
    touch(base / "train/good_reduced/0_good_lsm2.png")
    with pytest.raises(splits.SplitError, match="good_reduced"):
        split(isp)
    shutil.rmtree(base / "train/good_reduced")
    touch(base / "test/scratch/1.png")
    with pytest.raises(splits.SplitError, match="test/scratch"):
        split(isp)


def test_refuses_unknown_category(mvtec):
    data, cfg = mvtec
    with pytest.raises(splits.SplitError, match="not a category"):
        splits.make_split("mvtec_ad", "bolt", SEED, data_dir=data, config_path=cfg)


# --- check_manifest catches each rule -------------------------------------------------


def corrupt(rows, how):
    rows = [dict(r) for r in rows]
    first_support = next(i for i, r in enumerate(rows) if r["role"] == "support")
    if how == "duplicate":
        rows.insert(first_support + 1, dict(rows[first_support]))
    elif how == "defect_in_calib":
        i = next(i for i, r in enumerate(rows) if r["label"] == 1)
        rows[i]["role"] = "calib"
        rows.sort(key=lambda r: (splits.ROLES.index(r["role"]), r["image_id"]))
    elif how == "good_with_mask":
        rows[first_support]["mask_path"] = rows[-1]["mask_path"] or "x_mask.png"
    elif how == "label_mismatch":
        rows[first_support]["defect_type"] = "crack"
    elif how == "wrong_count":
        del rows[first_support]
    elif how == "wrong_sort":
        rows.reverse()
    elif how == "test_row_missing":
        del rows[-1]
    elif how == "not_from_pool":
        i = next(i for i, r in enumerate(rows) if r["role"] == "test" and r["label"] == 0)
        rows[i]["role"] = "support"
        rows.sort(key=lambda r: (splits.ROLES.index(r["role"]), r["image_id"]))
    elif how == "bad_header":
        rows = [{("id" if k == "image_id" else k): v for k, v in r.items()} for r in rows]
    return rows


@pytest.mark.parametrize(
    ("how", "message"),
    [
        ("duplicate", "rule 1"),
        ("defect_in_calib", "rule 2"),
        ("good_with_mask", "rule 3"),
        ("label_mismatch", "rule 3"),
        ("wrong_count", "rule 6"),
        ("wrong_sort", "rule 5"),
        ("test_row_missing", "rule 4/6"),
        ("not_from_pool", "not in the normal pool"),
        ("bad_header", "columns are"),
    ],
)
def test_check_manifest_catches(mvtec, how, message):
    errors = check(corrupt(split(mvtec), how), mvtec)
    assert any(message in e for e in errors), errors


def test_check_manifest_catches_missing_file(mvtec):
    rows = split(mvtec)
    data, _ = mvtec
    (data / rows[0]["path"]).unlink()
    errors = check(rows, mvtec)
    assert any("rule 7" in e and rows[0]["path"] in e for e in errors), errors


def test_check_manifest_on_isp(isp):
    rows = split(isp)
    assert check(rows, isp) == []
    # Point a train/good row at the same-named test/good file: the file exists, but the row
    # no longer describes its image_id (the filename collision must not go unnoticed).
    i = next(i for i, r in enumerate(rows) if "/train/good/" in r["path"])
    rows[i] = {**rows[i], "path": rows[i]["path"].replace("/train/", "/test/")}
    errors = check(rows, isp)
    assert any("image_id is not path without extension" in e for e in errors), errors


# --- CLI ------------------------------------------------------------------------------


def statuses(stdout, status):
    return sum(line.endswith("  " + status) for line in stdout.splitlines())


def run_cli(capsys, *args):
    code = cli.main(["splits", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_writes_reruns_refuses_and_checks(mvtec, tmp_path, capsys):
    data, cfg = mvtec
    out = tmp_path / "splits"
    base = ["--dataset", "mvtec_ad", "--data-dir", data, "--out-dir", out, "--config", cfg]

    code, stdout, _ = run_cli(capsys, *base)
    assert code == 0
    files = sorted(out.rglob("*.csv"))
    assert [f.relative_to(out).as_posix() for f in files] == [
        f"mvtec_ad/nut/seed{s}.csv" for s in SEEDS
    ]
    assert statuses(stdout, "written") == 3 and "support=4 calib=3 eval_normal=3 test=5" in stdout
    first = {f: f.read_bytes() for f in files}

    code, stdout, _ = run_cli(capsys, *base)
    assert code == 0 and statuses(stdout, "unchanged") == 3

    code, stdout, _ = run_cli(capsys, *base, "--check")
    assert code == 0 and "all manifests valid" in stdout

    # A changed split must not silently replace existing manifests.
    changed = {"support": 3, "calib": 4, "eval_normal": 3}
    write_config(cfg, "mvtec_ad", {"nut": MVTEC_COUNTS}, {"nut": changed})
    code, _, stderr = run_cli(capsys, *base)
    assert code == 1 and stderr.count("REFUSED") == 3 and "ids change role" in stderr
    assert {f: f.read_bytes() for f in files} == first
    code, stdout, _ = run_cli(capsys, *base, "--force")
    assert code == 0 and statuses(stdout, "overwritten (--force)") == 3
    assert {f: f.read_bytes() for f in files} != first


def test_cli_check_fails_on_corrupted_and_cross_seed_problems(mvtec, tmp_path, capsys):
    data, cfg = mvtec
    out = tmp_path / "splits"
    base = ["--dataset", "mvtec_ad", "--data-dir", data, "--out-dir", out, "--config", cfg]
    run_cli(capsys, *base)
    seed1, seed2, _ = (out / "mvtec_ad/nut" / f"seed{s}.csv" for s in SEEDS)

    good = seed2.read_bytes()
    seed2.write_bytes(seed1.read_bytes())  # two seeds with the same draw
    code, stdout, _ = run_cli(capsys, *base, "--check")
    assert code == 1 and "draw the same split" in stdout

    lines = good.decode().split("\n")
    lines[1], lines[2] = lines[2], lines[1]  # break the sort order
    seed2.write_text("\n".join(lines))
    code, stdout, _ = run_cli(capsys, *base, "--check")
    assert code == 1 and "rule 5" in stdout

    seed2.unlink()
    code, stdout, _ = run_cli(capsys, *base, "--check")
    assert code == 1 and "missing" in stdout


def test_cli_warns_on_seed_not_in_config(mvtec, tmp_path, capsys):
    data, cfg = mvtec
    args = ["--dataset", "mvtec_ad", "--data-dir", data, "--out-dir", tmp_path, "--config", cfg]
    code, stdout, stderr = run_cli(capsys, *args, "--seeds", 1)
    assert code == 0 and "seed 1 is not one of the config's seeds" in stderr
    assert (tmp_path / "mvtec_ad/nut/seed1.csv").is_file()


def test_cli_reports_failure_for_unverified_data(tmp_path, capsys):
    make_mvtec(tmp_path, "nut", train=9)
    cfg = write_config(tmp_path / "m.yaml", "mvtec_ad", {"nut": MVTEC_COUNTS}, {"nut": MVTEC_SPLIT})
    args = [
        "--dataset",
        "mvtec_ad",
        "--data-dir",
        tmp_path,
        "--out-dir",
        tmp_path / "o",
        "--config",
        cfg,
    ]
    code, _, stderr = run_cli(capsys, *args)
    assert code == 1 and "does not verify" in stderr
    assert not (tmp_path / "o").exists()


def test_argus_command_runs_outside_repo(tmp_path):
    exe = Path(sys.executable).with_name("argus")
    if not exe.exists():
        pytest.skip("argus console script not installed (run make install)")
    out = subprocess.run(
        [str(exe), "splits", "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert out.returncode == 0 and "--dataset" in out.stdout
