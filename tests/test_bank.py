import json

import numpy as np
import pytest

from argus.memory.bank import build_banks, load_bank, read_bank_json, save_banks

KW = dict(dataset="mvtec_ad", category="hazelnut", seed=20260801, preprocess="patchcore")


@pytest.fixture(scope="module")
def built():
    support = np.random.default_rng(0).standard_normal((2000, 64)).astype(np.float16)
    return support, *build_banks(support, **KW)


def test_one_greedy_run_gives_all_four(built):
    support, banks, coreset = built
    assert set(banks) == {"m100", "m10", "m1", "aso"}
    sizes = {s: b.info["coreset_size"] for s, b in banks.items()}
    assert sizes["m100"] == 2000 and sizes["m10"] == 200 and sizes["m1"] == 20
    assert 1 <= sizes["aso"] <= 200
    assert np.array_equal(banks["m100"].features, support)


def test_m1_is_the_first_tenth_of_m10(built):
    support, banks, coreset = built
    assert np.array_equal(banks["m1"].features, banks["m10"].features[:20])
    assert np.array_equal(banks["m10"].features, support[coreset.order[:200]])
    assert np.array_equal(
        banks["aso"].features, banks["m10"].features[: banks["aso"].info["coreset_size"]]
    )


def test_info_and_sizes(built):
    _, banks, _ = built
    info = banks["m10"].info
    assert info["input_size"] == [224, 224] and info["setting"] == "m10"
    assert info["bank_mb"] == round(200 * 64 * 4 / 2**20, 2)
    assert banks["m100"].setup_s == 0 and banks["m10"].setup_s >= banks["m1"].setup_s
    assert len({b.info["config_hash"] for b in banks.values()}) == 1


def test_save_and_reload_identical(built, tmp_path):
    _, banks, coreset = built
    paths = save_banks(banks, coreset, tmp_path)
    assert paths["m1"] == tmp_path / "banks/mvtec_ad/hazelnut/m1/seed20260801"
    for setting, bank in banks.items():
        again = load_bank(paths[setting])
        assert again.info == bank.info and np.array_equal(again.features, bank.features)
        assert again.features.dtype == np.float16
    body = json.loads((paths["m10"] / "bank.json").read_text())
    assert body["schema_version"] == 1 and "device" not in body
    assert (paths["m10"] / "greedy.npz").exists()


def test_bad_bank_folders_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_bank_json(tmp_path)
    (tmp_path / "bank.json").write_text(json.dumps({"schema_version": 9}))
    with pytest.raises(ValueError, match="schema_version"):
        read_bank_json(tmp_path)
