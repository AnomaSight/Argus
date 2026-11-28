"""Memory banks for every setting from one greedy run, saved per C3 (task A2.3).

m100 keeps every support patch. m10 is the first 10% of the greedy order, m1 the first 1%,
and aso the first k* (A2.2): all prefixes of one selection, so one greedy run gives all four.

Folder (C3): artifacts/banks/{dataset}/{category}/{setting}/seed{n}/
  bank.json      C3 info minus device; the only file other code may read
  features.npy   float16 (coreset_size, dim), the selected patches
  meta.json      internal: setup_s, coreset_fraction, n_support_patches
  greedy.npz     internal, m10 only: order, d, seconds (for ASO and the P4.2 figure)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from argus.data.transforms import input_size
from argus.features.cache import BACKBONE_CONFIG
from argus.memory.aso import aso
from argus.memory.greedy_coreset import Coreset, greedy_coreset

SCHEMA_VERSION = 1
SETTINGS = ("m100", "m10", "m1", "aso")
INFO_KEYS = (
    "dataset category setting seed config_hash preprocess input_size coreset_size bank_mb"
).split()


def config_hash(
    dataset: str, category: str, seed: int, preprocess: str, backbone: str = BACKBONE_CONFIG
) -> str:
    """Hash of what determines a run; A2.6 uses the same so banks match their C2 run.json."""
    text = f"{dataset}|{category}|{seed}|{preprocess}|{backbone}|m10=0.1|m1=0.01|aso"
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Bank:
    info: dict  # C3 info keys except device
    features: np.ndarray  # float16 (coreset_size, dim)
    setup_s: float
    coreset_fraction: float


def build_banks(
    support: np.ndarray,
    *,
    dataset: str,
    category: str,
    seed: int,
    preprocess: str,
    device: str = "cpu",
) -> tuple[dict[str, Bank], Coreset]:
    """support: (N, D) patch features of every support image. One greedy run, four banks."""
    support = np.asarray(support)
    n, dim = support.shape
    k10, k1 = max(1, round(0.10 * n)), max(1, round(0.01 * n))
    coreset = greedy_coreset(support, k10, seed=seed, device=device)
    k_aso = aso(coreset.d, n_total=n).k
    sizes = {"m100": n, "m10": k10, "m1": k1, "aso": k_aso}
    common = {
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "config_hash": config_hash(dataset, category, seed, preprocess),
        "preprocess": preprocess,
        "input_size": list(input_size(preprocess)),
    }
    banks = {}
    for setting, k in sizes.items():
        rows = support if setting == "m100" else support[coreset.order[:k]]
        info = {
            **common,
            "setting": setting,
            "coreset_size": int(k),
            "bank_mb": round(k * dim * 4 / 2**20, 2),  # as held for search: float32
        }
        banks[setting] = Bank(
            info={key: info[key] for key in INFO_KEYS},
            features=rows.astype(np.float16),
            setup_s=0.0 if setting == "m100" else coreset.setup_s(k),
            coreset_fraction=k / n,
        )
    return banks, coreset


def bank_dir(artifacts: Path, info: dict) -> Path:
    return (
        Path(artifacts)
        / "banks"
        / info["dataset"]
        / info["category"]
        / info["setting"]
        / f"seed{info['seed']}"
    )


def save_banks(banks: dict[str, Bank], coreset: Coreset, artifacts: Path) -> dict[str, Path]:
    paths = {}
    for setting, bank in banks.items():
        folder = bank_dir(artifacts, bank.info)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "features.npy", bank.features)
        meta = {
            "setup_s": bank.setup_s,
            "coreset_fraction": bank.coreset_fraction,
            "n_support_patches": len(banks["m100"].features),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        if setting == "m10":
            np.savez(
                folder / "greedy.npz", order=coreset.order, d=coreset.d, seconds=coreset.seconds
            )
        # bank.json last: a folder without it is incomplete and load_bank refuses it.
        body = {"schema_version": SCHEMA_VERSION, **bank.info}
        (folder / "bank.json").write_text(json.dumps(body, indent=2) + "\n")
        paths[setting] = folder
    return paths


def read_bank_json(folder: Path) -> dict:
    """bank.json, checked. Raises FileNotFoundError or ValueError (C3)."""
    path = Path(folder) / "bank.json"
    if not path.is_file():
        raise FileNotFoundError(f"no bank.json in {folder}")
    body = json.loads(path.read_text())
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bank.json schema_version {body.get('schema_version')!r} in {folder}"
        )
    missing = [k for k in INFO_KEYS if k not in body]
    if missing:
        raise ValueError(f"bank.json in {folder} is missing {missing}")
    return {k: body[k] for k in INFO_KEYS}


def load_bank(folder: Path) -> Bank:
    folder = Path(folder)
    info = read_bank_json(folder)
    meta = json.loads((folder / "meta.json").read_text())
    return Bank(info, np.load(folder / "features.npy"), meta["setup_s"], meta["coreset_fraction"])
