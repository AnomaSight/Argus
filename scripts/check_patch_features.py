"""A1.3 real-weight checks: official parity, determinism, cross-device agreement and cost.

Needs the IMAGENET1K_V1 weights (about 263 MB, downloaded once to ~/.cache/torch) and the
MVTec AD and ISP-AD data (A1.1). Run it on the Mac (CPU + MPS) and on Colab (CPU + CUDA):

    python scripts/check_patch_features.py [--data-dir DIR] [--devices cpu mps] [--json OUT]

1. Parity: ``patch_features`` against tests/_patchcore_reference.py (the official
   PatchCore code at fcaa92f) run on a hooked, full torchvision wide_resnet50_2.
2. Determinism: the same batch twice, and batch 1 against batch 16, per device.
3. Cross-device: every other device against CPU.
4. Cost: ms per image at batch 1 and 16, and memory of a batch-16 call (fresh process).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from argus.data.transforms import image_transform, load_image
from argus.features.backbone import Backbone, available_devices, device_name
from argus.features.patch_features import DIM, patch_features
from argus.utils import config

_REF = config.REPO_ROOT / "tests" / "_patchcore_reference.py"
_spec = importlib.util.spec_from_file_location("_patchcore_reference", _REF)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)

SAMPLES = {
    "patchcore": [
        "mvtec_ad/hazelnut/test/crack/000.png",
        "mvtec_ad/hazelnut/train/good/000.png",
        "mvtec_ad/screw/train/good/000.png",
    ],
    "deng": [
        "isp_ad/ASM/test/points/10_points_asm.png",
        "isp_ad/LSM_2/train/good/100_good_lsm2.png",
        "isp_ad/LSM_1/test/area/10_area_lsm1.png",
    ],
}
BATCH = 16


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.cpu().reshape(-1, DIM), b.cpu().reshape(-1, DIM)
    cos = F.cosine_similarity(a.double(), b.double(), dim=1)
    return {
        "equal": bool(torch.equal(a, b)),
        "min_cosine": cos.min().item(),
        "max_abs_diff": (a - b).abs().max().item(),
        "max_abs_value": a.abs().max().item(),
    }


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def ms_per_image(net, x: torch.Tensor, reps: int) -> float:
    dev = x.device
    patch_features(net, x)  # warm-up
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        patch_features(net, x)
    sync(dev)
    return (time.perf_counter() - t0) * 1000 / (reps * len(x))


def probe_memory(device: str, size: int) -> dict:
    """Memory of one batch-16 call, in this (fresh) process: see ``batch16_memory``."""
    dev = torch.device(device)
    net = Backbone().to(dev)
    x = torch.randn(BATCH, 3, size, size).to(dev)
    patch_features(net, x[:1])  # warm-up at batch 1
    sync(dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        patch_features(net, x)
        sync(dev)
        peak, api = torch.cuda.max_memory_allocated(dev), "torch.cuda.max_memory_allocated"
    elif dev.type == "mps":
        torch.mps.empty_cache()
        base = torch.mps.driver_allocated_memory()
        patch_features(net, x)
        sync(dev)
        # No peak counter on MPS. The caching allocator keeps the blocks the call needed, so
        # the driver allocation right after it approximates the peak (rounded up to heaps).
        peak, api = torch.mps.driver_allocated_memory(), "torch.mps.driver_allocated_memory"
    else:
        unit = 1 if platform.system() == "Darwin" else 1024  # ru_maxrss: bytes vs KiB
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        patch_features(net, x)
        peak, api = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit, "ru_maxrss"
    return {
        "api": api,
        "base_mb": base / 2**20,
        "peak_mb": peak / 2**20,
        "activations_mb": (peak - base) / 2**20,
    }


def batch16_memory(device: str, size: int) -> dict:
    """Run ``probe_memory`` in a fresh process, so earlier work does not hide the peak."""
    out = subprocess.run(
        [sys.executable, __file__, "--probe-memory", device, str(size)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def run(data_dir: Path, devices: list[str], reps: int) -> dict:
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    full = wide_resnet50_2(weights=weights).eval()
    report = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "devices": {d: device_name(torch.device(d)) for d in devices},
        "presets": {},
    }
    nets = {d: Backbone().to(d) for d in devices}
    for preset, paths in SAMPLES.items():
        t = image_transform(preset)
        x = torch.stack([t(load_image(data_dir, p)) for p in paths])
        x16 = x[torch.arange(BATCH) % len(x)]
        res = {"images": paths, "devices": {}}

        cpu_out, grid = patch_features(nets["cpu"], x)
        ref, shapes = reference.official_embed(full, x)
        res["grid"], res["shape"] = list(grid), list(cpu_out.shape)
        res["official_parity_cpu"] = compare(cpu_out, ref)
        assert tuple(shapes[0]) == grid

        for d in devices:
            net, dev = nets[d], torch.device(d)
            xd, x16d = x.to(dev), x16.to(dev)
            a, _ = patch_features(net, xd)
            b, _ = patch_features(net, xd)
            singles = torch.cat([patch_features(net, xd[i : i + 1])[0] for i in range(len(x))])
            batched, _ = patch_features(net, x16d)
            entry = {
                "repeat": compare(a, b),
                "batch1_vs_batch3": compare(singles, a),
                "batch1_vs_batch16": compare(singles[torch.arange(BATCH) % len(x)], batched),
                "vs_cpu": compare(a, cpu_out),
                "ms_per_image_batch1": ms_per_image(net, xd[:1], reps),
                "ms_per_image_batch16": ms_per_image(net, x16d, max(1, reps // 4)),
                "memory_batch16": batch16_memory(d, x.shape[-1]),
            }
            if dev.type != "cpu":
                try:
                    ref_d, _ = reference.official_embed(full.to(dev), xd)
                    entry["official_parity_on_device"] = compare(a, ref_d)
                except RuntimeError as e:  # the official code itself fails on MPS (A1.3 bug 1)
                    entry["official_parity_on_device"] = {"error": str(e).splitlines()[0]}
                finally:
                    full.to("cpu")
            res["devices"][d] = entry
        report["presets"][preset] = res
    return report


def show(report: dict) -> None:
    print(f"torch {report['torch']}, Python {report['python']}, {report['machine']}")
    print("devices:", ", ".join(report["devices"].values()))

    def fmt(c: dict) -> str:
        if "error" in c:
            return f"error: {c['error']}"
        eq = "identical" if c["equal"] else "differs"
        return f"{eq}, min cos {c['min_cosine']:.9f}, max |diff| {c['max_abs_diff']:.3g}"

    for preset, res in report["presets"].items():
        print(f"\n== {preset}: {res['shape']} grid {res['grid']}")
        print(f"  official parity (CPU): {fmt(res['official_parity_cpu'])}")
        for d, e in res["devices"].items():
            print(f"  [{d}] same batch twice:  {fmt(e['repeat'])}")
            print(f"  [{d}] batch 1 vs 3:      {fmt(e['batch1_vs_batch3'])}")
            print(f"  [{d}] batch 1 vs 16:     {fmt(e['batch1_vs_batch16'])}")
            if d != "cpu":
                print(f"  [{d}] vs CPU:            {fmt(e['vs_cpu'])}")
                print(f"  [{d}] official on {d}: {fmt(e['official_parity_on_device'])}")
            print(
                f"  [{d}] ms/image: batch 1 {e['ms_per_image_batch1']:.1f}, "
                f"batch 16 {e['ms_per_image_batch16']:.1f}"
            )
            m = e["memory_batch16"]
            print(
                f"  [{d}] memory at batch 16: +{m['activations_mb']:.0f} MB for the call "
                f"({m['base_mb']:.0f} -> {m['peak_mb']:.0f} MB, {m['api']}, fresh process)"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, help="default: $ARGUS_DATA_DIR, else <repo>/data")
    ap.add_argument("--devices", nargs="+", help="default: cpu plus every accelerator found")
    ap.add_argument("--reps", type=int, default=8, help="timing repetitions (default 8)")
    ap.add_argument("--json", type=Path, help="also write the full report here")
    ap.add_argument("--probe-memory", nargs=2, metavar=("DEVICE", "SIZE"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.probe_memory:
        print(json.dumps(probe_memory(args.probe_memory[0], int(args.probe_memory[1]))))
        return 0
    data_dir = args.data_dir or config.data_dir()
    missing = [p for ps in SAMPLES.values() for p in ps if not (data_dir / p).is_file()]
    if missing:
        print(f"error: missing under {data_dir}: {', '.join(missing)} (run A1.1's downloads)")
        return 2
    found = [d for d in available_devices() if ":" not in d]
    devices = ["cpu"] + [d for d in (args.devices or found) if d != "cpu"]
    report = run(data_dir, devices, args.reps)
    show(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
