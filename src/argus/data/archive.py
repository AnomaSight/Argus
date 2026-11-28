"""Archive helpers for the A1.1 download scripts (hashing, tar member planning, Zenodo check).

Usage:
    python -m argus.data.archive hash {md5,sha256} PATH [--no-cache]
    python -m argus.data.archive tar-plan LISTING CATEGORY [CATEGORY ...]
    python -m argus.data.archive zenodo-check RECORD_JSON FILE PINNED_MD5

Hashing is done here, not with md5/md5sum/shasum, because those differ between macOS and
Linux. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 4 << 20
MVTEC_SUBDIRS = {"train", "test", "ground_truth"}


def _cache_path(path: Path, algo: str) -> Path:
    return path.with_name(f"{path.name}.{algo}.json")


def file_digest(path: Path, algo: str = "sha256", cache: bool = True, progress=None) -> str:
    """Hex digest of ``path``. With ``cache``, the result is stored next to the archive in
    ``<name>.<algo>.json`` keyed by size and mtime, and reused while both match."""
    path = Path(path)
    st = path.stat()
    key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    cache_file = _cache_path(path, algo)
    if cache and cache_file.is_file():
        try:
            saved = json.loads(cache_file.read_text())
            if {k: saved.get(k) for k in key} == key and saved.get(algo):
                return saved[algo]
        except (OSError, ValueError):
            pass
    h = hashlib.new(algo)
    done, last = 0, -1
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            done += len(chunk)
            if progress and st.st_size:
                pct = done * 100 // st.st_size
                if pct != last:
                    progress(pct, done, st.st_size)
                    last = pct
    digest = h.hexdigest()
    if cache:
        try:
            cache_file.write_text(json.dumps({**key, algo: digest}) + "\n")
        except OSError:
            pass  # read-only location: just don't cache
    return digest


def plan_tar(members, categories) -> dict:
    """Find, in a tar listing, the member folder of each wanted MVTec AD category and the
    top-level ``license.txt``. Works with or without a wrapper directory (and with ``./``).

    Returns ``{"categories": {name: member}, "license": member or None, "prefix": [...]}``,
    where ``member`` is the name exactly as stored in the archive (for ``tar -x``).
    """
    wanted = set(categories)
    found, prefixes, licenses = {}, set(), []
    for raw in members:
        raw = raw.rstrip("\n").rstrip("/")
        parts = raw.split("/")
        lead = 1 if parts and parts[0] == "." else 0
        comps = parts[lead:]
        if comps and comps[-1] == "license.txt":
            licenses.append((len(comps), raw))
        for i in range(len(comps) - 1):
            if comps[i] in wanted and comps[i + 1] in MVTEC_SUBDIRS:
                found.setdefault(comps[i], "/".join(parts[: lead + i + 1]))
                prefixes.add(tuple(comps[:i]))
                break
    if len(prefixes) > 1:
        raise ValueError(
            f"categories sit under different folders in the archive: {sorted(prefixes)}"
        )
    prefix = list(next(iter(prefixes))) if prefixes else None
    license_member = None
    for depth, raw in sorted(licenses):
        if prefix is None or depth == len(prefix) + 1:
            license_member = raw
            break
    return {"categories": found, "license": license_member, "prefix": prefix}


def zenodo_check(record_json: Path, filename: str, pinned_md5: str, out=None) -> dict:
    """Print every file in a Zenodo record and check ``filename``'s published MD5."""
    out = out or sys.stderr
    record = json.loads(Path(record_json).read_text())
    files = record.get("files") or []
    print(
        f"Zenodo record {record.get('id')} (version {record.get('metadata', {}).get('version')}):",
        file=out,
    )
    for f in sorted(files, key=lambda f: f["key"]):
        mark = "  <- download" if f["key"] == filename else ""
        print(f"  {f['key']:<24} {f['size'] / 1e6:>9.1f} MB{mark}", file=out)
    entry = next((f for f in files if f["key"] == filename), None)
    if entry is None:
        raise SystemExit(
            f"{filename} is not in the Zenodo record any more; the record has changed."
        )
    published = entry.get("checksum", "").removeprefix("md5:")
    if published != pinned_md5:
        raise SystemExit(
            f"The Zenodo record has changed: {filename} is now published with MD5 {published}, "
            f"but the script pins {pinned_md5}. Check the new release before updating the pin."
        )
    print(f"Only {filename} is fetched; the other files are not used by this project.", file=out)
    return {
        "size": entry["size"],
        "url": entry["links"]["self"],
        "version": record.get("metadata", {}).get("version"),
    }


def _progress(pct: int, done: int, total: int) -> None:
    line = f"  hashing: {pct:3d}% ({done / 1e9:.2f} / {total / 1e9:.2f} GB)"
    if sys.stderr.isatty():  # one line, updated in place
        sys.stderr.write("\r" + line + ("\n" if done == total else ""))
    elif pct % 25 == 0:  # logs (Colab, CI): a few plain lines
        sys.stderr.write(line + "\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m argus.data.archive")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash")
    h.add_argument("algo", choices=["md5", "sha256"])
    h.add_argument("path", type=Path)
    h.add_argument("--no-cache", action="store_true")
    t = sub.add_parser("tar-plan")
    t.add_argument("listing", type=Path)
    t.add_argument("categories", nargs="+")
    z = sub.add_parser("zenodo-check")
    z.add_argument("record_json", type=Path)
    z.add_argument("file")
    z.add_argument("pinned_md5")
    args = ap.parse_args(argv)

    if args.cmd == "hash":
        big = args.path.stat().st_size > 256 << 20
        print(
            file_digest(
                args.path, args.algo, cache=not args.no_cache, progress=_progress if big else None
            )
        )
    elif args.cmd == "tar-plan":
        with open(args.listing) as f:
            plan = plan_tar(f, args.categories)
        for name, member in plan["categories"].items():
            print(f"{name}\t{member}")
        if plan["license"]:
            print(f"@license\t{plan['license']}")
    else:
        info = zenodo_check(args.record_json, args.file, args.pinned_md5)
        print(f"{info['size']}\t{info['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
