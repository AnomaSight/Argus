"""List ISP-AD images for one modality (A1.2).

Per contract C1 (amended in #68), ``train/good`` and ``test/good`` together are the normal
pool, and ``test/points`` and ``test/area`` are the fixed test rows, with masks of the same
filename under ``ground_truth/{points,area}/``. Nothing else is read. The same filename can
occur in ``train/good`` and ``test/good`` (different images): ``image_id`` keeps the folder.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from argus.data.records import ImageRecord, make_record
from argus.data.verify import list_pngs, list_subdirs

DATASET = "isp_ad"
DEFECT_DIRS = ("points", "area")


def list_images(data_dir: Path, category: str, root: str = DATASET) -> list[ImageRecord]:
    """Every usable image of modality ``category`` under ``data_dir/root``, sorted by id.

    Raises ``ValueError`` if ``train/good_reduced`` (duplicates of ``train/good``) or an unknown
    ``test/`` subfolder is present, so such a tree can never be split.
    """
    data_dir = Path(data_dir)
    base = data_dir / root / category
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; run `make data-isp` first")
    if (base / "train" / "good_reduced").exists():
        raise ValueError(
            f"{base}/train/good_reduced is present; it duplicates train/good images "
            "(leakage risk). Delete it before splitting."
        )
    unknown = [d.name for d in list_subdirs(base / "test") if d.name not in ("good", *DEFECT_DIRS)]
    if unknown:
        raise ValueError(f"{base}/test has unexpected folders: {', '.join(unknown)}")
    rel = PurePosixPath(root, category)
    records = []
    for folder in ("train/good", "test/good"):
        records += [
            make_record(rel / folder / p.name, 0, "good", "", "normal")
            for p in list_pngs(base / folder)
        ]
    for d in DEFECT_DIRS:
        for p in list_pngs(base / "test" / d):
            mask = rel / "ground_truth" / d / p.name
            mask_path = mask.as_posix() if (data_dir / mask).is_file() else ""
            records.append(make_record(rel / "test" / d / p.name, 1, d, mask_path, "test"))
    return sorted(records, key=lambda r: r.image_id)
