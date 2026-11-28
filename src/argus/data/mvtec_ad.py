"""List MVTec AD images for one category (A1.2).

``train/good`` is the normal pool; ``test/good`` and ``test/<type>`` are the fixed test rows,
with masks at ``ground_truth/<type>/<stem>_mask.png``. Images are not opened.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from argus.data.records import ImageRecord, make_record
from argus.data.verify import list_pngs, list_subdirs

DATASET = "mvtec_ad"


def list_images(data_dir: Path, category: str, root: str = DATASET) -> list[ImageRecord]:
    """Every image of ``category`` under ``data_dir/root``, sorted by ``image_id``."""
    data_dir = Path(data_dir)
    base = data_dir / root / category
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; run `make data-mvtec` first")
    rel = PurePosixPath(root, category)
    records = [
        make_record(rel / "train/good" / p.name, 0, "good", "", "normal")
        for p in list_pngs(base / "train" / "good")
    ]
    for d in list_subdirs(base / "test"):
        for p in list_pngs(d):
            if d.name == "good":
                records.append(make_record(rel / "test/good" / p.name, 0, "good", "", "test"))
                continue
            mask = rel / "ground_truth" / d.name / f"{p.stem}_mask.png"
            mask_path = mask.as_posix() if (data_dir / mask).is_file() else ""
            records.append(
                make_record(rel / "test" / d.name / p.name, 1, d.name, mask_path, "test")
            )
    return sorted(records, key=lambda r: r.image_id)
