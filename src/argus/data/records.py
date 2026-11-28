"""The image record shared by the dataset listings and the split generator (A1.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

POOLS = ("normal", "test")


@dataclass(frozen=True)
class ImageRecord:
    """One image as listed from disk. Paths are relative to ``ARGUS_DATA_DIR``, POSIX-style.

    ``pool`` is ``"normal"`` for good images that may be drawn into support / calib /
    eval_normal, and ``"test"`` for the fixed ``test`` rows (contract C1).
    """

    image_id: str
    path: str
    label: int
    defect_type: str
    mask_path: str
    pool: str

    def __post_init__(self):
        if self.pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {self.pool!r}")
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")


def make_record(rel_path: PurePosixPath, label: int, defect_type: str, mask: str, pool: str):
    path = rel_path.as_posix()
    return ImageRecord(
        image_id=rel_path.with_suffix("").as_posix(),
        path=path,
        label=label,
        defect_type=defect_type,
        mask_path=mask,
        pool=pool,
    )
