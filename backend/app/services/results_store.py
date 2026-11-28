"""Result tables for GET /results: every CSV under artifacts/results/ (C5)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from app.schemas import Cell, ResultTable


def _cell(text: str) -> Cell:
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        return None  # C5: missing values are null, never NaN
    return (
        int(value) if value.is_integer() and "." not in text and "e" not in text.lower() else value
    )


def load_tables(artifacts: Path) -> list[ResultTable]:
    root = artifacts / "results"
    tables = []
    for path in sorted(root.rglob("*.csv")):
        with path.open(newline="") as f:
            header, *rows = list(csv.reader(f)) or [[]]
        name = path.relative_to(root).with_suffix("").as_posix()
        tables.append(
            ResultTable(name=name, columns=header, rows=[[_cell(c) for c in r] for r in rows])
        )
    return sorted(tables, key=lambda t: t.name)
