"""Repo-wide path defaults shared by the CLI and the data modules.

``ARGUS_DATA_DIR`` and ``ARGUS_ARTIFACTS_DIR`` come from the environment; a relative value is
taken relative to the repo root, as in ``.env.example``. Unset, they default to ``<repo>/data``
and ``<repo>/artifacts``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_dir(name: str, default: str) -> Path:
    value = os.environ.get(name) or default
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def data_dir() -> Path:
    """``$ARGUS_DATA_DIR``, else ``<repo>/data``."""
    return _env_dir("ARGUS_DATA_DIR", "data")


def artifacts_dir() -> Path:
    """``$ARGUS_ARTIFACTS_DIR``, else ``<repo>/artifacts``."""
    return _env_dir("ARGUS_ARTIFACTS_DIR", "artifacts")
