"""Backend settings, read from the environment once at import."""

import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.environ.get("ARGUS_ARTIFACTS_DIR", "artifacts"))
# Comma-separated. Default is the Vite dev server (C5 rule 4).
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ARGUS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]
