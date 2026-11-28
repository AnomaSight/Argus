"""Decode an upload, score it, and render the C5 inspect response."""

from __future__ import annotations

import base64
import io
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.errors import ApiError
from app.schemas import InspectResponse, ModelInfo, decide_verdict
from app.services.model_registry import HEATMAP_RANGE, StubDetector

MAX_BYTES = 10 * 1024 * 1024
FORMATS = {"PNG", "JPEG", "BMP", "TIFF"}
MEDIA_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/x-ms-bmp", "image/tiff"}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Transparent -> red -> orange -> pale yellow, the same scale the frontend legend draws.
_STOPS = np.array(
    [
        [0.0, 215, 38, 61, 0],
        [0.35, 215, 38, 61, 150],
        [0.7, 255, 154, 46, 210],
        [1.0, 255, 243, 168, 240],
    ]
)
_LUT = np.stack(
    [np.interp(np.linspace(0, 1, 256), _STOPS[:, 0], _STOPS[:, i]) for i in range(1, 5)], axis=1
).astype(np.uint8)


def decode(data: bytes, filename: str, content_type: str) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise ApiError(413, "file_too_large", "The upload is over 10 MB.")
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if content_type not in MEDIA_TYPES and ext not in EXTENSIONS:
        raise ApiError(415, "unsupported_media_type", "Use a PNG, JPEG, BMP or TIFF image.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ApiError(422, "invalid_image", "The file could not be decoded as an image.") from e
    if image.format not in FORMATS:
        raise ApiError(415, "unsupported_media_type", "Use a PNG, JPEG, BMP or TIFF image.")
    return image


def _png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def colour_heatmap(heatmap: np.ndarray, lo: float, hi: float) -> Image.Image:
    """Fixed range per model (C5), so good images look calm instead of stretched."""
    idx = np.clip((heatmap - lo) / (hi - lo), 0, 1) * 255
    return Image.fromarray(_LUT[idx.astype(np.uint8)], mode="RGBA")


def inspect_image(image: Image.Image, meta: ModelInfo, detector: StubDetector) -> InspectResponse:
    start = time.perf_counter()
    with detector.lock:
        view = detector.input_view(image)
        score, heatmap = detector.score(image)
    lo, hi = HEATMAP_RANGE
    response = InspectResponse(
        model_id=meta.model_id,
        score=score,
        tau=meta.tau,
        review_band=meta.review_band,
        verdict=decide_verdict(score, meta.tau, meta.review_band),
        input_png_base64=_png(view),
        heatmap_png_base64=_png(colour_heatmap(heatmap, lo, hi)),
        heatmap_range=(lo, hi),
        latency_ms=0.0,
    )
    response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return response
