"""HTTP API types (contract C5, docs/architecture.md).

Must match frontend/src/api/types.ts field for field. Changing one changes C5.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Schema):
    status: Literal["ok"] = "ok"


class ReviewBand(_Schema):
    tau_low: float
    tau_high: float


class ModelInfo(_Schema):
    model_id: str
    dataset: str
    category: str
    setting: str
    seed: int
    coreset_size: int
    bank_mb: float
    tau: float
    alpha: float
    delta: float
    review_band: ReviewBand | None


Verdict = Literal["good", "defect", "review"]


class InspectResponse(_Schema):
    model_id: str
    score: float
    tau: float
    review_band: ReviewBand | None
    verdict: Verdict
    input_png_base64: str
    heatmap_png_base64: str
    heatmap_range: tuple[float, float]
    latency_ms: float


Cell = str | float | int | None


class ResultTable(_Schema):
    name: str
    columns: list[str]
    rows: list[list[Cell]]


ErrorCode = Literal[
    "bad_request",
    "model_not_found",
    "file_too_large",
    "unsupported_media_type",
    "invalid_image",
    "model_unavailable",
    "internal_error",
]


class ErrorDetail(_Schema):
    code: ErrorCode
    message: str


class ApiError(_Schema):
    error: ErrorDetail


def decide_verdict(score: float, tau: float, band: ReviewBand | None) -> Verdict:
    """Verdict rule from C5. Strict comparisons, as in C4."""
    if band is None:
        return "defect" if score > tau else "good"
    if score <= band.tau_low:
        return "good"
    if score > band.tau_high:
        return "defect"
    return "review"
