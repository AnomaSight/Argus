// HTTP API types (contract C5, docs/architecture.md).
// Must match backend/app/schemas.py field for field. Changing one changes C5.

export interface HealthResponse {
  status: "ok";
}

export interface ReviewBand {
  tau_low: number;
  tau_high: number;
}

export interface ModelInfo {
  model_id: string;
  dataset: string;
  category: string;
  setting: string;
  seed: number;
  coreset_size: number;
  bank_mb: number;
  tau: number;
  alpha: number;
  delta: number;
  review_band: ReviewBand | null;
}

export type ModelsResponse = ModelInfo[];

export type Verdict = "good" | "defect" | "review";

export interface InspectResponse {
  model_id: string;
  score: number;
  tau: number;
  review_band: ReviewBand | null;
  verdict: Verdict;
  input_png_base64: string;
  heatmap_png_base64: string;
  heatmap_range: [number, number];
  latency_ms: number;
}

export type Cell = string | number | null;

export interface ResultTable {
  name: string;
  columns: string[];
  rows: Cell[][];
}

export type ResultsResponse = ResultTable[];

export type ErrorCode =
  | "bad_request"
  | "model_not_found"
  | "file_too_large"
  | "unsupported_media_type"
  | "invalid_image"
  | "model_unavailable"
  | "internal_error";

export interface ApiError {
  error: {
    code: ErrorCode;
    message: string;
  };
}

// Verdict rule from C5, used when the threshold slider moves tau client-side.
// Strict comparisons, as in C4: flagged only if score > threshold.
export function decideVerdict(
  score: number,
  tau: number,
  band: ReviewBand | null,
): Verdict {
  if (band === null) return score > tau ? "defect" : "good";
  if (score <= band.tau_low) return "good";
  if (score > band.tau_high) return "defect";
  return "review";
}
