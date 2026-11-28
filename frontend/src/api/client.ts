import type { HealthResponse } from "./types";

const BASE = "/api";

export async function getHealth(): Promise<HealthResponse> {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`GET /health failed: ${r.status}`);
  return r.json();
}
