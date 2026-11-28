/// <reference types="vite/client" />
import * as mock from "./mock";
import type {
  ApiError,
  HealthResponse,
  InspectResponse,
  ModelsResponse,
  ResultsResponse,
} from "./types";

const BASE = "/api";

// Sample data until the stub backend (P1.6) exists. Set VITE_USE_MOCK=false to hit the API.
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init);
  if (!r.ok) {
    const body = (await r.json().catch(() => null)) as ApiError | null;
    throw new Error(body?.error.message ?? `${path} failed with status ${r.status}`);
  }
  return r.json();
}

export function getHealth(): Promise<HealthResponse> {
  return USE_MOCK ? Promise.resolve({ status: "ok" }) : call("/health");
}

export function getModels(): Promise<ModelsResponse> {
  return USE_MOCK ? mock.models() : call("/models");
}

export function inspect(image: File, modelId: string): Promise<InspectResponse> {
  if (USE_MOCK) return mock.inspect(image, modelId);
  const form = new FormData();
  form.append("image", image);
  form.append("model_id", modelId);
  return call("/inspect", { method: "POST", body: form });
}

export function getResults(): Promise<ResultsResponse> {
  return USE_MOCK ? mock.results() : call("/results");
}
