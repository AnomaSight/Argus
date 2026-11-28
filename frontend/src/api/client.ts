import type {
  ApiError,
  HealthResponse,
  InspectResponse,
  ModelsResponse,
  ResultsResponse,
} from "./types";

const BASE = "/api";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new Error("Can't reach the Argus API. Start it with `make api`.");
  }
  if (!r.ok) {
    const body = (await r.json().catch(() => null)) as ApiError | null;
    if (!body && r.status >= 500) throw new Error("Can't reach the Argus API. Start it with `make api`.");
    throw new Error(body?.error.message ?? `${path} failed with status ${r.status}`);
  }
  return r.json();
}

export const getHealth = (): Promise<HealthResponse> => call("/health");

export const getModels = (): Promise<ModelsResponse> => call("/models");

export function inspect(image: File, modelId: string): Promise<InspectResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("model_id", modelId);
  return call("/inspect", { method: "POST", body: form });
}

export const getResults = (): Promise<ResultsResponse> => call("/results");
