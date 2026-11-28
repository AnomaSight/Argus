// Sample responses shaped exactly like C5. Every number here is illustrative, not a result.
// Delete this file once the frontend runs against the backend.
import {
  decideVerdict,
  type InspectResponse,
  type ModelInfo,
  type ModelsResponse,
  type ResultsResponse,
} from "./types";

const SEED = 20260801;
const SETTINGS: [string, number, number][] = [
  // setting, coreset_size, bank_mb
  ["m100", 156800, 612.5],
  ["m10", 15680, 61.25],
  ["m1", 1568, 6.13],
  ["aso", 4210, 16.45],
];

const MODELS: ModelInfo[] = ["hazelnut", "screw"].flatMap((category, c) =>
  SETTINGS.map(([setting, coreset_size, bank_mb], s) => ({
    model_id: `mvtec_ad/${category}/${setting}/seed${SEED}`,
    dataset: "mvtec_ad",
    category,
    setting,
    seed: SEED,
    coreset_size,
    bank_mb,
    tau: 2.2 + c * 0.3 + s * 0.05,
    alpha: 0.05,
    delta: 0.05,
    review_band: null,
  })),
).sort((a, b) => a.model_id.localeCompare(b.model_id));

const wait = <T>(value: T, ms = 250) => new Promise<T>((r) => setTimeout(() => r(value), ms));

export const models = (): Promise<ModelsResponse> => wait(MODELS);

const SIZE = 224;

function canvas() {
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  return [c, c.getContext("2d")!] as const;
}

// The `patchcore` preset: resize the short side to 256, centre-crop 224.
async function inputView(file: File): Promise<string> {
  const bmp = await createImageBitmap(file);
  const scale = 256 / Math.min(bmp.width, bmp.height);
  const w = bmp.width * scale;
  const h = bmp.height * scale;
  const [c, ctx] = canvas();
  ctx.drawImage(bmp, (SIZE - w) / 2, (SIZE - h) / 2, w, h);
  return c.toDataURL("image/png").split(",")[1];
}

function heatmap(x: number, y: number, strength: number): string {
  const [c, ctx] = canvas();
  const g = ctx.createRadialGradient(x, y, 0, x, y, 70);
  g.addColorStop(0, `rgba(255, 243, 168, ${0.9 * strength})`);
  g.addColorStop(0.3, `rgba(255, 154, 46, ${0.8 * strength})`);
  g.addColorStop(0.7, `rgba(215, 38, 61, ${0.45 * strength})`);
  g.addColorStop(1, "rgba(215, 38, 61, 0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, SIZE, SIZE);
  return c.toDataURL("image/png").split(",")[1];
}

export async function inspect(file: File, modelId: string): Promise<InspectResponse> {
  const model = MODELS.find((m) => m.model_id === modelId);
  if (!model) throw new Error(`No model with id '${modelId}'.`);
  // Deterministic per file, so the same image always gets the same fake score.
  const h = [...file.name].reduce((a, ch) => (a * 31 + ch.charCodeAt(0)) >>> 0, file.size);
  const score = 1.5 + (h % 1000) / 600; // 1.5 to ~3.17
  const strength = Math.min(1, Math.max(0.15, (score - 1.5) / 1.5));
  return wait(
    {
      model_id: modelId,
      score,
      tau: model.tau,
      review_band: model.review_band,
      verdict: decideVerdict(score, model.tau, model.review_band),
      input_png_base64: await inputView(file),
      heatmap_png_base64: heatmap(40 + (h % 144), 40 + ((h >> 8) % 144), strength),
      heatmap_range: [1.5, 3.0],
      latency_ms: 40 + (h % 30),
    },
    600,
  );
}

export const results = (): Promise<ResultsResponse> =>
  wait([
    {
      name: "mvtec_ad/hazelnut/memory_sweep_alpha0.05",
      columns: ["setting", "fpr", "fpr_ci_low", "fpr_ci_high", "recall", "image_auroc", "bank_mb"],
      rows: [
        ["m100", 0.048, 0.021, 0.106, 0.99, 1.0, 612.5],
        ["m10", 0.052, 0.024, 0.11, 0.99, 1.0, 61.25],
        ["m1", 0.061, 0.031, 0.121, 0.97, 0.998, 6.13],
        ["aso", 0.055, 0.026, 0.114, 0.98, 0.999, 16.45],
      ],
    },
    {
      name: "mvtec_ad/screw/memory_sweep_alpha0.05",
      columns: ["setting", "fpr", "fpr_ci_low", "fpr_ci_high", "recall", "image_auroc", "bank_mb"],
      rows: [
        ["m100", 0.05, 0.021, 0.111, 0.93, 0.981, 501.2],
        ["m10", 0.059, 0.027, 0.124, 0.91, 0.977, 50.12],
        ["m1", 0.079, 0.04, 0.149, 0.86, 0.968, 5.01],
        ["aso", 0.064, 0.03, 0.13, 0.9, 0.974, 12.9],
      ],
    },
  ]);
