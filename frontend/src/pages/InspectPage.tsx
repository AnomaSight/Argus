import { useEffect, useRef, useState, type CSSProperties } from "react";

import { getModels, inspect, USE_MOCK } from "../api/client";
import type { InspectResponse, ModelInfo } from "../api/types";
import ImageUpload from "../components/ImageUpload";
import VerdictBadge from "../components/VerdictBadge";

const SETTING_NAMES: Record<string, string> = { m100: "100% memory", m10: "10% memory", m1: "1% memory", aso: "ASO" };

// The machine: the image enters on the left, is cut into an N x N grid of patches, the patches
// fly into the memory bank, come out on the right, reassemble, and the heatmap and verdict
// develop over them. Purely a visual of what the detector does; the numbers come from the API.
const N = 8;
const TILES = Array.from({ length: N * N }, (_, i) => ({ c: i % N, r: Math.floor(i / N) }));

// Paced on purpose, about 8 s end to end. Seeing the work makes the answer feel earned
// (Buell and Norton's labor illusion); progress that speeds up toward the end feels shorter
// (Harrison et al. 2007); a short held beat before the verdict builds anticipation, and the
// reveal is the peak people remember. Kept under the ~10 s attention limit, with a Skip.
// The detector itself takes milliseconds; this is a visual of what it does, not its runtime.
const READ_MIN = 1200; // scan line over the image, at least this long
const ORDER = ["idle", "reading", "grid", "split", "feed", "emit", "heat", "hold", "done"] as const;
type Phase = (typeof ORDER)[number];
const TIMELINE: [Phase, number][] = [
  // ms after reading ends
  ["split", 800], // grid lines have drawn; the image cracks into patches
  ["feed", 1300], // patches leave in reading order and fly into the memory bank
  ["emit", 3500], // matched patches come out the other side
  ["heat", 5100], // the heatmap develops
  ["hold", 6300], // a held beat before the answer
  ["done", 7100],
];
const TOTAL = READ_MIN + 7100;
const STATUS: Record<Phase, string> = {
  idle: "Waiting for an image",
  reading: "Reading the image",
  grid: "Cropping to 224 × 224",
  split: "Cutting into patches",
  feed: "Matching each patch to its nearest neighbour",
  emit: "Mapping distances back onto the image",
  heat: "Drawing the heatmap",
  hold: "Comparing the score with the threshold",
  done: "Inspection complete",
};
const pastClasses = (p: Phase) => ORDER.slice(2, ORDER.indexOf(p) + 1).map((x) => `past-${x}`);
type Offset = { dx: number; dy: number; sx: number; sy: number };

export default function InspectPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<InspectResponse | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [offsets, setOffsets] = useState<Offset[]>([]);
  const [error, setError] = useState("");
  const inPanel = useRef<HTMLDivElement>(null);
  const core = useRef<HTMLDivElement>(null);
  const outPanel = useRef<HTMLDivElement>(null);
  const verdictRef = useRef<HTMLDivElement>(null);
  const timersRef = useRef<number[]>([]);

  const skip = () => {
    timersRef.current.forEach(clearTimeout);
    setPhase("done");
  };

  useEffect(() => {
    getModels()
      .then((m) => {
        setModels(m);
        // Start on the full memory bank, the baseline every other setting is compared with.
        setModelId((id) => id || (m.find((x) => x.setting === "m100") ?? m[0])?.model_id || "");
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Score, then run the machine. Changing the model re-runs it on the same image.
  useEffect(() => {
    if (!file || !modelId) return;
    let live = true;
    const timers: number[] = [];
    timersRef.current = timers;
    setPhase("reading");
    setResult(null);
    setError("");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const minRead = new Promise((r) => setTimeout(r, reduce ? 0 : READ_MIN));
    Promise.all([inspect(file, modelId), minRead])
      .then(([r]) => {
        if (!live) return;
        setResult(r);
        if (reduce) return setPhase("done");
        setOffsets(measure());
        setPhase("grid");
        TIMELINE.forEach(([p, at]) => timers.push(window.setTimeout(() => setPhase(p), at)));
      })
      .catch((e: Error) => {
        if (!live) return;
        setError(e.message);
        setPhase("idle");
        setFile(null);
      });
    return () => {
      live = false;
      timers.forEach(clearTimeout);
    };
  }, [file, modelId]);

  // Bring the verdict into view once it lands.
  useEffect(() => {
    if (phase === "done") verdictRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [phase]);

  // Where each patch has to travel: from its place in the input to the core, and from the core
  // to its place in the output.
  function measure(): Offset[] {
    const a = inPanel.current!.getBoundingClientRect();
    const k = core.current!.getBoundingClientRect();
    const b = outPanel.current!.getBoundingClientRect();
    const kx = k.left + k.width / 2;
    const ky = k.top + k.height / 2;
    return TILES.map(({ c, r }) => ({
      dx: kx - (a.left + ((c + 0.5) * a.width) / N),
      dy: ky - (a.top + ((r + 0.5) * a.height) / N),
      sx: kx - (b.left + ((c + 0.5) * b.width) / N),
      sy: ky - (b.top + ((r + 0.5) * b.height) / N),
    }));
  }

  const model = models.find((m) => m.model_id === modelId);
  const running = phase !== "idle";
  const stage = ["machine", `phase-${phase}`, ...pastClasses(phase)].join(" ");
  const animating = running && phase !== "done";
  const tileBg = result ? `url(data:image/png;base64,${result.input_png_base64})` : "none";
  const tileStyle = (i: number, extra: CSSProperties): CSSProperties => ({
    ...extra,
    backgroundImage: tileBg,
    backgroundPosition: `${(TILES[i].c * 100) / (N - 1)}% ${(TILES[i].r * 100) / (N - 1)}%`,
  });

  return (
    <main className="page home">
      <header className="home-head">
        <div>
          <h1>Inspect a part</h1>
          <p>Feed in a photo. Argus cuts it into patches, checks each one against the good parts it remembers, and reports what it finds.</p>
        </div>
        <div className="field">
          <label htmlFor="model">Memory bank</label>
          <select id="model" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.category}, {SETTING_NAMES[m.setting] ?? m.setting}, seed {m.seed}
              </option>
            ))}
          </select>
        </div>
      </header>

      <section className={stage} aria-label="Inspection machine">
        {/* Input */}
        <div className="bay bay-in">
          <p className="bay-label">Input</p>
          <div className="panel" ref={inPanel}>
            {!running ? (
              <ImageUpload onFile={setFile} onError={setError} big />
            ) : (
              <>
                <img className="preview" src={preview} alt="The image being inspected" />
                {result && (
                  <div className="tiles">
                    {TILES.map(({ c, r }, i) => (
                      <span
                        key={i}
                        style={tileStyle(i, {
                          "--dx": `${offsets[i]?.dx ?? 0}px`,
                          "--dy": `${offsets[i]?.dy ?? 0}px`,
                          "--d": `${(r * N + c) * 22}ms`,
                        } as CSSProperties)}
                      />
                    ))}
                  </div>
                )}
                <span className="gridlines" aria-hidden="true" />
                {phase === "reading" && <span className="scanline" />}
              </>
            )}
          </div>
          {phase === "done" && (
            <div className="again">
              <ImageUpload onFile={setFile} onError={setError} />
            </div>
          )}
        </div>

        <span className="belt" aria-hidden="true" />

        {/* Core: the memory bank the patches are compared against */}
        <div className="core" ref={core}>
          <p className="bay-label">Memory bank</p>
          <div className="core-grid" aria-hidden="true">
            {Array.from({ length: 100 }, (_, i) => (
              <span key={i} style={{ "--r": `${(i * 37) % 23 * 45}ms` } as CSSProperties} />
            ))}
          </div>
          <dl className="readout">
            <dt>Stored</dt>
            <dd>{model ? model.coreset_size.toLocaleString() : "—"}</dd>
            <dt>Compared</dt>
            <dd><span className="count" /> / {N * N}</dd>
          </dl>
        </div>

        <span className="belt" aria-hidden="true" />

        {/* Output */}
        <div className="bay bay-out">
          <p className="bay-label">Output</p>
          <div className="panel out" ref={outPanel}>
            {result && running && phase !== "reading" ? (
              <>
                <div className="tiles">
                  {TILES.map(({ c, r }, i) => (
                    <span
                      key={i}
                      style={tileStyle(i, {
                        "--sx": `${offsets[i]?.sx ?? 0}px`,
                        "--sy": `${offsets[i]?.sy ?? 0}px`,
                        "--d": `${(r * N + c) * 16}ms`,
                      } as CSSProperties)}
                    />
                  ))}
                </div>
                <img className="heat" src={`data:image/png;base64,${result.heatmap_png_base64}`} alt="Anomaly heatmap" />
              </>
            ) : (
              <p className="waiting">{phase === "reading" ? "Reading image" : "The verdict appears here"}</p>
            )}
          </div>
          {result && phase === "done" && (
            <div className="verdict-block" ref={verdictRef}>
              <VerdictBadge verdict={result.verdict} />
              <dl className="stats">
                <dt>Score</dt>
                <dd>{result.score.toFixed(3)}</dd>
                <dt>Threshold τ</dt>
                <dd>{result.tau.toFixed(3)}</dd>
                <dt>Rule</dt>
                <dd>defect if score &gt; τ</dd>
                <dt>Latency</dt>
                <dd>{result.latency_ms.toFixed(1)} ms</dd>
              </dl>
              <div className="legend" aria-label={`Heatmap scale from ${result.heatmap_range[0]} to ${result.heatmap_range[1]}`}>
                <span>{result.heatmap_range[0].toFixed(2)}</span>
                <span className="legend-bar" />
                <span>{result.heatmap_range[1].toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>
        <div className="progress" aria-hidden="true">
          {running && <span key={file?.name + modelId} style={{ animationDuration: `${TOTAL}ms` }} />}
        </div>
        <div className="status" role="status">
          <span className={animating ? "status-text live" : "status-text"}>{STATUS[phase]}</span>
          {animating && (
            <button type="button" className="skip" onClick={skip}>
              Skip
            </button>
          )}
        </div>
      </section>

      {error && <p className="error" role="alert">{error}</p>}
      {USE_MOCK && (
        <p className="fine">Sample data: scores and heatmaps are generated in the browser until the trained models are served.</p>
      )}
    </main>
  );
}
