import { useState } from "react";

import { Link } from "../router";

// Shares of the memory bank the study compares. ASO is left out: its size is chosen per run.
const KEEP = [1, 0.1, 0.01];

export default function HowItWorksPage() {
  const [keep, setKeep] = useState(1);

  return (
    <main className="page how">
      <header className="page-head">
        <h1>How it works</h1>
        <p>
          PatchCore, a threshold with a stated false-positive rate, and the question of how much
          memory the detector needs before that rate stops holding.
        </p>
      </header>

      <section className="step">
        <div>
          <h2>It learns only what a good part looks like</h2>
          <p>
            PatchCore cuts photos of defect-free parts into small patches and stores what each
            patch looks like in a memory bank. Training never sees a defect.
          </p>
        </div>
        <div className="facts">
          <p><b>200</b>good hazelnuts</p>
          <p><b>160</b>good screws</p>
        </div>
      </section>

      <section className="step">
        <div>
          <h2>Anything it has never seen stands out</h2>
          <p>
            Each patch of a new image is matched to its nearest neighbour in memory. Patches far
            from everything stored light up on the heatmap, and the farthest one sets the
            image&apos;s anomaly score.
          </p>
        </div>
        <figure className="heat-scale">
          <span className="legend-bar" />
          <figcaption>
            <span>close to memory</span>
            <span>far from anything stored</span>
          </figcaption>
        </figure>
      </section>

      <section className="step">
        <div>
          <h2>How small can the memory get?</h2>
          <p>
            Keeping every patch is slow and heavy. We shrink the bank to <b>10%</b> and <b>1%</b>{" "}
            with greedy coreset sampling, let <b>ASO</b> choose a size on its own, and measure what
            each cut costs.
          </p>
        </div>
        <figure className="bank" style={{ "--mem": keep } as React.CSSProperties}>
          <div className="bank-grid" aria-hidden="true">
            {Array.from({ length: 100 }, (_, i) => (
              <span key={i} style={{ "--i": i } as React.CSSProperties} />
            ))}
          </div>
          <figcaption>
            <b>{keep * 100}%</b>of the memory bank kept
            <span className="toggle" role="group" aria-label="Memory kept">
              {KEEP.map((k) => (
                <button key={k} type="button" aria-pressed={k === keep} onClick={() => setKeep(k)}>
                  {k * 100}%
                </button>
              ))}
            </span>
          </figcaption>
        </figure>
      </section>

      <section className="step">
        <div>
          <h2>A threshold that comes with a guarantee</h2>
          <p>
            Good images held back for calibration set the cut-off. A distribution-free tolerance
            bound places it so that, with <b>95%</b> confidence, the false-positive rate stays under
            the target. The open question is whether that still holds once the memory shrinks.
          </p>
        </div>
        <div>
          <table className="need">
            <thead>
              <tr><th>FPR target</th><th>Good images needed</th></tr>
            </thead>
            <tbody>
              <tr><td>10%</td><td>29</td></tr>
              <tr><td>5%</td><td>59</td></tr>
              <tr><td>2%</td><td>149</td></tr>
            </tbody>
          </table>
          <p className="fine">
            MVTec AD holds back 100, enough for 10% and 5%. ISP-AD holds back 150 and runs all three.
          </p>
        </div>
      </section>

      <footer className="how-end">
        <Link className="cta" to="/">Try it on an image</Link>
        <p className="fine">
          Built by Arvind Kaushik and Pratyush Garg.{" "}
          <a href="https://github.com/AnomaSight/Argus">Source on GitHub</a>
        </p>
      </footer>
    </main>
  );
}
