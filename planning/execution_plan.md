# Execution Plan

**Team:** Arvind Kaushik (A), Pratyush Garg (P)
**Ownership:** see [work_split.md](work_split.md)

This plan is built so the two of you work in parallel and only meet at a few fixed handoff points. Everything between those handoffs is independent.

---

## How the plan avoids tangling

1. **Contracts first.** In Phase 0 you agree on the file formats and function signatures that cross between your halves. After that, each of you codes against the contract, not against the other person's code.
2. **Fakes until the real thing arrives.** Pratyush builds and tests everything on synthetic score files and a stub detector that follow the contract. When Arvind's real outputs arrive, they drop in with no code changes.
3. **Only five handoffs** in the whole project (listed below). Nobody waits on anyone else outside these points.
4. **Each task has one owner, one branch, and one pull request.** The other person reviews it.

### The five handoffs

| # | When | From → To | What |
|---|---|---|---|
| H1 | End of Phase 0 | Both | Signed-off contracts (C1–C5) |
| H2 | End of Phase 1 | P → A | Calibration and operating-point functions (`fit_threshold`, `operating_point`) |
| H3 | End of Phase 2 | A → P | `load_detector()` and the first real score file |
| H4 | During Phase 3 | A → P | MVTec AD score files (all settings and seeds) |
| H5 | During Phase 4 | A → P | ISP-AD and review zone result tables, for figures |

---

## Efficiency rules (apply to every phase)

- **Extract features once per image, not per split.** Cache each image's patch features (float16) under `artifacts/features/`. A split or seed is then just a list of image ids, so changing seeds never re-runs the CNN.
- **One greedy run gives three memory settings.** Greedy coreset adds one patch at a time, so 1% and ASO's size are prefixes of the 10% run. Store the selection order once and slice it. 100% needs no sampling.
- **Cache every pipeline stage** (features → coreset order → scores → thresholds → metrics), keyed by a hash of its config. Re-running with a new FPR target or review band only redoes the last stage, which takes seconds.
- **Lock the test set in code.** The evaluation step refuses to read test images until the threshold file for that run exists. This enforces the fair-test rule from the plan automatically.
- **Use the GPU only for features and nearest-neighbour search.** Everything after scoring runs on the CPU from saved files, so Pratyush never needs a GPU.

---

## Phase 0: Setup and contracts

**Goal:** both of you can work alone from here on.

| ID | Task | Owner | Done when |
|---|---|---|---|
| 0.1 | Fill `pyproject.toml` (torch, torchvision, numpy, scipy, pandas, scikit-learn, pyyaml, opencv-python, pillow, fastapi, uvicorn, pytest, ruff) and the `Makefile` targets (`install`, `test`, `lint`) | A | `make install && make test` runs on both laptops |
| 0.2 | Scaffold the frontend (Vite + React + TS) and the backend app (`/health` route) | P | `npm run dev` and `uvicorn` both start |
| 0.3 | Agree on contracts C1–C5 (below) and write them into `docs/architecture.md` | Both | Both approve the PR |
| 0.4 | Decide the MVTec AD categories and split sizes (see the note below) and write `configs/datasets/*.yaml` | Both | Configs merged |
| 0.5 | Write `docs/experiment_protocol.md`: seeds, FPR targets, confidence, fair-test rules | P | PR merged |

### Contracts to agree in 0.3

**C1. Split manifest** (A produces, both read)
One CSV per dataset, category/modality and seed: `image_id, path, role, label, defect_type, mask_path`.
`role` is one of `support`, `calib`, `eval_normal`, `test`.

**C2. Score file** (A produces, P reads)
One Parquet file per run, at `artifacts/scores/{dataset}/{category}/{setting}/seed{n}/scores.parquet`.
Columns: `image_id, role, label, defect_type, score, heatmap_path, mask_path`.
Plus `run.json` with: config hash, memory setting, coreset size, memory bank size (MB), setup time (s), mean time per image (ms).
Heatmaps are saved as float16 `.npy` files.

**C3. Detector interface** (A provides, P's backend and stress tests use)
```python
det = argus.load_detector(path)           # loads a saved memory bank
score, heatmap = det.score(image)         # one PIL image in, float + HxW array out
scores, heatmaps = det.score_batch(images)
```

**C4. Calibration interface** (P provides, A uses for ISP-AD and the review zone)
```python
t = fit_threshold(calib_scores, alpha=0.05, delta=0.05)   # -> tau, rank, achieved_confidence
op = operating_point(scores, labels, t.tau)               # -> fpr, recall, Wilson CIs, counts
```
Decision rule is strict: flagged if `score > tau`.

**C5. HTTP API** (P owns both ends)
`POST /inspect` (image, model id) → `score, verdict (good | defect | review), tau, heatmap (PNG, base64), latency_ms`.
`GET /models` lists available memory banks. `GET /results` serves result tables.

### Note for 0.4: MVTec AD category choice

- MVTec's own test set has only about 40 good images per category, which is too few to measure FPR precisely. Take extra **evaluation normals** from the training images, as Deng does. For example, from hazelnut's ~391 training images: about 200 support, 100 calibration, and the rest plus the test-set good images as evaluation normals. Check exact counts after downloading.
- 100 calibration images supports targets of 10% and 5%. The 2% target needs at least 149.
- Hazelnut scores 100% AUROC, so memory size may make no visible difference there. Add at least one harder category where shrinking memory hurt in the ASO paper (screw is the clearest case).

---

## Phase 1: Foundations (fully parallel)

**Goal:** Arvind can turn images into cached features; Pratyush has working, tested calibration code.

### Arvind

| ID | Task | Depends on | Done when |
|---|---|---|---|
| A1.1 | Download scripts for MVTec AD and the ISP-AD subset (Deng's per-modality split: 300 / 150 / 200 normals plus real defects) | 0.4 | Both datasets in `data/` |
| A1.2 | Dataset loaders and split-manifest generator (C1), seeded | A1.1 | Tests pass: roles disjoint, same seed gives the same split |
| A1.3 | Backbone and patch features: WRN-50 layers 2+3, 3×3 local average, layer 3 upsampled to layer 2's grid | 0.1 | Feature shape matches PatchCore (28×28 grid at 224 input) |
| A1.4 | Feature cache: one float16 file per image, keyed by image id and preprocessing config | A1.3 | Second run on the same images does no CNN pass |
| A1.5 | Exact nearest-neighbour search, chunked on the GPU (`torch.cdist`), CPU fallback | A1.3 | Matches brute-force NumPy on random data |

### Pratyush

| ID | Task | Depends on | Done when |
|---|---|---|---|
| P1.1 | `tolerance.py`: Wilks order-statistic threshold (C4) | 0.3 | Tests reproduce Deng's Table 6: m=150 gives rank 142 at 10% (confidence 0.9693), 148 at 5%, 150 at 2% |
| P1.2 | `sample_size.py`: minimum normals per target | — | Tests give 29 / 59 / 149 / 299 / 598 / 2995 for 10% → 0.1%, and α_min = 0.01977 at m=150 |
| P1.3 | `operating_point.py`: FPR, recall, Wilson 95% intervals, counts | — | Tested on hand-made examples |
| P1.4 | `image_metrics.py`: image AUROC | — | Matches scikit-learn |
| P1.5 | Fake score generator in `tests/conftest.py`, following C2 | 0.3 | Used by all of P's tests |
| P1.6 | Backend: stub detector + `/inspect` and `/models` returning contract-shaped fake responses | 0.2, 0.3 | Frontend can call it |
| P1.7 | Frontend: upload page calling the stub API, showing score and verdict | P1.6 | Works end to end with fake data |

**Handoff H2:** Pratyush merges P1.1–P1.3. Arvind can now calibrate.

---

## Phase 2: Detector and evaluation toolkit (fully parallel)

**Goal:** a working PatchCore that matches the paper, and a complete results toolkit tested on fake data.

### Arvind

| ID | Task | Depends on | Done when |
|---|---|---|---|
| A2.1 | Greedy coreset with random projection (to 128 dims); records the selection order and the representative distance d(k) at every step | A1.5 | Coverage on 2D toy data looks like PatchCore's Figure 3 |
| A2.2 | ASO: normalise k by k₁₀% and d by d₁₀%, pick k* minimising distance to the origin | A2.1 | Test on a synthetic curve; without normalisation it collapses to ~0, as the ASO paper says |
| A2.3 | Memory bank save/load: features plus selection order; settings 100 / 10 / 1 / ASO sliced from one run | A2.1, A2.2 | One greedy run produces all four banks |
| A2.4 | PatchCore scoring: nearest-patch distance, neighbourhood reweighting (Eq. 7), image score = max patch, heatmap upsampled and smoothed (σ = 4) | A2.3 | — |
| A2.5 | `load_detector()` and `score()` / `score_batch()` (C3) | A2.4 | Backend can import it |
| A2.6 | Pipeline runner: stage caching by config hash, timing and memory measurement, writes C2 score files | A2.4 | Re-running a finished config takes seconds |
| A2.7 | **Baseline check:** full training set, 10%, chosen categories | A2.6 | Image AUROC within ~1 point of the papers' per-category values |
| A2.8 | First real score file: one category, all four settings, seed 1 | A2.7 | Passes P's C2 schema check |

### Pratyush

| ID | Task | Depends on | Done when |
|---|---|---|---|
| P2.1 | `pixel_metrics.py`: pixel AUROC and PRO (integrated up to 30% FPR, normalised) | P1.5 | Tested on synthetic masks |
| P2.2 | `report.py`: reads a folder of C2 score files and builds the results table (FPR, recall, AUROC, RAM, time per image; mean ± range over seeds) as CSV and Markdown | P1.1–P1.5 | Produces the plan's four-row table from fake data |
| P2.3 | C2 schema checker (fails loudly on a missing column or a wrong role) | P1.5 | Run automatically by `report.py` |
| P2.4 | `perturbations.py`: brightness, contrast, Gaussian blur, JPEG compression, each at 3–5 fixed severity levels, deterministic | — | Visual check notebook plus unit tests |
| P2.5 | Frontend: heatmap overlay and verdict badge, still on the stub | P1.7 | — |

**Handoff H3:** Arvind merges A2.5 and A2.8. Pratyush swaps the stub detector for `load_detector()` and runs `report.py` on the real score file. If anything breaks, the contract is fixed, not worked around.

---

## Phase 3: Main experiments (parallel, one data-flow handoff)

**Goal:** both main results tables filled.

### Arvind

| ID | Task | Depends on | Done when |
|---|---|---|---|
| A3.1 | MVTec AD runs: chosen categories × 4 settings × 3 seeds → score files | A2.8 | All files pass the schema check (**handoff H4**) |
| A3.2 | ISP-AD reproduction: Deng's exact setup (WRN-50 layers 2+3, 256×256 with no crop, single nearest patch, 100% memory), 3 modalities × 3 seeds | A2.6 | AUROC and FPR close to Deng's Table 2 (PatchCore: AUROC 0.893 / 0.752 / 0.860; FPR 6.5–6.8% at the 10% target) |
| A3.3 | ISP-AD memory sweep: 10%, 1%, ASO on the same splits | A3.2 | Score files written |
| A3.4 | ISP-AD analysis with P's `report.py`: targets 10%, 5%, 2% | A3.3, H2 | ISP-AD results table done |

### Pratyush

| ID | Task | Depends on | Done when |
|---|---|---|---|
| P3.1 | MVTec AD analysis: one threshold per setting per seed, targets 10% and 5% (2% only if calibration has ≥149 images) | H4 | MVTec AD results table done |
| P3.2 | Backend on real banks: `model_registry` loads each bank once and keeps it in memory | H3 | `/inspect` returns real scores and heatmaps |
| P3.3 | Frontend inspect page finished, plus a threshold slider showing how the verdict changes | P3.2 | Demo works on real images |
| P3.4 | Stress tests, scoring: perturb calibration and evaluation images, score them with the existing banks via `score_batch()` | H3, P2.4 | Score files for each perturbation and severity |

**Gate G3:** both main tables exist. Short joint check: are the numbers plausible, and do they agree with the papers where they should?

---

## Phase 4: Studies (fully parallel)

**Goal:** the two extra findings, plus all figures.

### Arvind: human review zone

| ID | Task | Depends on | Done when |
|---|---|---|---|
| A4.1 | `review_zone.py`: a band around τ. Below the band → good, above → defect, inside → sent to a person. Band edges set from calibration scores only | H2 | Tested on fake scores |
| A4.2 | Sweep band widths on both datasets: errors caught versus share of items sent for review | A4.1, A3.1, A3.3 | Review zone table (**handoff H5**) |
| A4.3 | *(Optional)* ASO early stopping: stop greedy sampling once the curve flattens; measure setup time saved and whether k* changes | A2.2 | Comparison table |

### Pratyush: stress tests and figures

| ID | Task | Depends on | Done when |
|---|---|---|---|
| P4.1 | Stress test analysis: FPR and recall at the original τ under each perturbation; at what severity the guarantee breaks | P3.4 | Stress table |
| P4.2 | All figures: memory size vs FPR/recall, ASO curve, review zone trade-off, stress curves | H4, H5, P4.1 | Figures saved under `docs/figures/` |
| P4.3 | Frontend results page: shows the tables from `/results` | P2.2 | — |
| P4.4 | Review-zone verdict (`review`) shown in the demo | A4.1 | — |
| P4.5 | *(Optional)* Spatial lift (Deng's paired-minus-crossed test) on heatmaps | P2.1 | — |

---

## Phase 5: Write-up and demo

| ID | Task | Owner |
|---|---|---|
| 5.1 | Method and memory sections; ISP-AD results; review zone results | A |
| 5.2 | Calibration section; MVTec AD results; stress test results; demo section | P |
| 5.3 | `docker-compose.yml` runs backend and frontend with one command | P |
| 5.4 | `README.md`: how to reproduce every table from scratch (`make` targets) | A |
| 5.5 | Swap and review each other's sections | Both |
| 5.6 | Final check: every number in the write-up comes from a file in `artifacts/` | Both |

---

## Branches and pull requests

- Branch per task, named `<name>/<task-id>-<short-title>`, for example `arvind/A2.1-greedy-coreset` or `pratyush/P1.1-tolerance`.
- Every PR needs tests for its module and one approval from the other person.
- A change to a contract (C1–C5) needs both people to approve, and its own PR.
