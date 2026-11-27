# Work Split

**Team:** Arvind Kaushik, Pratyush Garg

The system has two halves:

- **Detector (Arvind):** looks at a product photo and gives it an anomaly score and a heatmap.
- **Decision layer and app (Pratyush):** takes those scores, decides what gets flagged, measures how well that works, and shows it in the web app.

Arvind builds the model (the memory banks). Pratyush never builds it; he uses Arvind's finished memory banks to score images (in the backend and the stress tests) and works from the score files Arvind produces.

---

## Arvind

**Code**

| Area | Files |
|---|---|
| Data loading and splits | `src/argus/data/`, `scripts/download_*.sh` |
| Feature extraction and caching | `src/argus/features/` |
| Greedy coreset, ASO, memory bank, nearest-neighbour search | `src/argus/memory/` |
| PatchCore scoring and heatmaps | `src/argus/scoring/` |
| Pipeline runner, CLI, utilities | `src/argus/pipeline/`, `src/argus/cli.py`, `src/argus/utils/` |
| Review zone | `src/argus/calibration/review_zone.py` |

**Model runs**

- Download and prepare both datasets.
- Build memory banks and produce score files for **both datasets** (MVTec AD and ISP-AD), for all four memory settings (100%, 10%, 1%, ASO).

**Research**

- **ISP-AD, end to end:** download Deng's subset only (per camera type: 300 train, 150 calibration, 200 evaluation normals, plus the real defects), reproduce Deng's 100% result, run the memory sweep, calibrate, analyse, and write the ISP-AD results.
- **Human review zone study (plan step 4):** how wide the "send to a person" band around the cut-off should be, how many mistakes it catches, and how many items it sends for review.

---

## Pratyush

**Code**

| Area | Files |
|---|---|
| Cut-off (Wilks rule) and sample-size table | `src/argus/calibration/tolerance.py`, `src/argus/calibration/sample_size.py` |
| AUROC, PRO, FPR and recall at the cut-off, spatial lift, results tables | `src/argus/evaluation/` |
| Stress-test perturbations (lighting, contrast, blur, compression) | `src/argus/stress/` |
| FastAPI backend | `backend/` |
| Frontend | `frontend/` |

**Research**

- **MVTec AD analysis:** take Arvind's MVTec score files, set the cut-offs, measure FPR and recall, fill in the main results table, and write the MVTec AD results.
- **Stress-test study (plan step 5), end to end:** change the images, score them with Arvind's existing memory banks, and analyse whether the cut-off still holds under changed lighting, contrast, blur and compression.
- **All charts and figures** for the write-up.

---

## Summary

| | MVTec AD | ISP-AD |
|---|---|---|
| Run the model and make the score files | Arvind | Arvind |
| Cut-off, FPR, recall, results table | Pratyush | Arvind |
| Write up its results | Pratyush | Arvind |

| Study | Owner |
|---|---|
| Memory sweep (model runs) | Arvind |
| ISP-AD reproduction of Deng's paper | Arvind |
| Human review zone | Arvind |
| Stress tests (perturbations, scoring and analysis) | Pratyush |

The ISP-AD analysis and the review zone study use Pratyush's cut-off and evaluation code, so both people rely on the same implementation.

---

## Shared

- **Configs and docs:** `configs/`, `docs/architecture.md`, `docs/experiment_protocol.md`.
- **Tests:** each person writes the tests for their own modules.
- **Write-up:** Arvind writes the method, memory, ISP-AD results and review zone sections; Pratyush writes the calibration, MVTec AD results, stress-test and demo sections.
- **Code review:** each reviews the other's pull requests.

---

## Agree on these first (Phase 0)

1. **Score file format.** This is the handoff between the two halves. Each record should hold: image id, split, label, image score, heatmap path, memory setting, seed.
2. **API shape.** What `/inspect` returns, so the frontend and backend can be built independently.

Until real score files exist, Pratyush can build and test calibration and evaluation on synthetic score arrays, and build the frontend against a mock API.

---

## Working rules

- One branch per task; merge into `main` through pull requests.
- Don't commit datasets or pipeline outputs (`data/` and `artifacts/` are gitignored).
