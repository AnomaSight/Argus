# Task issues

One file per task. ★ = on the critical path. GitHub issues are the live copy; these files are a snapshot.

## Phase 0: Setup and contracts

| Task | Issue | Title | Owner | Size | Depends on |
|---|---|---|---|---|---|
| [0.1](0.1-python-environment-dependencies-and-makefile.md) | [#1](https://github.com/AnomaSight/Argus/issues/1) | Python environment, dependencies and Makefile | Arvind | S ★ | – |
| [0.2](0.2-scaffold-the-frontend-and-backend.md) | [#2](https://github.com/AnomaSight/Argus/issues/2) | Scaffold the frontend and backend apps | Pratyush | S | 0.1 |
| [C1](C1-contract-c1-split-manifest-format.md) | [#3](https://github.com/AnomaSight/Argus/issues/3) | Contract C1: split manifest format | Both | S | – |
| [C2](C2-contract-c2-score-file-format.md) | [#4](https://github.com/AnomaSight/Argus/issues/4) | Contract C2: score file format | Both | S | – |
| [C3](C3-contract-c3-detector-interface-load.md) | [#5](https://github.com/AnomaSight/Argus/issues/5) | Contract C3: detector interface (`load_detector`) | Both | S | – |
| [C4](C4-contract-c4-calibration-interface-fit.md) | [#6](https://github.com/AnomaSight/Argus/issues/6) | Contract C4: calibration interface (`fit_threshold`, `operating_point`) | Both | S | – |
| [C5](C5-contract-c5-http-api.md) | [#7](https://github.com/AnomaSight/Argus/issues/7) | Contract C5: HTTP API | Both | S | – |
| [0.4](0.4-choose-mvtec-ad-categories-and.md) | [#8](https://github.com/AnomaSight/Argus/issues/8) | Choose MVTec AD categories and split sizes; write dataset configs | Both | M | – |
| [0.5](0.5-write-the-experiment-protocol.md) | [#9](https://github.com/AnomaSight/Argus/issues/9) | Write the experiment protocol | Pratyush | S | 0.4 |

## Phase 1: Foundations

| Task | Issue | Title | Owner | Size | Depends on |
|---|---|---|---|---|---|
| [A1.1](A1.1-dataset-download-scripts-mvtec-ad.md) | [#10](https://github.com/AnomaSight/Argus/issues/10) | Dataset download scripts (MVTec AD, ISP-AD subset) | Arvind | M | 0.1, 0.4 |
| [A1.2](A1.2-dataset-loaders-and-seeded-split.md) | [#11](https://github.com/AnomaSight/Argus/issues/11) | Dataset loaders and seeded split-manifest generator | Arvind | M | A1.1, C1 |
| [A1.3](A1.3-backbone-and-locally-aware-patch.md) | [#12](https://github.com/AnomaSight/Argus/issues/12) | Backbone and locally aware patch features | Arvind | M ★ | 0.1 |
| [A1.4](A1.4-per-image-feature-cache.md) | [#13](https://github.com/AnomaSight/Argus/issues/13) | Per-image feature cache | Arvind | S | A1.3 |
| [A1.5](A1.5-exact-nearest-neighbour-search.md) | [#14](https://github.com/AnomaSight/Argus/issues/14) | Exact nearest-neighbour search | Arvind | S ★ | A1.3 |
| [P1.1](P1.1-wilks-tolerance-threshold-fit-threshold.md) | [#15](https://github.com/AnomaSight/Argus/issues/15) | Wilks tolerance threshold (`fit_threshold`) | Pratyush | M | C4, 0.1 |
| [P1.2](P1.2-calibration-sample-size-planning.md) | [#16](https://github.com/AnomaSight/Argus/issues/16) | Calibration sample-size planning | Pratyush | S | 0.1 |
| [P1.3](P1.3-operating-point-fpr-recall-and.md) | [#17](https://github.com/AnomaSight/Argus/issues/17) | Operating point: FPR, recall and Wilson intervals | Pratyush | S | C4 |
| [P1.4](P1.4-image-auroc.md) | [#18](https://github.com/AnomaSight/Argus/issues/18) | Image AUROC | Pratyush | S | 0.1 |
| [P1.5](P1.5-fake-score-file-generator-c2.md) | [#19](https://github.com/AnomaSight/Argus/issues/19) | Fake score-file generator (C2) for tests | Pratyush | S | C2 |
| [P1.6](P1.6-backend-with-a-stub-detector.md) | [#20](https://github.com/AnomaSight/Argus/issues/20) | Backend with a stub detector | Pratyush | M | 0.2, C3, C5 |
| [P1.7](P1.7-frontend-inspect-page-on-the.md) | [#21](https://github.com/AnomaSight/Argus/issues/21) | Frontend inspect page (on the stub API) | Pratyush | M | P1.6 |

## Phase 2: Detector and evaluation toolkit

| Task | Issue | Title | Owner | Size | Depends on |
|---|---|---|---|---|---|
| [A2.1](A2.1-greedy-coreset-that-records-selection.md) | [#22](https://github.com/AnomaSight/Argus/issues/22) | Greedy coreset that records selection order and d(k) | Arvind | M ★ | A1.5 |
| [A2.2](A2.2-aso-automatic-memory-size.md) | [#23](https://github.com/AnomaSight/Argus/issues/23) | ASO: automatic memory size | Arvind | S ★ | A2.1 |
| [A2.3](A2.3-memory-bank-save-load-with.md) | [#24](https://github.com/AnomaSight/Argus/issues/24) | Memory bank save/load with all four settings from one run | Arvind | M ★ | A2.1, A2.2, A1.4 |
| [A2.4](A2.4-patchcore-scoring-and-heatmaps.md) | [#25](https://github.com/AnomaSight/Argus/issues/25) | PatchCore scoring and heatmaps | Arvind | M ★ | A2.3, A1.5 |
| [A2.5](A2.5-load-detector-and-the-detector.md) | [#26](https://github.com/AnomaSight/Argus/issues/26) | `load_detector()` and the Detector class (C3) | Arvind | S | A2.4, C3 |
| [A2.6](A2.6-pipeline-runner-with-stage-caching.md) | [#27](https://github.com/AnomaSight/Argus/issues/27) | Pipeline runner with stage caching | Arvind | L ★ | A2.4, A1.2, A1.4, C2 |
| [A2.7](A2.7-baseline-check-against-the-papers.md) | [#28](https://github.com/AnomaSight/Argus/issues/28) | Baseline check against the papers | Arvind | M ★ | A2.6, P1.4 |
| [A2.8](A2.8-first-real-score-file-one.md) | [#29](https://github.com/AnomaSight/Argus/issues/29) | First real score file (one category, four settings, seed 1) | Arvind | S ★ | A2.7, P2.3 |
| [P2.1](P2.1-pixel-auroc-and-pro.md) | [#30](https://github.com/AnomaSight/Argus/issues/30) | Pixel AUROC and PRO | Pratyush | M | P1.5 |
| [P2.2](P2.2-results-table-builder-report-py.md) | [#31](https://github.com/AnomaSight/Argus/issues/31) | Results-table builder (`report.py`) | Pratyush | L | P1.1, P1.3, P1.4, P1.5, P2.3 |
| [P2.3](P2.3-c2-schema-checker.md) | [#32](https://github.com/AnomaSight/Argus/issues/32) | C2 schema checker | Pratyush | S | C2, P1.5 |
| [P2.4](P2.4-image-perturbations-for-stress-tests.md) | [#33](https://github.com/AnomaSight/Argus/issues/33) | Image perturbations for stress tests | Pratyush | S | 0.1 |
| [P2.5](P2.5-heatmap-overlay-and-verdict-badge.md) | [#34](https://github.com/AnomaSight/Argus/issues/34) | Heatmap overlay and verdict badge | Pratyush | M | P1.7 |

## Phase 3: Main experiments

| Task | Issue | Title | Owner | Size | Depends on |
|---|---|---|---|---|---|
| [A3.1](A3.1-mvtec-ad-runs-all-categories.md) | [#35](https://github.com/AnomaSight/Argus/issues/35) | MVTec AD runs: all categories × 4 settings × 3 seeds | Arvind | L ★ | A2.8 |
| [A3.2](A3.2-isp-ad-reproduction-of-deng.md) | [#36](https://github.com/AnomaSight/Argus/issues/36) | ISP-AD reproduction of Deng's PatchCore result | Arvind | L | A2.6, A1.2, P2.2 |
| [A3.3](A3.3-isp-ad-memory-sweep-10.md) | [#37](https://github.com/AnomaSight/Argus/issues/37) | ISP-AD memory sweep (10%, 1%, ASO) | Arvind | M | A3.2 |
| [A3.4](A3.4-isp-ad-analysis-and-results.md) | [#38](https://github.com/AnomaSight/Argus/issues/38) | ISP-AD analysis and results table | Arvind | M | A3.3, P2.2 |
| [P3.1](P3.1-mvtec-ad-analysis-and-results.md) | [#39](https://github.com/AnomaSight/Argus/issues/39) | MVTec AD analysis and results table | Pratyush | M | A3.1, P2.2, P2.1 |
| [P3.2](P3.2-backend-on-real-memory-banks.md) | [#40](https://github.com/AnomaSight/Argus/issues/40) | Backend on real memory banks | Pratyush | M | A2.5, P1.6 |
| [P3.3](P3.3-finish-the-inspect-page-with.md) | [#41](https://github.com/AnomaSight/Argus/issues/41) | Finish the inspect page with a threshold slider | Pratyush | M | P3.2, P2.5 |
| [P3.4](P3.4-stress-tests-score-perturbed-images.md) | [#42](https://github.com/AnomaSight/Argus/issues/42) | Stress tests: score perturbed images | Pratyush | M ★ | A2.5, P2.4, A3.1 |

## Phase 4: Studies

| Task | Issue | Title | Owner | Size | Depends on |
|---|---|---|---|---|---|
| [A4.1](A4.1-review-zone-implementation.md) | [#43](https://github.com/AnomaSight/Argus/issues/43) | Review zone: implementation | Arvind | S | P1.1 |
| [A4.2](A4.2-review-zone-study-on-both.md) | [#44](https://github.com/AnomaSight/Argus/issues/44) | Review zone study on both datasets | Arvind | M | A4.1, A3.1, A3.3, P2.2 |
| [A4.3](A4.3-aso-early-stopping.md) | [#45](https://github.com/AnomaSight/Argus/issues/45) | (Optional) ASO early stopping | Arvind | M | A2.2 |
| [P4.1](P4.1-stress-test-analysis.md) | [#46](https://github.com/AnomaSight/Argus/issues/46) | Stress test analysis | Pratyush | M ★ | P3.4, P1.3 |
| [P4.2](P4.2-all-figures-for-the-write.md) | [#47](https://github.com/AnomaSight/Argus/issues/47) | All figures for the write-up | Pratyush | L ★ | P3.1, A3.4, A4.2, P4.1 |
| [P4.3](P4.3-results-page.md) | [#48](https://github.com/AnomaSight/Argus/issues/48) | Results page | Pratyush | M | P2.2, P3.3, C5 |
| [P4.4](P4.4-show-the-review-verdict-in.md) | [#49](https://github.com/AnomaSight/Argus/issues/49) | Show the review verdict in the demo | Pratyush | S | A4.1, P3.3 |
| [P4.5](P4.5-spatial-lift-on-heatmaps.md) | [#50](https://github.com/AnomaSight/Argus/issues/50) | (Optional) Spatial lift on heatmaps | Pratyush | L | P2.1, A3.1 |

## Phase 5: Write-up and demo

| Task | Issue | Title | Owner | Size | Depends on |
|---|---|---|---|---|---|
| [5.1](5.1-write-up-method-memory-isp.md) | [#51](https://github.com/AnomaSight/Argus/issues/51) | Write-up: method, memory, ISP-AD results, review zone | Arvind | L ★ | A3.4, A4.2, P4.2 |
| [5.2](5.2-write-up-calibration-mvtec-ad.md) | [#52](https://github.com/AnomaSight/Argus/issues/52) | Write-up: calibration, MVTec AD results, stress tests, demo | Pratyush | L | P3.1, P4.1, P4.2, P3.3 |
| [5.3](5.3-docker-compose-for-the-demo.md) | [#53](https://github.com/AnomaSight/Argus/issues/53) | docker-compose for the demo | Pratyush | M | P3.3, P4.3 |
| [5.4](5.4-readme-reproduce-every-table-from.md) | [#54](https://github.com/AnomaSight/Argus/issues/54) | README: reproduce every table from scratch | Arvind | M | A3.1, A3.4, A4.2, P4.2 |
| [5.5](5.5-cross-review-each-other-s.md) | [#55](https://github.com/AnomaSight/Argus/issues/55) | Cross-review each other's write-up | Both | M ★ | 5.1, 5.2 |
| [5.6](5.6-final-number-check.md) | [#56](https://github.com/AnomaSight/Argus/issues/56) | Final number check | Both | M ★ | 5.5, 5.4 |
