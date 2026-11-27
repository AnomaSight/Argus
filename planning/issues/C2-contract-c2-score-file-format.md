# [C2] Contract C2: score file format

GitHub issue: [#4](https://github.com/AnomaSight/Argus/issues/4)

**Labels:** `size: S`, `type: design`, `blocks other person`

**Task:** C2 · **Owner:** Both · **Milestone:** Phase 0: Setup and contracts
**Branch:** `shared/C2-contract-c2-score-file-format`

## Summary
Agree on the exact format of the score files that pass from the detector (Arvind) to the analysis (Pratyush).

## Why this matters
This is the main handoff in the project. Once it is fixed, Pratyush can build all analysis on fake files and Arvind's real files drop in unchanged.

## What to do
1. Location: `artifacts/scores/{dataset}/{category}/{setting}/seed{n}/`, where `setting` is `m100`, `m10`, `m1` or `aso`.
2. `scores.parquet` columns: `image_id` (str), `role` (str), `label` (int8), `defect_type` (str), `score` (float32), `heatmap_path` (str, relative), `mask_path` (str).
3. Rows: every `calib`, `eval_normal` and `test` image of that split (support images are not scored).
4. Heatmaps: one float16 `.npy` per image, shape H×W at model input resolution, in `heatmaps/` next to the parquet.
5. `run.json`: `config_hash, dataset, category, setting, seed, coreset_size, coreset_fraction, bank_mb, setup_s, ms_per_image, device, git_commit`.
6. Stress-test variant: same layout under `artifacts/scores_stress/{perturbation}/{severity}/...` (used by P3.4).

## Deliverables
- C2 section in `docs/architecture.md` with a column table and a sample `run.json`

## Done when
- [ ] Both approve the PR
- [ ] Reviewed and approved by the other person

## Depends on (blocked by)
None. Can start immediately.

## Unblocks
- [#19](https://github.com/AnomaSight/Argus/issues/19) P1.5: Fake score-file generator (C2) for tests (Pratyush)
- [#27](https://github.com/AnomaSight/Argus/issues/27) A2.6: Pipeline runner with stage caching (Arvind)
- [#32](https://github.com/AnomaSight/Argus/issues/32) P2.3: C2 schema checker (Pratyush)

## Files
- `docs/architecture.md`

## Notes
- Lead: Arvind. Reviewer: Pratyush. Pratyush's fake generator (P1.5) and schema checker (P2.3) implement exactly this spec.

---
_Plan: `planning/execution_plan.md` · Ownership: `planning/work_split.md`_
