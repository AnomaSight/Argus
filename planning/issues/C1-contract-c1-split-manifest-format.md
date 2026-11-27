# [C1] Contract C1: split manifest format

GitHub issue: [#3](https://github.com/AnomaSight/Argus/issues/3)

**Labels:** `size: S`, `type: design`, `blocks other person`

**Task:** C1 · **Owner:** Both · **Milestone:** Phase 0: Setup and contracts
**Branch:** `shared/C1-contract-c1-split-manifest-format`

## Summary
Agree on the file that says which image plays which role (support / calib / eval_normal / test) for each dataset, category and seed.

## Why this matters
Both halves read the same splits. If roles are defined once in a file, nobody can accidentally calibrate or train on test images.

## What to do
1. Format: one CSV per (dataset, category or modality, seed) at `artifacts/splits/{dataset}/{category}/seed{n}.csv`.
2. Columns: `image_id, path, role, label, defect_type, mask_path`. `label` is 0 (good) / 1 (defect); `defect_type` is `good` for normals; `mask_path` empty for normals.
3. `role` values: `support` (builds the memory bank), `calib` (sets the cut-off), `eval_normal` (measures FPR), `test` (original test set: good + defects).
4. Rule: roles are disjoint within a seed; defect images only ever appear as `test`.
5. `image_id` is stable across seeds (e.g. relative path without extension), so features can be cached per image.
6. Write the spec into `docs/architecture.md` under "Contracts".

## Deliverables
- C1 section in `docs/architecture.md`
- One example manifest (5 rows) in the doc

## Done when
- [ ] Both approve the PR
- [ ] Reviewed and approved by the other person

## Depends on (blocked by)
None. Can start immediately.

## Unblocks
- [#11](https://github.com/AnomaSight/Argus/issues/11) A1.2: Dataset loaders and seeded split-manifest generator (Arvind)

## Files
- `docs/architecture.md`

## Notes
- Lead: Arvind (he generates manifests in A1.2). Reviewer: Pratyush.

---
_Plan: `planning/execution_plan.md` · Ownership: `planning/work_split.md`_
