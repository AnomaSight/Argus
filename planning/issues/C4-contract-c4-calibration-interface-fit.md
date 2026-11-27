# [C4] Contract C4: calibration interface (`fit_threshold`, `operating_point`)

GitHub issue: [#6](https://github.com/AnomaSight/Argus/issues/6)

**Labels:** `size: S`, `type: design`, `blocks other person`

**Task:** C4 · **Owner:** Both · **Milestone:** Phase 0: Setup and contracts
**Branch:** `shared/C4-contract-c4-calibration-interface-fit`

## Summary
Agree on the functions that turn calibration scores into a cut-off and measure FPR/recall at it.

## Why this matters
Arvind uses these for the ISP-AD analysis and the review zone, so one implementation serves both halves.

## What to do
1. `fit_threshold(calib_scores, alpha, delta=0.05) -> Threshold` with fields `tau, rank, m, alpha, delta, achieved_confidence`. Raises a clear error if `m` is too small for `alpha`.
2. `operating_point(scores, labels, tau) -> OperatingPoint` with `fpr, fpr_ci, recall, recall_ci, n_normal, n_defect, fp, tp` (Wilson 95% intervals).
3. Decision rule fixed as strict: flagged if `score > tau`.
4. Thresholds saved as `threshold.json` next to the score file they came from.

## Deliverables
- C4 section in `docs/architecture.md`

## Done when
- [ ] Both approve the PR
- [ ] Reviewed and approved by the other person

## Depends on (blocked by)
None. Can start immediately.

## Unblocks
- [#15](https://github.com/AnomaSight/Argus/issues/15) P1.1: Wilks tolerance threshold (`fit_threshold`) (Pratyush)
- [#17](https://github.com/AnomaSight/Argus/issues/17) P1.3: Operating point: FPR, recall and Wilson intervals (Pratyush)

## Files
- `docs/architecture.md`

## Notes
- Lead: Pratyush (he implements it in P1.1 and P1.3). Reviewer: Arvind.

---
_Plan: `planning/execution_plan.md` · Ownership: `planning/work_split.md`_
