# [C3] Contract C3: detector interface (`load_detector`)

GitHub issue: [#5](https://github.com/AnomaSight/Argus/issues/5)

**Labels:** `size: S`, `type: design`, `blocks other person`

**Task:** C3 · **Owner:** Both · **Milestone:** Phase 0: Setup and contracts
**Branch:** `shared/C3-contract-c3-detector-interface-load`

## Summary
Agree on the Python interface that lets anyone score images with an already-built memory bank.

## Why this matters
Pratyush's backend and stress tests score images but never build models. This interface is the only thing they need from the detector code.

## What to do
1. `argus.load_detector(path: str | Path, device: str = 'auto') -> Detector` loads a saved memory bank and its preprocessing config.
2. `Detector.score(image: PIL.Image) -> tuple[float, np.ndarray]`: image score and H×W heatmap (float32).
3. `Detector.score_batch(images: list[PIL.Image], batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]`.
4. `Detector.info -> dict`: dataset, category, setting, coreset size, input size.
5. Preprocessing (resize, crop, normalisation) happens inside the detector; callers pass raw images.
6. Bank location: `artifacts/banks/{dataset}/{category}/{setting}/seed{n}/`.

## Deliverables
- C3 section in `docs/architecture.md` with signatures and a usage example

## Done when
- [ ] Both approve the PR
- [ ] Reviewed and approved by the other person

## Depends on (blocked by)
None. Can start immediately.

## Unblocks
- [#20](https://github.com/AnomaSight/Argus/issues/20) P1.6: Backend with a stub detector (Pratyush)
- [#26](https://github.com/AnomaSight/Argus/issues/26) A2.5: `load_detector()` and the Detector class (C3) (Arvind)

## Files
- `docs/architecture.md`

## Notes
- Lead: Arvind. Reviewer: Pratyush.

---
_Plan: `planning/execution_plan.md` · Ownership: `planning/work_split.md`_
