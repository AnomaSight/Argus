# [C5] Contract C5: HTTP API

GitHub issue: [#7](https://github.com/AnomaSight/Argus/issues/7)

**Labels:** `size: S`, `type: design`, `blocks other person`

**Task:** C5 · **Owner:** Both · **Milestone:** Phase 0: Setup and contracts
**Branch:** `shared/C5-contract-c5-http-api`

## Summary
Agree on the web API between the backend and the frontend.

## Why this matters
Pratyush owns both ends, but writing the API down first lets the frontend and backend be built and tested separately, and lets Arvind review what the demo will expose.

## What to do
1. `GET /health` → `{status}`.
2. `GET /models` → list of `{model_id, dataset, category, setting, seed, tau, alpha}`.
3. `POST /inspect` (multipart: `image`, `model_id`) → `{score, tau, verdict: good|defect|review, heatmap_png_base64, latency_ms}`.
4. `GET /results` → list of result tables (`{name, columns, rows}`) read from `artifacts/results/`.
5. Error format: `{error: {code, message}}` with proper HTTP status codes.
6. Mirror the types in `frontend/src/api/types.ts`.

## Deliverables
- C5 section in `docs/architecture.md`
- `frontend/src/api/types.ts` matching it

## Done when
- [ ] Both approve the PR
- [ ] Reviewed and approved by the other person

## Depends on (blocked by)
None. Can start immediately.

## Unblocks
- [#20](https://github.com/AnomaSight/Argus/issues/20) P1.6: Backend with a stub detector (Pratyush)
- [#48](https://github.com/AnomaSight/Argus/issues/48) P4.3: Results page (Pratyush)

## Files
- `docs/architecture.md`
- `frontend/src/api/types.ts`
- `backend/app/schemas.py`

## Notes
- Lead: Pratyush. Reviewer: Arvind.

---
_Plan: `planning/execution_plan.md` · Ownership: `planning/work_split.md`_
