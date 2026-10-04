# Protected front-selection — OUTCOME (2026-08-25)

## What changed
`_select_front_member` (calib.py) gained an optional `protect` set of objective names, wired via env
`KDT_CALIB_PROTECT` (comma-separated). When no Pareto member passes the full holdout gate, instead of
committing the raw minimax knee — which on a strongly-conflicting front can wreck one objective the
user cares about — it commits a member that keeps every PROTECTED objective admissible on holdout and
is best on the FREE objectives. Three admissibility tiers, best-first:
- **A** protected objective passes its own holdout gate (`ok=True`)
- **B** protected objective beats the uncalibrated baseline out-of-sample
- **C** protected objective degrades no more than `protect_tol` (default 0.25) beyond baseline holdout loss

Tier C is the key case: when the UNCALIBRATED model is already best on a protected objective (baseline
unbeatable → A/B infeasible), a small bounded give-back there can buy a large gain on the free
objective. Among admissible members, rank by minimax over the free objectives' holdout losses. If no
tier has a member → INFEASIBLE, keep the minimax knee. **Never promotes** (protected pick stays
not-promotable — downstream `promotable = holdout.passed is True`).

## Motivating case — CRHM @ Blue River 08LB038 (discharge + SWE)
The uncalibrated baseline has discharge r 0.82 (best on the whole front) but SWE NSE −0.27 (broken).
The raw minimax knee (#23) committed discharge holdout r **0.12** to fit SWE. With
`KDT_CALIB_PROTECT="basinflow_s:temporal_pattern_match,basinflow_s:magnitude_accuracy"` the selector
commits **member #24** (tier C):

| Objective (holdout) | Protected #24 | Raw knee #23 | Baseline |
|---|---|---|---|
| Discharge magnitude | 0.265 (beats) | 0.403 | 0.480 |
| Discharge temporal  | 0.664 (r 0.713) | 1.16 (r 0.123) | 0.555 |
| SWE magnitude       | 0.493 (beats) | 1.26 | 0.743 |
| SWE temporal        | 0.586 (beats) | 3.08 | 1.267 |

Protection rescued discharge timing from r 0.12 → **0.71** while beating baseline on 3 of 4 objectives.
Still not-promotable (discharge timing can't beat the unbeatable uncalibrated baseline) — an honest,
defensible trade-off instead of a discharge-wrecking artifact.

## Review
- **codex: APPROVE** (2026-08-25). First pass REQUEST-CHANGES found 3 real bugs — all fixed:
  1. `_num(...) or inf` treated a valid 0.0 free loss as missing → explicit `None` check.
  2. inf free loss → nan in normalization could win argmin → rank only finite-free candidates, else
     fall back to the most-balanced admissible member.
  3. Unknown protect names left `prot` empty but re-ranked the front anyway → intersect with real
     objective names; skip the block (with a logged note) when none match.
- **kimi: two findings — both FALSE POSITIVES.** Its snippet was truncated at `idx = order[0]`, so it
  did not see lines 593-600 where the INFEASIBLE branch falls through to the shared knee-commit that
  sets `result.best_x`/`best_loss` and returns `(idx, top_holdout)`. Verified by code inspection and
  by the passing `test_protect_infeasible_keeps_knee` / `test_protect_unknown_names_ignored_no_reranking`.

## Tests
`calibration_kit/test_front_select.py` — 8 pass (2 pre-existing + 6 new: tier C over knee, tier A
precedence, infeasible keeps knee, zero-free-loss wins, missing-free-loss never wins, unknown names
ignored).
