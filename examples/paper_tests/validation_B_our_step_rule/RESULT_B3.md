# Validation result (part B3) — 2026-10-03, judged by the frozen judge.py on batch 1 (seeds 1000-1079)

Pre-registration: PREREGISTRATION_B.md (codex B2 round 6 SOUND); files frozen (FROZEN_AT_START.sha256). Full per-search
results: results/{sce,nsga}_{fixed,boot}.json. No invalid runs; every run carried the frozen dependency digest.

| optimizer | tolerance | judged | premature | safe | undecidable | no stop | decidable | 95 % upper bound | result |
|---|---|---|---|---|---|---|---|---|---|
| SCE-UA (HYMOD) | T-fixed 0.01 | 59 | 0 | 59 | 0 | 0 | 59 | 4.95 % | **PASS** |
| SCE-UA (HYMOD) | T-boot | 59 | 1 | 58 | 0 | 0 | 59 | 7.79 % | FAIL |
| NSGA-II (HYMOD, 2 obj.) | T-fixed 0.01 | 80 | 43 | 7 | 0 | 30 | 50 | 93 % | FAIL |
| NSGA-II (HYMOD, 2 obj.) | T-boot | 80 | 40 | 11 | 0 | 29 | 51 | 87 % | FAIL |

- SCE-UA, T-fixed: median stop at loop 14; the stop would have saved a median 59 % of the budget; nothing moved
  beyond tolerance in any of the >= 10 later loops of any of the 59 searches.
- SCE-UA, T-boot (bootstrap tolerances r 0.049, alpha 0.100, beta 0.145, lnNSE 0.136 — wide): stops earlier
  (median loop 11); seed 1018 stopped at loop 7 and the best loss then improved by 0.68 % (> 0.5 %) at loop 8.
- NSGA-II: the archive front keeps creeping after the stop (median eps after the stop 0.011 fixed / 0.015 boot, vs
  eps_front 0.01), and with fixed tolerances the compromise's scores also shift; 30 / 29 of 80 searches never stopped
  in 98 generations.

Consequences, as fixed in advance (section 5, fallback — no re-tuning):
- SCE-UA paper searches may get validated verdicts with T-fixed tolerances only (after the B4 confirmation passes).
- T-boot is not used for any verdict.
- NSGA-II (the six-model campaign and the other NSGA-II searches) gets NO validated convergence verdict; it is
  reported descriptively (last change of the best losses, watched scores and front) and this failed validation is
  reported honestly.
- B4 (confirmation, never used to change anything): only the SAC-SMA SCE-UA set is still meaningful; the NSGA-II
  confirmation sets cannot rescue a failed B3.
