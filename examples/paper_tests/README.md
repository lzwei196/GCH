# Paper tests (examples)

The scripts, pre-registrations and result summaries behind the convergence work for the GRL paper. They are kept as
worked examples of using and testing GCH. **They will not run as-is elsewhere**: they point at absolute paths of
our server (the paper-frozen kit, the KIs, observations, the study record), and the run data (per-call records and
simulated series, several GB, some under redistribution limits) is not included. `PROCESS_LOG.md` records every step
and result in order.

## 1. Reruns of the paper's experiments — `reruns/`

Each paper search rerun with the paper's own (frozen) kit and runner, every call recorded, and checked to reproduce
the paper's result.

| folder | experiment | result (`checks/` = per-run reproduction checks) |
|---|---|---|
| `hymod_tagged/` | HYMOD budget sweep, 36 runs (SI Table S8a) | 36 / 36 reproduce |
| `e2/` | SAC-SMA vs CAMELS (E2) | reproduces call for call |
| `e1/` | GR4J vs airGR (E1) | reproduces exactly |
| `sacsma_cmp/` | SAC-SMA optimizer comparison, 27 cells | 27 / 27 reproduce |
| `gr4j/` | GR4J repeated formulation (Test C), 21 cells, through the paper's sealed launcher | 20 / 21 exactly; C3_seed2: two reruns identical to each other, the original diverged after call 855 |
| `six_model/` | six-model NSGA-II campaign, seeds 0-2 | run with today's KI tools (tested same inputs); in progress at the time of writing |

`../recorder/frozen_recorder.py` wraps the frozen kit from outside (no kit code changed) and records every model run
and scoring call with its phase, seed and SCE-UA loop / pymoo generation.

## 2. Validation of convergence rules — pre-registered, new seeds, a fixed judge

| folder | rules tested | result |
|---|---|---|
| `validation_B_our_step_rule/` | our step-end rule (quiet loss + watched scores for m steps, adaptive m) | SCE-UA + fixed tolerance PASS 0/59; bootstrap tolerance FAIL 1/59; NSGA-II FAIL |
| `validation_C_optimizers_own_rules/` | SPOTPY's own SCE-UA test (loss only; loss + scores); pymoo's own termination (period 50, period 5) | SPOTPY both PASS 0/59 (median stop loop 19 / 24); pymoo period 5 FAIL 31/59; period 50 never stopped in 200 runs |

Each folder: the pre-registration (written and reviewed before any run), the run driver, the judge, the dependency
manifest script and `results/*.json` (every search judged). The judge counts a stop as premature when the best loss,
a watched score or the front still moved beyond tolerance at ANY later step end.

## 3. Applying the rules to recorded searches — `apply/`

`apply_rule.py` applies the step rule and the settle point to a recorded search (part D of the plan); the kit's own
`calibration_kit/replay.py: replay_native` is the maintained version.
