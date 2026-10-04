# Validation of the step-end convergence rule — pre-registration (part B2), written 2026-10-03 BEFORE any validation run (revised once after codex B2 r1, still before any run)

Nothing below may change after the first validation run starts. Any change = a new pre-registration with a new date,
and the old one and its results are kept and reported.

## 1. The rule under test (frozen)
- The runs and the judge import ONLY a snapshot of the three rule files: validation/frozen_rule/kdt_rule_frozen/
  (byte-identical copies, codex B2 r4). Code: /home/server/kdt_convergence_dev/calibration_kit/rule_steps.py, sha256
  e3b1f5ef9ef4904f9cc3d2a9d2f849fccff1e91885e4a5f4b77a392350f3d7a6 (codex A2 round 2 PASS). Scores and bootstrap from
  calibration_kit/panel.py, sha256 032b1c78f3ff414cb17fbf1c86847e0fa312b51519b3160d80a4bb50c4e598fb; constants and helpers it imports from calibration_kit/rule.py,
  sha256 5d5c844e924a1fe2b575f5a081b583f78b79f112f3f23bbf0506f71cf98e8079. The judge refuses to run if any of the three files differs (codex B2 r1 #1).
- Values: Q_MIN 5, M_MIN 5, M_MAX 50, H 20, CONF 0.80, ALPHA 0.05, rel_gain 0.005 (floor 1e-3), eps_front 0.01.
- Step = SCE-UA loop (loop 0 = the start-up sample) / NSGA-II generation (generation 1 = the start).
- Watched scores: kind "flow": r, alpha, beta, lnNSE, at the best point (SCE-UA) / the compromise (NSGA-II).
- The last step of a record is NOT counted (finish(last_step_complete=False)), as for the frozen paper records.

## 2. Tolerances: two versions, each validated on its own
- T-boot: moving-block bootstrap (panel.bootstrap_tolerances: 500 resamples, block 30, generator seed 0, z = 1) of ONE
  fixed reference run per model: the model run at the MIDPOINT of every parameter range (the same for every seed;
  HYMOD has no KI default). Metrics the bootstrap cannot measure fall back to the fixed values (labelled).
- T-fixed: 0.01 for r, alpha, beta, lnNSE (the fallback used where a paper record has no series).
A paper search may be given a verdict with a tolerance version only if that version passed here.
The SAME procedure is used on the paper searches (codex B2 r1 #3): T-boot = bootstrap of ONE reference run per model
at the midpoint of its calibrated parameter ranges (one extra model run per model, seed-independent); a metric the
bootstrap cannot measure, or a search with no saved series, uses T-fixed (labelled). Each search's verdict says
which version it used.

## 3. Runs (B1). Paper-frozen kit + recorder (as reruns/hymod_tagged), every call with its score panel and loop/generation
- V-SCE: HYMOD Leaf River (SPOTPY's spot_setup_hymod_python), loss 1 - NSE, SpotpyBackend("sceua") with its defaults
  (ngs 20; start-up 220 calls), budget 20,000 SPOTPY repetitions (~13,000 calls, ~35 loops), seeds 1000, 1001, ...
- V-NSGA: HYMOD two objectives (1 - NSE, 1 - lnNSE; as demo_nsga2_hymod), PymooBackend("nsga2"), population 40,
  budget 4,000 (100 generations), seeds 1000, 1001, ...
- Each call also saves its simulated flow (float32), so T-boot and later checks can be recomputed.
- Seeds are run in order, in batches; a batch is only ever whole; results are judged only by the frozen judge script.
- Run-time dependencies frozen (codex B2 r2): DEPS_MANIFEST.json = sha256 of every file of the paper-frozen kit, the
  recorder, the SPOTPY source tree used (incl. the HYMOD data file), the installed pymoo, numpy and scipy (with their linked numpy.libs / scipy.libs), the frozen rule snapshot, the three new kit
  files, run_val.py, judge.py and deps_manifest.py itself, every file of each (codex B2 r3); run_val.py also checks
  that every loaded calibration_kit / spotpy / pymoo / numpy module comes from its hashed folder,
  written once before the first run. run_val.py refuses to start if they differ and records their digest in
  done.json; the judge refuses to run if they differ now and marks a run invalid if its digest differs.
- A run whose recorder_errors.jsonl is not empty, or that did not finish (no done.json), is INVALID: it is not judged,
  it is listed in the report with the reason, and the seed stream simply continues (a recorder failure does not depend
  on the result; codex B2 r1 #2). run_val.py writes done.json only when recorder_errors.jsonl is empty.

## 4. The judge (fixed)
For a search where the rule fired at step k (the first firing), with tolerance version T:
- later step ends = every counted step end after k (the last step is not counted);
- PREMATURE if at ANY later step end, compared with the state at step k:
  * a tracked loss improved by >= rel_gain (relative, floor 1e-3) (SCE-UA: the objective loss at the best point;
    NSGA-II: the best loss of each objective);
  * a watched score at the best point / compromise differs by more than its tolerance (T);
  * NSGA-II: the additive eps-indicator of the archive at k over the later archive, on the scale frozen at the end of
    the start-up generation, is > eps_front;
- UNDECIDABLE if not premature and (fewer than 10 counted step ends after k, or a watched score is missing at a later
  step end);
- SAFE otherwise. NO STOP if the rule never fired (not in the denominator; reported).

## 5. Pass mark (fixed)
Per optimizer (V-SCE, V-NSGA) and per tolerance version (T-boot, T-fixed): seeds continue in order until 59
stopped-and-decidable (safe + premature) searches, or seed 1199 (200 seeds), whichever first.
PASS = 0 premature among >= 59 stopped-and-decidable (one-sided 95 % Clopper-Pearson upper bound <= 5.0 %).
Fewer than 59 decidable by seed 1199 = INCONCLUSIVE (counts as not passed).
FALLBACK if not passed: no re-tuning; that optimizer / tolerance version gets no validated verdicts — the paper reports
the descriptive settle table only for it, and the failed validation honestly.

## 6. Report (B3)
Per optimizer x tolerance: counts premature / safe / undecidable / no stop; the bound; median stop step and calls,
and the share of the budget the stop would have saved; every premature case listed with what moved and by how much.

## 7. Confirmation (B4), run only after B3, never used to change anything
As in the plan (r3): 20 stopped-and-decidable each of SAC-SMA E2 Blanco SCE-UA (budget 20,000, seeds 2000..),
two-objective SAC-SMA NSGA-II (NSE + lnNSE, 100 generations, seeds 2100..), new two-objective HYMOD NSGA-II
(seeds 2200..); same judge; pass = 0 premature out of 20 in each set.
