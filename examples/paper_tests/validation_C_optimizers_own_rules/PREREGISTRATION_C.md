# Validation of the optimizers' OWN (built-in) stopping rules on our scores — pre-registration (part C), 2026-10-04
# written BEFORE any run of the seeds below

Why: Leo (2026-10-04): "why didn't we just use their rule, switch out the variable from NSE to our score". The first
validation (part B, RESULT_B3.md) tested OUR step-end rule: it passed for SCE-UA with fixed tolerances and failed for
NSGA-II. Part B and its results stay as they are and are reported. Part C tests the optimizers' own rules, unchanged,
with the objective(s) the search optimizes being our scores. Nothing below changes after the first run of seed 3000.

## 1. The rules under test (four variants, each validated on its own)
- spotpy_loss: SPOTPY SCE-UA's own test, verbatim (sceua.py: gnrng < peps, or for nloop >= kstop
  |c[n-1] - c[n-kstop]| * 100 / mean(|c[n-kstop:n]|) <= pcento, c = best objective at each loop end), with the
  classic SCE-UA settings kstop 10, pcento 0.1, peps 0.001 (SPOTPY's own defaults kstop 100, pcento 1e-7, peps 1e-7
  never fired in any part-B run: every run ended at the trial limit).
- spotpy_scores: the same test, also applied to each watched score of the best point at each loop end (r, alpha,
  beta, lnNSE); fires only when the objective AND every watched score pass (or gnrng < peps).
- pymoo_p50: pymoo's own DefaultMultiObjectiveTermination as shipped (xtol 0.0005, ftol 0.005, n_skip 5, period 50),
  on the two objectives the search optimizes (1 - NSE, 1 - lnNSE). Convergence = its design-space (x) or
  objective-space (f) test reaching 1; its built-in max-generation / max-evaluation caps are budget limits, never
  convergence.
- pymoo_p5: the same with period 5 (Leo asked).
The rule's state comes from what the optimizer itself computes: SPOTPY's gnrng and best value at each loop end
(recorded by sceua_recording.py, a copy of SPOTPY's sceua.py with two recording-only lines), pymoo's own termination
objects updated after each generation exactly as pymoo updates its own (a read-only callback, run_native.py).
Checked on seed 998: both recording paths give call-for-call the same search as the unmodified kit.

## 2. Runs (paper-frozen kit + recorder, as part B)
- SCE-UA: HYMOD, loss 1 - NSE, SpotpyBackend("sceua") defaults (ngs 20), budget 40,000 SPOTPY trials (~65 loops).
- NSGA-II: HYMOD two objectives (1 - NSE, 1 - lnNSE), PymooBackend("nsga2"), population 40, budget 40,000
  (1,000 generations = pymoo's own max-generation default).
- Seeds 3000, 3001, ... in order (new; part B used 1000-1079 and tracers 997-999); batches run whole.
- Every call recorded with its score panel; SCE-UA calls also with their simulated flow.
- Run-time dependencies frozen: DEPS_MANIFEST.json (deps_native.py: every file of the frozen kit, recorder, SPOTPY
  source, pymoo, numpy, scipy and their .libs, the frozen rule snapshot used for scores and states, run_native.py,
  judge_native.py, sceua_recording.py, deps_native.py). Runs refuse / are invalid on any difference, as in part B.

## 3. The judge (the same "premature" as part B, fixed tolerances only)
At the step k where the built-in rule first fires (SCE-UA: loop k; NSGA-II: generation k), compared with every later
counted step end (the last step never counted): PREMATURE if a tracked loss improved by >= 0.5 % (relative, floor
1e-3), or a watched score at the best point / compromise moved by more than 0.01, or (NSGA-II) the archive front's
additive eps-indicator (scale frozen at the end of generation 1) exceeds 0.01; UNDECIDABLE if not premature and
fewer than 10 later counted step ends or a missing score; SAFE otherwise; NO STOP if the rule never fires.
(T-boot is not used: it failed in part B.)

## 4. Pass mark and fallback (as part B)
Per variant: seeds continue until 59 stopped-and-decidable searches or seed 3199. PASS = 0 premature among >= 59.
Fewer than 59 by seed 3199 = INCONCLUSIVE (not passed). A variant that does not pass gives no validated verdicts.
No re-tuning; any change = a new pre-registration with new seeds; this one and its results are kept and reported.

## 5. How a passed variant is used
- SCE-UA paper searches (GR4J, E2, SAC-SMA comparison): spotpy_loss / spotpy_scores can be replayed on the saved
  records EXCEPT the gnrng part (frozen paper records have no per-loop population): applied without gnrng it can only
  fire later or not at all, never earlier; this is stated with the verdict.
- NSGA-II paper searches (2-13 generations) are shorter than period 5 + n_skip 5 or period 50: "too short to judge"
  for any pymoo variant; they keep the descriptive "last change" report. A passed pymoo variant is for the kit's
  future (live) use.
- ONE rule per optimizer in the paper, by this order fixed now (Leo's stated preference: the optimizer's own rule
  on our scores): SCE-UA: spotpy_scores if it passes, else spotpy_loss if it passes, else part B's validated rule
  (SCE-UA, T-fixed). NSGA-II: pymoo_p50 if it passes, else pymoo_p5 if it passes, else no validated rule. Both
  validations (B and C) are reported in full whichever is used.

## 6. Wording (codex C review, before any run)
- pymoo variants: "a replay of pymoo's DefaultMultiObjectiveTermination on the same generation states, updated as
  pymoo updates its termination" — not the live termination object of the run (the kit ran to an evaluation budget).
- SCE-UA variants: "the SCE-UA objective test with classic settings (kstop 10, pcento 0.1, peps 0.001)" — never
  "SPOTPY defaults" (those are kstop 100, pcento 1e-7, peps 1e-7).
