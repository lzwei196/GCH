# Recorder for reruns with the paper-frozen kit (2026-10-03)

frozen_recorder.py wraps the frozen kit from the outside (backends' optimize, the Evaluator's evaluate, a driver's
problem) and writes one line per call: parameters, losses, the full metrics payload (calibrate() runs), cache hit,
rejected-before-run, split, phase, seed, search number, SCE-UA loop (0 = start-up sample), pymoo batch/generation.
Nothing the optimizer sees is changed.

Tests: test_frozen_recorder.py, 9 tests (each in a fresh process): recording does not change the search (DDS,
SCE-UA, NSGA-II: same calls, same order, same result with and without it); SCE-UA loop 0 = exactly ngs(2d+1) calls;
NSGA-II 40 calls per generation; DDS no step tags; driver records and several searches; the frozen Evaluator path
(model run, cache hit, call rejected before any run); refuses a kit that is not the frozen one. Planted bugs: 14
applied, 14 caught (1 not applicable).

Tracer bullets:
- tracer_hymod: HYMOD SCE-UA, budget 2000, seed 0: repetitions 2503 and objective 0.328424 = the original log;
  1,623 calls, call-for-call identical to the earlier recorded rerun; loop 0 = 220 calls; loops start at calls
  0, 220, 536, 893, 1254.
- tracer_e2: E2 SAC-SMA / CAMELS (SCE-UA, seed 0, real Sac-SMA binary): 2,702 calls, parameters identical in order to
  the September recorded rerun; the result file identical except the wall-clock time; loop 0 = 460 calls
  (20 x 23 for 11 parameters); loops start at 0, 460, 1088, 1859.
Open: calibrate()-driven experiments (six-model campaign) use the paper's runners, which reply only their own
scores (no alpha, lnNSE, series). How to record the missing scores without changing the search is still to decide.
