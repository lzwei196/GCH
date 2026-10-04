# pymoo stop-hook wiring — OUTCOME (2026-08-30)

## Defect (found in the architecture review; flagged by codex + kimi)
`calibration_kit/backends/pymoo_backend.py` (NSGA-II/NSGA-III/MOEA-D) ignored the `on_eval(history)`
stop hook that every other backend honours. So the convention-floor **stopping** layer (`stop.py`)
never fired for multi-objective runs — and **every** paper multi-variable case uses NSGA-II. Stopping
was silently budget-only for all MV cases.

## Fix
- `optimize()` now reads `on_eval` from kw, builds the shared-shape history `[{"x","loss"}]`, calls the
  hook after every evaluation, and raises `EarlyStop` to unwind pymoo's `minimize()`; returns
  `history` + `stopped_early`.
- Multi-objective scalar loss for the plateau/top-set signal = **sum of the finite loss vector**, or
  `inf` when any objective is non-finite (so `stop.py`'s `loss < 1e29` finite mask excludes penalized
  points). The main-stat convergence + convention floor come from `stop.py`'s `main_stat_of` hook.
- **Finiteness is judged on the RAW evaluator result** (`bool(losses) and len==n_obj and all finite`) —
  a failed/empty eval never enters `finite_pts` and gets `history` loss `inf`. pymoo's `F` row is
  clamped to `_PENALTY` **separately** so its dominance sort still gets finite numbers.
- On early stop / no returned front, the candidate set is rebuilt from all finite evals and labeled
  "finite candidate set" (not "Pareto front"); front-select gates every member on holdout anyway.

## Review
- **codex: APPROVE-WITH-NITS.** First pass REQUEST-CHANGES on one real bug — a `[_PENALTY]*n_obj` fill
  made failed evals read as finite (polluting the stop rule); fixed by judging finiteness on the raw
  result. Re-review confirmed resolved.
- **kimi: APPROVE-WITH-NITS** — same core finding; labeling + scale-assumption nits addressed/noted.

## Tests
`calibration_kit/test_stop.py` +3 (NSGA-II honours the hook → stops early + returns a front; no-hook
runs to cap; failed-eval guard → inf history rows, front all-finite & in-region). 17 pass across
test_stop + test_front_select + test_madr_backend.

## Note
This wires the *mechanism*. It only takes effect when a contract declares `strategy.stop` with a
`floor: {source: ki_validation_convention, ...}`. Our MV case contracts do NOT yet declare `stop` —
adding it (per model, against each KI's cited convention) is the next replan step, together with the
convention-**gate** adoption in the case drivers.
