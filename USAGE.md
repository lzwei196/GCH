# GCH usage guide

GCH (general calibration harness) splits calibration into two jobs:

1. **Formulation** (judgement, done once): an agent — or a person — reads the model's Knowledge Infrastructure (KI)
   and writes a **contract** (`calibration.yaml`) and a **runner** (`tools/calib_run.py`).
2. **Search** (no judgement, repeatable): the **engine** (`calibration_kit/`) runs the optimizer against that
   contract and runner, checks every candidate, and reports the result, its holdout check and whether it converged.

This guide covers the engine. `authoring/` holds the agent side as reference.

## 1. Install

Python 3.12 with: `numpy`, `scipy`, `pandas`, `pyyaml`, `spotpy==1.6.7` (DDS, SCE-UA, DREAM) and `pymoo==0.6.1.6`
(NSGA-II / NSGA-III / MOEA-D). Put the repository root on `PYTHONPATH`.

## 2. What a model needs: contract + runner

`<ki_path>/calibration.yaml` — full field list in `calibration_kit/CALIBRATION_YAML_SCHEMA.md`. The core:

```yaml
model_id: MyModel
targets:
  - var: Q                      # a model output that has observations
    weight: 1.0
parameters:
  - name: k1
    type: continuous
    default: 0.5
    range: [0.1, 0.9]
    transform: identity         # identity | log | logit
    address: {kind: api, setter: runner}    # runner mode: the runner writes the value itself
runner:
  kind: subprocess
  command: ["python", "tools/calib_run.py", "--workdir", "{workdir}", "--out", "{metrics_json}"]
  metrics_file: "{metrics_json}"
strategy:
  default_algorithm: sceua      # dds | sceua | dream | nsga2 | nsga3 | moead
  max_evaluations: 3000         # the budget (a cap)
  budget: {seeds: 3}            # independent searches; the cap is per seed
  convergence:
    mode: keep_going            # keep_going (record the rule) | stop (the rule ends the search)
    variables: {Q: {kind: flow}}   # flow | series | snapshot | categorical
  holdout: {kind: years, fraction: 0.3}
```

The **runner** reads the candidate from the file named in `KDT_CALIB_PARAMS`, runs the model with the model's own
tools, scores the simulated series against observations, and writes a JSON metrics file, e.g.
`{"Q": {"nse": 0.71, "kge": 0.66, "r": 0.85, ...}, "__kdt__": {"applied_params": {...}}}`.
Echoing `applied_params` lets the kit check that what it asked for is what the model used.
`calibration_kit/metrics_panel.py` / `panel_block()` give the kit's own score definitions for the watched scores
(r, alpha, beta, lnNSE for flow; r, alpha, beta for series; PBIAS, NRMSE for snapshots).

## 3. Run

```python
from calibration_kit import calib

report = calib.calibrate(
    ki_path="/path/to/model_KI",          # holds calibration.yaml + tools/calib_run.py
    workdir="/path/to/run_folder",        # a persistent folder: every evaluation is kept there
    obs_shape_by_var={"Q": "point_time_series"},
    budget=None,                          # None = the contract's max_evaluations / budget block
    seed=0,
)
print(report["status"], report["best_params"], report["holdout_validated"])
```

Every evaluation is written to `<workdir>/eval_history.jsonl` (phase, seed, parameters, losses, scores).
Never use a throw-away folder for a real run.

## 4. Convergence: what the kit reports and how it decides

The budget is the planner's guess (often the agent's, from the KI) of what convergence needs. The convergence rule
checks that guess from what the search actually did. Each optimizer uses **its own** rule, applied to **our
scores**:

| optimizer | rule | validated? |
|---|---|---|
| SCE-UA | SPOTPY's own loop-end test, on the objective AND every watched score of the best point; classic settings kstop 10, pcento 0.1 (%), peps 0.001 | yes: 0 premature in 59 independent searches (pre-registered) |
| NSGA-II / NSGA-III / MOEA-D | pymoo's own `DefaultMultiObjectiveTermination` (period 50, n_skip 5, xtol 0.0005, ftol 0.005) | no: it never fired in 200 tests of 1,000 generations (period 5 stopped too early). It is recorded; no "converged" claim is made from it |
| DDS (and random, LLM-in-loop) | none of its own; the settle point "settled by run N of M" (last run where the best loss improved by >= rel_gain or a watched score moved beyond its tolerance) | descriptive only |
| DREAM | its own R-hat < 1.2 | — |

- `mode: keep_going` records the rule; `mode: stop` ends the search at the end of the loop / generation where it
  fires. SPOTPY's own internal settings stay at its defaults; the kit applies the test through a loop-end hook in
  its copy of SPOTPY's SCE-UA (`calibration_kit/backends/sceua_hooked.py`).
- SCE-UA needs at least 10 loops (about `ngs x (2d+1) x 21` evaluations) before the rule can fire; shorter
  searches are "too short to judge", and the kit warns when the budget is too small.

Where to read it in the report (`report["convergence"]`, for the returned seed; every seed in
`report["convergence"]["seeds"]["slots"]`):

| field | meaning |
|---|---|
| `verdict.verdict` | `converged` / `not_converged` / `unknown` (too short, no rule, not tracked) — the kit's answer |
| `verdict.how` | in words, e.g. "SCE-UA objective test on the objective and watched scores: fired at loop 14" |
| `native` | the rule's details: settings, `fired_at`, `stopped_the_search` |
| `settle` | the settle point: `settled_by_run`, `share_after`, `wording` |
| `ended.text` | what ended the search ("converged at loop 14 ...", "ran to the cap ...") |
| `run_verdict` | across seeds (converged only if every seed converged and the seeds agree on the scores) |
| `step_rule_verdict` | the older per-call step rule, recorded only (not the kit's answer) |

## 5. Re-judging a saved run

```python
from calibration_kit import replay
rows = [r for r in replay.read_history(workdir) if r.get("phase") == "search" and r.get("seed") == 0]
cv = report["convergence"]
objs = list(zip(cv["rule"]["objectives"], cv["rule"]["objective_vars"]))
tol = {v: dict(r.get("tol") or {}) for v, r in cv["tolerance_records"].items()}
out = replay.replay_native(rows, cv, report["algorithm"], objs, cv["kinds"], tol, weights=cv["rule"].get("weights"))
```

It gives the same verdict as the live run. For another seed, pass that slot's `optimizer_termination`. Records
from older tools without SPOTPY's per-loop record are replayed without the population-spread part (it can only make
the rule fire later, never earlier) and say so.

## 6. Tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q calibration_kit -p no:cacheprovider
```

829 pass on our machine; two tests in `calibration_kit/tests/test_entry_points_checkpoint.py` (G1, G2) are known
open items unrelated to convergence. Some tests in `calibration_kit/tests/` refer to absolute paths of the
development server.

## 7. Examples

`examples/paper_tests/` holds the drivers, pre-registrations and result summaries of the tests behind the GRL
paper (see its README).
