# Using GCH

GCH executes a model-specific calibration contract and runner. The [preparation guide](docs/AUTHORING.md) describes the scientific decisions the agent makes before execution. The engine itself does not require a live language-model connection for ordinary numerical search.

## Installation and first run

Use Python 3.12 and install the current-engine dependencies from `requirements.txt` in a virtual environment. Run the [synthetic example](examples/minimal/README.md) first:

```bash
python -m pip install -r requirements.txt
python examples/minimal/run.py --output ./outputs/minimal
```

When calling the engine from another directory, add this repository to `PYTHONPATH`. Native models have their own software and data requirements; installing these Python packages does not install those models.

## A model workflow

A KI supplies the model interface, graph, observation definitions, model-specific guidance and assessment conventions. The prepared workflow contains a `calibration.yaml` contract and a runner, usually `tools/calib_run.py`. See the complete [contract schema](calibration_kit/CALIBRATION_YAML_SCHEMA.md) and [example contract](calibration_kit/calibration.example.yaml).

A simplified contract has these fields:

```yaml
model_id: MyModel
targets:
  - var: Q
    weight: 1.0
parameters:
  - name: k1
    type: continuous
    default: 0.5
    range: [0.1, 0.9]
    transform: identity
injection:
  mode: runner
runner:
  kind: subprocess
  command: ["python", "tools/calib_run.py", "--workdir", "{workdir}", "--out", "{metrics_json}"]
  metrics_file: "{metrics_json}"
strategy:
  default_algorithm: sceua
  max_evaluations: 3000
  budget: {mode: fixed, seeds: 3}
  convergence:
    mode: keep_going
    variables: {Q: {kind: flow}}
  holdout: {kind: years, fraction: 0.3}
```

This is a field illustration, not a complete runnable model. The synthetic example includes its actual contract, graph, observations and runner.

The runner reads candidate parameters from the JSON file named by `KDT_CALIB_PARAMS`, applies them to the model and scores the requested split. It returns metrics and `__kdt__.applied_params`, and identifies the split it actually scored. The read-back must reflect applied model values. Return the watched score panel and the simulated/observed series when requested so GCH can check the reply and record convergence evidence.

Use the interpreter from the active environment for subprocess runners. Ordinary subprocess and Python-callable runners are independent of the original host application. Detached execution and live agent repair depend on external host tools and are not demonstrated by this standalone checkout.

## Run a prepared workflow

```python
from calibration_kit import calib

report = calib.calibrate(
    ki_path="/path/to/model_KI",
    workdir="/path/to/new_run_folder",
    obs_shape_by_var={"Q": "point_time_series"},
    budget=None,
    seed=0,
)
print(report["status"], report.get("best_params"), report.get("holdout_validated"))
```

Use a persistent, separate output directory. GCH writes `eval_history.jsonl` and run artifacts there. Retain the engine commit, contract, runner and runtime identity with the outputs. A completed workflow, a converged search and a scientifically accepted result are distinct outcomes.

## Budgets and convergence

The contract can specify a fixed cap with a rationale or a wall-clock allowance using `strategy.budget.mode: measured`. Measured allocation uses a pilot, model runtime, overhead and available parallelism. The cap is per seed; the default is three seeds. This estimates affordable computation, not convergence.

The default `strategy.convergence.mode: keep_going` records the convergence diagnostics without ending search through the GCH convergence hook. Explicit `stop` mode can stop at an optimizer boundary. The backend can also end for its own recorded reason.

| Optimizer | Current convergence evidence |
| --- | --- |
| SCE-UA | At evolution-loop boundaries, normalized geometric population range below 0.001, or objective and watched-score percent changes no greater than 0.1% over ten loops. The population-range branch can fire earlier. GCH uses classic SCE-UA settings through its loop hook; these differ from SPOTPY's internal defaults. |
| NSGA-II, NSGA-III, MOEA/D | pymoo design-space or objective-space termination, with `period=50`, `n_skip=5`, `xtol=0.0005`, `ftol=0.005`; evaluation/generation caps are excluded from the convergence verdict. The retained validation did not establish reliable early stopping for this configuration. |
| DDS | No native convergence rule. A retrospective settle point describes the last material change within the observed search. Its budget-dependent schedule is preserved. |
| DREAM | Native R-hat criterion below 1.2. This is separate from the original paper's demonstration, which did not evaluate sampling convergence. |

Scientific acceptance against observational standards is assessed separately. Reaching a budget, returning fewer evaluations than requested or meeting a fit threshold does not by itself establish convergence. Short or incomplete records may give an `unknown` verdict.

The report's `convergence` object includes:

| Field | Meaning |
| --- | --- |
| `verdict.verdict`, `verdict.how` | Per-seed decision and its explanation |
| `native` | Rule settings and first recorded firing |
| `ended` | Why execution ended |
| `seeds.slots` | Evidence for individual seeds |
| `run_verdict` | Across-seed decision, including agreement of watched calibration scores |
| `settle` | Descriptive last-change information |
| `step_rule_verdict` | Older per-call diagnostic; not the current native-rule verdict |

## Reassess a saved search

```python
from calibration_kit import replay

rows = [r for r in replay.read_history(workdir)
        if r.get("phase") == "search" and r.get("seed") == 0]
cv = report["convergence"]
objectives = list(zip(cv["rule"]["objectives"], cv["rule"]["objective_vars"]))
tolerances = {v: dict(r.get("tol") or {})
              for v, r in cv["tolerance_records"].items()}
verdict = replay.replay_native(
    rows, cv, report["algorithm"], objectives, cv["kinds"], tolerances,
    weights=cv["rule"].get("weights"),
)
```

Use the matching seed's termination record when replaying another seed. Older histories without population-spread records cannot reproduce that part of the SCE-UA rule; the replay reports this limitation.

This operation reassesses a stored search history. It does not rerun a physical model or reproduce the paper's eleven single-evaluation environment checks. See [paper/README.md](paper/README.md) for those records and [docs/TESTING.md](docs/TESTING.md) for software tests.
