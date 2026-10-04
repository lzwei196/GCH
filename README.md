# GCH — a general calibration harness for process-based models

GCH lets a language-model agent **formulate** a model calibration — which parameters, how simulated outputs map onto
observations, which metrics and acceptance levels, which optimizer and budget — as a written **contract**
(`calibration.yaml`) and **runner** (`tools/calib_run.py`), drawn from and checked against the model's Knowledge
Infrastructure (KI). A fixed, model-agnostic **engine** then does the numerical search without the agent: it checks
every candidate, searches with a conventional optimizer, gates the result on held-out data and says whether the
search converged. This is the framework called GCH in the GRL manuscript "Delegating calibration formulation to
language model agents".

**Start here: [USAGE.md](USAGE.md).**

## Layout

| folder | what |
|---|---|
| `calibration_kit/` | the engine (self-contained): contract loading, parameter checks, pilot runs and budget, optimizer backends (SPOTPY DDS / SCE-UA / DREAM, pymoo NSGA-II / NSGA-III / MOEA-D, optional PEST++ / surrogate), score panel, convergence rules, seeds, holdout gate, replay; tests |
| `authoring/` | the agent side (reference): how a calibration agent authors the contract and runner from a KI; depends on the host's agent-spawning helper |
| `examples/paper_tests/` | drivers, pre-registrations and result summaries of the tests behind the GRL paper |

## Convergence, in one table

| optimizer | the kit's rule (its own, on our scores) | validated |
|---|---|---|
| SCE-UA | SPOTPY's loop-end test on the objective and watched scores (kstop 10, pcento 0.1, peps 0.001) | 0 premature / 59 |
| NSGA-II / III, MOEA-D | pymoo's DefaultMultiObjectiveTermination, recorded | not validated (never fired in 200 tests) |
| DDS | settle point "settled by run N of M" | descriptive |
| DREAM | R-hat < 1.2 | — |

`strategy.convergence.mode: keep_going` records the rule, `stop` lets it end the search. Details in USAGE.md §4.

## Design documents

- `calibration_kit/CALIBRATION_YAML_SCHEMA.md` — the contract (every field)
- `calibration_kit/CALIBRATION_FRAMEWORK_DESIGN.md` — architecture and rationale
- `calibration_kit/calibration.example.yaml` — a worked contract
- `calibration_kit/backends/sceua_hooked.py` — the kit's copy of SPOTPY 1.6.7 SCE-UA with a loop-end hook (MIT)

## Status

Private research code. Version history: the engine is developed in the Knowledge Dissection Toolkit
(`calibration_kit/`, branch `convergence-panel-2026-09`, commit `a5bdb16`). The code the paper's experiments ran on
is the frozen copy in the study record (toolkit commit `8acd3d7`, archived 2026-09-07).
