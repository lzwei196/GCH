# GCH — General Calibration Harness

GCH turns a calibration formulation into a reusable numerical workflow. An agent uses a model's knowledge infrastructure (KI) to choose parameters, observation mappings, objectives, an optimizer and a computational budget, then writes a calibration contract and model runner. GCH checks and executes that workflow, records the search and assesses its results.

The agent works during preparation. The numerical engine runs independently after the contract and runner are fixed. The [agent preparation guide](docs/AUTHORING.md) explains this handoff without requiring a particular agent provider or orchestration system.

## Start with a runnable example

Use Python 3.12 in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/minimal/run.py --output ./outputs/minimal
```

The [minimal example](examples/minimal/README.md) uses synthetic model inputs and observations. It demonstrates the contract, runner and recorded numerical search; it is separate from the paper's experiments. See [USAGE.md](USAGE.md) for the engine interface and convergence reporting.

## Repository contents

| Location | Purpose |
| --- | --- |
| `calibration_kit/` | Numerical engine, optimizer adapters, checks and implementation tests |
| `docs/AUTHORING.md` | How an agent prepares and checks the contract and model runner |
| `examples/minimal/` | A complete synthetic example that runs from this checkout |
| `paper/` | Index of reported tests and portable checks of the companion study package |
| `examples/paper_tests/` | Retained research scripts and convergence-validation records; these still require their original study inputs |

The former `authoring/` directory contained adapters for a larger host application whose orchestration modules were not included here. Those adapters are retained in Git history at commit `5722f2b`; the provider-independent preparation guide is the supported entry point in this repository.

## Inspect the paper's experiments

```bash
python paper/reproduce.py list
python paper/reproduce.py validate --paper-root /path/to/study-package
python paper/reproduce.py check-saved --paper-root /path/to/study-package
```

The [paper guide](paper/README.md) maps the reported tests to their contracts, KIs, data, environments and results. The companion study package is supplied separately; a fresh clone does not contain its model binaries, observations or full run histories. The checks distinguish file integrity and saved-result consistency from a new numerical or agent experiment. Case-specific setup and known unavailable historical inputs remain explicit.

The current engine and the engine used for the original paper experiments have different version identities. Preserve the frozen paper engine when reproducing those results. See [versions and reproduction scope](docs/REPRODUCIBILITY.md).

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

[Testing notes](docs/TESTING.md) describe the portable suite, optional host integrations and known historical checks. Changes to scoring, stopping or selection require separate scientific validation; repository organization alone does not update the paper's results.
