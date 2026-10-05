# Minimal synthetic calibration

From the repository root, after installing `requirements.txt` in a Python 3.12 environment:

```bash
python examples/minimal/run.py --output ./outputs/minimal
```

This complete example uses a linear reservoir: forcing adds `gain × input` to storage and `release × storage` leaves each step. The checked-in 240-step synthetic observations were generated with `release=0.28`, `gain=1.2`, deterministic forcing and no observation noise. The runner discards the first 40 steps as warm-up, scores steps 40–179 for calibration, and reserves 180–239 for holdout. The parameter domain and defaults are in `calibration.yaml`; `dag.yaml` maps discharge `Q` to a temporal-pattern objective.

DDS searches for 60 evaluations with seed 7 and one seed slot. This small fixed cap keeps the demonstration inexpensive; it is not a scientifically justified budget for any paper model. The engine also performs commissioning, consumption checks, pilot and assessment calls. DDS has no native convergence rule; do not interpret completion at the cap as convergence or a cross-seed reproducibility test.

The entry point copies the workflow into a new output folder, sets its runner to the active Python interpreter, and runs GCH through a real subprocess runner. It refuses to overwrite an existing output directory. No language-model call, network access, external model executable or companion package is required. Source example files are preserved.

Inspect `report.json`, `run/eval_history.jsonl` and the other run artifacts for the declared parameters, actual evaluations, phase counts, calibration/holdout assessment and convergence reporting. The report is the numerical result; the printed summary alone is not evidence of scientific acceptance. This is an integration demonstration of the current engine, separate from every empirical paper experiment.
