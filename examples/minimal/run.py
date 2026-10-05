"""Run a small synthetic calibration from a clean checkout; no model downloads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from calibration_kit import calib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing runs are not overwritten")
    inherited = sorted(k for k in os.environ if k.startswith("KDT_CALIB_"))
    if inherited:
        parser.error("Run in a shell without KDT_CALIB_* overrides: " + ", ".join(inherited))
    ki = output / "workflow"
    ki.mkdir(parents=True)
    for name in ("calibration.yaml", "dag.yaml", "calib_run.py", "model.py", "observations.csv"):
        shutil.copyfile(HERE / name, ki / name)
    contract = yaml.safe_load((ki / "calibration.yaml").read_text())
    contract["runner"]["command"][0] = os.path.abspath(sys.executable)
    (ki / "calibration.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
    os.environ["PYTHONPATH"] = str(REPO) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    report = calib.calibrate(str(ki), str(output / "run"), {"Q": "point_time_series"}, seed=7)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {"status": report["status"], "best_params": report.get("best_params"),
               "budget_used": report.get("budget_used"),
               "holdout_validated": report.get("holdout_validated"),
               "consumption_coverage": report.get("consumption_coverage"),
               "convergence": report.get("convergence", {}).get("verdict"),
               "report": str(output / "report.json")}
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
