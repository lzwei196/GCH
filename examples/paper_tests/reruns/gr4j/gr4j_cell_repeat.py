"""GR4J repeated-formulation (Test C) cell, rerun with the recorder. Started ONLY through the paper's own sealed
launcher (sealed_exec/sealed_python.sh), so the engine, SPOTPY copy, packages, interpreter and empty environment are
the paper's. What is left out: the paper campaign's ledger / gate / panel bookkeeping (nothing in the paper record is
written). The calibrate() call is the one in run_contract.py section 7, with the same arguments.

usage: sealed_python.sh gr4j_cell.py <tag>      e.g. C1_seed0, C1_seed0_nse
Writes runs/<tag>/: rec/ (recorder), run_workdir/ (the engine's work folder, incl. qsim_<id>.csv per call),
calib_report.json, check.json (does it reproduce the paper cell?).
"""
import json
import os
import sys
from pathlib import Path

import sealed_boot

HERE = Path(__file__).resolve().parent
REC_DIR = HERE.parent.parent / "recorder"
FULL = Path("/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/02_tests/"
            "C_stability_gr4j/full")
SE = FULL / "sealed_exec"
ORIG = FULL / "runs" / "panel1"
sealed_boot.allow_root(HERE)
sealed_boot.allow_root(REC_DIR)

tag = sys.argv[1]
pm = json.loads((FULL / "PANEL_MANIFEST.json").read_text())
cell = pm["cells"][tag]
label, seed, m = cell["contract"], int(cell["seed"]), cell["effective_objective"]
D = HERE / "runs_repeat" / tag          # REPEAT: second rerun of a cell that differed
if D.exists():
    sys.exit(f"REFUSED: {D} exists")
workdir = D / "run_workdir"
workdir.mkdir(parents=True)

sys.path.insert(0, str(REC_DIR))
import frozen_recorder as R  # noqa: E402
R.FROZEN = str(SE / "engine")                 # the sealed engine = the paper-frozen kit code (same calib, evaluator,
R.install(str(D / "rec"))                     # runner, spotpy_backend files; checked 2026-10-03)

import spotpy  # noqa: E402
from calibration_kit import calib  # noqa: E402
assert Path(spotpy.__file__).resolve().is_relative_to(SE), spotpy.__file__
assert Path(calib.__file__).resolve().is_relative_to(SE), calib.__file__

os.environ["KDT_CALIB_CONTRACT"] = str(FULL / "contracts_wired" / f"{label}_calibration_wired.yaml")
report = calib.calibrate(ki_path=str(SE / "ki"), workdir=str(workdir), obs_shape_by_var={"Qsim": "point_time_series"},
                         determining_metric=m, budget=None, seed=seed, expected_case_id=None)
(D / "calib_report.json").write_text(json.dumps(report, indent=2, default=str))

# ── does it reproduce the paper cell? ─────────────────────────────────────────────────────────────
o = json.loads((ORIG / tag / "calib_report.json").read_text())
keys = ("status", "algorithm", "n_evaluations", "best_params", "best_value", "best_metrics", "objectives")


def _cache(p):
    fs = sorted(Path(p).glob("eval_metrics_cache_*.jsonl"))
    return [json.loads(x) for f in fs for x in f.read_text().splitlines() if x.strip()]


oc, nc = _cache(ORIG / tag / "run_workdir"), _cache(workdir)
chk = {"tag": tag, "same": {k: o.get(k) == report.get(k) for k in keys},
       "orig": {k: o.get(k) for k in keys}, "rerun": {k: report.get(k) for k in keys},
       "cache_lines": [len(oc), len(nc)], "cache_identical": oc == nc}
chk["identical"] = all(chk["same"].values()) and chk["cache_identical"]
(D / "check.json").write_text(json.dumps(chk, indent=2, default=str))
print("CHECK", tag, "identical" if chk["identical"] else "DIFFERS", chk["same"], chk["cache_lines"], chk["cache_identical"])
