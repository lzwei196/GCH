"""E1 (GR4J vs airGR, DDS 1,000, seed 0) rerun with the recorder: the paper's run_e1.py call, with the paper-frozen
kit, the record's copy of the frozen agent_authored.yaml (identical to the FINAL_2026-09-01 copy it ran with; its
runner command points at the identical FINAL_2026-09-01 calib_run.py and forcing), the KI unchanged since (files
dated 2026-07/08), and a work folder here. Each call's simulated flow is written by the paper's runner
(run_workdir/qsim_<id>.csv). Writes calib_report.json and check.json (reproduces the paper's E1?)."""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
FROZEN = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code"
REC = Path("/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/03_supporting_evidence/"
           "E1_E2_reproductions/E1_airGR")
sys.path.insert(0, FROZEN)
sys.path.insert(0, str(HERE.parent.parent / "recorder"))
import frozen_recorder as R
R.install(str(HERE / "rec"))
from calibration_kit import calib
assert calib.__file__.startswith(FROZEN), calib.__file__
os.environ["KDT_CALIB_CONTRACT"] = str(REC / "agent_authored.yaml")
workdir = HERE / "run_workdir"
workdir.mkdir(exist_ok=False)
report = calib.calibrate(ki_path="/mnt/disk1/Hydrocraft_server/models/GR4J___airGR/knowledge_infrastructure",
                         workdir=str(workdir), obs_shape_by_var={"Qsim": "point_time_series"},
                         determining_metric="nse", budget=1000, seed=0, expected_case_id=None)
(HERE / "calib_report.json").write_text(json.dumps(report, indent=2, default=str))
o = json.loads((REC / "calib_report.json").read_text())
keys = ("status", "algorithm", "n_evaluations", "best_params", "best_value", "best_metrics", "objectives")


def _cache(p):
    return [json.loads(x) for f in sorted(Path(p).glob("eval_metrics_cache_*.jsonl")) for x in f.read_text().splitlines() if x.strip()]


oc, nc = _cache(REC / "run_workdir"), _cache(workdir)
chk = {"same": {k: o.get(k) == report.get(k) for k in keys}, "orig": {k: o.get(k) for k in keys},
       "rerun": {k: report.get(k) for k in keys}, "cache_lines": [len(oc), len(nc)], "cache_identical": oc == nc}
chk["identical"] = all(chk["same"].values()) and chk["cache_identical"]
(HERE / "check.json").write_text(json.dumps(chk, indent=2, default=str))
print("CHECK", "identical" if chk["identical"] else "DIFFERS", chk["same"], chk["cache_lines"], chk["cache_identical"])
