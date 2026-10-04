"""Six-model campaign rerun (paper section "six model families"), one model x one seed, recorded.
The paper's driver (six_model_families/rerun_conv.py) with: the paper-frozen kit (commit 8acd3d7, archived 2026-09-07;
the runs ended before the 2026-09-01 record copy) instead of the then-live kit; the record's copy of each KI as used
(copied here per run, so seeds never share files); the record's validation_convention.py (toolkit_at_3ed4e991) loaded by
path — the headline objectives it derives are checked against the line the paper's rerun.log printed; an EMPTY
work folder (the paper runs started on earlier caches, so call-for-call equality is not expected); the same env flags
(KDT_VALIDATION_CONVENTION, KDT_CALIB_FRONT_SELECT, KDT_CALIB_PROTECT per case) and the agent-declared budget.
Recording only: the frozen-kit recorder (every model run and scoring call, generation tags) and the shared scorer's
own switch KDT_SERIES_DUMP_DIR, set to a NEW folder for every model run (dumps/run_<n>/), so each run's scored obs/sim
pairs are tied to it; dumps/index.jsonl joins each folder to the recorder's model_run_i and lists the files found
(HBV, SUMMA, VIC write them; a model run whose expected dump is missing is flagged, never silently passed). A cache-hit
call makes no model run and so no dump (kit design). CRHM's runner writes its own dated _series/; MODFLOW6 and WOFOST:
the scores in the cache (owner 2026-10-04: MODFLOW6 reports the scores it already has).
KI tools (owner 2026-10-04 + tests, PROCESS_LOG step 63): today's tools are used for all six. WOFOST's runner copy is
updated in ONE line: its weather step file is build_pcse_weather_from_source.py (where load_source / build_from_forcing
live since models commit a818fd45) instead of create_csv_weather_file.py; tested: identical weather and soil files.
Environment: every KDT_* variable that the kit or runners read is refused if already set, except the ones set here.
VIC seeds run one at a time (exclusive lock); one model at a time is the operator's rule (fleet rule).
usage: python run_model.py <WOFOST|HBV|SUMMA|VIC|MODFLOW6|CRHM> <seed>   -> runs/<MODEL>_s<seed>/"""
import importlib.util, json, os, re, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REC = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07"
FROZEN = REC + "/01_framework/framework_code"
SMF = REC + "/03_supporting_evidence/six_model_families"
VC = REC + "/09_kdt_repo_proofs/toolkit_at_3ed4e991/validation_convention.py"
CASES = {   # verbatim from rerun_conv.py
 "HBV": {"dir": "02_HBV", "obs": {"Q": "point_time_series", "SWE": "point_time_series"}, "primary": "Q",
         "protect": ["Q:temporal_pattern_match", "Q:magnitude_accuracy"]},
 "SUMMA": {"dir": "03_SUMMA", "obs": {"scalarTotalRunoff": "point_time_series", "scalarSWE": "point_time_series"},
           "primary": "scalarTotalRunoff"},
 "VIC": {"dir": "04_VIC", "obs": {"OUT_DISCHARGE": "point_time_series", "OUT_EVAP": "point_time_series"},
         "primary": "OUT_DISCHARGE", "protect": ["OUT_DISCHARGE:temporal_pattern_match", "OUT_DISCHARGE:magnitude_accuracy"]},
 "MODFLOW6": {"dir": "05_MODFLOW6", "obs": {"streamflow": "flow_duration_curve", "baseflow": "flow_duration_curve"},
              "primary": "streamflow", "case_id": "USGS:10343500"},
 "WOFOST": {"dir": "01_WOFOST", "obs": {"TWSO": "point_snapshot", "TAGP": "point_snapshot"}, "primary": "TWSO"},
 "CRHM": {"dir": "06_CRHM", "obs": {"basinflow_s": "point_time_series", "SWE": "point_time_series"},
          "primary": "basinflow_s", "protect": ["basinflow_s:temporal_pattern_match", "basinflow_s:magnitude_accuracy"]},
}
if len(sys.argv) != 3 or sys.argv[1] not in CASES:
    sys.exit("usage: run_model.py <" + "|".join(CASES) + "> <seed>")
name, seed = sys.argv[1], int(sys.argv[2])
c = CASES[name]
OUT = os.path.join(HERE, "runs", f"{name}_s{seed}")
if os.path.exists(OUT):
    sys.exit(f"REFUSED: {OUT} exists")
_dirty = sorted(k for k in os.environ if k.startswith("KDT_") or "TIMEOUT" in k.upper())
if _dirty:                                            # codex six-driver r1 #1: a dirty shell must not change the run
    sys.exit(f"REFUSED: environment variables already set: {_dirty}")
if name == "VIC":                                     # VIC seeds share /dev/shm caches: one at a time
    import fcntl
    _lock = open(os.path.join(HERE, "runs", "VIC.lock"), "w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("REFUSED: another VIC seed is running")
SRC_KI = os.path.join(SMF, c["dir"], "ki")
if name == "WOFOST":                                  # check the runner line BEFORE any folder is made
    if open(os.path.join(SRC_KI, "tools", "calib_run.py")).read().count(
            'lt("s3", "s3_weather_prep", "create_csv_weather_file.py")') != 1:
        sys.exit("REFUSED: WOFOST runner line to update not found exactly once")
# headline objectives from the record KI, checked against the paper log BEFORE any folder is made
_sp = importlib.util.spec_from_file_location("validation_convention", VC)
vc = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(vc)
import yaml
strat = (yaml.safe_load(open(os.path.join(SRC_KI, "calibration.yaml"))).get("strategy") or {})
budget = int(strat.get("max_evaluations") or 200)
conv = vc.load(SRC_KI); ho = {}
for b in (conv or {}).get("validation", []):
    q = b.get("dag_variable")
    if q in c["obs"] and str(b.get("obs_shape", "")) == str(c["obs"][q]):
        objs = vc.headline_objectives(b)
        if objs:
            ho.setdefault(q, []).extend(objs)
ho_line = json.dumps({k: [(o["metric"], o["target"]) for o in v] for k, v in ho.items()})
paper_log = open(os.path.join(SMF, c["dir"], "rerun.log")).read()
m = re.search(r"convention headline_objectives = (\{.*\})", paper_log)
same_ho = bool(m) and json.loads(m.group(1)) == json.loads(ho_line)
print(f">>> {name} s{seed}: budget {budget}; headline_objectives {ho_line}; same as paper log: {same_ho}", flush=True)
if not same_ho:
    sys.exit(f"REFUSED: headline objectives differ from the paper log ({m.group(1) if m else 'none in log'})")
# ── all checks passed: only now is the run folder made (codex six-driver r2 #2) ───────────────────────────
KI, WD, DUMP = os.path.join(OUT, "ki"), os.path.join(OUT, "work"), os.path.join(OUT, "dumps")
shutil.copytree(SRC_KI, KI, symlinks=True)
os.makedirs(WD); os.makedirs(DUMP)
if name == "WOFOST":                                  # the one-line runner update (owner 2026-10-04, tested)
    _rp = os.path.join(KI, "tools", "calib_run.py")
    _src = open(_rp).read()
    _old = 'lt("s3", "s3_weather_prep", "create_csv_weather_file.py")'
    _new = 'lt("s3", "s3_weather_prep", "build_pcse_weather_from_source.py")'
    open(_rp, "w").write(_src.replace(_old, _new))
    open(os.path.join(OUT, "RUNNER_UPDATE.txt"), "w").write(
        f"tools/calib_run.py of this run's KI copy: {_old} -> {_new} (models commit a818fd45 moved load_source /\n"
        "build_from_forcing there; tested 2026-10-04: identical weather CSV and soil JSON for the paper cell)\n")
os.environ["KDT_VALIDATION_CONVENTION"] = "1"
os.environ["KDT_CALIB_FRONT_SELECT"] = "1"
if c.get("protect"):
    os.environ["KDT_CALIB_PROTECT"] = ",".join(c["protect"])
else:
    os.environ.pop("KDT_CALIB_PROTECT", None)
sys.path.insert(0, FROZEN)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "recorder"))
import frozen_recorder as R
R.install(os.path.join(OUT, "rec"))
from calibration_kit import runner as RN
EXPECT_DUMPS = {"HBV": 2, "SUMMA": 2, "VIC": 2}.get(name, 0)   # two scored variables each   # files per real model run (OUTPUT_FILES_PER_MODEL.md)
_n = {"i": 0, "missing": 0}
_idx = open(os.path.join(DUMP, "index.jsonl"), "a")
_rec_mrm = RN.make_run_model                                  # the recorder's wrapper


def make_run_model(runner_spec, ki_path, workdir):
    inner = _rec_mrm(runner_spec, ki_path, workdir)

    def run(*a, **k):                                         # RECORDING ONLY: a fresh dump folder per model run
        d = os.path.join(DUMP, f"run_{_n['i']:05d}"); _n["i"] += 1
        os.makedirs(d)
        os.environ["KDT_SERIES_DUMP_DIR"] = d
        try:
            return inner(*a, **k)
        finally:
            os.environ.pop("KDT_SERIES_DUMP_DIR", None)
            files = sorted(f for f in os.listdir(d) if f.endswith(".csv"))
            ok = len(files) >= EXPECT_DUMPS
            _n["missing"] += 0 if ok else 1
            _idx.write(json.dumps({"dump_dir": os.path.basename(d), "model_run_i": R._state.get("last_run_i"),
                                   "phase": R._state["phases"][-1] if R._state.get("phases") else None,
                                   "files": files, "expected": EXPECT_DUMPS, "ok": ok}) + "\n")
            _idx.flush()
    run._kdt_recorded = True        # the recorder already wraps `inner`; it must not wrap again (staged searches)
    return run


RN.make_run_model = make_run_model
from calibration_kit import calib
if not calib.__file__.startswith(FROZEN):
    sys.exit(f"REFUSED: not the frozen kit: {calib.__file__}")
kw = dict(ki_path=KI, workdir=WD, obs_shape_by_var=c["obs"], budget=budget, seed=seed, headline_objectives=ho)
if c.get("case_id"):
    kw["expected_case_id"] = c["case_id"]
r = calib.calibrate(**kw)
json.dump(r, open(os.path.join(OUT, "rerun_conv_report.json"), "w"), default=str, indent=1)
# ── compare with the paper's run (seed 0 only; not expected to be identical, see the docstring) ─────────
if seed == 0:
    o = json.load(open(os.path.join(SMF, c["dir"], "rerun_conv_report.json")))
    keys = ("status", "algorithm", "n_evaluations", "best_params", "best_metrics", "promotable", "objectives")
    json.dump({"same": {k: o.get(k) == r.get(k) for k in keys}, "paper": {k: o.get(k) for k in keys},
               "rerun": {k: r.get(k) for k in keys}}, open(os.path.join(OUT, "compare_paper.json"), "w"),
              default=str, indent=1)
R._state["fe"].flush()
n_err = os.path.getsize(os.path.join(OUT, "rec", "recorder_errors.jsonl"))
json.dump({"model": name, "seed": seed, "done": True, "recorder_errors_bytes": n_err, "budget": budget,
           "model_runs_wrapped": _n["i"], "dumps_missing": _n["missing"]},
          open(os.path.join(OUT, "done.json"), "w"))
print(f">>> {name} s{seed} DONE status={r.get('status')} n_eval={r.get('n_evaluations')} recorder_error_bytes={n_err} "
      f"model_runs={_n['i']} dumps_missing={_n['missing']}",
      flush=True)
