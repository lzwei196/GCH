"""The recorder must not change the search, and must record every call with its phase, seed and step.
Run each test in a FRESH process (the recorder patches the frozen kit once): pytest -p no:cacheprovider --forked is
not available, so each test spawns its own python."""
import json, os, subprocess, sys, textwrap
import pytest
PY = "/mnt/disk1/Hydrocraft_server/python_env/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code"

PRELUDE = f"""
import sys, json, os
sys.path.insert(0, {FROZEN!r}); sys.path.insert(0, {HERE!r})
sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/models/spotpy/source/repo/src")
import numpy as np
from calibration_kit.backends.spotpy_backend import SpotpyBackend
from calibration_kit.backends.pymoo_backend import PymooBackend
class Toy:
    names = ["a", "b", "c", "d", "e"]; lower = [0.0] * 5; upper = [1.0] * 5; is_multi_objective = False
    objective_names = ["f1"]
    def evaluate(self, x):
        return [float(sum((v - 0.3) ** 2 for v in x))]
class Toy2(Toy):
    is_multi_objective = True
    objective_names = ["f1", "f2"]
    def evaluate(self, x):
        return [float(sum((v - 0.3) ** 2 for v in x)), float(sum((v - 0.7) ** 2 for v in x))]
"""


def run(body, out=None):
    code = PRELUDE + textwrap.dedent(body)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", KDT_RECORDER_STRICT="1"))
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


SEARCH = """
res = {algo_cls}({algo!r}).optimize({prob}(), budget={budget}, seed=1)
print("RESULT", json.dumps({{"n": res.n_evaluations, "best": [float(v) for v in res.best_loss],
                             "hist": [[round(float(z), 12) for z in h["x"]] for h in res.history]}}))
"""


@pytest.mark.parametrize("algo_cls, algo, prob, budget", [("SpotpyBackend", "dds", "Toy", 300),
                                                          ("SpotpyBackend", "sceua", "Toy", 600),
                                                          ("PymooBackend", "nsga2", "Toy2", 200)])
def test_recording_does_not_change_the_search(tmp_path, algo_cls, algo, prob, budget):
    s = SEARCH.format(algo_cls=algo_cls, algo=algo, prob=prob, budget=budget)
    plain = run(s)
    rec = run(f"import frozen_recorder as R; R.install({str(tmp_path)!r})\n" + s)
    a = json.loads(plain.split("RESULT ", 1)[1]); b = json.loads(rec.split("RESULT ", 1)[1])
    assert a == b                                                    # same calls, same order, same result
    calls = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert [c["i"] for c in calls] == list(range(len(calls))) and len(calls) == b["n"]
    assert [[round(z, 12) for z in c["x"]] for c in calls] == b["hist"]
    assert all(c["phase"] == "search" and c["seed"] == 1 and c["algorithm"] == algo for c in calls)
    srch = [json.loads(l) for l in open(tmp_path / "searches.jsonl")]
    assert srch == [{"search": 0, "algorithm": algo, "budget": budget, "seed": 1, "lines": b["n"], "first_line": 0}]


def test_sceua_loop_numbers(tmp_path):
    run(f"import frozen_recorder as R; R.install({str(tmp_path)!r})\n"
        + SEARCH.format(algo_cls="SpotpyBackend", algo="sceua", prob="Toy", budget=900))
    calls = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    loops = [c["loop"] for c in calls]
    start = 20 * (2 * 5 + 1)                                         # SPOTPY's start-up sample: ngs (2d + 1)
    assert loops[:start] == [0] * start and loops[start] == 1      # loop 0 = exactly the start-up sample
    assert loops == sorted(loops) and max(loops) >= 2


def test_pymoo_generation_numbers(tmp_path):
    run(f"import frozen_recorder as R; R.install({str(tmp_path)!r})\n"
        + SEARCH.format(algo_cls="PymooBackend", algo="nsga2", prob="Toy2", budget=200))
    calls = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    gens = [c["gen"] for c in calls]
    assert gens == [g for g in range(1, 6) for _ in range(40)]      # 5 generations of 40


def test_dds_has_no_loop_or_generation(tmp_path):
    run(f"import frozen_recorder as R; R.install({str(tmp_path)!r})\n"
        + SEARCH.format(algo_cls="SpotpyBackend", algo="dds", prob="Toy", budget=100))
    calls = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert all(c["loop"] is None and c["gen"] is None for c in calls)


def test_a_driver_record_and_several_searches(tmp_path):
    body = f"""
import frozen_recorder as R; R.install({str(tmp_path)!r})
class T(Toy):
    def evaluate(self, x):
        l = Toy.evaluate(self, x); self.last_record = {{"panel": {{"Q": {{"nse": 1 - l[0]}}}}}}; return l
SpotpyBackend("dds").optimize(T(), budget=50, seed=0)
R.set_phase("holdout")
SpotpyBackend("dds").optimize(T(), budget=30, seed=2)
"""
    run(body)
    calls = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert len(calls) == 80 and [c["search"] for c in calls] == [0] * 50 + [1] * 30
    assert [c["seed"] for c in calls] == [0] * 50 + [2] * 30
    assert all(abs(c["extra"]["panel"]["Q"]["nse"] - (1 - c["losses"][0])) < 1e-12 for c in calls)


def test_refuses_a_kit_that_is_not_the_frozen_one(tmp_path):
    code = (f"import sys; sys.path.insert(0, '/home/server/kdt_convergence_dev'); sys.path.insert(0, {HERE!r})\n"
            f"import frozen_recorder as R\ntry:\n    R.install({str(tmp_path)!r})\nexcept RuntimeError as e:\n    print('REFUSED', e)\n")
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    assert "REFUSED" in r.stdout


EVAL = """
import frozen_recorder as R; R.install(OUT)
from calibration_kit.evaluator import Evaluator
from calibration_kit.objectives import Objective
runs = []
def run_model():
    p = json.load(open(os.environ["KDT_CALIB_PARAMS"]))
    runs.append(p)
    return {"nse": 1 - (p["a"] - 0.3) ** 2, "kge": 0.5, "__kdt__": {"applied_params": p, "panel": {"Q": {"r": 0.9}}}}
ev = Evaluator(ki_path=OUT, workdir=OUT, parameters=[{"name": "a", "range": [0, 1]}],
               objectives=[Objective("Q:nse", "Q", "temporal_pattern_match", "nse")], transform_inv={},
               run_model=run_model, injection_mode="runner",
               constraints_ok=lambda named: named["a"] < 0.9)
print("L", ev.evaluate([0.5]), ev.evaluate([0.5]), ev.evaluate([0.95]), ev.evaluate([0.2]), len(runs))
"""


def test_the_frozen_evaluator_path(tmp_path):
    """calibrate() runs: a model run, a cache hit, a call rejected before any run, another run."""
    out = run(f"OUT = {str(tmp_path)!r}\n" + EVAL)
    assert out.strip().split()[-1] == "2"                            # the stub ran the model exactly twice
    calls = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert [c["kind"] for c in calls] == ["evaluator"] * 4
    a, b, c, d = calls
    assert a["ran_model"] and not a["cache_hit"] and a["eval_id"] == 1 and a["metrics"]["nse"] == 1 - 0.04
    assert a["params"] == {"a": 0.5} and a["split"] == "calibration" and a["phase"] == "outside"
    assert (not b["ran_model"]) and b["cache_hit"] and b["metrics"]["nse"] == a["metrics"]["nse"] and b["eval_id"] is None
    assert (not c["ran_model"]) and not c["cache_hit"] and c["metrics"] is None and c["rejected_before_run"]
    assert c["losses"] == ["inf"]                                    # JSON-safe spelling of +inf
    assert d["ran_model"] and d["eval_id"] == 2 and abs(d["metrics"]["nse"] - (1 - 0.01)) < 1e-12
    assert d["metrics"]["__kdt__"]["panel"]["Q"]["r"] == 0.9         # the whole payload is kept



RUNNER = """
import json, os, sys
out = sys.argv[sys.argv.index("--out") + 1]
p = json.load(open(os.environ["KDT_CALIB_PARAMS"]))
if p["a"] > 0.8:                                  # a failed run: no metrics file
    sys.exit(3)
json.dump({"nse": 1 - (p["a"] - 0.3) ** 2, "__kdt__": {"applied_params": p}}, open(out, "w"))
"""


def test_every_model_run_is_recorded_and_metrics_are_never_from_an_earlier_call(tmp_path):
    """codex recorder r1 #1, #2: runs outside evaluate() are recorded; a failed run carries no earlier metrics."""
    (tmp_path / "runner.py").write_text(RUNNER)
    body = f"""
import frozen_recorder as R; R.install({str(tmp_path / 'rec')!r})
from calibration_kit import runner as RN
from calibration_kit.evaluator import Evaluator
from calibration_kit.objectives import Objective
rm = RN.make_run_model({{"kind": "subprocess", "command": [sys.executable, {str(tmp_path / 'runner.py')!r}, "--out", "{{metrics_json}}"]}},
                       {str(tmp_path)!r}, {str(tmp_path)!r})
json.dump({{"a": 0.5}}, open({str(tmp_path / 'p.json')!r}, "w")); os.environ["KDT_CALIB_PARAMS"] = {str(tmp_path / 'p.json')!r}
with R.phase("certification"):
    rm()                                           # a run outside evaluate(), as certify_runner makes
os.environ.pop("KDT_CALIB_PARAMS")
ev = Evaluator(ki_path={str(tmp_path)!r}, workdir={str(tmp_path)!r}, parameters=[{{"name": "a", "range": [0, 1]}}],
               objectives=[Objective("Q:nse", "Q", "temporal_pattern_match", "nse")], transform_inv={{}},
               run_model=rm, injection_mode="runner")
print("L", ev.evaluate([0.5]), ev.evaluate([0.9]), ev.evaluate([0.5]))
"""
    run(body)
    L = [json.loads(l) for l in open(tmp_path / "rec" / "calls.jsonl")]
    kinds = [(x["kind"], x["phase"]) for x in L]
    assert kinds == [("model_run", "certification"), ("model_run", "outside"), ("evaluator", "outside"),
                     ("model_run", "outside"), ("evaluator", "outside"), ("evaluator", "outside")]
    cert, run1, ev1, run2, ev2, ev3 = L
    assert cert["params_handed"] == {"a": 0.5} and cert["payload"]["nse"] == 1 - 0.04
    assert ev1["ran_model"] and ev1["model_run_i"] == run1["i"] and ev1["metrics"]["nse"] == 1 - 0.04
    assert run2["payload"] == {} and ev2["ran_model"] and ev2["model_run_i"] == run2["i"]
    assert ev2["metrics"] == {} and ev2["losses"] == ["inf"]          # NOT the earlier call's metrics
    assert ev3["cache_hit"] and not ev3["ran_model"] and ev3["metrics"]["nse"] == 1 - 0.04
    assert ev3["model_run_i"] is None and ev3["eval_id"] is None    # a cache hit points at no model run


def test_a_write_failure_never_reaches_the_search(tmp_path):
    """codex recorder r1 #3: best-effort recording (not strict)."""
    body = f"""
import frozen_recorder as R; R.install({str(tmp_path)!r})
R._state["fh"].close()                             # every later write fails
res = SpotpyBackend("dds").optimize(Toy(), budget=50, seed=0)
print("N", res.n_evaluations)
"""
    code = PRELUDE + textwrap.dedent(body)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=dict(os.environ, KDT_RECORDER_STRICT="0"))
    assert r.returncode == 0 and "N 50" in r.stdout, r.stderr[-2000:]
    errs = (tmp_path / "recorder_errors.jsonl").read_text().splitlines()
    assert len(errs) >= 50 and "write" in errs[0]


def test_phases_of_the_frozen_kit_are_wrapped(tmp_path):
    """codex recorder r1 #5"""
    body = f"""
import frozen_recorder as R; R.install({str(tmp_path)!r})
from calibration_kit import calib as CA, holdout as HO, sensitivity as SE, evaluator as EV
print("W", all(hasattr(f, "__wrapped__") for f in (CA.certify_runner, CA._probe_runner_metrics, CA._select_front_member,
      HO.validate_holdout, SE.morris_screen, EV.Evaluator.responsiveness_check)))
with R.phase("holdout"):
    SpotpyBackend("dds").optimize(Toy(), budget=20, seed=0)
"""
    out = run(body)
    assert "W True" in out
    calls = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert all(c["phase"] == "search" for c in calls)                 # the search inside a phase is "search"



def test_a_model_callable_handed_to_calibrate_is_recorded(tmp_path):
    """codex recorder r2 #1: calibrate(..., run_model=my_callable) — every call of it is a model_run line."""
    body = f"""
from calibration_kit import calib as CA
CA.calibrate = lambda ki_path, workdir, obs, run_model=None, **k: run_model      # stand-in for the frozen calibrate
CA.calibrate_staged = lambda ki_path, workdir, obs, run_model=None, **k: run_model
import frozen_recorder as R; R.install({str(tmp_path)!r})
f = lambda: {{"nse": 0.5}}
g1 = CA.calibrate("ki", "wd", {{}}, run_model=f); g2 = CA.calibrate("ki", "wd", {{}}, f); g3 = CA.calibrate_staged("ki", "wd", {{}}, run_model=f)
print("R", g1(), g2(), g3(), g1 is not f, g2 is not f, CA.calibrate("ki", "wd", {{}}, run_model=g1) is g1)
"""
    out = run(body)
    assert "R {'nse': 0.5} {'nse': 0.5} {'nse': 0.5} True True True" in out
    L = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert [x["kind"] for x in L] == ["model_run"] * 3 and all(x["payload"] == {"nse": 0.5} for x in L)


def test_what_was_handed_is_read_before_the_run(tmp_path):
    """codex recorder r2 #2: a runner that rewrites its params file and the environment."""
    body = f"""
import frozen_recorder as R; R.install({str(tmp_path / 'rec')!r})
def sneaky():
    json.dump({{"a": 999}}, open(os.environ["KDT_CALIB_PARAMS"], "w")); os.environ["KDT_CALIB_SPLIT"] = "holdout"
    return {{"nse": 0.1}}
f = R._recorded_run_model(sneaky)
json.dump({{"a": 0.5}}, open({str(tmp_path / 'p.json')!r}, "w"))
os.environ["KDT_CALIB_PARAMS"] = {str(tmp_path / 'p.json')!r}; os.environ["KDT_CALIB_SPLIT"] = "calibration"
f()
"""
    run(body)
    x = json.loads(open(tmp_path / "rec" / "calls.jsonl").readline())
    assert x["params_handed"] == {"a": 0.5} and x["split_env"] == "calibration"


def test_numpy_non_finite_losses_are_recorded(tmp_path):
    """codex recorder r2 #4"""
    body = f"""
import frozen_recorder as R; R.install({str(tmp_path)!r})
class N(Toy):
    def evaluate(self, x):
        return [np.float32(np.nan)] if x[0] > 0.5 else [np.float32(np.inf)]
SpotpyBackend("dds").optimize(N(), budget=20, seed=0)
"""
    run(body)
    L = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert len(L) == 20 and {x["losses"][0] for x in L} <= {"nan", "inf"}


def test_two_searches_at_once_are_refused(tmp_path):
    """codex recorder r2 #3: the guard holds under threads."""
    body = f"""
import frozen_recorder as R, threading, time; R.install({str(tmp_path)!r})
class Slow(Toy):
    def evaluate(self, x):
        time.sleep(0.01); return Toy.evaluate(self, x)
errs = []
def go(s):
    try:
        SpotpyBackend("dds").optimize(Slow(), budget=30, seed=s)
    except RuntimeError as e:
        errs.append(str(e))
ts = [threading.Thread(target=go, args=(s,)) for s in (0, 1)]
[t.start() for t in ts]; [t.join() for t in ts]
print("E", len(errs), errs[:1])
"""
    out = run(body)
    assert "E 1 ['recorder: a second search started" in out



def test_a_model_run_that_raises_is_recorded_as_attempted(tmp_path):
    """codex recorder r3: a python model callable raises; the kit catches it; the record says a run was attempted."""
    body = f"""
import frozen_recorder as R; R.install({str(tmp_path)!r})
from calibration_kit.evaluator import Evaluator
from calibration_kit.objectives import Objective
def boom():
    raise ValueError("model crashed")
rm = R._recorded_run_model(boom)
ev = Evaluator(ki_path={str(tmp_path)!r}, workdir={str(tmp_path)!r}, parameters=[{{"name": "a", "range": [0, 1]}}],
               objectives=[Objective("Q:nse", "Q", "temporal_pattern_match", "nse")], transform_inv={{}},
               run_model=rm, injection_mode="runner")
print("L", ev.evaluate([0.5]))
"""
    run(body)
    L = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert [x["kind"] for x in L] == ["model_run", "evaluator"]
    mr, ev = L
    assert mr["payload"] is None and "model crashed" in mr["raised"]
    assert ev["ran_model"] and ev["run_raised"] and not ev["rejected_before_run"] and ev["model_run_i"] == mr["i"]
    assert ev["metrics"] is None and ev["losses"] == ["inf"]
