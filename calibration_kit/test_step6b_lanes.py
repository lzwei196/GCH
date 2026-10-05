"""Build step 6b (design §1.3, §1.7): seeds in parallel lanes. When the machine probe measured more than
one lane (a parallel_safe subprocess runner), the runner can be rebuilt per lane from the contract and the
backend reports each call, seeds run min(S, L) at a time in forked processes on cloned workdirs; each lane's
call log and cache are merged back. Results must equal the one-lane run (the same seeds, the same searches)."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import compute                                                       # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402

RUN_PY = """
import json, os, sys, time
sys.path.insert(0, {root!r})
from calibration_kit import test_calibrate_convergence_e2e as E
out, wd, log = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
time.sleep(0.05)
m = E.Runner(wd)()
open(log, "a").write(json.dumps({{"wd": wd, "t0": t0, "t1": time.time(),
                                  "eid": os.environ.get("KDT_CALIB_EVAL_ID")}}) + "\\n")
json.dump(m, open(out, "w"), default=float)
"""


PROBE_KS: list = []


def _pin_lanes(monkeypatch, lanes=3, eff=None, edit_contract=None):
    """A machine probe that measured `lanes` lanes (the real probe stays at 1 without parallel_safe)."""
    def fake(pilot, *, parallel_safe=True, runner_mode="", clone_and_eval=None, ks=(1, 2, 4)):
        PROBE_KS.append(tuple(ks))
        if edit_contract is not None:
            edit_contract()
        if not parallel_safe or clone_and_eval is None:        # like the real probe: 1 until measured
            return {"machine": {}, "ceiling": 8, "ceiling_why": {}, "lanes": 1, "efficiency": 1.0,
                    "measured": None, "notes": ["not parallel-safe or not measured"]}
        return {"machine": {"cores": 8, "load1": 0.0, "free_cores": 8, "mem_available_gb": 64.0}, "ceiling": 8,
                "ceiling_why": {"binding": "cpu"}, "lanes": lanes, "efficiency": (eff or {}).get(lanes, 0.9),
                "measured": {"efficiency": eff or {1: 1.0, 2: 0.95, 3: 0.9}, "probe_wall_s": 0.1}, "notes": []}
    monkeypatch.setattr(compute, "probe_lanes", fake)


def _setup(tmp, parallel_safe=True, seeds=3, cap=20):
    ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "1h", "parallel_safe": parallel_safe,
                              "seeds": seeds, "pilot_runs": 3}, None, cap)
    Path(tmp, "run.py").write_text(RUN_PY.format(root=str(ROOT)))
    log = Path(tmp, "calls.log")
    c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
    c["runner"] = {"kind": "subprocess", "command": [sys.executable, str(Path(tmp, "run.py")), "{metrics_json}",
                                                    "{workdir}", str(log)]}
    yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
    return ki, wd, log


def _calib(ki, wd, run_model=None):
    with E._clean_env(), contextlib.redirect_stdout(io.StringIO()) as buf:
        rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=run_model, budget=None, seed=0)
    hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    return rep, hist, buf.getvalue()


def _slots(rep):
    return [(s["seed"], s["incumbent_call"], tuple(s["incumbent_losses"] or []))
            for s in rep["convergence"]["seeds"]["slots"]]


def test_parallel_lanes_give_the_same_seeds_as_one_lane(monkeypatch):
    _pin_lanes(monkeypatch)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    tmp2 = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, log = _setup(tmp, parallel_safe=True)
        par, hist, _ = _calib(ki, wd)
        ki2, wd2, _ = _setup(tmp2, parallel_safe=False)
        one, hist1, _ = _calib(ki2, wd2)
        lanes_left = [p for p in Path(wd).parent.iterdir() if "_seed_lanes" in p.name]
        calls = [json.loads(l) for l in log.read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)
    sd = par["convergence"]["seeds"]
    assert par["status"] == "completed" and sd["seeds_parallel"] == 3 and not sd["run_one_after_another"]
    assert par["budget_plan"]["waves"] == 1 and par["budget_plan"]["efficiency"] == pytest.approx(0.9)
    assert one["convergence"]["seeds"]["seeds_parallel"] == 1
    assert _slots(par) == _slots(one)                                   # the same searches, lane or not
    assert sd["returned"] == one["convergence"]["seeds"]["returned"]
    assert par["best_params"] == pytest.approx(one["best_params"])
    # the lanes read their OWN runs: every search call's panel, and each incumbent's panel, equal the one-lane run
    def _panels(h):
        return {(r["seed"], k): r["panel"] for s_ in (0, 1, 2)
                for k, r in enumerate([x for x in h if x["phase"] == "search" and x.get("seed") == s_])}
    assert _panels(hist) == _panels(hist1) and all(v for v in _panels(hist).values())
    assert [s["incumbent_panel"] for s in sd["slots"]] == [s["incumbent_panel"] for s in one["convergence"]["seeds"]["slots"]]
    # the returned incumbent's metrics are known to the parent (merged cache), so the report and the holdout match
    assert par["train_metrics"] and par["train_metrics"] == one["train_metrics"]
    assert par["holdout"]["passed"] == one["holdout"]["passed"]
    # each lane has its own KDT_CALIB_EVAL_ID stream: no id is used twice
    lane_ids = [c["eid"] for c in calls if "_seed_lanes" in c["wd"]]
    assert lane_ids and len(lane_ids) == len(set(lane_ids))
    # every search call is in the main log once, tagged with its seed and lane; one numbering
    search = [h for h in hist if h["phase"] == "search"]
    assert len(search) == sd["total_search_calls"] == 3 * 20
    assert {(h["seed"], h["lane"]) for h in search} == {(0, 1), (1, 2), (2, 3)}
    assert [h["i"] for h in hist] == list(range(len(hist)))
    assert lanes_left == []                                              # the lane workdirs are removed
    # the lanes really ran at the same time
    by_lane = {}
    for c in calls:
        if "_seed_lanes" in c["wd"]:
            by_lane.setdefault(Path(c["wd"]).name, []).append((c["t0"], c["t1"]))
    spans = [(min(a for a, _ in v), max(b for _, b in v)) for v in by_lane.values()]
    assert len(spans) == 3 and max(a for a, _ in spans) < min(b for _, b in spans)


def test_a_resume_replays_every_lane_from_the_merged_cache(monkeypatch):
    _pin_lanes(monkeypatch)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, log = _setup(tmp)
        a, _, _ = _calib(ki, wd)
        n_before = len(log.read_text().splitlines())
        b, hist, _ = _calib(ki, wd)
        n_after = len(log.read_text().splitlines())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert b["budget_plan"].get("resumed_cap") and b["convergence"]["seeds"]["seeds_parallel"] == 3
    assert _slots(a) == _slots(b)
    resumed = [h for h in hist[len(hist) - b["convergence"]["seeds"]["total_search_calls"]:] if h["phase"] == "search"]
    assert resumed and all(h["cache_hit"] for h in resumed)            # nothing re-run in the lanes
    assert n_after - n_before <= 5                                      # only the probe / proof runs again


class _Crash:
    bad: set = set()

    def __init__(self, inner):
        self.inner = inner

    def available(self):
        return self.inner.available()

    def optimize(self, problem, budget, seed, **kw):
        if seed in self.bad:
            hook = kw.get("on_eval")

            def boom(history):
                if len(history) >= 7:
                    raise RuntimeError("planted crash after 7 calls")
                return hook(history) if hook else False
            kw = dict(kw, on_eval=boom)
        return self.inner.optimize(problem, budget=budget, seed=seed, **kw)


def test_a_crashed_lane_is_replaced_once(monkeypatch):
    _pin_lanes(monkeypatch)

    class B(_Crash):
        bad = {1}
    real = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algo: B(real(algo)))
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, hist, log = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sd = rep["convergence"]["seeds"]
    assert [s["seed"] for s in sd["slots"]] == [0, 4, 2] and sd["slots"][1]["replaced"]["calls"] == 7
    assert sd["total_search_calls"] == rep["phase_counts"]["search"]["n"] == 3 * 20 + 7
    assert any("re-run once with seed 4" in w for w in rep["budget_plan"]["warnings"])


def test_a_lane_process_that_dies_is_a_crashed_attempt(monkeypatch):
    _pin_lanes(monkeypatch)

    class Die(_Crash):
        def optimize(self, problem, budget, seed, **kw):
            if seed == 2:
                hook = kw.get("on_eval")

                def die(history):
                    if len(history) >= 5:
                        os._exit(3)                       # the lane process dies without a word
                    return hook(history) if hook else False
                kw = dict(kw, on_eval=die)
            return self.inner.optimize(problem, budget=budget, seed=seed, **kw)
    real = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algo: Die(real(algo)))
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, hist, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    s3 = rep["convergence"]["seeds"]["slots"][2]
    assert s3["replaced"]["seed"] == 2 and "ended without a result" in s3["replaced"]["error"]
    assert s3["replaced"]["calls"] == 5 and s3["seed"] == 5
    assert rep["status"] == "completed"


def test_a_caller_given_runner_keeps_the_seeds_in_one_lane(monkeypatch):
    _pin_lanes(monkeypatch)
    from calibration_kit import test_step1_panel as S1
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, _, _ = _calib(ki, wd, run_model=S1._Runner(wd))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["convergence"]["seeds"]["seeds_parallel"] == 1 and rep["budget_plan"]["waves"] == 3


def _spans(log):
    by = {}
    for c in [json.loads(l) for l in Path(log).read_text().splitlines()]:
        if "_seed_lanes" in c["wd"]:
            by.setdefault(Path(c["wd"]).name, []).append((c["t0"], c["t1"]))
    return {k: (min(a for a, _ in v), max(b for _, b in v)) for k, v in by.items()}


def _max_overlap(spans):
    ev = sorted([(a, 1) for a, _ in spans.values()] + [(b, -1) for _, b in spans.values()])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def test_two_lanes_for_three_seeds_never_run_three_at_once(monkeypatch):
    _pin_lanes(monkeypatch, lanes=2)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, log = _setup(tmp)
        rep, hist, _ = _calib(ki, wd)
        spans = _spans(log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bp = rep["budget_plan"]
    assert rep["convergence"]["seeds"]["seeds_parallel"] == 2 and bp["waves"] == 2
    assert bp["efficiency"] == pytest.approx(0.95)                     # e measured at 2 lanes
    search = [h for h in hist if h["phase"] == "search"]
    assert {h["lane"] for h in search} == {1, 2} and {h["slot"] for h in search} == {1, 2, 3}
    assert len(spans) == 3 and _max_overlap(spans) == 2                # at most min(S, L) at once
    s3 = [v for k, v in spans.items() if k.startswith("slot3")][0]
    assert s3[0] >= min(v[1] for k, v in spans.items() if not k.startswith("slot3"))



def test_a_lanes_warnings_reach_the_plan(monkeypatch):
    """The live time check runs inside each lane: its warning must come back to the parent's plan."""
    _pin_lanes(monkeypatch)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"]["budget"]["allowance"] = "0.01s"
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=None, budget=20, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["convergence"]["seeds"]["seeds_parallel"] == 3
    w = [x for x in rep["budget_plan"]["warnings"] if "share of the allowance is 0 s" in x]
    assert sorted(x.split(":")[0] for x in w) == ["seed 0", "seed 1", "seed 2"]



def test_a_replacement_gets_its_own_eval_ids_and_takes_the_next_free_lane(monkeypatch):
    _pin_lanes(monkeypatch, lanes=2)

    class B(_Crash):
        bad = {0}
    real = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algo: B(real(algo)))
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, log = _setup(tmp)
        rep, _, _ = _calib(ki, wd)
        calls = [json.loads(l) for l in log.read_text().splitlines()]
        spans = _spans(log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ids = [c["eid"] for c in calls if "_seed_lanes" in c["wd"]]
    assert len(ids) == len(set(ids))                                   # the crashed attempt's ids are not reused
    assert [s["seed"] for s in rep["convergence"]["seeds"]["slots"]] == [3, 1, 2]
    assert _max_overlap(spans) <= 2 and len(spans) == 4
    # QUEUE, not waves (Opus 6b r2 #7): slot 3 starts in the lane the crash freed, before slot 2 ends; the
    # replacement starts as soon as a lane is free, before the later of slots 2 and 3 ends
    sp = {k.split("_")[0] + "_" + k.split("_")[1]: v for k, v in spans.items()}
    assert sp["slot3_seed2"][0] < sp["slot2_seed1"][1]
    assert sp["slot1_seed3"][0] < max(sp["slot2_seed1"][1], sp["slot3_seed2"][1])


def test_lanes_are_not_used_when_the_process_has_other_threads(monkeypatch):
    import threading
    _pin_lanes(monkeypatch)
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, _, _ = _calib(ki, wd)
    finally:
        stop.set()
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["convergence"]["seeds"]["seeds_parallel"] == 1
    assert any("Python threads" in w for w in rep["budget_plan"]["warnings"])


def test_a_second_calibration_in_a_busy_workdir_is_refused():
    import fcntl
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        held = open(Path(wd, ".kdt_calibrate.lock"), "a")
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=None, budget=None, seed=0)
        held.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["status"] == "workdir_busy"


@pytest.mark.parametrize("threaded", [False, True])        # True: this call runs its seeds one after another
def test_a_killed_runs_lane_folders_are_merged_and_removed(monkeypatch, threaded):
    import threading
    _pin_lanes(monkeypatch)
    stop = threading.Event()
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        a, _, _ = _calib(ki, wd)                                        # a full run: its cache lines
        cache = sorted(Path(wd).glob("eval_metrics_cache_*"))[0]
        lines = cache.read_text().splitlines(keepends=True)
        old = Path(wd).parent / f"{Path(wd).name}_seed_lanes_dead" / "slot1_seed0"
        old.mkdir(parents=True)
        (old / cache.name).write_text("".join(lines) + '{"key": "torn')   # plus a torn last line
        cache.write_text("")                                            # the main cache lost them
        if threaded:
            threading.Thread(target=stop.wait, daemon=True).start()
        b, hist, _ = _calib(ki, wd)
        left = [p for p in Path(wd).parent.iterdir() if "_seed_lanes" in p.name]
    finally:
        stop.set()
        shutil.rmtree(tmp, ignore_errors=True)
    assert left == []
    assert b["convergence"]["seeds"]["seeds_parallel"] == (1 if threaded else 3)
    assert any("of an earlier, killed calibration" in w for w in b["budget_plan"]["warnings"])
    if threaded:                                       # the resume says why it runs fewer lanes than its plan
        assert any("this resume runs 1 seed(s) at a time; the saved plan (and its cap) counted 3" in w
                   for w in b["budget_plan"]["warnings"])
    search = [h for h in hist[-b["convergence"]["seeds"]["total_search_calls"]:] if h["phase"] == "search"]
    assert search and all(h["cache_hit"] for h in search)             # the dead lanes' work was kept
    assert _slots(a) == _slots(b)


def test_a_contract_edited_during_the_run_still_merges_the_lane_caches(monkeypatch):
    holder = {}

    def edit():
        p = Path(holder["ki"], "calibration.yaml")
        p.write_text(p.read_text() + "\n# edited while the run was going\n")
    _pin_lanes(monkeypatch, edit_contract=edit)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        holder["ki"] = ki
        rep, _, _ = _calib(ki, wd)
        n_cache = sum(len(f.read_text().splitlines()) for f in Path(wd).glob("eval_metrics_cache_*"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["train_metrics"] and n_cache > 3 * 20 * 0.5
    assert not any("no cache lines to merge" in w for w in rep["budget_plan"]["warnings"])


def test_a_lane_that_cannot_be_set_up_runs_its_seed_in_the_process(monkeypatch):
    _pin_lanes(monkeypatch)

    def broken(*a, **k):
        raise OSError("no space left on device (planted)")
    monkeypatch.setattr(compute, "clone_workdir", broken)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, _, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sd = rep["convergence"]["seeds"]
    assert rep["status"] == "completed" and [s["seed"] for s in sd["slots"]] == [0, 1, 2]
    assert not any(s["crashed"] or s["replaced"] for s in sd["slots"])
    assert sum("could not be set up" in w for w in rep["budget_plan"]["warnings"]) == 3
    assert sd["seeds_parallel"] == 1 and sd["run_one_after_another"]        # what really ran


def test_a_seed_run_in_the_process_after_a_failed_set_up_is_still_replaced_when_it_crashes(monkeypatch):
    _pin_lanes(monkeypatch)

    def broken(*a, **k):
        raise OSError("no space left on device (planted)")
    monkeypatch.setattr(compute, "clone_workdir", broken)

    class B(_Crash):
        bad = {0}
    real = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algo: B(real(algo)))
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, _, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    s1 = rep["convergence"]["seeds"]["slots"][0]
    assert s1["seed"] == 3 and s1["replaced"]["seed"] == 0 and not s1["crashed"]


def test_no_lane_folder_can_be_made_so_the_seeds_run_one_after_another(monkeypatch):
    _pin_lanes(monkeypatch)
    real_mk = C.tempfile.mkdtemp

    def mk(*a, **k):
        if "_seed_lanes_" in str(k.get("prefix", "")):
            raise PermissionError("read-only folder (planted)")
        return real_mk(*a, **k)
    monkeypatch.setattr(C.tempfile, "mkdtemp", mk)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, _, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sd = rep["convergence"]["seeds"]
    assert rep["status"] == "completed" and [s["seed"] for s in sd["slots"]] == [0, 1, 2]
    assert sd["seeds_parallel"] == 1 and any("lane folder could not be made" in w for w in rep["budget_plan"]["warnings"])


def test_a_kept_exception_does_not_keep_the_workdir_lock():
    """Opus 6b r2 #6: the lock is closed in a finally, not when Python frees calibrate()'s frame."""
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp, parallel_safe=False, seeds=1)

        def interrupted():
            raise KeyboardInterrupt
        kept = None
        try:
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=interrupted, budget=None, seed=0)
        except KeyboardInterrupt as e:
            kept = e                                     # like the REPL's sys.last_value
        assert kept is not None
        rep, _, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["status"] != "workdir_busy"


def test_the_probe_measures_at_the_seed_count_or_the_plan_says_it_did_not(monkeypatch):
    PROBE_KS.clear()
    _pin_lanes(monkeypatch, lanes=4, eff={1: 1.0, 2: 0.95, 4: 0.8})
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, _, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert PROBE_KS and 3 in PROBE_KS[-1]
    bp = rep["budget_plan"]
    assert rep["convergence"]["seeds"]["seeds_parallel"] == 3 and bp["efficiency"] == pytest.approx(0.8)
    assert any("not measured at 3 lanes" in w for w in bp["warnings"])


LANE_PRINT_PY = """
import contextlib, io, json, os, sys, tempfile, shutil
sys.path.insert(0, {root!r})
from calibration_kit import calib as C, compute
from calibration_kit import test_step6b_lanes as T
from calibration_kit import test_calibrate_convergence_e2e as E
class MP:
    def setattr(self, obj, name, val):
        setattr(obj, name, val)
T._pin_lanes(MP())
tmp = tempfile.mkdtemp(prefix="kdt_s6b_p_")
ki, wd, _ = T._setup(tmp)
with E._clean_env():
    rep = C.calibrate(ki, wd, {{"streamflow": "point_time_series"}}, run_model=None, budget=None, seed=0)
print("STATUS", rep["status"], rep["convergence"]["seeds"]["seeds_parallel"])
shutil.rmtree(tmp, ignore_errors=True)
"""


def test_what_a_lane_prints_reaches_the_output(tmp_path):
    import subprocess
    script = tmp_path / "run_lanes.py"
    script.write_text(LANE_PRINT_PY.format(root=str(ROOT)))
    out = tmp_path / "out.txt"
    with open(out, "w") as fh:
        subprocess.run([sys.executable, str(script)], stdout=fh, stderr=subprocess.STDOUT, timeout=900)
    txt = out.read_text()
    assert "STATUS completed 3" in txt, txt[-2000:]
    # the lanes' own search output (spotpy's progress / start lines) is there, not lost at os._exit
    assert txt.count("maximal objective function") + txt.count("Initializing the") >= 3, txt[-2000:]


ORPHAN_PY = """
import sys, tempfile
sys.path.insert(0, {root!r})
from calibration_kit import calib as C
from calibration_kit import test_step6b_lanes as T
from calibration_kit import test_calibrate_convergence_e2e as E
class MP:
    def setattr(self, obj, name, val):
        setattr(obj, name, val)
T._pin_lanes(MP())
if {handler!r}:                                  # a host that handles SIGTERM itself ("stop soon")
    import signal
    signal.signal(signal.SIGTERM, lambda *a: None)
ki, wd, log = T._setup({tmp!r}, cap=400)
with E._clean_env():
    C.calibrate(ki, wd, {{"streamflow": "point_time_series"}}, run_model=None, budget=None, seed=0)
"""


@pytest.mark.parametrize("handler", [False, True])
def test_the_lanes_of_a_killed_parent_stop_and_free_the_workdir(tmp_path, handler):
    import signal
    import subprocess
    import time as _t
    work = tmp_path / "work"
    work.mkdir()
    script = tmp_path / "orphan.py"
    script.write_text(ORPHAN_PY.format(root=str(ROOT), tmp=str(work), handler=handler))
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = work / "calls.log"
    t_end = _t.time() + 300
    while _t.time() < t_end:
        n = sum(1 for l in (log.read_text().splitlines() if log.exists() else []) if "_seed_lanes" in l)
        if n >= 15:
            break
        _t.sleep(0.5)
    assert n >= 15, "the lanes never started"
    p.send_signal(signal.SIGKILL)
    p.wait()
    _t.sleep(5)                                    # a runner call in flight may still finish
    n1 = len(log.read_text().splitlines())
    _t.sleep(6)
    n2 = len(log.read_text().splitlines())
    assert n2 == n1, f"lanes kept running after their parent died ({n2 - n1} more model runs)"
    # and the workdir is free again: a resume is not refused as busy
    import fcntl
    lock = open(work / "wd" / ".kdt_calibrate.lock", "a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)      # raises if a lane still holds it
    lock.close()



LOCKING_RUN_PY = """
import fcntl, json, os, sys
sys.path.insert(0, {root!r})
from calibration_kit import test_calibrate_convergence_e2e as E
out, wd = sys.argv[1], sys.argv[2]
lk = open(os.path.join(wd, ".kdt_calib.lock"), "a")      # what the HBV / Noah-MP runners do per evaluation
fcntl.flock(lk.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # fails at once if the kit held this file
json.dump(E.Runner(wd)(), open(out, "w"), default=float)
"""


def test_the_kits_workdir_lock_does_not_block_a_runner_that_locks_its_own_file():
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1, "pilot_runs": 3}, None, 10)
        Path(tmp, "run.py").write_text(LOCKING_RUN_PY.format(root=str(ROOT)))
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["runner"] = {"kind": "subprocess", "command": [sys.executable, str(Path(tmp, "run.py")), "{metrics_json}",
                                                        "{workdir}"]}
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        rep, hist, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["status"] == "completed", rep.get("reason")
    assert all(h["ok"] for h in hist if h["phase"] == "search")


def test_each_calibration_gets_its_own_lane_folder(monkeypatch):
    """Lane folders are unique per calibrate() call (mkdtemp), never a fixed name another call could share."""
    _pin_lanes(monkeypatch)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, log = _setup(tmp)
        roots = []
        for _ in range(2):
            for c in Path(wd).glob("eval_metrics_cache_*"):
                c.unlink()                                              # so the second call runs again
            n0 = len(log.read_text().splitlines()) if log.exists() else 0
            _calib(ki, wd)
            new = [json.loads(l) for l in log.read_text().splitlines()[n0:]]
            roots.append({str(Path(r["wd"]).parent) for r in new if "_seed_lanes" in r["wd"]})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert len(roots[0]) == 1 and len(roots[1]) == 1
    assert roots[0] != roots[1]


def test_ctrl_c_keeps_the_calls_the_lanes_finished(tmp_path):
    """Opus 6b r2 #2: an interrupt ends the lanes and their model runs, and their finished calls are merged
    into the main log and cache (as a kill -9 leaves them for the next call's leftover merge)."""
    import signal
    import subprocess
    import time as _t
    work = tmp_path / "work"
    work.mkdir()
    script = tmp_path / "ctrlc.py"
    script.write_text(ORPHAN_PY.format(root=str(ROOT), tmp=str(work), handler=False))
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)          # its own group, like a terminal's foreground job
    log = work / "calls.log"
    t_end = _t.time() + 300
    n = 0
    while _t.time() < t_end:
        n = sum(1 for l in (log.read_text().splitlines() if log.exists() else []) if "_seed_lanes" in l)
        if n >= 30:
            break
        _t.sleep(0.5)
    assert n >= 30, "the lanes never started"
    assert p.pid > 1 and os.getpgid(p.pid) == p.pid
    os.killpg(p.pid, signal.SIGINT)                     # what Ctrl-C sends: the whole group
    p.wait(timeout=120)
    _t.sleep(3)
    n1 = len(log.read_text().splitlines())
    _t.sleep(5)
    assert len(log.read_text().splitlines()) == n1, "model runs went on after the interrupt"
    lane_calls = sum(1 for l in log.read_text().splitlines() if "_seed_lanes" in l)
    wd = work / "wd"
    cache_lines = sum(len(c.read_text().splitlines()) for c in wd.glob("eval_metrics_cache_*"))
    search_rows = sum(1 for l in (wd / "eval_history.jsonl").read_text().splitlines() if '"search"' in l)
    assert cache_lines >= lane_calls - 3 and search_rows >= lane_calls - 3      # at most the 3 in flight lost
    assert [q for q in work.iterdir() if "_seed_lanes" in q.name] == []


# ── review round 3 (Opus 6b) ────────────────────────────────────────────────────────────────────
SLOWKILL_PY = """
import sys, tempfile
sys.path.insert(0, {root!r})
from pathlib import Path
from calibration_kit import calib as C
from calibration_kit import test_step6b_lanes as T
from calibration_kit import test_calibrate_convergence_e2e as E
class MP:
    def setattr(self, obj, name, val):
        setattr(obj, name, val)
T._pin_lanes(MP())
ki, wd, log = T._setup({tmp!r}, cap=40)
Path({tmp!r}, "run.py").write_text(T.SLOW_RUN_PY.format(root={root!r}, tmp={tmp!r}))
with E._clean_env():
    rep = C.calibrate(ki, wd, {{"streamflow": "point_time_series"}}, run_model=None, budget=None, seed=0)
print("STATUS", rep["status"], flush=True)
"""

SLOW_RUN_PY = """
import json, os, sys, time
sys.path.insert(0, {root!r})
from calibration_kit import test_calibrate_convergence_e2e as E
out, wd, log = sys.argv[1], sys.argv[2], sys.argv[3]
arm = os.path.join({tmp!r}, "arm")
if "_seed_lanes" in wd:
    try:
        os.rename(arm, arm + ".taken")                      # exactly one lane call takes it
        open(os.path.join({tmp!r}, "slow.json"), "w").write(json.dumps({{"pid": os.getpid(), "ppid": os.getppid()}}))
        time.sleep(90)                                      # a long model call, in flight when its lane is killed
    except FileNotFoundError:
        pass
m = E.Runner(wd)()
st = open("/proc/self/status").read()
ign = int([l for l in st.splitlines() if l.startswith("SigIgn:")][0].split()[1], 16)
open(log, "a").write(json.dumps({{"wd": wd, "t0": 0, "t1": 0, "eid": None, "sigign": ign}}) + "\\n")
json.dump(m, open(out, "w"), default=float)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc signal records and parent-death handling")
def test_a_lane_killed_from_outside_takes_its_model_run_with_it(tmp_path):
    """Opus 6b r3 #1: the OOM killer ends a LANE (SIGKILL, no handler runs); its model run in flight must not go
    on beside the replacement lane."""
    import signal
    import subprocess
    import time as _t
    work = tmp_path / "work"
    work.mkdir()
    script = tmp_path / "slowkill.py"
    script.write_text(SLOWKILL_PY.format(root=str(ROOT), tmp=str(work)))
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = work / "calls.log"
    t_end = _t.time() + 300
    while _t.time() < t_end and not (log.exists() and sum("_seed_lanes" in l for l in log.read_text().splitlines()) >= 6):
        _t.sleep(0.3)
    (work / "arm").write_text("")
    while _t.time() < t_end and not (work / "slow.json").exists():
        _t.sleep(0.2)
    slow = json.loads((work / "slow.json").read_text())
    assert slow["ppid"] > 1 and slow["ppid"] != os.getpid()
    os.kill(slow["ppid"], signal.SIGKILL)                     # the lane, as the OOM killer would
    _t.sleep(4)
    with pytest.raises(ProcessLookupError):
        os.kill(slow["pid"], 0)                              # its model run is gone too
    out, _ = p.communicate(timeout=600)
    assert "STATUS completed" in out, out[-2000:]


@pytest.mark.skipif(sys.platform != "linux", reason="Runner inspects Linux /proc/self/status")
def test_model_runs_in_a_lane_keep_the_default_ctrl_c(monkeypatch):
    """Opus 6b r3 #2: a lane must not hand an IGNORED SIGINT to the model runs it starts (exec keeps SIG_IGN)."""
    _pin_lanes(monkeypatch)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, log = _setup(tmp)
        Path(tmp, "run.py").write_text(SLOW_RUN_PY.format(root=str(ROOT), tmp=tmp))
        rep, _, _ = _calib(ki, wd)
        calls = [json.loads(l) for l in log.read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lane = [c for c in calls if "_seed_lanes" in c["wd"]]
    assert rep["convergence"]["seeds"]["seeds_parallel"] == 3 and lane
    assert all(not (c["sigign"] & (1 << (2 - 1))) for c in lane)          # SIGINT (2) is not ignored


def test_the_leftover_search_treats_the_workdir_name_literally():
    """Opus 6b r3 #3: workdir `run[1]` must neither remove `run1_seed_lanes_*` (another workdir's) nor miss its own."""
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, _ = E._fixture(tmp, {"seeds": 1}, None, 12)
        wd = Path(tmp, "run[1]")
        wd.mkdir()
        other = Path(tmp, "run1_seed_lanes_abc")
        other.mkdir()
        (other / "keep.txt").write_text("another workdir's lane folder")
        own = Path(tmp, "run[1]_seed_lanes_xyz")
        own.mkdir()
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            from calibration_kit import test_step1_panel as S1
            rep = C.calibrate(ki, str(wd), {"streamflow": "point_time_series"}, run_model=S1._Runner(str(wd)),
                              budget=None, seed=0)
        assert other.exists() and not own.exists()
        assert any("run[1]_seed_lanes_xyz" in w for w in rep["budget_plan"]["warnings"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _state(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return "?"


@pytest.mark.parametrize("orphaned", [False, True])
@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc process-state records")
def test_ctrl_z_pauses_a_shell_job_and_never_freezes_an_orphaned_one(tmp_path, orphaned):
    """Opus 6b r3 #4 / r4 #1. A shell job (its own group inside the caller's session): Ctrl-Z stops the main
    process AND the lanes; fg resumes them. An orphaned group (setsid, ssh -t, docker exec): the kernel drops the
    stop for the main process, so the lanes must keep running too — never stopped with nothing to resume them."""
    import signal
    import subprocess
    import time as _t
    work = tmp_path / "work"
    work.mkdir()
    script = tmp_path / "ctrlz.py"
    script.write_text(ORPHAN_PY.format(root=str(ROOT), tmp=str(work), handler=False))
    kw = {"start_new_session": True} if orphaned else {"process_group": 0}
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    log = work / "calls.log"

    def n_lane():
        return sum(1 for l in (log.read_text().splitlines() if log.exists() else []) if "_seed_lanes" in l)
    try:
        t_end = _t.time() + 300
        while _t.time() < t_end and n_lane() < 15:
            _t.sleep(0.5)
        assert n_lane() >= 15, "the lanes never started"
        assert p.pid > 1 and os.getpgid(p.pid) == p.pid
        os.killpg(p.pid, signal.SIGTSTP)                       # Ctrl-Z
        _t.sleep(2)
        n1 = n_lane()
        _t.sleep(5)
        if orphaned:
            assert _state(p.pid) != "T" and n_lane() > n1, "an orphaned job's lanes were frozen"
        else:
            assert _state(p.pid) == "T", "the main process did not stop"
            assert n_lane() == n1, "the lanes kept running after Ctrl-Z"
            os.killpg(p.pid, signal.SIGCONT)                   # fg
            _t.sleep(6)
            assert n_lane() > n1, "the lanes did not resume"
    finally:
        try:
            os.killpg(p.pid, signal.SIGCONT)
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=120)
        except Exception:
            p.kill()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires util-linux script command options")
def test_lanes_do_not_stop_on_a_terminal_with_tostop(tmp_path):
    """Opus 6b r4 #2: a lane is a background group for the terminal; with `stty tostop` its first print would stop
    it and the run would wait forever. Run with lanes inside a real pty with tostop, with a time limit."""
    import shutil as _sh
    import subprocess
    if not _sh.which("script"):
        pytest.skip("util-linux `script` not available")
    py = tmp_path / "run_lanes.py"
    py.write_text(LANE_PRINT_PY.format(root=str(ROOT)))
    out = tmp_path / "typescript.txt"
    # the lanes print to the TERMINAL (script's pty), which is what tostop acts on; `script` records it
    cmd = f"stty tostop; {sys.executable} {py}"
    try:
        r = subprocess.run(["script", "-qec", cmd, str(out)], stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=600)
    except subprocess.TimeoutExpired:
        pytest.fail("the run with lanes hung on a tostop terminal")
    txt = out.read_text(errors="replace") if out.exists() else ""
    assert "STATUS completed 3" in txt, (r.returncode, txt[-1500:])
    assert "maximal objective function" in txt or "Initializing the" in txt      # the lanes' own prints got out


def test_seeds_parallel_counts_only_lanes_that_really_searched_together(monkeypatch):
    """Opus 6b r4 #5: 2 of 3 lanes fail set-up; one lane searched, then two seeds ran in this process after it —
    only one search at a time, so seeds_parallel is 1."""
    _pin_lanes(monkeypatch)
    real = compute.clone_workdir

    def some_fail(src, dst, *a, **k):
        if "slot1_" not in str(dst):
            raise OSError("no space left on device (planted)")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(compute, "clone_workdir", some_fail)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, _, _ = _calib(ki, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sd = rep["convergence"]["seeds"]
    assert rep["status"] == "completed" and [s["seed"] for s in sd["slots"]] == [0, 1, 2]
    assert sd["seeds_parallel"] == 1 and sd["run_one_after_another"]


def _sigblk():
    return int([l for l in open("/proc/self/status").read().splitlines() if l.startswith("SigBlk:")][0].split()[1], 16)


def _pool_task(tmp):
    ki, wd, _ = _setup(tmp)
    before = _sigblk()
    try:
        rep, _, _ = _calib(ki, wd)
        return rep["status"], rep["convergence"]["seeds"]["seeds_parallel"], before, _sigblk(), None
    except BaseException as e:                                  # report, do not hang the pool
        return None, None, before, _sigblk(), f"{type(e).__name__}: {e}"


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc/self/status signal masks")
def test_a_pool_worker_runs_its_seeds_one_after_another_and_keeps_its_signal_mask(monkeypatch):
    """Opus 6b r5 #1: a daemonic process (a multiprocessing.Pool worker) cannot start lanes; calibrate() falls back to
    seeds one after another, and never leaves Ctrl-C / Ctrl-Z blocked in the worker."""
    import multiprocessing as mp
    _pin_lanes(monkeypatch)                                      # the fork copies the pinned probe into the worker
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        with mp.get_context("fork").Pool(1) as pool:
            status, par, before, after, err = pool.apply(_pool_task, (tmp,))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert err is None, err
    assert status == "completed" and par == 1
    assert after == before


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc/self/status signal masks")
def test_an_unexpected_error_from_a_lanes_start_restores_the_signal_mask(monkeypatch):
    """codex 6b r1 #2: not an OSError (which falls back to in-process) — any other error out of Process.start()
    must still put the caller's signal mask back (Ctrl-C / Ctrl-Z are blocked only around the fork)."""
    import multiprocessing.context as mpc
    _pin_lanes(monkeypatch)

    def boom(self):
        raise RuntimeError("planted: start() failed")
    monkeypatch.setattr(mpc.ForkProcess, "start", boom)
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        before = _sigblk()
        with pytest.raises(RuntimeError, match="planted"):
            _calib(ki, wd)
        after = _sigblk()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert after == before


def test_the_machine_probe_is_skipped_when_lanes_cannot_be_used(monkeypatch):
    """codex 6b r1 #1: with a caller-given run_model (or one seed, a thread, a daemonic worker …) lanes are never
    used, so the probe must not spend model runs measuring them."""
    calls = []
    real = compute.probe_lanes

    def spy(pilot, **kw):
        calls.append(kw.get("clone_and_eval") is not None)
        return real(pilot, **kw)
    monkeypatch.setattr(compute, "probe_lanes", spy)
    from calibration_kit import test_step1_panel as S1
    tmp = tempfile.mkdtemp(prefix="kdt_s6b_")
    try:
        ki, wd, _ = _setup(tmp)
        rep, hist, _ = _calib(ki, wd, run_model=S1._Runner(wd))       # a caller-given runner: no lanes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert calls == [False]                                             # probed without clones: no lane runs
    assert not [h for h in hist if h["phase"] == "machine_probe"]
    assert any("machine probe skipped (the caller passed its own run_model)" in w for w in rep["budget_plan"]["warnings"])
