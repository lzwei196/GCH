"""Build step 5 (design §1.3; gaps 2k, 2m, 2i, 2v): the pilot always on, the machine probe, the budget
plan with ONE allowance ledger, t_slow, the pilot size, the live time check and the refusal below 20."""
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit.budget import plan_budget, t_slow_from, FINAL_OVERHEAD_RUNS          # noqa: E402
from calibration_kit.compute import concurrency_test, make_clone_and_eval, probe_lanes    # noqa: E402
from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402
from calibration_kit import test_step1_panel as S1                                        # noqa: E402

PILOT = {"median_s": 8.0, "p90_s": 10.0, "max_s": 11.0, "fail_rate": 0.0}


# ── t_slow and the ledger (§1.3) ──────────────────────────────────────────────────────────────────
def test_t_slow_from_the_pilot_or_the_users_run_time():
    assert t_slow_from(PILOT) == pytest.approx(12.0)                       # 1.2 x p90
    assert t_slow_from(PILOT, run_time_s=20.0) == pytest.approx(24.0)      # 1.2 x the user's time
    assert t_slow_from(PILOT, run_time_s=5.0) == pytest.approx(13.2)       # 1.2 x the slowest of the runs


def test_shared_and_final_overhead_are_charged_once_not_per_seed():
    # 3 seeds in 1 lane: 3 waves; 600 s shared, 120 s final, t_slow 12 s, 3 h allowance
    p = plan_budget("3h", PILOT, lanes=1, efficiency=1.0, n_params=4, n_seeds=3,
                    shared_overhead_s=600.0, final_overhead_s=120.0)
    assert p["cap_per_seed"] == math.floor((10800 - 600 - 120) / 3 * 1.0 / 12.0)      # 280
    per_seed_charged = math.floor((10800 / 3 - 600 - 120) * 1.0 / 12.0)               # the wrong way: 240
    assert p["cap_per_seed"] != per_seed_charged
    L = p["ledger"]
    assert (L["shared_overhead_s"], L["final_overhead_s"], L["search_s"]) == (600.0, 120.0, 10080.0)
    assert L["per_seed_search_s"] == pytest.approx(3360.0)


def test_final_overhead_defaults_to_five_slow_runs_and_efficiency_scales_the_cap():
    p = plan_budget("1h", PILOT, lanes=2, efficiency=0.5, n_params=4, n_seeds=2, shared_overhead_s=0.0)
    assert p["ledger"]["final_overhead_s"] == pytest.approx(FINAL_OVERHEAD_RUNS * 12.0)
    assert p["waves"] == 1 and p["cap_per_seed"] == math.floor((3600 - 60) / 1 * 0.5 / 12.0)


def test_a_measured_cap_under_20_is_refused_a_fixed_one_is_not():
    p = plan_budget("312s", PILOT, lanes=1, efficiency=1.0, n_params=4, n_seeds=1, shared_overhead_s=0.0)
    assert p["cap_per_seed"] == 21 and not p["refuse"]                     # floor((312-60)/12) = 21
    q = plan_budget("290s", PILOT, lanes=1, efficiency=1.0, n_params=4, n_seeds=1, shared_overhead_s=0.0)
    assert q["cap_per_seed"] == 19 and q["refuse"] and "under 20" in q["refuse_reason"]
    f = plan_budget(None, PILOT, lanes=1, efficiency=1.0, n_params=4, mode="fixed", max_evaluations=10)
    assert f["cap_per_seed"] == 10 and not f["refuse"]


# ── through calibrate() ───────────────────────────────────────────────────────────────────────────
class _Sleepy:
    """The fixture runner, with a real sleep per run: fixture runs take < 1 ms, which rounds to 0 s in
    phase_counts and leaves timing tests to chance (Opus round 2)."""
    SLEEP = 0.015

    def __init__(self, wd):
        self.inner = S1._Runner(wd)

    def __call__(self):
        time.sleep(self.SLEEP)
        return self.inner()


class _Sleepy20(_Sleepy):
    SLEEP = 0.02


def _calib(budget_block, budget=None, max_evaluations=100000, runner=None, strategy=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s5_")
    try:
        ki, wd = E._fixture(tmp, budget_block, None, max_evaluations)
        if strategy:
            c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strategy)
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        buf = io.StringIO()
        with E._clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=(runner or S1._Runner)(wd), budget=budget, seed=0)
        return rep, buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_ledger_reproduces_the_cap_in_a_real_run():
    """Gap 2v: shared overhead (pilot + commissioning + proof + machine probe, measured) and the final
    overhead are subtracted ONCE; the reported numbers give back the cap exactly."""
    rep, _ = _calib({"mode": "measured", "allowance": "6s", "pilot_runs": 10, "seeds": 3}, runner=_Sleepy)
    bp = rep["budget_plan"]; L = bp["ledger"]
    assert set(L["shared_parts_s"]) >= {"commission", "consume_proof", "pilot", "machine_probe"}
    assert L["shared_overhead_s"] == pytest.approx(sum(L["shared_parts_s"].values()), abs=1e-9)
    for ph in ("commission", "consume_proof", "pilot"):            # each measured phase, exactly
        assert L["shared_parts_s"][ph] == pytest.approx(rep["phase_counts"].get(ph, {}).get("wall_s", 0.0), abs=1e-9)
    for ph in ("commission", "consume_proof", "pilot"):            # each one really charged
        assert L["shared_parts_s"][ph] >= _Sleepy.SLEEP
    expect = math.floor((L["allowance_s"] - L["shared_overhead_s"] - L["final_overhead_s"])
                        / bp["waves"] * bp["efficiency"] / bp["t_slow_s"])
    assert bp["waves"] == 3
    assert bp["cap_per_seed"] == expect                                # the ledger is unrounded
    per_seed_charged = math.floor((L["allowance_s"] / bp["waves"] - L["shared_overhead_s"] - L["final_overhead_s"])
                                  * bp["efficiency"] / bp["t_slow_s"])
    assert bp["cap_per_seed"] > per_seed_charged + 1                   # overhead is NOT charged per seed
    assert L["final_overhead_s"] == pytest.approx(FINAL_OVERHEAD_RUNS * bp["t_slow_s"], rel=1e-12)


def test_the_pilot_runs_even_when_the_caller_gives_the_budget():
    rep, _ = _calib(None, budget=25)
    assert rep["phase_counts"]["pilot"]["n"] == 10
    assert rep["budget_used"]["cap"] == 25


def test_a_given_run_time_means_a_3_run_pilot_and_t_slow_from_it():
    rep, _ = _calib({"mode": "measured", "allowance": "10m", "run_time": "2s"})
    bp = rep["budget_plan"]
    assert rep["phase_counts"]["pilot"]["n"] == 3
    assert bp["run_time_given_s"] == 2.0 and bp["t_slow_s"] == pytest.approx(2.4, rel=1e-3)


def test_a_too_small_allowance_is_refused_before_the_search():
    rep, _ = _calib({"mode": "measured", "allowance": "1s", "pilot_runs": 4}, runner=E._slow_runner)
    assert rep["status"] == "budget_exhausted" and "under 20" in rep["reason"]
    assert "search" not in rep["phase_counts"]


def test_the_live_time_check_warns_when_the_projected_finish_passes_the_allowance():
    class _Slow:
        def __init__(self, wd):
            self.inner = S1._Runner(wd)

        def __call__(self):
            time.sleep(0.02)
            return self.inner()
    # the allowance gives the seed ~0 s of search time; the caller's budget of 30 calls overruns it
    rep, log = _calib({"mode": "measured", "allowance": "0.6s", "pilot_runs": 3}, budget=30, runner=_Slow)
    assert rep["status"] == "completed"
    assert any("share of the allowance" in w for w in rep["budget_plan"]["warnings"])
    assert "share of the allowance" in log


def test_without_parallel_safe_true_there_is_no_concurrency_test():
    rep, _ = _calib({"mode": "measured", "allowance": "10m"}, budget=25)
    bp = rep["budget_plan"]
    assert bp["lanes"] == 1 and any("parallel_safe is not true" in n for n in bp["lane_notes"])


# ── the machine probe with a real subprocess runner (gap 2m) ─────────────────────────────────────
RUNNER_PY = """
import json, os, sys, time
p = json.load(open(os.environ["KDT_CALIB_PARAMS"]))
assert os.environ.get("KDT_CALIB_SPLIT") == "calibration"
time.sleep(0.3)
json.dump({"nse": 1.0 - (p["a"] - 1.0) ** 2}, open(sys.argv[1], "w"))
"""


# A contract runner for tests of the machine probe (codex 6b r1 #1: the probe runs only when lanes could be used —
# no caller run_model, several seeds): inside a probe copy it runs `probe_body`; for the search it is the full
# fixture runner (applied params, case, split).
PROBE_AND_SEARCH_PY = """
import json, os, sys, time
sys.path.insert(0, {root!r})
out, wd = sys.argv[1], sys.argv[2]
if "kdt_lane_probe_" in out:
{probe_body}
else:
    from calibration_kit import test_calibrate_convergence_e2e as E
    json.dump(E.Runner(wd)(), open(out, "w"), default=float)
"""
PROBE_SLEEP = "    time.sleep(0.3)\n    json.dump({'nse': 0.5}, open(out, 'w'))"


def probe_contract(tmp, ki, probe_body=PROBE_SLEEP):
    """Make the fixture contract's runner a subprocess running PROBE_AND_SEARCH_PY."""
    root = str(Path(__file__).resolve().parents[1])
    Path(tmp, "run.py").write_text(PROBE_AND_SEARCH_PY.format(root=root, probe_body=probe_body))
    c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
    c["runner"] = {"kind": "subprocess",
                   "command": [sys.executable, str(Path(tmp, "run.py")), "{metrics_json}", "{workdir}"]}
    yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))


def test_make_clone_and_eval_runs_separate_copies_and_measures_efficiency(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    (wd / "model_input.txt").write_text("x")
    (tmp_path / "run.py").write_text(RUNNER_PY)
    spec = {"kind": "subprocess", "command": [sys.executable, str(tmp_path / "run.py"), "{metrics_json}"]}
    ce = make_clone_and_eval(spec, str(tmp_path), str(wd), {"a": 0.8, "b": 2.0})
    try:
        m = concurrency_test(ce, ks=(1, 2))
    finally:
        ce.cleanup()
    assert m["efficiency"][1] == 1.0 and 0.5 < m["efficiency"][2] <= 1.3
    assert m["probe_wall_s"] >= 0.6                                     # both rounds are counted
    assert not any(p.name.startswith("kdt_lane_probe_") for p in tmp_path.iterdir())   # cleaned up
    assert make_clone_and_eval({"kind": "python", "callable": "x:y"}, str(tmp_path), str(wd), {}) is None


def test_probe_lanes_uses_the_measurement():
    calls = []

    def fake(i):
        calls.append(i)
        time.sleep(0.05)
        return 0.05
    pilot = {"threads_per_run": 1.0, "peak_rss_mb": 10.0}
    out = probe_lanes(pilot, parallel_safe=True, clone_and_eval=fake, ks=(1, 2))
    if out["ceiling"] >= 2:
        assert out["measured"] is not None and out["lanes"] in (1, 2) and calls
    out2 = probe_lanes(pilot, parallel_safe=False, clone_and_eval=fake)
    assert out2["lanes"] == 1 and out2["measured"] is None


def test_calibrate_measures_lanes_for_a_parallel_safe_subprocess_runner_and_cleans_up(monkeypatch):
    # a busy server (1-minute load above the core count) leaves 1 free core and no probe: pin the machine
    from calibration_kit import compute as _cmp
    monkeypatch.setattr(_cmp, "probe_machine", lambda: {"cores": 192, "load1": 0.0, "free_cores": 192,
                                                        "mem_available_gb": 64.0})
    tmp = tempfile.mkdtemp(prefix="kdt_s5_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "parallel_safe": True, "seeds": 2},
                            None, 30)
        probe_contract(tmp, ki)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=None, budget=None, seed=0)
        bp = rep["budget_plan"]
        assert bp["ledger"]["shared_parts_s"]["machine_probe"] > 0      # the probe ran and was charged
        leftovers = [p for p in Path(wd).parent.iterdir() if p.name.startswith("kdt_lane_probe_")]
        assert leftovers == []                                         # the lane clones are removed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── round-1 review fixes (Opus 2026-09-29) ───────────────────────────────────────────────────────
def test_a_cap_of_exactly_20_is_not_refused_and_a_cut_names_its_ceiling():
    p = plan_budget("300s", PILOT, lanes=1, efficiency=1.0, n_params=4, n_seeds=1, shared_overhead_s=0.0)
    assert p["cap_per_seed"] == 20 and not p["refuse"]                     # floor((300-60)/12) = 20
    q = plan_budget("3h", PILOT, lanes=1, efficiency=1.0, n_params=4, shared_overhead_s=0.0, max_evaluations=10)
    assert q["refuse"] and "max_evaluations (10)" in q["refuse_reason"]
    h = plan_budget("3h", PILOT, lanes=1, efficiency=1.0, n_params=4, shared_overhead_s=0.0, hard_max=15)
    assert h["refuse"] and "hard_max (15)" in h["refuse_reason"]
    both = plan_budget("3h", PILOT, lanes=1, efficiency=1.0, n_params=4, shared_overhead_s=0.0,
                       max_evaluations=10, hard_max=15)
    assert "max_evaluations (10)" in both["refuse_reason"] and "hard_max (15)" in both["refuse_reason"]
    # the allowance itself buys only 19: the allowance is the reason, and the low ceiling is named too
    a = plan_budget("290s", PILOT, lanes=1, efficiency=1.0, n_params=4, n_seeds=1, shared_overhead_s=0.0,
                    max_evaluations=10)
    assert "the allowance buys 19 search calls" in a["refuse_reason"] and "raise the allowance" in a["refuse_reason"]
    assert "max_evaluations (10) must also be raised" in a["refuse_reason"]


def test_a_measured_contract_without_max_evaluations_has_no_hidden_ceiling():
    tmp = tempfile.mkdtemp(prefix="kdt_s5_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "30s"}, None, 100000)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); del c["strategy"]["max_evaluations"]
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=25, seed=0)
        bp = rep["budget_plan"]
        assert bp["cap_per_seed"] > 200                                   # the formula is the cap
        assert not any("max_evaluations" in w for w in bp["warnings"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_earlier_attempts_in_the_same_workdir_are_not_charged_again():
    tmp = tempfile.mkdtemp(prefix="kdt_s5_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "3h"}, None, 100000)
        with open(Path(wd, "eval_history.jsonl"), "w") as fh:        # an older attempt: 5000 s of commissioning
            for ph in ("commission", "pilot", "consume_proof"):
                fh.write(json.dumps({"phase": ph, "ok": True, "wall_s": 5000.0}) + "\n")
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=25, seed=0)
        parts = rep["budget_plan"]["ledger"]["shared_parts_s"]
        assert parts["commission"] < 100 and parts["pilot"] < 100 and parts["consume_proof"] < 100
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _FakeEv:
    """Just enough of an Evaluator for run_pilot: evaluate(x) fails where `bad(k)` says so."""
    def __init__(self, bad):
        self.bad, self.k, self.workdir, self._last_metrics = bad, 0, tempfile.mkdtemp(prefix="kdt_s5_"), {}

    @contextlib.contextmanager
    def phase_as(self, _ph):
        yield

    def evaluate(self, x):
        k, self.k = self.k, self.k + 1
        return [float("inf")] if self.bad(k) else [0.5]


def _pilot_with(bad, n):
    from calibration_kit.pilot import run_pilot
    import numpy as np
    ev = _FakeEv(bad)
    try:
        return run_pilot(ev, np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]), n=n,
                         emit_series=False)
    finally:
        shutil.rmtree(ev.workdir, ignore_errors=True)


def test_a_small_pilot_is_judged_by_every_non_default_run_failing_not_by_the_default():
    one_bad = _pilot_with(lambda k: k == 2, 3)
    assert one_bad["status"] == "ok" and "1 of 3 pilot runs failed" in one_bad["warning"]
    dflt_bad = _pilot_with(lambda k: k == 0, 3)
    # (build step 8) a failed default is no longer a pilot stop: the triage decides (no_baseline)
    assert dflt_bad["status"] == "ok" and "the DEFAULT run failed" in dflt_bad["warning"]
    rest_bad = _pilot_with(lambda k: k > 0, 3)                            # only the defaults run
    assert rest_bad["status"] == "pilot_unstable" and "every non-default" in rest_bad["reason"]
    three_of_four = _pilot_with(lambda k: k in (1, 2, 3), 5)                  # 3 of 4 non-default fail
    assert three_of_four["status"] == "ok" and three_of_four.get("warning")
    all_four = _pilot_with(lambda k: k > 0, 5)                                 # 4 of 4 non-default fail
    assert all_four["status"] == "pilot_unstable" and "4 of 4" in all_four["reason"]
    two_runs = _pilot_with(lambda k: k == 1, 2)                     # one non-default point: no rule
    assert two_runs["status"] == "ok"
    full = _pilot_with(lambda k: k in (1, 2, 3, 4), 10)
    assert full["status"] == "pilot_unstable" and "40%" in full["reason"]      # the rate rule for 10 runs
    fine = _pilot_with(lambda k: k in (1, 2), 10)
    assert fine["status"] == "ok"


def test_a_zero_time_share_warns_at_the_first_call():
    # final overhead alone is 4 x 1.2 x >= 20 ms = 0.096 s, more than the 0.01 s allowance
    rep, log = _calib({"mode": "measured", "allowance": "0.01s", "pilot_runs": 3}, budget=30, runner=_Sleepy20)
    assert rep["budget_plan"]["ledger"]["per_seed_search_s"] == 0.0     # the overhead used it all
    assert any("would have refused" in x for x in rep["budget_plan"]["warnings"])
    w = [x for x in rep["budget_plan"]["warnings"] if "share of the allowance is 0 s" in x]
    assert len(w) == 3 and w[0].startswith("seed 0: ")              # once per seed (3 by default)


def test_the_inputs_are_restored_when_the_run_ends_before_the_search(monkeypatch):
    calls = []
    from calibration_kit.evaluator import Evaluator as _Ev
    orig = _Ev.restore_originals

    def spy(self):
        calls.append(1)
        return orig(self)
    monkeypatch.setattr(_Ev, "restore_originals", spy)
    rep, _ = _calib({"mode": "measured", "allowance": "1s", "pilot_runs": 4}, runner=E._slow_runner)
    assert rep["status"] == "budget_exhausted" and calls
    calls.clear()
    monkeypatch.setattr(C, "_make_backend", lambda algo: None)
    rep, _ = _calib(None, budget=25)
    assert rep["status"] == "backend_unavailable" and calls


LANE_PY = """
import json, os, sys
open(sys.argv[2], "a").write(os.environ["KDT_CALIB_PARAMS"] + " " + os.getcwd() + "\\n")
json.dump({"nse": 0.5}, open(sys.argv[1], "w"))
"""


def test_probe_lanes_are_really_separate(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    (tmp_path / "run.py").write_text(LANE_PY)
    log = tmp_path / "lanes.log"
    spec = {"kind": "subprocess", "cwd": "{workdir}",
            "command": [sys.executable, str(tmp_path / "run.py"), "{metrics_json}", str(log)]}
    ce = make_clone_and_eval(spec, str(tmp_path), str(wd), {"a": 0.8})
    try:
        concurrency_test(ce, ks=(1, 2))
    finally:
        ce.cleanup()
    rows = [ln.split() for ln in log.read_text().splitlines()]
    k2 = rows[1:3]                                                   # the two copies run at once
    assert len(k2) == 2 and k2[0][0] != k2[1][0] and k2[0][1] != k2[1][1]
    assert all(Path(r[0]).parent == Path(r[1]) for r in rows)       # each copy reads its own params file


def test_a_failed_probe_round_is_still_charged():
    def ce(i):
        time.sleep(0.2)
        if i == 1:
            raise RuntimeError("copy 2 broke")
        return 0.2
    m = concurrency_test(ce, ks=(1, 2))
    assert 2 in m["errors"] and m["probe_wall_s"] >= 0.4

    m3 = concurrency_test(lambda i: (time.sleep(0.2), None)[1], ks=(1,))    # a round with no timing
    assert m3["errors"][1] == "no timing returned" and m3["probe_wall_s"] >= 0.2


def test_the_live_clock_starts_with_the_search(monkeypatch):
    """Fake clock: every model call takes 10 s, the seed's share is ~280 s, the caller budget is 30
    calls (300 s). Timed from the search's start the warning comes after exactly 5 calls, at 300 s;
    a clock started at the first call's END would say 16 calls / 281 s."""
    import types
    clock = {"t": 1000.0}
    monkeypatch.setattr(C, "time", types.SimpleNamespace(perf_counter=lambda: clock["t"], time=time.time,
                                                         sleep=time.sleep))
    inner_holder = {}

    def runner_factory(wd):
        inner_holder["r"] = S1._Runner(wd)

        def run():
            clock["t"] += 10.0
            return inner_holder["r"]()
        return run
    rep, _ = _calib({"mode": "measured", "allowance": "280.3s", "pilot_runs": 3, "seeds": 1}, budget=30,
                    runner=runner_factory)
    w = [x for x in rep["budget_plan"]["warnings"] if "projected search time" in x]
    assert len(w) == 1 and w[0].startswith("seed 0: after 5 calls the projected search time is 300 s")


def test_the_probe_is_not_run_again_on_resume(monkeypatch):
    from calibration_kit import compute
    tmp = tempfile.mkdtemp(prefix="kdt_s5_")
    try:
        monkeypatch.setattr(compute, "probe_machine", lambda: {"cores": 192, "load1": 0.0, "free_cores": 192,
                                                               "mem_available_gb": 64.0})
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "parallel_safe": True, "seeds": 2},
                            None, 30)
        probe_contract(tmp, ki)
        made = []
        real = compute.make_clone_and_eval
        monkeypatch.setattr(compute, "make_clone_and_eval", lambda *a, **k: made.append(1) or real(*a, **k))
        for _ in range(2):
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=None,
                                  budget=None, seed=0)
        assert len(made) == 1 and rep["budget_plan"].get("resumed_cap")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _ConsEv(_FakeEv):
    """A _FakeEv with a contract constraint: points with x0 > 0.5 are rejected without a run."""
    def __init__(self, bad, default_ok=True, a_max=0.5):
        super().__init__(bad)
        self.default_ok, self.ran, self.a_max = default_ok, [], a_max

    def _named(self, x):
        return {"a": float(x[0]), "b": float(x[1])}

    def constraints_ok(self, named):
        return named["a"] <= self.a_max

    def evaluate(self, x):
        self.ran.append(list(x))
        if not self.constraints_ok(self._named(x)):
            self.k += 1
            return [float("inf")]
        return super().evaluate(x)


def test_points_the_constraints_reject_are_not_failed_runs():
    from calibration_kit.pilot import run_pilot
    import numpy as np
    for n in (3, 10):
        ev = _ConsEv(lambda k: False)
        try:
            out = run_pilot(ev, np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.2, 0.5]), n=n,
                            emit_series=False)
        finally:
            shutil.rmtree(ev.workdir, ignore_errors=True)
        assert out["status"] == "ok" and out["fail_rate"] == 0.0
        assert out["n"] == n and out["n_rejected_by_constraints"] > 0     # replaced by fresh draws
        assert all(x[0] <= 0.5 for x in ev.ran)                           # a rejected point never ran
        assert len({tuple(x) for x in ev.ran}) == n                       # fresh draws, not repeats
        assert "warning" not in out
    for n in (1, 3, 10):                  # a default that breaks the constraints: same reason at every size
        ev = _ConsEv(lambda k: False)
        try:
            out = run_pilot(ev, np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.9, 0.5]), n=n,
                            emit_series=False)
        finally:
            shutil.rmtree(ev.workdir, ignore_errors=True)
        assert out["status"] == "pilot_unstable" and "break the contract's constraints" in out["reason"]
        assert out["fail_rate"] == 0.0                                    # the rejected default is not a failure
    # a tiny feasible region (exactly 2 non-default points ever pass): fewer points than asked, said so
    for bad, both in ((lambda k: False, False), (lambda k: k == 1, True)):
        ev = _TwoPointEv(bad)
        try:
            out = run_pilot(ev, np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]), n=5,
                            emit_series=False)
        finally:
            shutil.rmtree(ev.workdir, ignore_errors=True)
        assert out["n"] == 3 and "only 2 of 4 non-default pilot points satisfy" in out["warning"]
        assert ("1 of 3 pilot runs failed" in out["warning"]) is both          # joined, not replaced


class _TwoPointEv(_ConsEv):
    """Only the defaults and the first two other points ever drawn satisfy the constraints."""
    def __init__(self, bad):
        super().__init__(bad)
        self.ok_pts = set()

    def constraints_ok(self, named):
        key = (named["a"], named["b"])
        if key == (0.5, 0.5) or key in self.ok_pts:
            return True
        if len(self.ok_pts) < 2:
            self.ok_pts.add(key)
            return True
        return False


def test_a_measured_contract_with_a_constraint_is_not_refused_by_the_pilot():
    for pilot_runs in (3, 10):
        for sd in range(4):
            tmp = tempfile.mkdtemp(prefix="kdt_s5_")
            try:
                ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "1h", "pilot_runs": pilot_runs},
                                    None, 100000)
                c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["constraints"] = ["b >= 5 * (a - 1)"]
                yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
                with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                    rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                      budget=25, seed=sd)
                assert rep["status"] != "pilot_unstable", (pilot_runs, sd, rep.get("reason"))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)


def test_an_unreadable_or_missing_allowance_is_refused_before_any_model_run():
    for block, word in (({"mode": "measured", "allowance": "90min"}, "could not be read"),
                        ({"mode": "measured"}, "no allowance")):
        tmp = tempfile.mkdtemp(prefix="kdt_s5_")
        try:
            ki, wd = E._fixture(tmp, block, None, 100000)
            ran = []
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                  run_model=lambda: ran.append(1), budget=None, seed=0)
            assert rep["status"] == "invalid_contract" and word in rep["reason"] and not ran
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):     # a caller's budget= still wins
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=25, seed=0)
            assert rep["status"] == "completed", rep.get("reason")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    for block, word in (({"mode": "measured", "allowance": 0}, "must be a positive time"),
                        ({"mode": "measured", "allowance": "0s"}, "must be a positive time"),
                        ({"mode": "measured", "allowance": -600}, "must be a positive time"),
                        ({"mode": "measure", "allowance": "1h"}, "is not measured or fixed"),
                        ({"mode": "measured", "allowance": float("inf")}, "must be a positive time"),
                        ({"mode": "measured", "allowance": float("nan")}, "must be a positive time"),
                        ({"allowance": 0}, "must be a positive time"),                  # no mode: still measured
                        ("72h", "must be a mapping")):
        tmp = tempfile.mkdtemp(prefix="kdt_s5_")
        try:
            ki, wd = E._fixture(tmp, block, None, 100000)
            ran = []
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                  run_model=lambda: ran.append(1), budget=None, seed=0)
            assert rep["status"] == "invalid_contract" and word in rep["reason"] and not ran, (block, rep)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    q = plan_budget(None, PILOT, lanes=1, efficiency=1.0, n_params=4, mode="measured")
    assert not any("falling back to the contract's max_evaluations" in w for w in q["warnings"])


def test_a_fixed_budget_pilot_problem_reaches_the_plan_warnings(monkeypatch):
    from calibration_kit import pilot as PL
    real = PL.run_pilot

    def unstable(*a, **k):
        out = real(*a, **k)
        out.update(status="pilot_unstable", reason="40% of 10 pilot evaluations failed (planted)")
        return out
    monkeypatch.setattr(PL, "run_pilot", unstable)
    rep, _ = _calib(None, budget=25)                                   # fixed budget: continues
    assert rep["status"] == "completed"
    assert "pilot: 40% of 10 pilot evaluations failed (planted) — continuing (fixed budget)" in \
        rep["budget_plan"]["warnings"]


def test_a_mode_typo_is_refused_even_under_a_caller_budget_and_spellings_are_accepted():
    for block, want in (({"mode": "measure", "allowance": "1h"}, "invalid_contract"),
                        ({"mode": " Measured ", "allowance": "10m"}, "completed"),
                        ({"mode": " FIXED "}, "completed")):
        rep, _ = _calib(block, budget=25)
        assert rep["status"] == want, (block, rep.get("reason"))
        if want == "completed":                    # the spelling is read, not just let through
            assert rep["budget_plan"]["mode"] == str(block["mode"]).strip().lower()


def test_defaults_that_break_the_constraints_are_refused_at_load_in_measured_mode():
    for block, want in (({"mode": "measured", "allowance": "1h"}, "invalid_contract"),
                        ({"mode": "fixed"}, "no_baseline")):          # fixed mode: triage (build step 8)
        tmp = tempfile.mkdtemp(prefix="kdt_s5_")
        try:
            ki, wd = E._fixture(tmp, block, None, 100000)
            c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["constraints"] = ["a >= 1.0"]  # default a=0.8
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
            ran = []
            inner = S1._Runner(wd)
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                  run_model=lambda: (ran.append(1), inner())[1], budget=25, seed=0)
            assert rep["status"] == want, (block, rep.get("reason"))
            if want == "invalid_contract":
                assert "break the contract's constraints" in rep["reason"] and not ran
            else:
                # the default never ran, so there is no baseline: triage stops before the search and says why
                assert "no finite objective metric" in rep["reason"] and "break the contract's constraints" in rep["reason"]
                assert rep["pilot"]["fail_rate"] == 0.0                            # the rejected default
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_a_rejected_default_is_named_in_the_no_baseline_reason():
    tmp = tempfile.mkdtemp(prefix="kdt_s5_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "fixed", "pilot_runs": 3}, None, 60)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["strategy"].update({"multi_objective": True, "default_algorithm": "nsga2",
                              "protect": {"streamflow": ["r"]}})
        c["constraints"] = ["a >= 1.0"]                                   # the defaults (a = 0.8) break it
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        # build step 8: the default never ran -> `no_baseline` before the search, with the pilot's reason
        assert rep["status"] == "no_baseline", rep.get("reason")
        assert "break the contract's constraints" in rep["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_load_check_sees_the_default_the_run_uses():
    """An integer parameter's default is rounded by the run (3.5 -> 4); a parameter with no default uses
    the midpoint. The load check must judge the same point the pilot runs (Opus round 6)."""
    cases = ((3.5, ["b >= 4"], "completed"),           # rounded to 4: allowed
             (0, ["b <= 0.5"], "completed"),           # a default of 0 is a default, not "none"
             (None, ["b >= 4"], "completed"),          # midpoint 3.5 -> 4: allowed
             (None, ["b <= 2"], "invalid_contract"))   # midpoint -> 4 breaks it
    for dflt, cons, want in cases:
        tmp = tempfile.mkdtemp(prefix="kdt_s5_")
        try:
            ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m"}, None, 100000)
            c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
            pb = {"name": "b", "range": [-5, 7] if dflt == 0 else [0, 7], "type": "integer"}
            if dflt is not None:
                pb["default"] = dflt
            c["parameters"][1] = pb
            c["constraints"] = cons
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=25, seed=0)
            assert rep["status"] == want, (dflt, cons, rep.get("reason"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_a_non_finite_allowance_never_crashes_the_planner():
    for a in (float("inf"), float("nan"), "inf"):
        p = plan_budget(a, PILOT, lanes=1, efficiency=1.0, n_params=4, mode="measured", max_evaluations=50)
        assert p["cap_per_seed"] == 50 and p["mode"] == "fixed"          # no allowance: the declared ceiling


def test_the_load_check_maps_a_log_parameter_the_way_the_run_does():
    tmp = tempfile.mkdtemp(prefix="kdt_s5_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m"}, None, 100000)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["parameters"][0]["transform"] = "log"                           # a in [0.5, 1.5], default 0.8
        c["constraints"] = ["a <= 1.0", "a >= 0.6"]
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=25, seed=0)
        assert rep["status"] == "completed", rep.get("reason")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
