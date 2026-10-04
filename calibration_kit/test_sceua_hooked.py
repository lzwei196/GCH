"""A3b: the kit's copy of SPOTPY 1.6.7 sceua.py with a loop-end hook (backends/sceua_hooked.py)."""
import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spotpy = pytest.importorskip("spotpy")
from calibration_kit.backends.sceua_hooked import sceua_hooked          # noqa: E402


class _Setup:
    """A small 4-parameter test function that records every point SPOTPY asks for."""
    def __init__(self):
        self.calls = []

    def parameters(self):
        return spotpy.parameter.generate([spotpy.parameter.Uniform(f"x{i}", -2.0, 2.0) for i in range(4)])

    def simulation(self, v):
        x = np.asarray(v, float)
        self.calls.append(tuple(np.round(x, 12)))
        return [float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))]

    def evaluation(self):
        return [0.0]

    def objectivefunction(self, simulation, evaluation):
        return simulation[0]


def _run(cls, hook=None, reps=3000, seed=7):
    st = _Setup()
    s = cls(st, dbname="t", dbformat="ram", save_sim=False, random_state=seed, optimization_direction="minimize")
    if hook is not None:
        s.loop_hook = hook
    with contextlib.redirect_stdout(io.StringIO()) as out:
        s.sample(reps, ngs=4, kstop=100, pcento=1e-7, peps=1e-7)
    return st.calls, s, out.getvalue()


def test_no_hook_is_spotpy_call_for_call():
    a, _, _ = _run(spotpy.algorithms.sceua)
    b, s, _ = _run(sceua_hooked)
    assert len(a) > 500 and a == b
    assert len(s.loop_record) > 3 and [d["loop"] for d in s.loop_record] == list(range(1, len(s.loop_record) + 1))


def test_hook_answering_no_changes_nothing():
    a, _, _ = _run(spotpy.algorithms.sceua)
    seen = []
    b, s, _ = _run(sceua_hooked, hook=lambda n, g, f: seen.append((n, g, f)) or False)
    loops = [d["loop"] for d in s.loop_record]
    # asked at every loop end, except a last loop that SPOTPY's own budget check ends first (status.stop -> break)
    assert a == b and [x[0] for x in seen] == loops[:len(seen)] and len(seen) >= len(loops) - 1
    assert all(np.isfinite(g) and np.isfinite(f) for _, g, f in seen)


def test_hook_stops_at_the_end_of_the_loop():
    full, sfull, _ = _run(sceua_hooked)
    calls, s, out = _run(sceua_hooked, hook=lambda n, g, f: n == 3)
    assert s.stopped_by_kit_at_loop == 3 and len(s.loop_record) == 3
    assert "STOPPED BY THE CALIBRATION KIT'S CONVERGENCE RULE AT THE END OF LOOP 3" in out
    # the calls made are exactly the full search's first calls, up to the end of loop 3
    assert calls == full[:len(calls)] and len(calls) < len(full)
    assert s.loop_record[-1]["rep"] == sfull.loop_record[2]["rep"]


def test_loop_record_matches_spotpys_own_numbers():
    _, s, out = _run(sceua_hooked)
    # SPOTPY prints the final normalized geometric range; the last record must carry the same value
    import re
    g = float(re.search(r"NORMALIZED GEOMETRIC RANGE = ([0-9.eE+-]+)", out).group(1))
    assert abs(s.loop_record[-1]["gnrng"] - g) < 1e-6


def _problem():
    from calibration_kit.backends.base import Problem
    def ev(x):
        x = np.asarray(x, float)
        return [float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))]
    return Problem(names=[f"x{i}" for i in range(4)], lower=[-2.0] * 4, upper=[2.0] * 4,
                   objective_names=["f"], evaluate=ev, is_multi_objective=False)


def test_backend_uses_the_copy_and_reports_the_loops():
    from calibration_kit.backends.spotpy_backend import SpotpyBackend
    with contextlib.redirect_stdout(io.StringIO()):
        r = SpotpyBackend("sceua").optimize(_problem(), budget=2000, seed=3, ngs=4)
    t = r.termination
    assert t["stopped_by_kit_at_loop"] is None and len(t["loop_record"]) >= 3
    assert t["status"] != "stopped_by_kit_rule" and t["early_stopped_by_rule"] is False


def test_backend_loop_hook_stops_and_says_so():
    from calibration_kit.backends.spotpy_backend import SpotpyBackend
    seen = []

    def on_loop_end(n, g, f, calls):
        seen.append((n, calls))
        return n == 2
    with contextlib.redirect_stdout(io.StringIO()):
        r = SpotpyBackend("sceua").optimize(_problem(), budget=5000, seed=3, ngs=4, on_loop_end=on_loop_end)
    t = r.termination
    assert t["stopped_by_kit_at_loop"] == 2 and t["status"] == "stopped_by_kit_rule"
    assert t["early_stopped_by_rule"] is True and r.stopped_early is True
    assert t["converged_by_own_rule"] is False and len(t["loop_record"]) == 2
    assert seen[-1] == (2, r.n_evaluations)             # the hook saw every call made, up to the loop end


# ── A3c: pymoo's own rule as the kit's NSGA-II rule, with a generation-end hook ─────────────────────────
def _two_obj():
    from calibration_kit.backends.base import Problem
    def ev(x):
        x = np.asarray(x, float)
        g = 1 + 9 * float(np.mean(x[1:]))
        return [float(x[0]), float(g * (1 - np.sqrt(x[0] / g)))]
    return Problem(names=[f"p{i}" for i in range(5)], lower=[0.0] * 5, upper=[1.0] * 5,
                   objective_names=["f1", "f2"], evaluate=ev, is_multi_objective=True)


def test_pymoo_no_hook_and_hook_saying_no_change_nothing():
    pytest.importorskip("pymoo")
    from calibration_kit.backends.pymoo_backend import PymooBackend
    a = PymooBackend("nsga2", pop_size=20).optimize(_two_obj(), budget=600, seed=2)
    seen = []
    b = PymooBackend("nsga2", pop_size=20).optimize(_two_obj(), budget=600, seed=2,
                                                    on_generation_end=lambda g, n, f, c: seen.append((g, n)) or False)
    assert [h["x"] for h in a.history] == [h["x"] for h in b.history]
    assert [g for g, _ in seen] == list(range(1, len(seen) + 1)) and seen[-1][1] == 600
    assert b.termination["stopped_by_kit_at_generation"] is None and b.termination["status"] == "max_evals"


def test_pymoo_hook_stops_at_a_generation_end():
    pytest.importorskip("pymoo")
    from calibration_kit.backends.pymoo_backend import PymooBackend
    full = PymooBackend("nsga2", pop_size=20).optimize(_two_obj(), budget=600, seed=2)
    r = PymooBackend("nsga2", pop_size=20).optimize(_two_obj(), budget=600, seed=2,
                                                    on_generation_end=lambda g, n, f, c: g == 4)
    t = r.termination
    assert t["stopped_by_kit_at_generation"] == 4 and t["status"] == "stopped_by_kit_rule"
    assert r.stopped_early is True and r.n_evaluations == 80                  # 4 whole generations of 20
    assert [h["x"] for h in r.history] == [h["x"] for h in full.history][:80]
    assert len(r.pareto_f) >= 1                                              # front rebuilt from the finite calls


def test_pymoo_native_rule_fires_and_can_stop_the_search():
    pytest.importorskip("pymoo")
    from calibration_kit.backends.pymoo_backend import PymooBackend
    kw = dict(front_period=5, front_ftol=0.05, front_xtol=0.05)             # loose so it fires in a short test
    obs = PymooBackend("nsga2", pop_size=20).optimize(_two_obj(), budget=4000, seed=2, **kw)
    fg = obs.termination["native_verdict"]["fired_gen"]
    assert fg is not None and obs.n_evaluations == 4000                      # recorded only: runs to the cap
    r = PymooBackend("nsga2", pop_size=20).optimize(_two_obj(), budget=4000, seed=2,
                                                    on_generation_end=lambda g, n, f, c: c, **kw)
    assert r.termination["stopped_by_kit_at_generation"] == fg and r.n_evaluations == fg * 20
