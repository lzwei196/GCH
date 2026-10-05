"""Build step 4 (design §1.2, §1.6, §2.8, §5.1b; gaps 2a, 2b, 2h): one termination policy, the cap
counted in calls for every backend, stop / keep-going modes, the old strategy.stop translated, and each
optimizer's native verdict recorded."""
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit.rule import resolve_mode, sceua_settings, STOPPABLE                 # noqa: E402
from calibration_kit.backends.base import Problem                                       # noqa: E402
from calibration_kit.backends.spotpy_backend import SpotpyBackend                       # noqa: E402
from calibration_kit import test_step3_verdict as T3                                    # noqa: E402
from calibration_kit import test_step1_panel as S1                                      # noqa: E402


# ── modes and the old strategy.stop (§1.2, §1.6, §5.1b) ────────────────────────────────────────────
def test_mode_default_is_keep_going_and_old_names_are_translated():
    assert resolve_mode({}) == ("keep_going", [], None)
    m, w, t = resolve_mode({"strategy": {"convergence": {"mode": "stop"}}})
    assert m == "stop" and not w
    for old, new in (("observe", "keep_going"), ("enforce", "stop")):
        m, w, _ = resolve_mode({"strategy": {"convergence": {"mode": old}}})
        assert m == new and w and "old name" in w[0]
    with pytest.raises(ValueError):
        resolve_mode({"strategy": {"convergence": {"mode": "fast"}}})


def test_an_old_strategy_stop_block_is_translated_not_an_error():
    c = {"strategy": {"stop": {"main_stat": "NSE", "floor": {"source": "ki_validation_convention", "band": "good"},
                               "converge": {"patience": 50}, "escalate": ["x"]}}}
    m, w, t = resolve_mode(c)
    assert m == "stop" and t == {"mode": "stop", "main_stat": "nse", "band": "good", "dag_variable": None}
    assert "ignored" in w[0] and "converge" in w[0] and "escalate" in w[0]
    c["strategy"]["convergence"] = {"mode": "keep_going"}              # an explicit mode wins
    m, w, t = resolve_mode(c)
    assert m == "keep_going" and t["mode"] == "keep_going" and "explicit" in w[0]
    assert "not a mapping" in resolve_mode({"strategy": {"stop": "yes"}})[1][0]


def test_sceua_settings_per_mode():
    """2026-10-04: SPOTPY's own INTERNAL settings stay at its defaults in every mode; the kit's rule is applied at
    each loop end through the loop hook (native_rules.SpotpyTest)."""
    for mode in ("stop", "keep_going"):
        s = sceua_settings(mode, 3000, 4)                  # pop = 20 * 9 = 180
        assert (s["kstop"], s["pcento"], s["peps"]) == (100, 1e-7, 1e-7)
        assert s["repetitions"] == 2 * 3000 + 180
        assert s["loops_available"] == pytest.approx((3000 - 180) / 360, abs=0.005)
    assert STOPPABLE == ("sceua", "nsga2", "nsga3", "moead")


# ── the cap is counted in calls, for SCE-UA too (§1.1, gap 2b) ────────────────────────────────────
def _toy(n=3):
    def ev(x):
        return [sum((v - 0.3) ** 2 for v in x)]
    return Problem(names=[f"p{i}" for i in range(n)], lower=[0.0] * n, upper=[1.0] * n,
                   objective_names=["l"], evaluate=ev, is_multi_objective=False)


@pytest.mark.parametrize("cap", [150, 239, 400])
def test_sceua_never_exceeds_the_cap_when_the_kit_hook_counts_calls(cap):
    s = sceua_settings("keep_going", cap, 3)
    hook = lambda h: len(h) >= cap                          # noqa: E731 — the kit's rule, as in calib
    r = SpotpyBackend("sceua").optimize(_toy(), budget=cap, seed=1, on_eval=hook,
                                        **{k: s[k] for k in ("ngs", "kstop", "pcento", "peps", "repetitions")})
    assert r.n_evaluations == cap
    loops = [h["loop"] for h in r.history]
    assert loops[0] == 0 and loops == sorted(loops)          # loop index per call, burn-in = 0
    if cap >= 2 * 20 * 7:                                     # past the burn-in population: real loops
        assert max(loops) >= 1
    assert r.termination["repetitions_given"] == 2 * cap + 20 * 7


def test_without_the_hook_spotpys_counter_ends_sceua_below_the_cap():
    """Why the kit counts calls itself: SPOTPY's SCE-UA counter counts every trial plus one per
    step, so given repetitions = cap it stops BELOW the cap."""
    r = SpotpyBackend("sceua").optimize(_toy(), budget=400, seed=1, repetitions=400, kstop=100)
    assert r.n_evaluations < 400


# ── through calibrate(): what ended the search, and why (§2.8) ─────────────────────────────────────
def _calib(strategy=None, env=None, cap=400, runner=S1._Runner, families=None, conv=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s4_")
    try:
        return T3._calib(tmp, runner, strategy=strategy, env=env, cap=cap, families=families, conv=conv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_keep_going_runs_every_optimizer_to_the_cap_or_its_own_test():
    rep, hist = _calib(env={"KDT_CALIB_ALGO": "sceua"})
    e = rep["convergence"]["ended"]
    assert rep["convergence"]["mode"] == "keep_going"
    assert e["calls"] == 400 and e["reason"] == "cap" and e["ended_at_call"] == 399
    assert len([h for h in hist if h["phase"] == "search"]) == 400
    assert not rep["budget_used"]["stopped_early"] and not rep["stopped_early"]
    assert "stop" not in rep                                  # kit_report["stop"] is no longer produced


def test_stop_mode_ends_sceua_at_spotpys_own_test_at_a_loop_end():
    rep, hist = _calib(strategy={"convergence": {"mode": "stop"}}, env={"KDT_CALIB_ALGO": "sceua"}, cap=8000)
    cv = rep["convergence"]; e = cv["ended"]; nat = cv["native"]; t = cv["optimizer_termination"]
    assert e["reason"] == "SCE-UA own test (kit)" and nat["stopped_the_search"] is True
    k = nat["fired_at"]["loop"]
    assert k >= 10 and t["status"] == "stopped_by_kit_rule" and t["stopped_by_kit_at_loop"] == k
    assert len(t["loop_record"]) == k                     # it ended at the end of loop k
    assert len([h for h in hist if h["phase"] == "search"]) == e["calls"] < 8000
    assert rep["stopped_early"] is True and rep["budget_used"]["stopped_early"] is True
    assert e["text"].startswith(f"converged at loop {k} by SCE-UA's own rule on the objective and watched scores")
    assert nat["settings"]["kstop"] == 10 and "not SPOTPY's shipped defaults" in nat["settings"]["note"]


def test_keep_going_records_spotpys_own_test_and_runs_on():
    rep, hist = _calib(env={"KDT_CALIB_ALGO": "sceua"}, cap=8000)
    cv = rep["convergence"]; e = cv["ended"]; nat = cv["native"]
    assert nat["stopped_the_search"] is False and nat["fired_at"] is not None
    assert e["reason"] != "SCE-UA own test (kit)" and rep["stopped_early"] is False
    assert f"own rule fired at loop {nat['fired_at']['loop']} (recorded only)" in e["text"]
    assert cv["settle"]["settled_by_run"] is not None


def test_stop_mode_never_ends_dds():
    rep, _ = _calib(strategy={"convergence": {"mode": "stop"}}, env={"KDT_CALIB_ALGO": "dds"})
    e = rep["convergence"]["ended"]
    assert e["reason"] == "cap" and e["calls"] == 400 and e["stop_point"] is not None
    assert rep["convergence"]["optimizer_termination"]["status"] == "budget_schedule_complete"
    assert not rep["budget_used"]["stopped_early"]
    nat = rep["convergence"]["native"]; st = rep["convergence"]["settle"]
    assert nat["rule"] is None and "no convergence rule of its own" in nat["verdict"]
    assert st["wording"].startswith("settled by run ") and st["wording"].endswith(" of 400")


def test_dream_is_capped_at_the_call_level_and_r_hat_is_recorded():
    rep, hist = _calib(env={"KDT_CALIB_ALGO": "dream"}, cap=1500)
    cv = rep["convergence"]; t = cv["optimizer_termination"]; e = cv["ended"]
    # deterministic (seed 0): R-hat < 1.2 is reached at call 174, then 100 more calls
    assert e["calls"] <= 1500 and t["rhat_recorded"] is True and t["rhat_reached"] is True
    assert t["rhat_converged_at_call"] == 174 and e["reason"] == "DREAM convergence"
    assert cv["verdict"]["verdict"] == "converged" and cv["verdict"]["how"] == "DREAM: R-hat < 1.2 reached"
    assert e["text"].startswith("DREAM reached R-hat < 1.2 at call 174")
    # at a cap of 200 DREAM's post-convergence calls are cut, and the text says so
    rep2, _ = _calib(env={"KDT_CALIB_ALGO": "dream"}, cap=200)
    e2 = rep2["convergence"]["ended"]
    assert e2["calls"] == 200 and e2["reason"] == "cap"
    assert "ended by cap during DREAM's post-convergence calls" in e2["text"]
    assert rep2["convergence"]["verdict"]["verdict"] == "converged"


def test_stop_mode_never_ends_dream_by_our_rule():
    rep, _ = _calib(strategy={"convergence": {"mode": "stop"}}, env={"KDT_CALIB_ALGO": "dream"}, cap=600)
    assert rep["convergence"]["ended"]["reason"] != "our rule"


def test_pymoo_records_its_native_verdict_and_the_cap_is_not_an_early_stop():
    pytest.importorskip("pymoo")
    rep, hist = _calib(strategy={"multi_objective": True, "default_algorithm": "nsga2"},
                       runner=T3._PullApart, families=["temporal_pattern_match", "magnitude_accuracy"], cap=400)
    t = rep["convergence"]["optimizer_termination"]
    assert t["status"] == "max_evals" and "native_verdict" in t and t["native_verdict"]["settings"]["n_skip"] == 5
    assert rep["convergence"]["ended"]["reason"] == "cap"


def test_the_ended_text_follows_section_2_8():
    # no stop point at the cap, evidence complete -> "ran to the cap without converging"
    rep, _ = _calib(env={"KDT_CALIB_ALGO": "dds"}, cap=40)
    e = rep["convergence"]["ended"]
    assert e["stop_point"] is None and e["text"] == "ran to the cap without converging"
    # a required metric never recorded (this runner gives no alpha) -> the unknown text
    pytest.importorskip("pymoo")
    rep2, _ = _calib(strategy={"multi_objective": True, "default_algorithm": "nsga2"},
                     runner=T3._PullApart, families=["temporal_pattern_match", "magnitude_accuracy"], cap=400)
    assert rep2["convergence"]["ended"]["text"] == (
        "ran to the cap; convergence unknown (streamflow:alpha was not recorded or had no tolerance)")


def test_an_unknown_mode_is_refused_at_load():
    rep, hist = _calib(strategy={"convergence": {"mode": "fast"}})
    assert rep["status"] == "invalid_contract" and "mode" in rep["reason"] and hist == []


def test_an_old_strategy_stop_contract_runs_in_stop_mode_and_its_band_reaches_the_standards():
    conv = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5, "good": 0.99}}]}]}
    rep, _ = _calib(strategy={"stop": {"main_stat": "nse", "floor": {"band": "good"}}}, conv=conv,
                    env={"KDT_CALIB_ALGO": "sceua"})
    cv = rep["convergence"]
    assert cv["mode"] == "stop" and cv["translated_from_strategy_stop"]["band"] == "good"
    st = cv["verdict"]["standards"]["streamflow"]["calibration"]["metrics"]["nse"]
    assert st["band"] == "good" and st["threshold"] == 0.99
    assert "stop" not in rep


# ── review round 1 (Opus 5.5) ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("algo", ["nsga2", "nsga3", "moead"])
def test_an_nsga_run_at_the_cap_keeps_pymoos_own_front_and_result(algo):
    pytest.importorskip("pymoo")
    rep, _ = _calib(strategy={"multi_objective": True, "default_algorithm": algo},
                    runner=T3._PullApart, families=["temporal_pattern_match", "magnitude_accuracy"], cap=800)
    assert rep["convergence"]["ended"]["reason"] == "cap"
    F = rep["pareto_f"]
    assert 2 <= len(F) <= 40                                        # a population's front, not every call
    for i, a in enumerate(F):                                       # nothing in it is dominated
        assert not any(all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a))
                       for j, b in enumerate(F) if j != i)
    assert "stopped early" not in (rep.get("backend_notes") or "")


@pytest.mark.parametrize("algo", ["nsga2", "nsga3", "moead"])
def test_stop_mode_ends_nsga_at_pymoos_own_rule(algo):
    pytest.importorskip("pymoo")
    rep, _ = _calib(strategy={"multi_objective": True, "default_algorithm": algo,
                              "convergence": {"mode": "stop", "front_period": 5, "front_ftol": 0.05}},
                    families=["temporal_pattern_match", "magnitude_accuracy"], cap=6000)
    cv = rep["convergence"]; e = cv["ended"]; nat = cv["native"]; t = cv["optimizer_termination"]
    assert e["reason"] == "pymoo own rule (kit)" and nat["stopped_the_search"] is True
    assert t["status"] == "stopped_by_kit_rule" and t["stopped_by_kit_at_generation"] == nat["fired_at"]["n_gen"]
    assert rep["stopped_early"] is True and rep["algorithm"] == algo and e["calls"] < 6000
    assert e["text"].startswith(f"converged at generation {nat['fired_at']['n_gen']} by pymoo's own rule")
    F = rep["pareto_f"]                                             # a rebuilt front is non-dominated
    for i, a in enumerate(F):
        assert not any(all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a))
                       for j, b in enumerate(F) if j != i)


def test_the_per_call_step_rule_never_ends_a_search():
    """2026-10-04: the per-call step rule is recorded only; in stop mode its stop point does not end SCE-UA."""
    rep, hist = _calib(strategy={"convergence": {"mode": "stop", "window": 20}}, env={"KDT_CALIB_ALGO": "sceua"},
                       cap=400)
    e = rep["convergence"]["ended"]
    assert e["stop_point"] is not None and e["reason"] == "cap" and e["calls"] == 400


def test_too_few_loops_is_too_short_to_judge():
    rep, _ = _calib(strategy={"convergence": {"mode": "stop"}}, env={"KDT_CALIB_ALGO": "sceua"}, cap=400)
    nat = rep["convergence"]["native"]
    assert nat["fired_at"] is None and nat["verdict"] == "too short to judge" and not nat["stopped_the_search"]


def test_loop_and_generation_of_every_call_are_saved():
    rep, _ = _calib(env={"KDT_CALIB_ALGO": "sceua"}, cap=400)
    ls = rep["convergence"]["optimizer_termination"]["loop_starts"]
    assert ls[0] == [0, 0] and [x[1] for x in ls] == sorted(x[1] for x in ls) and len(ls) >= 2
    assert ls[1] == [100, 1]                                  # the burn-in is ngs(2d+1) = 20 * 5 calls
    pytest.importorskip("pymoo")
    rep2, _ = _calib(strategy={"multi_objective": True, "default_algorithm": "nsga2"},
                     runner=T3._PullApart, families=["temporal_pattern_match", "magnitude_accuracy"], cap=200)
    gs = rep2["convergence"]["optimizer_termination"]["generation_starts"]
    assert gs[0] == [0, 1] and gs[1] == [40, 2]


def test_a_cap_end_is_never_labelled_an_early_stop_anywhere():
    for algo in ("dds", "sceua", "dream"):
        rep, _ = _calib(env={"KDT_CALIB_ALGO": algo}, cap=200)
        t = rep["convergence"]["optimizer_termination"]
        assert rep["convergence"]["ended"]["reason"] in ("cap", "DREAM convergence")
        assert rep["stopped_early"] is False and not t.get("early_stopped_by_rule")
        assert "stopped early" not in (rep.get("backend_notes") or "")


def test_stop_mode_warns_that_dds_and_dream_are_not_stopped():
    for algo in ("dds", "dream"):
        rep, _ = _calib(strategy={"convergence": {"mode": "stop"}}, env={"KDT_CALIB_ALGO": algo}, cap=200)
        assert any(f"mode stop does not end {algo}" in w for w in rep["convergence"]["warnings"])


def test_an_old_floor_band_applies_to_its_own_variable_only():
    conv = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5, "good": 0.99}}]}]}
    rep, _ = _calib(strategy={"stop": {"main_stat": "nse", "floor": {"band": "good", "dag_variable": "swe"}}},
                    conv=conv, env={"KDT_CALIB_ALGO": "sceua"})
    st = rep["convergence"]["verdict"]["standards"]["streamflow"]["calibration"]["metrics"]["nse"]
    assert st["band"] == "satisfactory"


# ── review round 2 (Opus 5.5) ─────────────────────────────────────────────────────────────────────
def test_the_old_floor_band_on_the_validation_side_respects_its_variable():
    conv = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5, "good": 0.99}}]}]}
    for dv, band in (("SWE", "satisfactory"), ("STREAMFLOW", "good"), (None, "good")):     # case-insensitive
        rep, _ = _calib(strategy={"stop": {"main_stat": "nse", "floor": {"band": "good", "dag_variable": dv}}},
                        conv=conv, runner=T3._SplitRunner, env={"KDT_CALIB_ALGO": "sceua"})
        st = rep["convergence"]["verdict"]["standards"]["streamflow"]
        assert isinstance(st["validation"], dict), st["validation"]
        assert st["validation"]["metrics"]["nse"]["band"] == band
        assert st["calibration"]["metrics"]["nse"]["band"] == band


def test_no_call_is_reported_as_such(monkeypatch):
    from calibration_kit import calib as C
    from calibration_kit.backends.base import CalibResult

    class _NoCalls:
        @staticmethod
        def available():
            return True

        def optimize(self, problem, budget, seed=0, **kw):
            return CalibResult(best_x=[], best_loss=[float("inf")], n_evaluations=0,
                               backend="zero-call stand-in", notes="no evaluations made")

    original = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algorithm: _NoCalls() if algorithm == "smt" else original(algorithm))
    rep, _ = _calib(strategy={"default_algorithm": "smt"})
    e = rep["convergence"]["ended"]
    assert e["calls"] == 0
    assert e["text"] == "no call was made" and e["ended_at_call"] is None


def test_unavailable_backend_is_reported_before_search(monkeypatch):
    from calibration_kit import calib as C

    class _Unavailable:
        @staticmethod
        def available():
            return False

        def optimize(self, *args, **kwargs):
            raise AssertionError("An unavailable backend must never execute")

    original = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algorithm: _Unavailable() if algorithm == "smt" else original(algorithm))
    rep, history = _calib(strategy={"default_algorithm": "smt"})
    assert rep["status"] == "backend_unavailable" and rep["algorithm"] == "smt"
    assert "not importable" in rep["reason"]
    assert not any(row.get("phase") == "search" for row in history)


def test_the_rebuilt_front_filter():
    from calibration_kit.backends.pymoo_backend import _nondominated
    pts = [([0], [1.0, 3.0]), ([1], [3.0, 1.0]), ([2], [2.0, 2.0]), ([3], [2.0, 2.0]),   # a copy
           ([4], [2.0, 3.0]),                                                            # weakly dominated
           ([5], [4.0, 4.0])]                                                            # dominated
    X, F = _nondominated(pts)
    assert sorted(map(tuple, F.tolist())) == [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]
    assert [int(x[0]) for x in X.tolist()] == [0, 1, 2]


def test_the_dream_end_call_in_the_text():
    rep, _ = _calib(env={"KDT_CALIB_ALGO": "dream"}, cap=1500)
    e = rep["convergence"]["ended"]
    assert e["text"] == f"DREAM reached R-hat < 1.2 at call 174; ended at call {e['ended_at_call']} (DREAM convergence)"
    assert e["ended_at_call"] == e["calls"] - 1


# ── review round 3 (Opus 5.5) — proposed ─────────────────────────────────────────────────────────
def test_an_untracked_backend_gets_the_not_tracked_text(monkeypatch):
    from calibration_kit import calib as C
    from calibration_kit.backends.base import CalibResult

    class _FiveCalls:                        # stands in for a backend that does not report each call
        @staticmethod
        def available():
            return True

        def optimize(self, problem, budget, seed=0, **kw):
            xs = [[lo + (hi - lo) * f for lo, hi in zip(problem.lower, problem.upper)]
                  for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
            ls = [problem.evaluate(x) for x in xs]
            i = min(range(5), key=lambda k: ls[k][0])
            return CalibResult(best_x=xs[i], best_loss=ls[i], n_evaluations=5, backend="stand-in")

    orig = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda a: _FiveCalls() if a == "smt" else orig(a))
    rep, _ = _calib(strategy={"default_algorithm": "smt"})
    assert rep["convergence"]["ended"]["text"] == (
        "convergence not tracked (smt does not report each call); the search ended at call 4 "
        "(optimizer ended on its own)")


def test_sceua_ending_on_peps_is_named(monkeypatch):
    from calibration_kit import rule as R
    orig = R.sceua_settings
    monkeypatch.setattr(R, "sceua_settings", lambda *a, **k: {**orig(*a, **k), "peps": 0.999})
    rep, _ = _calib(env={"KDT_CALIB_ALGO": "sceua"}, cap=400)
    e = rep["convergence"]["ended"]
    assert e["reason"] == "SCE-UA peps" and e["calls"] == 100      # peps met right after the burn-in


def test_pymoo_at_the_cap_without_its_rule_firing_is_the_cap():
    pytest.importorskip("pymoo")
    rep, _ = _calib(strategy={"multi_objective": True, "default_algorithm": "nsga2",
                              "convergence": {"mode": "stop"}},
                    families=["temporal_pattern_match", "magnitude_accuracy"], cap=128)
    e = rep["convergence"]["ended"]
    assert e["reason"] == "cap" and rep["convergence"]["optimizer_termination"]["status"] == "max_evals"
    assert rep["convergence"]["native"]["verdict"] == "too short to judge"


def test_moead_calls_per_generation():
    pytest.importorskip("pymoo")
    rep, _ = _calib(strategy={"multi_objective": True, "default_algorithm": "moead"},
                    runner=T3._PullApart, families=["temporal_pattern_match", "magnitude_accuracy"], cap=200)
    t = rep["convergence"]["optimizer_termination"]
    assert t["calls_per_generation"] == 13 and t["generations_available"] == pytest.approx(200 / 13, abs=0.01)


def test_spotpys_own_test_is_fed_the_best_points_watched_scores(monkeypatch):
    """The loop hook hands SpotpyTest the best point's watched scores (not only the objective)."""
    from calibration_kit import native_rules as NR
    seen = []
    orig = NR.SpotpyTest.add_loop

    def spy(self, loop, bestf, gnrng=None, scores=None):
        seen.append(scores)
        return orig(self, loop, bestf, gnrng, scores)
    monkeypatch.setattr(NR.SpotpyTest, "add_loop", spy)
    rep, _ = _calib(env={"KDT_CALIB_ALGO": "sceua"}, cap=3000)
    var = rep["convergence"]["rule"]["objective_vars"][0]
    assert seen and all(isinstance(sc, dict) and sc.get(var) for sc in seen)
    req = rep["convergence"]["required_panel"][var]
    assert all(set(req) <= set(sc[var]) for sc in seen)


def test_keep_going_records_pymoos_own_rule_and_runs_to_the_cap():
    pytest.importorskip("pymoo")
    rep, _ = _calib(strategy={"multi_objective": True, "default_algorithm": "nsga2",
                              "convergence": {"front_period": 5, "front_ftol": 0.05}},
                    families=["temporal_pattern_match", "magnitude_accuracy"], cap=6000)
    nat = rep["convergence"]["native"]; e = rep["convergence"]["ended"]
    assert nat["fired_at"] is not None and nat["stopped_the_search"] is False
    assert e["reason"] == "cap" and e["calls"] == 6000 and rep["stopped_early"] is False
    assert "(recorded only)" in e["text"]


def test_the_kits_verdict_is_its_convergence_rule_and_the_step_rule_is_labelled():
    """codex A3d #1: convergence.verdict and the run verdict come from the kit's rule (SPOTPY's own test here); the
    per-call step rule's verdict is kept as step_rule_verdict, labelled recorded only."""
    rep, _ = _calib(env={"KDT_CALIB_ALGO": "sceua"}, cap=8000)
    cv = rep["convergence"]
    assert cv["native"]["verdict"] == "converged"
    assert cv["verdict"]["verdict"] == "converged" and cv["verdict"]["how"].startswith("SCE-UA objective test")
    assert "recorded only" in cv["step_rule_verdict"]["note"]
    assert cv["run_verdict"]["verdict"] in ("converged", "unknown")       # several seeds must also agree
    assert all(s["verdict"] == "converged" for s in cv["seeds"]["slots"] if not s["crashed"])
    assert "step_rule_run_verdict" in cv and "step_rule_verdict" in cv["seeds"]["slots"][0]
