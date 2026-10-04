"""Build step 3 (design §2.7 verdicts, §2.9 standards, §2.10 safety; gaps 2f, 2q): each test checks
one rule. Rules are driven call by call; `inc` = each call is a new (tiny) best unless noted, so the
incumbent's panel is that call's panel."""
import contextlib
import io
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

from calibration_kit import panel as P                                              # noqa: E402
from calibration_kit.rule import ConvergenceRule                                    # noqa: E402
from calibration_kit.verdict import (CONVERGED, NOT_CONVERGED, UNKNOWN, aggregate,  # noqa: E402
                                     final_value_settle_points, safety, seed_verdict,
                                     sole_blocker_calls, standards, variable_verdict)

TOL = {"r": 0.01, "alpha": 0.01, "beta": 0.01}
FLAT = {"r": 0.9, "alpha": 1.0, "beta": 1.0}


def _rule(W=3, kinds=None, objectives=None, tol=None, **kw):
    kinds = kinds or {"Q": "series"}
    pn = P.Panel(list(kinds), kinds=dict(kinds))
    objectives = objectives or [(f"{v}:f", v) for v in kinds]
    return ConvergenceRule(objectives, pn, tol if tol is not None else {v: dict(TOL) for v in kinds}, W, **kw)


def _feed(r, rows):
    """rows: list of (losses, panel); each call a new incumbent (tiny improvement)."""
    for k, (loss, pan) in enumerate(rows):
        l = [x - 1e-9 * k if x is not None else None for x in loss]
        r.add(l, None, pan)
    return r


def _q(**kw):
    return {"Q": dict(FLAT, **kw)}


# ── seed verdict (§2.7) ──────────────────────────────────────────────────────────────────────────
def test_rule_triggered_when_our_rule_ended_the_search():
    r = _feed(_rule(), [([1.0], _q())] * 4)
    assert r.stop_point == 3
    v = seed_verdict(r, ended_at_stop=True)
    assert v["verdict"] == CONVERGED and v["how"] == "rule triggered"
    assert v["variables"]["Q"]["verdict"] == CONVERGED


def test_confirmed_when_it_ran_past_and_nothing_moved():
    r = _feed(_rule(), [([1.0], _q())] * 8)
    v = seed_verdict(r)
    assert v["verdict"] == CONVERGED and v["how"].startswith("confirmed")
    assert v["safety"]["verdict"] == "safe" and v["safety"]["calls_after"] == 4


def test_premature_when_the_loss_or_a_metric_moved_after_the_stop_point():
    r = _feed(_rule(), [([1.0], _q())] * 5 + [([0.9], _q())] + [([0.9], _q())] * 3)
    assert r.stop_point == 3
    v = seed_verdict(r)
    assert v["verdict"] == NOT_CONVERGED and v["how"] == "premature"
    assert v["safety"]["largest_move_after"]["loss:Q:f"] == pytest.approx(0.1, rel=1e-6)
    r2 = _feed(_rule(), [([1.0], _q())] * 5 + [([1.0], _q(beta=1.05))] * 4)
    assert seed_verdict(r2)["safety"]["moved"] == ["Q:beta"]


def test_undecidable_when_too_few_calls_follow_or_a_metric_is_missing_after_but_movement_decides_first():
    r = _feed(_rule(), [([1.0], _q())] * 5)                    # s = 3, 1 call after < W
    assert seed_verdict(r)["verdict"] == UNKNOWN and seed_verdict(r)["how"].startswith("undecidable")
    r2 = _feed(_rule(), [([1.0], _q())] * 4 + [([1.0], {"Q": {"r": 0.9, "alpha": 1.0}})] * 4)
    assert seed_verdict(r2)["how"].startswith("undecidable") and "Q:beta" in seed_verdict(r2)["how"]
    r3 = _feed(_rule(), [([1.0], _q())] * 4 + [([0.5], {"Q": {"r": 0.9, "alpha": 1.0}})])
    assert seed_verdict(r3)["how"] == "premature"             # few calls AND missing, but it moved


def test_no_stop_point_only_because_a_metric_was_never_recorded_is_unknown():
    no_beta = {"Q": {"r": 0.9, "alpha": 1.0}}
    r = _feed(_rule(), [([1.0], no_beta)] * 8)
    assert r.stop_point is None and sole_blocker_calls(r)
    v = seed_verdict(r)
    assert v["verdict"] == UNKNOWN and "Q:beta" in v["how"]


def test_no_stop_point_but_the_search_moved_after_the_missing_metric_blocked_is_not_converged():
    no_beta = {"Q": {"r": 0.9, "alpha": 1.0}}
    rows = [([1.0], no_beta)] * 6 + [([1.0 - 0.02 * k], no_beta) for k in range(1, 6)]
    r = _feed(_rule(), rows)
    assert seed_verdict(r)["verdict"] == NOT_CONVERGED


def test_no_stop_point_with_full_evidence_is_not_converged():
    rows = [([1.0 - 0.05 * k], _q()) for k in range(10)]         # the loss never flattens
    r = _feed(_rule(), rows)
    assert r.stop_point is None and seed_verdict(r)["verdict"] == NOT_CONVERGED


def test_variables_that_settle_but_never_at_the_same_call_are_not_converged():
    kinds = {"Q": "series", "S": "series"}
    r = _rule(W=2, kinds=kinds)
    # Q's beta changes at calls k % 4 == 0, S's at k % 4 == 2: with W = 2 a variable is unsettled at
    # the call of its change and the next one, so the two are never settled at the same call
    q_beta, s_beta = 1.0, 1.0
    for k in range(16):
        if k % 4 == 0 and k:
            q_beta = 2.05 - q_beta
        if k % 4 == 2:
            s_beta = 2.05 - s_beta
        r.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT, beta=q_beta), "S": dict(FLAT, beta=s_beta)})
    assert r.stop_point is None and all(r.first_settle.values())
    assert seed_verdict(r)["verdict"] == NOT_CONVERGED


def test_a_short_search_is_not_unknown_early_calls_are_not_missing_evidence():
    r = _feed(_rule(W=5), [([1.0], _q())] * 4)                  # fewer than W calls in all
    assert r.stop_point is None and sole_blocker_calls(r) == []
    assert seed_verdict(r)["verdict"] == NOT_CONVERGED


def test_the_unknown_cap_for_the_variable_and_the_seed_but_not_for_recorded_only_metrics():
    pn = P.Panel(["Q"], kinds={"Q": "series"})
    pn.apply_pilot({"kit_missing": {"Q": {"beta": "mean near zero"}}})
    r = ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL)}, 3)
    _feed(r, [([1.0], {"Q": {"r": 0.9, "alpha": 1.0}})] * 8)
    v = seed_verdict(r)
    assert v["variables"]["Q"]["verdict"] == UNKNOWN and "mean near zero" in v["variables"]["Q"]["reason"]
    assert v["verdict"] == UNKNOWN and "capped" in v["how"] and "confirmed" in v["how"]
    pn2 = P.Panel(["Q"], kinds={"Q": "series"})
    pn2.set_kit_missing("Q", "nrmse", "2n mismatch")               # recorded-only: no cap
    r2 = ConvergenceRule([("Q:f", "Q")], pn2, {"Q": dict(TOL)}, 3)
    _feed(r2, [([1.0], _q())] * 8)
    assert seed_verdict(r2)["verdict"] == CONVERGED


def test_dream_seed_verdict_comes_from_r_hat():
    r = _feed(_rule(), [([1.0 - 0.05 * k], _q()) for k in range(8)])     # our rule: not converged
    assert seed_verdict(r, dream={"rhat_recorded": True, "rhat_reached": True})["verdict"] == CONVERGED
    assert seed_verdict(r, dream={"rhat_recorded": True, "rhat_reached": False})["verdict"] == NOT_CONVERGED
    assert seed_verdict(r, dream={"rhat_recorded": False})["verdict"] == UNKNOWN
    # the cap never changes a DREAM seed verdict
    pn = P.Panel(["Q"], kinds={"Q": "series"}); pn.apply_pilot({"kit_missing": {"Q": {"beta": "mean near zero"}}})
    r2 = _feed(ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL)}, 3), [([1.0], {"Q": {"r": 0.9, "alpha": 1.0}})] * 8)
    assert seed_verdict(r2, dream={"rhat_recorded": True, "rhat_reached": True})["verdict"] == CONVERGED


# ── per-variable verdict (§2.7 rules 0-5) ────────────────────────────────────────────────────────
def test_rule_0_variable_is_unknown_with_its_reason():
    pn = P.Panel(["Q", "ev"], kinds={"Q": "series", "ev": "categorical"})
    r = _feed(ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL)}, 3), [([1.0], _q())] * 8)
    v = seed_verdict(r)
    assert v["variables"]["ev"] == {"variable": "ev", "first_settle": None, "verdict": UNKNOWN,
                                    "reason": "no objective and no panel metric"}
    assert v["verdict"] == CONVERGED                                # rule 0 never blocks


def test_a_variable_is_judged_from_the_stop_point_not_its_first_settle():
    """SWE settles, moves (before s), settles again; Q settles later -> s. Judged from s: converged."""
    kinds = {"Q": "series", "S": "series"}
    r = _rule(W=2, kinds=kinds)
    for k in range(14):
        s_beta = 1.05 if 4 <= k < 6 else 1.0
        q_r = 0.9 if k >= 8 else 0.5 + 0.05 * k
        r.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT, r=q_r), "S": dict(FLAT, beta=s_beta)})
    assert r.first_settle["S"] < 4 and r.stop_point >= 9
    assert variable_verdict(r, "S")["verdict"] == CONVERGED


def test_without_a_stop_point_a_settled_variable_is_judged_from_every_sole_blocker_call_too():
    """Q (flow) never gets lnnse -> no stop point. SWE settles early, moves at call 9, flat again:
    from its settle point it moved, from later sole-blocker calls it did not -> unknown."""
    kinds = {"Q": "flow", "S": "series"}
    r = _rule(W=2, kinds=kinds, tol={"Q": dict(TOL, lnnse=0.01), "S": dict(TOL)})
    for k in range(20):
        s_beta = 1.0 if k < 9 else 1.05
        r.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT, beta=s_beta)})
    assert r.stop_point is None and r.first_settle["S"] is not None
    v = variable_verdict(r, "S")
    assert v["verdict"] == UNKNOWN, v
    assert variable_verdict(r, "Q")["verdict"] == UNKNOWN


def test_a_variable_that_never_settles_only_for_a_missing_metric_is_unknown_else_not_converged():
    kinds = {"Q": "flow"}
    r = _feed(_rule(W=2, kinds=kinds, tol={"Q": dict(TOL, lnnse=0.01)}), [([1.0], _q())] * 8)
    assert r.first_settle["Q"] is None and variable_verdict(r, "Q")["verdict"] == UNKNOWN
    r2 = _feed(_rule(W=2, kinds=kinds, tol={"Q": dict(TOL, lnnse=0.01)}),
               [([1.0 - 0.1 * k], _q()) for k in range(8)])      # its own loss keeps moving
    assert variable_verdict(r2, "Q")["verdict"] == NOT_CONVERGED


def test_fewer_than_w_calls_after_the_reference_is_unknown():
    r = _feed(_rule(W=3), [([1.0], _q())] * 5)
    assert variable_verdict(r, "Q")["verdict"] == UNKNOWN


# ── §2.10 details ────────────────────────────────────────────────────────────────────────────────
def test_front_excursion_after_the_stop_point_is_premature_in_a_trade_off_search():
    objs = [("Q:a", "Q"), ("S:b", "S")]
    pn = P.Panel(["Q", "S"], kinds={"Q": "categorical", "S": "categorical"})
    r = ConvergenceRule(objs, pn, {}, 2, trade_off=True)
    pts = [[1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [0.5, 0.5]]
    for p in pts:
        r.add(p, None, {})
    assert r.stop_point is not None
    sa = safety(r)
    assert sa["verdict"] == "premature" and "front" in sa["moved"] and sa["largest_move_after"]["front:eps"] > r.eps_front


def test_final_value_settle_points():
    rows = [([1.0], _q(beta=b)) for b in (1.2, 1.1, 1.05, 1.0, 1.0, 1.0)]
    r = _feed(_rule(W=2), rows)
    fv = final_value_settle_points(r)
    assert fv["Q:beta"] == 3 and fv["Q:r"] == 0


# ── standards (§2.9) ─────────────────────────────────────────────────────────────────────────────
CONV = {"validation": [
    {"dag_variable": "Q", "obs_shape": "point_time_series", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5, "good": 0.65}, "pass_band": "good"},
        {"metric": "pbias", "direction": "zero_centered", "bands": {"satisfactory": 15}},
        {"metric": "rmse", "direction": "minimize", "bands": {"satisfactory": 2.0}, "pass_band": "satisfactory"}]},
    {"dag_variable": "Q", "obs_shape": "point_snapshot", "headline_metrics": [
        {"metric": "nrmse", "direction": "lower_is_better", "bands": {"satisfactory": 30}}]},
    {"dag_variable": "E", "obs_shape": "point_time_series", "headline_metrics": []}]}


def test_standards_outcomes():
    ok = {"nse": 0.7, "pbias": -14.0, "rmse": 1.5}
    assert standards(CONV, "Q", "point_time_series", ok)["outcome"] == "meets"
    assert standards(CONV, "Q", "point_time_series", dict(ok, nse=0.6))["outcome"] == "fails"   # pass_band good
    assert standards(CONV, "Q", "point_time_series", dict(ok, pbias=-16.0))["outcome"] == "fails"
    assert standards(CONV, "Q", "point_time_series", dict(ok, rmse=2.5))["outcome"] == "fails"
    assert standards(CONV, "Q", "point_time_series", {"nse": 0.7, "pbias": 1.0})["outcome"] == "unknown"
    assert standards(CONV, "Q", "point_time_series", {"nse": 0.1, "pbias": 1.0})["outcome"] == "fails"   # fails wins
    assert standards(CONV, "Q", "point_snapshot", {"nrmse": 29.0})["outcome"] == "meets"
    assert standards(CONV, "E", "point_time_series", ok)["outcome"] == "no standard declared"
    assert standards(CONV, "X", "point_time_series", ok)["outcome"] == "no standard declared"
    assert standards(CONV, "Q", "spatial_snapshot", ok)["outcome"] == "no standard declared"   # shape must match
    bad = {"validation": [{"dag_variable": "Q", "obs_shape": "s", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"good": 0.6}}]}]}             # no satisfactory band
    assert standards(bad, "Q", "s", {"nse": 0.9})["metrics"]["nse"]["result"] == "band unusable"


def test_standards_read_the_variables_own_metrics():
    m = {"Q": {"nse": 0.7, "pbias": 1.0, "rmse": 1.0}, "nse": 0.1}
    assert standards(CONV, "Q", "point_time_series", m, single_variable=False)["outcome"] == "meets"
    assert standards(CONV, "Q", "point_time_series", {"nse": 0.7, "pbias": 1.0, "rmse": 1.0},
                     single_variable=False)["outcome"] == "unknown"      # flat keys only for one variable


def test_aggregate():
    assert aggregate([CONVERGED, CONVERGED]) == CONVERGED
    assert aggregate([CONVERGED, UNKNOWN]) == UNKNOWN
    assert aggregate([UNKNOWN, NOT_CONVERGED]) == NOT_CONVERGED
    assert aggregate([CONVERGED], exceeded=True) == NOT_CONVERGED
    assert aggregate([]) == UNKNOWN


# ── wiring ───────────────────────────────────────────────────────────────────────────────────────
def _run_with_convention(tmp, conv):
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_step1_panel as S1
    ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10}, None, 60)
    if conv is not None:
        Path(ki, "docs").mkdir(exist_ok=True)
        yaml.safe_dump(conv, open(Path(ki, "docs", "validation_convention.yaml"), "w"))
    with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
        rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                          budget=None, seed=0)
    return rep


def test_the_report_carries_the_verdict_safety_and_standards():
    conv = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5}}]}]}
    tmp = tempfile.mkdtemp(prefix="kdt_s3_")
    try:
        rep = _run_with_convention(tmp, conv)
        vd = rep["convergence"]["step_rule_verdict"]          # the per-call step rule (recorded)
        # the fixture is deterministic (seed 0): the stop point is premature (the loss fell afterwards)
        assert vd["verdict"] == NOT_CONVERGED and vd["how"] == "premature", vd["how"]
        assert vd["stop_point"] == rep["convergence"]["rule"]["stop_point"]
        assert set(vd["variables"]) == {"streamflow"} and "final_value_settle" in vd
        st = vd["standards"]["streamflow"]
        assert st["calibration"]["outcome"] == "meets" and st["assessed_call"] == rep["convergence"]["rule"]["incumbent_final"]
        assert st["calibration"]["metrics"]["nse"]["value"] > 0.5
        # this fixture's runner scores both splits alike, so the holdout gate is inconclusive
        assert st["validation"].startswith("not assessed: holdout inconclusive")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_safety_bounds_exactly_at_the_limits():
    # the loss bound is "< rel_gain": a move of exactly rel_gain is movement (binary-exact numbers)
    r = _rule(W=1, rel_gain=0.25)
    for loss in (4.0, 4.0, 4.0, 3.0, 3.0):
        r.add([loss], None, _q())
    assert r.stop_point == 1 and safety(r)["verdict"] == "premature"
    # the metric bound is "<= tol": a move of exactly tol is not movement
    r2 = _rule(W=1, tol={"Q": {"r": 0.125, "alpha": 0.125, "beta": 0.125}})
    for k, rv in enumerate((0.5, 0.5, 0.5, 0.625, 0.625)):
        r2.add([1.0 - 1e-9 * k], None, _q(r=rv))
    assert r2.stop_point == 1 and safety(r2)["verdict"] == "safe"


def test_early_calls_are_not_missing_evidence_for_a_variable_without_objectives():
    """A declared variable whose objectives the probe dropped: its own loss test is empty, so only
    'early' separates the first W calls from a missing metric."""
    pn = P.Panel(["Q", "S"], kinds={"Q": "series", "S": "series"})
    r = ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL), "S": dict(TOL)}, 5)
    for k in range(4):                                             # fewer than W calls in all
        r.add([1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT)})
    assert sole_blocker_calls(r, "S") == []
    assert variable_verdict(r, "S")["verdict"] == NOT_CONVERGED


def test_wiring_standards_are_read_at_the_final_incumbent():
    conv = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5}}]}]}
    tmp = tempfile.mkdtemp(prefix="kdt_s3_")
    try:
        from calibration_kit import calib as C
        from calibration_kit import test_calibrate_convergence_e2e as E
        from calibration_kit import test_step1_panel as S1
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 60)
        Path(ki, "docs").mkdir(exist_ok=True)
        yaml.safe_dump(conv, open(Path(ki, "docs", "validation_convention.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        search = [h for h in hist if h["phase"] == "search"]
        vd = rep["convergence"]["verdict"]; ru = rep["convergence"]["rule"]
        fin = ru["incumbent_final"]
        stop_inc_nse = None
        st = vd["standards"]["streamflow"]
        assert st["calibration"]["metrics"]["nse"]["value"] == pytest.approx(search[fin]["panel"]["streamflow"]["nse"])
        if st["at_stop_point"] is not None:
            assert st["at_stop_point"]["metrics"]["nse"]["value"] <= st["calibration"]["metrics"]["nse"]["value"] + 1e-12
            assert st["at_stop_point"]["metrics"]["nse"]["value"] != st["calibration"]["metrics"]["nse"]["value"], \
                "fixture must move the incumbent after the stop point"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── review round 1 (K3 + Opus 5.5) ────────────────────────────────────────────────────────────────
def test_rule_0_comes_before_rule_1():
    pn = P.Panel(["Q", "ev"], kinds={"Q": "series", "ev": "categorical"})
    r = _feed(ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL)}, 3), [([1.0], _q())] * 4)
    assert seed_verdict(r, ended_at_stop=True)["variables"]["ev"]["verdict"] == UNKNOWN


def test_movement_decides_before_missing_in_rules_3_to_5():
    r = _feed(_rule(W=2), [([1.0], _q())] * 3 + [([1.0], {"Q": {"r": 0.5, "alpha": 1.0}})] * 4)
    v = variable_verdict(r, "Q")
    assert v["verdict"] == NOT_CONVERGED and "Q:r" in v["reason"]


def test_a_never_settled_variable_is_judged_on_its_own_sole_blockers_and_its_own_movement():
    # S never settles (beta missing), Q fully recorded but its loss keeps moving -> seed J empty;
    # S's own sole-blocker calls exist and nothing of S moved -> S unknown
    kinds = {"Q": "series", "S": "series"}
    r = _rule(W=2, kinds=kinds)
    for k in range(10):
        r.add([1.0 - 0.1 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": {"r": 0.9, "alpha": 1.0}})
    assert sole_blocker_calls(r) == [] and sole_blocker_calls(r, "S")
    assert variable_verdict(r, "S")["verdict"] == UNKNOWN
    # the same, but S's own loss keeps moving after its sole-blocker calls -> not converged
    r2 = _rule(W=2, kinds=kinds)
    for k in range(10):
        s_loss = 1.0 if k < 5 else 1.0 - 0.1 * (k - 4)
        r2.add([1.0 - 0.1 * k, s_loss - 1e-9 * k], None, {"Q": dict(FLAT), "S": {"r": 0.9, "alpha": 1.0}})
    assert variable_verdict(r2, "S")["verdict"] == NOT_CONVERGED


def test_exactly_w_calls_after_the_reference_is_enough():
    r = _feed(_rule(W=3), [([1.0], _q())] * 7)                # s = 3, 3 calls after
    assert safety(r)["verdict"] == "safe" and safety(r)["calls_after"] == 3
    assert variable_verdict(r, "Q")["verdict"] == CONVERGED


def _front(W=2):
    pn = P.Panel(["Q", "S"], kinds={"Q": "series", "S": "categorical"})
    return ConvergenceRule([("Q:a", "Q"), ("S:b", "S")], pn, {"Q": dict(TOL)}, W, trade_off=True)


def test_trade_off_seed_rule_3_needs_the_front_test_at_j_and_counts_a_front_move_after_j():
    no_beta = {"Q": {"r": 0.9, "alpha": 1.0}}
    # front settled, only Q:beta missing -> unknown
    r = _front()
    for p in ([1.0, 3.0], [3.0, 1.0], [2.0, 2.0]) + ([2.0, 2.0],) * 5:
        r.add(p, None, no_beta)
    assert seed_verdict(r)["verdict"] == UNKNOWN
    # the front keeps improving (condition 4 fails at every j) -> not converged
    r2 = _front()
    for k, p in enumerate(([1.0, 3.0], [3.0, 1.0]) + tuple([2.0 - 0.3 * k, 2.0 - 0.3 * k] for k in range(1, 7))):
        r2.add(p, None, no_beta)
    assert seed_verdict(r2)["verdict"] == NOT_CONVERGED
    # the front settles, then jumps later: from the quiet j's the front moved -> not converged
    r3 = _front(W=1)
    for p in ([1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [0.5, 0.5]):
        r3.add(p, None, no_beta)
    assert sole_blocker_calls(r3) and seed_verdict(r3)["verdict"] == NOT_CONVERGED


def test_the_cap_never_turns_not_converged_into_unknown():
    pn = P.Panel(["Q"], kinds={"Q": "series"})
    pn.apply_pilot({"kit_missing": {"Q": {"beta": "mean near zero"}}})
    r = ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL)}, 3)
    _feed(r, [([1.0 - 0.05 * k], {"Q": {"r": 0.9, "alpha": 1.0}}) for k in range(8)])
    v = seed_verdict(r)
    assert v["verdict"] == NOT_CONVERGED and v["variables"]["Q"]["verdict"] == NOT_CONVERGED


def test_the_cap_also_applies_to_a_rule_triggered_seed():
    pn = P.Panel(["Q"], kinds={"Q": "series"})
    pn.apply_pilot({"kit_missing": {"Q": {"beta": "mean near zero"}}})
    r = _feed(ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL)}, 3), [([1.0], {"Q": {"r": 0.9, "alpha": 1.0}})] * 4)
    v = seed_verdict(r, ended_at_stop=True)
    assert v["verdict"] == UNKNOWN and v["how"].startswith("rule triggered; capped: Q:beta missing")


def test_the_every_j_rule_when_all_reference_points_agree():
    """No stop point (Q's lnnse never recorded); S settled early and never moved again: judged from
    its settle point and every sole-blocker call, all say converged."""
    kinds = {"Q": "flow", "S": "series"}
    r = _rule(W=2, kinds=kinds, tol={"Q": dict(TOL, lnnse=0.01), "S": dict(TOL)})
    for k in range(10):
        r.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT)})
    J = sole_blocker_calls(r)
    assert any(len(r.F) - 1 - j < r.W for j in J)          # blocker calls inside the last W calls exist
    # Leo 2026-09-29 (option b): those have no vote, so a flat S reads converged
    v = variable_verdict(r, "S")
    assert v["verdict"] == CONVERGED and "all agree" in v["reason"]


def test_standards_boundaries_and_unknown_direction():
    c = {"validation": [{"dag_variable": "Q", "obs_shape": "s", "headline_metrics": [
        {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5}},
        {"metric": "rmse", "direction": "minimize", "bands": {"satisfactory": 2.0}},
        {"metric": "kge", "direction": "sideways", "bands": {"satisfactory": 0.5}}]}]}
    got = standards(c, "Q", "s", {"nse": 0.5, "rmse": 2.0, "kge": 0.9})
    assert got["metrics"]["nse"]["result"] == "meets" and got["metrics"]["rmse"]["result"] == "meets"
    assert got["metrics"]["kge"]["result"] == "band unusable" and got["outcome"] == "unknown"
    assert standards(c, "Q", None, {"nse": 0.9})["why"].startswith("the variable's obs_shape is not known")


def test_final_value_settle_boundaries_and_gaps():
    rows = [([1.0], _q(beta=b)) for b in (1.25, 1.125, 1.0, 1.0)]   # 1.125 differs by exactly tol
    r = _feed(_rule(W=2, tol={"Q": {"r": 0.125, "alpha": 0.125, "beta": 0.125}}), rows)
    assert final_value_settle_points(r)["Q:beta"] == 1
    rows = [([1.0], _q()), ([1.0], {"Q": {"r": 0.9, "alpha": 1.0}}), ([1.0], _q()), ([1.0], _q())]
    r2 = _feed(_rule(W=2), rows)
    assert final_value_settle_points(r2)["Q:beta"] == 2               # a gap counts as "not settled"


def test_not_tracked_backends_and_empty_searches():
    r = _rule()
    assert seed_verdict(r)["how"].startswith("convergence not tracked")
    r2 = _feed(_rule(), [([1.0], _q())] * 8)
    for algo in ("surrogate", "smt", "pestpp_ies", "madr"):
        assert seed_verdict(r2, algorithm=algo)["how"].startswith("convergence not tracked")
    assert seed_verdict(r2, algorithm="dds")["verdict"] == CONVERGED
    r3 = _rule()
    for _ in range(5):
        r3.add(None, None, {})                                     # every call failed
    assert seed_verdict(r3)["how"] == "convergence not tracked (the rule saw no call with a finite loss)"


def test_the_missing_metric_is_named():
    r = _feed(_rule(W=2), [([1.0], _q())] * 3 + [([1.0], {"Q": {"r": 0.9, "alpha": 1.0}})] * 4)
    assert "Q:beta" in variable_verdict(r, "Q")["reason"]
    assert safety(r)["missing_after"] == ["Q:beta"]


def test_long_searches_stay_fast():
    import time
    kinds = {"Q": "flow", "S": "series"}
    pn = P.Panel(list(kinds), kinds=kinds)
    tol = {v: {m: 0.01 for m in P.KIND_TABLE[k]["recorded"]} for v, k in kinds.items()}
    r = ConvergenceRule([("Q:f", "Q"), ("S:f", "S")], pn, tol, 50)
    for i in range(5000):
        r.add([1.0 - 1e-9 * i] * 2, None, {v: dict(FLAT) for v in kinds})
    t = time.time()
    seed_verdict(r)
    assert time.time() - t < 5.0


# ── wiring (round 1) ──────────────────────────────────────────────────────────────────────────────
def _calib(tmp, runner_factory, strategy=None, families=None, conv=None, budget_block="default", cap=60,
           env=None):
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    # one search per run: these tests are about a single seed's rule and verdicts (seeds: build step 6)
    ki, wd = E._fixture(tmp, ({"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}
                              if budget_block == "default" else budget_block), None, cap)
    c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strategy or {})
    yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
    if families:
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = families
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
    if conv is not None:
        Path(ki, "docs").mkdir(exist_ok=True)
        yaml.safe_dump(conv, open(Path(ki, "docs", "validation_convention.yaml"), "w"))
    with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
        for k, v in (env or {}).items():
            os.environ[k] = v
        try:
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=runner_factory(wd),
                              budget=None, seed=0)
        finally:
            for k in (env or {}):
                os.environ.pop(k, None)
    hp = Path(wd, "eval_history.jsonl")
    hist = [json.loads(l) for l in hp.read_text().splitlines()] if hp.exists() else []
    return rep, hist


CONV_NSE = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
    {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5}}]}]}


class _PullApart:
    """nse best at a = 1, |pbias| best at a = 1.5: a real two-point front."""
    def __init__(self, wd):
        self.wd = wd

    def __call__(self):
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        a = float(p["a"])
        return {"nse": 1.0 - (a - 1.0) ** 2, "pbias": 100.0 * (a - 1.5) ** 2, "r": 0.9, "kge": 0.8,
                "__kdt__": {"applied_params": dict(p), "case_id": "SITE:fixture"}}


def test_wiring_trade_off_standards_are_known_whichever_archive_member_is_the_compromise():
    pytest.importorskip("pymoo")
    tmp = tempfile.mkdtemp(prefix="kdt_s3_")
    try:
        rep, hist = _calib(tmp, _PullApart, strategy={"multi_objective": True, "default_algorithm": "nsga2"},
                           families=["temporal_pattern_match", "magnitude_accuracy"], conv=CONV_NSE)
        ru = rep["convergence"]["rule"]; vd = rep["convergence"]["verdict"]
        assert ru["trade_off"] is True
        st = vd["standards"]["streamflow"]
        search = [h for h in hist if h["phase"] == "search"]
        fin = ru["incumbent_final"]
        assert st["calibration"]["metrics"]["nse"]["value"] == pytest.approx(1.0 - (search[fin]["x"]["a"] - 1.0) ** 2)
        # every call with a finite loss is kept, so ANY archive member can be assessed
        assert st["calibration"]["outcome"] in ("meets", "fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_dream_seed_verdict_comes_from_r_hat():
    tmp = tempfile.mkdtemp(prefix="kdt_s3_")
    try:
        from calibration_kit import test_step1_panel as S1
        rep, _ = _calib(tmp, S1._Runner, env={"KDT_CALIB_ALGO": "dream"}, cap=200)
        assert rep["algorithm"] == "dream", rep.get("algorithm")
        # step 4: R-hat is recorded now; the DREAM seed verdict comes from it
        t = rep["convergence"]["optimizer_termination"]
        assert t["rhat_recorded"] is True and rep["convergence"]["verdict"]["how"].startswith("DREAM: ")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_a_surrogate_search_is_not_tracked():
    tmp = tempfile.mkdtemp(prefix="kdt_s3_")
    try:
        from calibration_kit import test_step1_panel as S1
        from calibration_kit import calib as C
        be = C._make_backend("smt")
        if be is None or not be.available():
            pytest.skip("surrogate backend not available")
        rep, _ = _calib(tmp, S1._Runner, strategy={"default_algorithm": "smt"})
        # whether or not the surrogate library works here, the report never judges convergence
        assert rep["algorithm"] == "smt"
        assert rep["convergence"]["verdict"]["how"].startswith("convergence not tracked (smt")
        e = rep["convergence"]["ended"]
        assert e["text"] == "no call was made" if e["calls"] == 0 else e["text"].startswith("convergence not tracked")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _SplitRunner:
    """Honours the split: on holdout, sim is 30 % high, so the holdout gate is not inconclusive and the
    reported member has holdout metrics."""
    def __init__(self, wd):
        from calibration_kit import test_calibrate_convergence_e2e as E
        from calibration_kit import panel as PN
        self.E, self.PN = E, PN

    def __call__(self):
        E, PN = self.E, self.PN
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        hold = os.environ.get("KDT_CALIB_SPLIT") == "holdout"
        sl = E.HOLD if hold else E.CAL
        sim = (float(p["a"]) * E.OBS + float(p["b"]))[sl] * (1.3 if hold else 1.0)
        obs = E.OBS[sl]
        full = PN.panel_from_series(sim, obs, "flow")
        return {**{k: full[k] for k in ("nse", "kge", "r", "pbias")},
                "__kdt__": {"applied_params": dict(p), "case_id": "SITE:fixture"}}


def test_wiring_validation_standards_come_from_the_holdout_split():
    tmp = tempfile.mkdtemp(prefix="kdt_s3_")
    try:
        rep, hist = _calib(tmp, _SplitRunner, conv=CONV_NSE)
        st = rep["convergence"]["verdict"]["standards"]["streamflow"]
        assert isinstance(st["validation"], dict), st["validation"]
        cal = st["calibration"]["metrics"]["nse"]["value"]; val = st["validation"]["metrics"]["nse"]["value"]
        assert val < cal - 0.05                        # holdout sim is 30 % high: a worse nse
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── round 1, second pass: cases the first tests could not separate ─────────────────────────────────
def test_seed_rule_3_needs_the_front_test_at_j_even_when_every_best_loss_is_flat():
    """The front keeps changing in the MIDDLE (B_k flat) until the last call: condition 4 never holds,
    so no call is a sole blocker -> not converged."""
    no_beta = {"Q": {"r": 0.9, "alpha": 1.0}}
    r = _front(W=2)
    mids = [[1.0, 3.0], [3.0, 1.0]] + [[2.0 - 0.1 * k, 2.0 + 0.1 * k] if k % 2 else [2.0 + 0.1 * k, 2.0 - 0.1 * k]
                                        for k in range(1, 9)]
    for p in mids:
        r.add(p, None, no_beta)
    assert all(rec["cond4"] is not True for rec in r.cond)
    assert sole_blocker_calls(r) == [] and seed_verdict(r)["verdict"] == NOT_CONVERGED


def test_seed_rule_3_counts_a_front_move_after_j_even_when_every_best_loss_is_flat():
    no_beta = {"Q": {"r": 0.9, "alpha": 1.0}}
    r = _front(W=1)
    for p in ([1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [1.5, 1.5]):   # B stays [1, 1]
        r.add(p, None, no_beta)
    assert r.V[-1] == [1.0, 1.0] and sole_blocker_calls(r)
    assert seed_verdict(r)["verdict"] == NOT_CONVERGED


def test_the_seed_cap_never_turns_a_not_converged_seed_into_unknown():
    pn = P.Panel(["Q", "S"], kinds={"Q": "series", "S": "series"})
    pn.apply_pilot({"kit_missing": {"Q": {"beta": "mean near zero"}}})
    r = ConvergenceRule([("Q:f", "Q"), ("S:f", "S")], pn, {"Q": dict(TOL), "S": dict(TOL)}, 2)
    for k in range(10):
        r.add([1.0 - 1e-9 * k, 1.0 - 0.1 * k], None, {"Q": {"r": 0.9, "alpha": 1.0}, "S": dict(FLAT)})
    v = seed_verdict(r)
    assert v["variables"]["Q"].get("capped") and v["variables"]["S"]["verdict"] == NOT_CONVERGED
    assert v["verdict"] == NOT_CONVERGED


def test_wiring_the_runner_metrics_of_every_finite_call_are_kept():
    from calibration_kit import test_step1_panel as S1
    tmp = tempfile.mkdtemp(prefix="kdt_s3_")
    try:
        rep, hist = _calib(tmp, S1._Runner)
        search = [h for h in hist if h["phase"] == "search"]
        finite = [h for h in search if h.get("losses") and any(
            x is not None and math.isfinite(float(x)) for x in h["losses"])]
        assert rep["convergence"]["rule"]["calls_with_runner_metrics"] == len(finite) > 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── round 2 (Opus 5.5) ────────────────────────────────────────────────────────────────────────────
def _qs(W, tol=None):
    kinds = {"Q": "flow", "S": "series"}
    pn = P.Panel(list(kinds), kinds=kinds)
    tol = tol if tol is not None else {"Q": dict(TOL, lnnse=0.01), "S": dict(TOL)}
    return ConvergenceRule([("Q:f", "Q"), ("S:f", "S")], pn, tol, W)


def test_a_metric_missing_at_the_reference_call_is_missing_evidence():
    """A seed-level sole-blocker call can come BEFORE the variable's first settle point."""
    r, n = _qs(2), 14
    for k in range(n):
        q_loss = 1.0 if k < n - 2 else 0.5
        s_pan = {"r": 0.9, "alpha": 1.0} if k <= 2 else dict(FLAT)
        r.add([q_loss - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": s_pan})
    J = sole_blocker_calls(r)
    assert J and min(J) < r.first_settle["S"]
    v = variable_verdict(r, "S")
    assert v["verdict"] == UNKNOWN and "S:beta" in v["reason"]


def test_a_metric_missing_at_one_call_after_the_stop_point_is_undecidable():
    r = _rule(W=3)
    for k, pan in enumerate([FLAT] * 5 + [{"r": 0.9, "alpha": 1.0}] + [FLAT] * 3):
        r.add([1.0 - 1e-9 * k], None, {"Q": dict(pan)})
    v = seed_verdict(r)
    assert r.stop_point == 3 and v["verdict"] == UNKNOWN and v["how"].startswith("undecidable")
    assert "Q:beta" in v["how"]


def test_the_every_j_rule_includes_the_first_settle_point():
    r, n = _qs(2), 20
    for k in range(n):
        q_loss = 2.0 - 0.1 * k if k < 10 else (1.0 if k < n - 2 else 0.5)
        s_beta = 1.0 if k < 6 else 1.05
        r.add([q_loss - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT, beta=s_beta)})
    v = variable_verdict(r, "S")
    assert v["verdict"] == UNKNOWN and f"moved after call {r.first_settle['S']}" in v["reason"]


def test_the_safety_loss_bound_uses_the_floor_near_a_perfect_fit():
    r = _rule(W=3)
    for loss in [1e-5] * 5 + [0.8e-5] * 4:
        r.add([loss], None, _q())
    v = seed_verdict(r)
    assert v["verdict"] == CONVERGED and v["how"].startswith("confirmed")
    assert v["safety"]["largest_move_after"]["loss:Q:f"] == pytest.approx(2e-6 / 1e-3)


def test_reasons_name_the_blocking_metric_of_another_variable_and_a_missing_tolerance():
    r = _qs(2)
    for k in range(10):
        r.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT)})
    assert "Q:lnnse" in variable_verdict(r, "S")["reason"]
    r2 = _qs(2, tol={"Q": {}, "S": dict(TOL)})                    # the tolerance step failed for Q
    for k in range(10):
        r2.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT, lnnse=0.5), "S": dict(FLAT)})
    how = seed_verdict(r2)["how"]
    assert "Q:r (no tolerance)" in how and "a metric" not in how


# ── Leo's rule-2 decision (2026-09-29, option b) ──────────────────────────────────────────────────
def test_late_blocker_calls_have_no_vote_but_a_late_jump_is_still_caught():
    # the example: S flat from its settle point to the end; Q:lnnse never recorded
    r = _qs(20)
    for k in range(120):
        r.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT)})
    assert variable_verdict(r, "S")["verdict"] == CONVERGED
    assert seed_verdict(r)["verdict"] == UNKNOWN and variable_verdict(r, "Q")["verdict"] == UNKNOWN
    # S jumps at call 100 (inside the searches' last stretch but with W calls after some blockers)
    r2 = _qs(10)
    for k in range(120):
        beta = 1.0 if k < 100 else 1.05
        r2.add([1.0 - 1e-9 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT, beta=beta)})
    assert variable_verdict(r2, "S")["verdict"] == NOT_CONVERGED


def test_when_every_blocker_call_is_in_the_last_w_the_settle_point_alone_decides():
    """S settles early; the search (Q's loss) keeps improving until the last W calls, so every blocker
    call is inside the last W: S is judged from its first settle point alone."""
    r, n, W = _qs(5), 40, 5
    for k in range(n):
        flat_from = n - W - 2                    # blocker calls only at the last 2 calls
        q_loss = 2.0 - 0.02 * k if k < flat_from else 2.0 - 0.02 * flat_from
        r.add([q_loss, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT)})
    J = sole_blocker_calls(r)
    assert J and all(len(r.F) - 1 - j < W for j in J)
    v = variable_verdict(r, "S")
    assert v["verdict"] == CONVERGED and "alone" in v["reason"] and "Q:lnnse" in v["reason"]
    assert seed_verdict(r)["verdict"] == UNKNOWN               # seed rule 3 is not filtered


def test_a_blocker_call_with_exactly_w_calls_after_it_still_votes():
    """The earliest blocker j has exactly W calls after it. S drifts within tol of its settle value but
    by more than tol from its value at j: from the settle point converged, from j moved -> they
    disagree -> unknown. Without j's vote it would read converged."""
    W, n = 5, 40
    r = _qs(W, tol={"Q": dict(TOL, lnnse=0.01), "S": {"r": 0.01, "alpha": 0.01, "beta": 0.01}})
    f = n - 1 - 2 * W                                  # Q's loss flat from f -> blockers from f + W = n-1-W
    j0 = n - 1 - W
    for k in range(n):
        q_loss = 2.0 - 0.02 * min(k, f)
        beta = 1.0 if k < j0 else (1.008 if k < n - 1 else 0.997)
        r.add([q_loss - 1e-12 * k, 1.0 - 1e-9 * k], None, {"Q": dict(FLAT), "S": dict(FLAT, beta=beta)})
    J = sole_blocker_calls(r)
    assert J and min(J) == j0
    v = variable_verdict(r, "S")
    assert v["verdict"] == UNKNOWN and "Q:lnnse" in v["reason"]


def test_a_blocker_call_with_w_minus_1_calls_after_it_has_no_vote_even_when_it_shows_a_move():
    """The mirror: the drift starts one call later, so only the call with W-1 calls after it would see
    the move — it has no vote (Leo, option b), so S reads converged."""
    W, n = 5, 40
    r = _qs(W)
    f, j0 = n - 1 - 2 * W, n - 1 - W
    for k in range(n):
        beta = 1.0 if k <= j0 else (1.008 if k < n - 1 else 0.997)
        r.add([2.0 - 0.02 * min(k, f) - 1e-12 * k, 1.0 - 1e-9 * k], None,
              {"Q": dict(FLAT), "S": dict(FLAT, beta=beta)})
    J = sole_blocker_calls(r)
    assert min(J) == j0 and j0 + 1 in J
    assert variable_verdict(r, "S")["verdict"] == CONVERGED
