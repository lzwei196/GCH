"""Build step 2 (design HANDOFF_CONVERGENCE_2026-09-27_v2.md §2.2-§2.6; gaps 2g, 2d, 2e): each test
checks one rule of the convergence rule (rule.py)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit import panel as P                                              # noqa: E402
from calibration_kit.rule import (ConvergenceRule, admissible, check_protect, design_window,  # noqa: E402
                                  eps_indicator, resolve_protect)

TOL = {"r": 0.01, "alpha": 0.01, "beta": 0.01}


def _panel(vars_kinds):
    return P.Panel(list(vars_kinds), kinds=dict(vars_kinds))


def _rule(W=5, objectives=(("Q:f", "Q"),), kinds=None, tol=None, **kw):
    kinds = kinds or {"Q": "series"}
    pn = _panel(kinds)
    tol = tol or {v: dict(TOL) for v in kinds}
    return ConvergenceRule(list(objectives), pn, tol, W, **kw)


PAN = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}}


# ── 2g: window (§2.3) ────────────────────────────────────────────────────────────────────────────
def test_window_formula():
    assert design_window(2, 1000) == 20            # max(20, 5*3=15)
    assert design_window(10, 1000) == 55           # 5*11
    assert design_window(10, 60) == 30             # capped at floor(cap/2)
    assert design_window(3, 30) == 15              # W < 20 only when the cap is below 40
    assert design_window(3, 39) == 19 and design_window(3, 40) == 20
    assert design_window(0, 1) == 1


# ── 2d: condition 1, the loss test (§2.2) ─────────────────────────────────────────────────────────
def test_loss_test_threshold_floor_and_window():
    r = _rule(W=3)
    for loss in (1.0, 1.0, 1.0):
        rec = r.add([loss], None, PAN)
        assert rec["cond1"] == [False]            # i < W: never
    rec = r.add([0.996], None, PAN)               # 0.4 % below the value W calls ago
    assert rec["cond1"] == [True]
    r2 = _rule(W=1)
    r2.add([1.0], None, PAN)
    assert r2.add([0.994], None, PAN)["cond1"] == [False]      # 0.6 % -> still improving
    # near a perfect fit the denominator is floored at 1e-3
    r4 = _rule(W=1)
    r4.add([2e-6], None, PAN)
    assert r4.add([0.0], None, PAN)["cond1"] == [True]        # |2e-6|/1e-3 = 0.002 < 0.005
    r5 = _rule(W=1)
    r5.add([1e-5], None, PAN)
    assert r5.add([0.0], None, PAN)["cond1"] == [False]       # 1e-5/1e-3 = 0.01 >= 0.005


def test_single_objective_tracks_each_component_at_the_incumbent_not_the_scalar():
    objs = (("Q:a", "Q"), ("Q:b", "Q"))
    r = _rule(W=2, objectives=objs)
    r.add([1.0, 1.0], 1.0, PAN)
    r.add([1.0, 1.0], 1.0, PAN)
    # a new incumbent whose scalar is only 0.3 % better: component a moved 0.1 %, b moved 0.7 %
    rec = r.add([1.001, 0.993], 0.997, PAN)
    assert rec["inc"] == 2 and rec["cond1"] == [True, False]
    assert not rec["stop"]


def test_single_objective_incumbent_is_the_lowest_loss_ties_earliest_and_failures_never_win():
    r = _rule(W=50)
    r.add([2.0], None, PAN); r.add([1.0], None, PAN); r.add([1.0], None, PAN)
    assert r.inc[-1] == 1
    r.add(None, None, {})                           # a failed / rejected call
    r.add([float("nan")], None, PAN)
    assert r.inc[-1] == 1 and r.X[-1] == PAN


def test_trade_off_tracks_the_best_loss_of_each_objective_so_far():
    objs = (("Q:a", "Q"), ("S:b", "S"))
    r = _rule(W=50, objectives=objs, kinds={"Q": "series", "S": "series"}, trade_off=True)
    r.add([3.0, 1.0], None, {}); r.add([1.0, 3.0], None, {}); r.add([2.0, 2.0], None, {})
    assert r.V[-1] == [1.0, 1.0]


# ── 2d: conditions 2-3, the panel test (§2.5) ──────────────────────────────────────────────────────
def _feed(r, panels, loss=1.0):
    out = None
    for p in panels:
        out = r.add([loss], None, p)
    return out


def test_panel_needs_every_call_of_the_window_and_uses_an_inclusive_bound():
    base = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}}
    r = _rule(W=3)
    rec = _feed(r, [base] * 4)
    assert rec["cond2"]["Q"] == {"r": "ok", "alpha": "ok", "beta": "ok"} and rec["stop"]
    # exactly at the tolerance is still flat (inclusive); binary-exact numbers, and the second call is
    # a new incumbent so its panel is the one read
    r2 = _rule(W=1, tol={"Q": {"r": 0.125, "alpha": 0.125, "beta": 0.125}})
    r2.add([1.0], None, {"Q": {"r": 0.5, "alpha": 1.0, "beta": 1.0}})
    rec = r2.add([0.999], None, {"Q": {"r": 0.625, "alpha": 1.0, "beta": 1.0}})
    assert rec["inc"] == 1 and rec["cond2"]["Q"]["r"] == "ok"
    rec = r2.add([0.998], None, {"Q": {"r": 0.75 + 1e-9, "alpha": 1.0, "beta": 1.0}})
    assert rec["cond2"]["Q"]["r"] == "moved"


def test_a_missing_value_anywhere_in_the_window_is_unknown_and_blocks():
    base = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}}
    hole = {"Q": {"r": 0.9, "alpha": 1.0}}               # beta missing at one call
    r = _rule(W=3)
    # the incumbent's panel is read, so the hole must be at the incumbent: make each call a new best
    for k, p in enumerate([base, hole, base, base]):
        rec = r.add([1.0 - 1e-6 * k], None, p)
    assert rec["cond2"]["Q"]["beta"] == "unknown" and rec["cond3"] is False and not rec["stop"]


def test_an_oscillation_that_returns_is_still_movement():
    r = _rule(W=3)
    vals = [0.9, 0.95, 0.9, 0.9]
    for k, v in enumerate(vals):
        rec = r.add([1.0 - 1e-6 * k], None, {"Q": {"r": v, "alpha": 1.0, "beta": 1.0}})
    assert rec["cond2"]["Q"]["r"] == "moved" and not rec["stop"]


def test_only_required_metrics_are_tested_and_recorded_ones_never_block():
    r = _rule(W=2)
    for k in range(3):
        rec = r.add([1.0], None, {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0, "nse": 0.1 * k}})
    assert set(rec["cond2"]["Q"]) == {"r", "alpha", "beta"} and rec["stop"]


# ── several variables: each settles on its own, all at the same call (§1.5, §2.5) ─────────────────
def test_variables_are_never_combined_and_the_stop_needs_all_at_the_same_call():
    objs = (("Q:f", "Q"), ("S:f", "S"))
    r = _rule(W=2, objectives=objs, kinds={"Q": "series", "S": "series"})
    flat = {"r": 0.9, "alpha": 1.0, "beta": 1.0}
    # every call is a (tiny) new best, so the incumbent's panel is each call's panel
    for k in range(8):
        s_beta = 1.0 + 0.05 * min(k, 4)                      # S settles only after call 4
        rec = r.add([1.0 - 1e-6 * k, 1.0 - 1e-6 * k], 1.0 - 1e-6 * k, {"Q": dict(flat), "S": dict(flat, beta=s_beta)})
    assert r.first_settle["Q"] == 2 and r.first_settle["S"] == 6
    assert r.stop_point == 6


def test_a_rule0_variable_never_blocks_and_a_categorical_one_is_judged_on_its_loss_only():
    objs = (("Q:f", "Q"), ("fld:csi", "fld"))
    pn = P.Panel(["Q", "fld", "ev"], kinds={"Q": "series", "fld": "categorical", "ev": "categorical"})
    r = ConvergenceRule(list(objs), pn, {"Q": dict(TOL)}, 2)
    assert r.variables == ["Q", "fld"] and r.rule0 == ["ev"]
    for k in range(4):
        rec = r.add([1.0, 0.5], 0.75, dict(PAN))
    assert rec["settled"] == {"Q": True, "fld": True} and rec["stop"]
    r2 = ConvergenceRule(list(objs), pn, {"Q": dict(TOL)}, 2)
    for k in range(4):
        rec = r2.add([1.0 - 1e-6 * k, 0.5 * (1 - 0.1 * k)], 0.75 - 1e-6 * k, dict(PAN))
    assert rec["settled"]["fld"] is False and not rec["stop"]


def test_a_variable_with_a_panel_but_no_objective_is_judged_on_its_panel():
    pn = P.Panel(["Q", "S"], kinds={"Q": "series", "S": "series"})
    r = ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL), "S": dict(TOL)}, 2)
    assert r.variables == ["Q", "S"]
    for k in range(4):
        rec = r.add([1.0 - 1e-6 * k], None, {"Q": dict(PAN["Q"]), "S": {"r": 0.5 + 0.1 * k, "alpha": 1, "beta": 1}})
    assert rec["settled"]["S"] is False and not rec["stop"]


# ── 2d: condition 4, the front test (§2.6) ─────────────────────────────────────────────────────────
def test_eps_indicator_definition():
    s = [1.0, 1.0]
    assert eps_indicator([[0, 1], [1, 0]], [[0, 1], [1, 0]], s) == 0.0
    # the new front improved objective 0 by 0.2 at one point: the old front misses it by 0.2
    assert eps_indicator([[0.5, 0.5]], [[0.3, 0.5]], s) == pytest.approx(0.2)
    # a new front the old one dominates is covered with a negative margin
    assert eps_indicator([[0.3, 0.3]], [[0.5, 0.5]], s) == pytest.approx(-0.2)
    # scale matters per objective
    assert eps_indicator([[0.5, 0.5]], [[0.3, 0.5]], [0.1, 1.0]) == pytest.approx(2.0)


def _front_rule(W=3, **kw):
    objs = (("Q:a", "Q"), ("S:b", "S"))
    return _rule(W=W, objectives=objs, kinds={"Q": "categorical", "S": "categorical"},
                 tol={}, trade_off=True, **kw)


def test_scale_is_frozen_once_at_t0_with_its_floors():
    r = _front_rule(W=3)
    r.add([1.0, 5.0], None, {}); r.add([2.0, 5.0], None, {})           # dominated: archive 1 point
    r.add([1.5, 5.0], None, {}); assert r.t0 is None
    r.add([3.0, 4.0], None, {})                                          # i = 3 >= W, archive = 2
    assert r.t0 == 3 and r.z == [1.0, 4.0] and r.s == [2.0, 1.0]
    r.add([0.0, 0.0], None, {})                                          # later points never re-freeze
    assert r.z == [1.0, 4.0] and r.s == [2.0, 1.0]
    # floors: three objectives, two DISTINCT points, objective 2 constant at t0
    objs3 = (("Q:a", "Q"), ("S:b", "S"), ("T:c", "T"))
    q = _rule(W=1, objectives=objs3, kinds={"Q": "categorical", "S": "categorical", "T": "categorical"},
              tol={}, trade_off=True)
    q.add([1.0, 2.0, 7.0], None, {}); q.add([2.0, 1.0, 7.0], None, {})
    assert q.t0 == 1 and q.s[:2] == [1.0, 1.0] and q.s[2] == pytest.approx(7e-6)
    z = _rule(W=1, objectives=objs3, kinds={"Q": "categorical", "S": "categorical", "T": "categorical"},
              tol={}, trade_off=True)
    z.add([1.0, 2.0, 0.0], None, {}); z.add([2.0, 1.0, 0.0], None, {})
    assert z.s[2] == 1e-12                                              # the absolute floor


def test_front_test_holds_only_after_t0_and_blocks_a_moving_front():
    r = _front_rule(W=2)
    pts = [[1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]
    recs = [r.add(p, None, {}) for p in pts]
    assert recs[1]["cond4"] is None                                      # before t0
    assert r.t0 == 2 and recs[4]["cond4"] is True and r.stop_point == 4
    r2 = _front_rule(W=2)
    for k, p in enumerate(pts[:3] + [[1.5, 1.5], [1.2, 1.2], [1.0, 1.0]]):
        rec = r2.add(p, None, {})
    assert rec["cond4"] is False and r2.stop_point is None


# ── 2e: compromise and protection (§2.4) ───────────────────────────────────────────────────────────
def test_compromise_is_minimax_on_the_frozen_scale_then_sum_then_earliest():
    r = _front_rule(W=1)
    r.add([0.0, 10.0], None, {}); r.add([10.0, 0.0], None, {})         # t0 = 1: z = [0,0], s = [10,10]
    assert r.t0 == 1
    r.add([4.0, 6.0], None, {})
    assert r.inc[-1] == 2                                                # worst normalized 0.6 < 1.0
    r.add([6.0, 4.0], None, {})
    assert r.inc[-1] == 2                                                # same max and sum -> earliest
    r.add([5.0, 5.0], None, {})
    assert r.inc[-1] == 4                                                # 0.5 < 0.6


def test_before_t0_the_compromise_is_the_smallest_raw_sum():
    r = _front_rule(W=10)
    r.add([0.0, 10.0], None, {}); r.add([6.0, 3.0], None, {})
    assert r.t0 is None and r.inc[-1] == 1


def test_admissibility_formulas():
    d = {"Q": {"r": 0.8, "beta": 1.1, "alpha": 0.9, "pbias": 10.0, "nrmse": 20.0, "lnnse": 0.5}}
    t = {"Q": {"r": 0.01, "beta": 0.01, "alpha": 0.01, "pbias": 1.0, "nrmse": 1.0, "lnnse": 0.01}}
    act = {"Q": ["r", "beta", "alpha", "pbias", "nrmse", "lnnse"]}
    ok = {"Q": {"r": 0.79, "beta": 0.895, "alpha": 1.105, "pbias": -11.0, "nrmse": 21.0, "lnnse": 0.49}}
    assert admissible(ok, act, d, t)
    for m, bad in (("r", 0.78), ("beta", 0.88), ("alpha", 1.12), ("pbias", 11.5), ("nrmse", 21.5), ("lnnse", 0.48)):
        cand = {"Q": dict(ok["Q"], **{m: bad})}
        assert not admissible(cand, act, d, t), m
    missing = {"Q": {k: v for k, v in ok["Q"].items() if k != "beta"}}
    assert not admissible(missing, act, d, t)


def test_protection_picks_among_admissible_and_flags_infeasible():
    objs = (("Q:a", "Q"), ("S:b", "S"))
    pn = _panel({"Q": "series", "S": "series"})
    tol = {"Q": {"r": 0.01, "alpha": 0.01, "beta": 0.01}, "S": dict(TOL)}
    dflt = {"Q": {"r": 0.8}}
    r = ConvergenceRule(list(objs), pn, tol, 1, trade_off=True, protect={"Q": ["r"]}, default_panel=dflt)
    r.add([0.0, 10.0], None, {"Q": {"r": 0.85}}); r.add([10.0, 0.0], None, {"Q": {"r": 0.85}})
    r.add([5.0, 5.0], None, {"Q": {"r": 0.5}})                          # most balanced, but loses r
    assert r.inc[-1] in (0, 1) and r.infeasible[-1] is False
    r2 = ConvergenceRule(list(objs), pn, tol, 1, trade_off=True, protect={"Q": ["r"]}, default_panel=dflt)
    r2.add([0.0, 10.0], None, {"Q": {"r": 0.5}}); r2.add([10.0, 0.0], None, {"Q": {"r": 0.5}})
    r2.add([5.0, 5.0], None, {"Q": {"r": 0.5}})
    assert r2.infeasible[-1] is True and r2.inc[-1] == 2                 # most balanced, flagged
    assert r2.summary()["protection_infeasible_final"] is True


def test_protect_load_check():
    pn = P.Panel(["Q", "fld"], kinds={"Q": "flow", "fld": "categorical"})
    assert check_protect({"Q": ["r", "BETA", "lnnse"]}, pn) == {"Q": ["r", "beta", "lnnse"]}
    assert check_protect(None, pn) == {}
    for bad in ({"X": ["r"]}, {"fld": ["r"]}, {"Q": ["csi"]}, {"Q": "r"}, ["Q"]):
        with pytest.raises(ValueError):
            check_protect(bad, pn)


def test_protected_metrics_missing_at_the_default_run_are_dropped_with_the_reason():
    pn = P.Panel(["Q", "T"], kinds={"Q": "flow", "T": "series"})
    pn.apply_pilot({"kinds": {"Q": "series"}, "kit_missing": {"T": {"beta": "mean near zero"}}})
    act, dropped = resolve_protect({"Q": ["r", "lnnse", "nse"], "T": ["beta", "r"]}, pn,
                                   {"Q": {"r": 0.8}, "T": {"r": 0.7}})
    assert act == {"Q": ["r"], "T": ["r"]}
    why = {(d["var"], d["metric"]): d["reason"] for d in dropped}
    assert "no longer recorded" in why[("Q", "lnnse")]
    assert "did not emit" in why[("Q", "nse")] and why[("T", "beta")] == "mean near zero"


def test_protection_is_not_used_in_a_single_objective_search():
    r = _rule(W=1, protect={"Q": ["r"]}, default_panel={"Q": {"r": 0.99}})
    r.add([1.0], None, {"Q": {"r": 0.1, "alpha": 1, "beta": 1}})
    assert r.inc[-1] == 0 and r.protect == {}


# ── wiring ─────────────────────────────────────────────────────────────────────────────────────────
def test_the_report_carries_the_rule_with_the_design_window_and_a_stop_point():
    from calibration_kit import test_step1_panel as S1
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, *_ = S1._calibrate(tmp, conv={"mode": "observe"})
        assert rep["status"] == "completed", rep.get("reason")
        ru = rep["convergence"]["rule"]
        cap = rep["budget_used"]["cap"]
        assert ru["window"] == design_window(2, cap) and ru["errors"] == []
        assert ru["calls"] == rep["budget_used"]["n_evaluations"]
        assert ru["stop_point"] is not None and ru["first_settle"]["streamflow"] is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_bad_protect_block_is_a_clean_refusal():
    from calibration_kit import test_step1_panel as S1
    import tempfile, shutil, yaml
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        from calibration_kit import calib as C
        from calibration_kit import test_calibrate_convergence_e2e as E
        import io, contextlib
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 30)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"]["protect"] = {"swe": ["r"]}
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        assert rep["status"] == "invalid_contract" and "swe" in rep["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── review round 1 (K3 + Opus 5.5) ────────────────────────────────────────────────────────────────
def test_b_k_counts_every_finite_component_even_when_another_is_missing():
    objs = (("Q:a", "Q"), ("S:b", "S"))
    r = _rule(W=50, objectives=objs, kinds={"Q": "series", "S": "series"}, trade_off=True)
    r.add([1.0, 1.0], None, {}); r.add([0.5, None], None, {})
    assert r.V[-1] == [0.5, 1.0]
    assert r._arch == [0]                                     # a partial vector never joins the archive


def test_copies_never_count_as_two_archive_points():
    r = _front_rule(W=1)
    r.add([0.5, 0.5], None, {}); r.add([0.5, 0.5], None, {}); r.add([0.5, 0.5], None, {})
    assert r._arch == [0] and r.t0 is None
    r.add([0.4, 0.6], None, {})
    assert r.t0 == 3 and r.s == [pytest.approx(0.1), pytest.approx(0.1)]


def test_a_one_point_archive_never_freezes_t0_and_no_front_test_before_t0_even_after_w():
    r = _front_rule(W=1)
    recs = [r.add([1.0 - 0.1 * k, 1.0 - 0.1 * k], None, {}) for k in range(4)]   # each dominates
    assert r.t0 is None and all(rec["cond4"] is None for rec in recs) and r.stop_point is None


def test_eps_front_bound_is_inclusive():
    r = _front_rule(W=1, eps_front=0.25)
    r.add([0.0, 1.0], None, {}); r.add([1.0, 0.0], None, {})           # t0 = 1, s = [1, 1]
    rec = r.add([0.75, 0.25], None, {})                                 # needs eps 0.25 exactly
    assert rec["cond4"] is True
    r2 = _front_rule(W=1, eps_front=0.25)
    r2.add([0.0, 1.0], None, {}); r2.add([1.0, 0.0], None, {})
    assert r2.add([0.5, 0.25], None, {})["cond4"] is False


def test_dominated_points_never_join_and_dominated_members_leave():
    r = _front_rule(W=50)
    r.add([1.0, 1.0], None, {}); r.add([2.0, 2.0], None, {})
    assert r._arch == [0]
    r.add([0.5, 0.5], None, {})
    assert r._arch == [2]


def test_compromise_order_minimax_then_sum_with_points_that_disagree():
    # after t0 (z = [0, 0], s = [10, 10]):  A (4, 7): max .7 sum 1.1 ; B (1, 7.5): max .75 sum .85
    r = _front_rule(W=1)
    r.add([0.0, 10.0], None, {}); r.add([10.0, 0.0], None, {})
    r.add([4.0, 7.0], None, {}); r.add([1.0, 7.5], None, {})
    assert r.inc[-1] == 2                                               # minimax first, not the sum
    # equal max, different sums: C (7, 1) max .7 sum .8 beats A
    r.add([7.0, 1.0], None, {})
    assert r.inc[-1] == 4


def test_after_t0_the_frozen_normalization_is_used_not_raw_losses():
    # objective 2 lives on a scale 100x larger: raw minimax would pick differently
    r = _front_rule(W=1)
    r.add([0.0, 1000.0], None, {}); r.add([10.0, 0.0], None, {})        # s = [10, 1000]
    r.add([3.0, 600.0], None, {})                                       # norm max .6 ; raw max 600
    r.add([6.5, 50.0], None, {})                                        # norm max .65; raw max 50
    assert r.inc[-1] == 2


def test_before_t0_the_raw_sum_not_the_raw_minimax():
    r = _front_rule(W=10)
    r.add([0.0, 9.0], None, {}); r.add([5.0, 5.0], None, {})           # sums 9 vs 10; max 9 vs 5
    assert r.t0 is None and r.inc[-1] == 0


def test_admissible_pbias_uses_the_absolute_value():
    d = {"Q": {"pbias": 10.0}}; t = {"Q": {"pbias": 1.0}}; act = {"Q": ["pbias"]}
    assert not admissible({"Q": {"pbias": -30.0}}, act, d, t)
    assert admissible({"Q": {"pbias": -10.5}}, act, d, t)


def test_loss_test_uses_the_value_w_calls_ago_as_denominator_is_strict_and_needs_both_finite():
    r = _rule(W=1, rel_gain=0.25)
    r.add([4.0], None, PAN)
    assert r.add([3.0], None, PAN)["cond1"] == [False]          # |1|/4 = 0.25: strict
    r2 = _rule(W=1, rel_gain=0.3)
    r2.add([4.0], None, PAN)
    assert r2.add([3.0], None, PAN)["cond1"] == [True]          # 1/4 = 0.25 < 0.3 (1/3 would fail)
    r3 = _rule(W=1, rel_gain=0.3)
    r3.add(None, None, {})
    assert r3.add([3.0], None, PAN)["cond1"] == [False]         # nothing finite W calls ago


def test_protect_empty_forms_and_undeclared_targets():
    pn = P.Panel(["Q", "S"], kinds={"Q": "series", "S": "series"})
    for empty in (None, False, [], {}):
        assert check_protect(empty, pn) == {}
    with pytest.raises(ValueError):
        check_protect({"S": ["r"]}, pn, declared=["Q"])


def test_no_default_panel_drops_protection_with_that_reason():
    pn = P.Panel(["Q"], kinds={"Q": "series"})
    act, dropped = resolve_protect({"Q": ["r"]}, pn, None, no_default_reason="no pilot")
    assert act == {} and dropped == [{"var": "Q", "metric": "r", "reason": "no pilot"}]


def test_single_objective_weighted_scalar_fallback():
    objs = (("Q:a", "Q"), ("Q:b", "Q"))
    r = _rule(W=50, objectives=objs, weights=[3.0, 1.0])
    r.add([1.0, 0.0], None, PAN)          # weighted 0.75
    r.add([0.0, 2.0], None, PAN)          # weighted 0.5 (plain mean would be 1.0 and lose)
    assert r.inc[-1] == 1


# ── wiring through calibrate() ──────────────────────────────────────────────────────────────────────
import contextlib, io, json, os, shutil, tempfile, yaml                        # noqa: E402


def _run(tmp, strategy=None, runner_kw=None, targets=None, families=None, budget_block="default"):
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_step1_panel as S1
    ki, wd = E._fixture(tmp, ({"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}
                              if budget_block == "default" else budget_block), None, 60)
    c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
    c["strategy"].update(strategy or {})
    if targets:
        c["targets"] = targets
    yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
    if families:
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = families
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
    with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
        rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                          run_model=S1._Runner(wd, **(runner_kw or {})), budget=None, seed=0)
    hp = Path(wd, "eval_history.jsonl")                  # absent when the run is refused at load
    hist = [json.loads(l) for l in hp.read_text().splitlines()] if hp.exists() else []
    plan = json.loads(Path(wd, "kdt_budget_plan.json").read_text()) if Path(wd, "kdt_budget_plan.json").exists() else None
    return rep, hist, plan, ki, wd


TRADE = {"multi_objective": True, "default_algorithm": "nsga2"}
FAMS = ["temporal_pattern_match", "magnitude_accuracy"]


def test_wiring_trade_off_search_feeds_a_trade_off_rule_with_resolved_protection_and_saved_anchor():
    pytest.importorskip("pymoo")
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, hist, plan, ki, wd = _run(tmp, strategy=dict(TRADE, protect={"streamflow": ["r", "nse"]}),
                                       families=FAMS)
        assert rep["status"] == "completed", rep.get("reason")
        ru = rep["convergence"]["rule"]
        assert ru["trade_off"] is True and ru["errors"] == []
        # (both objectives peak at the same parameters here, so the archive may stay one point and
        # the scale never freezes; the compromise and protection still run on the archive)
        assert ru["protect_active"] == {"streamflow": ["r", "nse"]} and ru["protect_dropped"] == []
        # nse is RECORDED (not required): its tolerance must reach the rule, or nothing is admissible
        assert ru["protection_infeasible_final"] is False
        assert plan["panel_setup"]["default_panel"]["streamflow"]["nse"] is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_losses_are_aligned_with_the_search_calls():
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        from calibration_kit import test_step1_panel as S1
        rep, hist, *_ = _run(tmp, runner_kw={})
        ru = rep["convergence"]["rule"]
        search = [h for h in hist if h["phase"] == "search"]
        assert ru["calls"] == len(search)
        losses = [(h.get("losses") or [None])[0] for h in search]
        fin = [(k, l) for k, l in enumerate(losses) if l is not None and math.isfinite(l)]
        best = min(fin, key=lambda t: (t[1], t[0]))[0]
        assert ru["incumbent_final"] == best
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_the_scalar_is_weighted_and_a_fixed_window_is_used():
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, *_ = _run(tmp, strategy={"convergence": {"window": 7}}, families=FAMS,
                       targets=[{"var": "streamflow", "weight": 1.0,
                                 "family_weights": {"temporal_pattern_match": 3.0, "magnitude_accuracy": 1.0}}])
        ru = rep["convergence"]["rule"]
        assert ru["window"] == 7 and ru["trade_off"] is False and ru["errors"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_the_rule_uses_the_design_window():
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, *_ = _run(tmp)
        assert rep["convergence"]["rule"]["window"] == design_window(2, rep["budget_used"]["cap"])
        assert "window" not in rep["convergence"]            # the old marker's window is gone
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_pilot_runs_zero_still_anchors_protection():
    """Gap 2k: the pilot is always on, so even pilot_runs: 0 runs the default run and protection keeps
    its anchor (before step 5, "no pilot" dropped protection)."""
    pytest.importorskip("pymoo")
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, *_ = _run(tmp, strategy=dict(TRADE, protect={"streamflow": ["r"]}), families=FAMS,
                       budget_block={"pilot_runs": 0})
        ru = rep["convergence"]["rule"]
        assert ru["protect_active"] == {"streamflow": ["r"]} and ru["protect_dropped"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_resume_reuses_the_saved_default_panel():
    pytest.importorskip("pymoo")
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        from calibration_kit import calib as C
        from calibration_kit import test_calibrate_convergence_e2e as E
        from calibration_kit import test_step1_panel as S1
        rep, hist, plan, ki, wd = _run(tmp, strategy=dict(TRADE, protect={"streamflow": ["r"]}), families=FAMS)
        # tamper with the saved anchor: a resumed run must use the SAVED one (r impossible to keep)
        plan["panel_setup"]["default_panel"]["streamflow"]["r"] = 2.0
        Path(wd, "kdt_budget_plan.json").write_text(json.dumps(plan))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep2 = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                               budget=None, seed=0)
        assert rep2["convergence"]["rule"]["protection_infeasible_final"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _pilot_ev(tmp_path, fail_default=False, junk=False):
    from calibration_kit.evaluator import Evaluator
    from calibration_kit.objectives import Objective
    wd = str(tmp_path)
    params = [{"name": n, "address": "none", "default": 1.0, "range": [0.0, 4.0]} for n in ("a", "b")]
    state = {"i": -1}

    def run():
        state["i"] += 1
        named = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        if fail_default and state["i"] == 1:          # the pilot's default run (after one earlier run)
            return {}
        m = {"nse": 1.0 - (named["a"] - 2.0) ** 2 / 10.0, "r": 0.9,
             "__kdt__": {"applied_params": dict(named), "panel": {"Q": {"r": 0.9 + 0.01 * state["i"]}}}}
        if junk:
            m["sim_series"] = np.arange(5)             # not a metric, not JSON
        return m
    obj = Objective(name="Q:temporal_pattern_match", var="Q", family="temporal_pattern_match", metric_key="nse")
    return Evaluator(ki_path=wd, workdir=wd, parameters=params, objectives=[obj], transform_inv={},
                     run_model=run, injection_mode="runner"), state


def test_pilot_block_survives_a_non_metric_value_and_a_failed_default_run_leaves_no_stale_anchor(tmp_path):
    from calibration_kit.pilot import run_pilot
    (tmp_path / "a").mkdir()
    ev, st = _pilot_ev(tmp_path / "a", junk=True)
    p = run_pilot(ev, [0.0, 0.0], [4.0, 4.0], [1.0, 1.0], n=3, seed=0)
    assert p["default_panel_block"]["Q"]["r"] == pytest.approx(0.9)
    assert "sim_series" not in p["default_metrics"] and p["default_metrics"]["nse"] is not None
    # an earlier good run, then the pilot's default run fails: nothing is kept from the earlier one
    (tmp_path / "b").mkdir()
    ev2, st2 = _pilot_ev(tmp_path / "b", fail_default=True)
    ev2.evaluate([3.0, 3.0])                           # state i = 0: a good run before the pilot
    p2 = run_pilot(ev2, [0.0, 0.0], [4.0, 4.0], [1.0, 1.0], n=3, seed=0)   # i = 1 fails
    assert p2["default_panel_block"] == {} and p2["default_metrics"] == {}


def test_a_variable_whose_metrics_are_all_kit_set_missing_is_judged_not_rule0():
    pn = P.Panel(["Q", "Y"], kinds={"Q": "series", "Y": "snapshot"})
    pn.apply_pilot({"kit_missing": {"Y": {"pbias": "mean near zero", "nrmse": "mean near zero"}}})
    r = ConvergenceRule([("Q:f", "Q")], pn, {"Q": dict(TOL)}, 2)
    assert "Y" in r.variables and r.rule0 == []


class _Flaky:
    """Fails (no metrics) whenever a > 1.3; otherwise the step-1 runner."""
    def __init__(self, wd):
        from calibration_kit import test_step1_panel as S1
        self.inner = S1._Runner(wd)

    def __call__(self):
        a = float(json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())["a"])
        return {} if a > 1.3 else self.inner()


def test_wiring_failed_calls_are_counted_by_the_rule():
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 60)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_Flaky(wd),
                              budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        search = [h for h in hist if h["phase"] == "search"]
        assert any(not h["ok"] for h in search), "fixture must produce failed calls"
        assert rep["convergence"]["rule"]["calls"] == len(search)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _TwoPull:
    """nse peaks at a = 1, |pbias| at a = 1.5: the weighted (3:1) and plain best calls differ."""
    def __init__(self, wd):
        self.wd = Path(wd)

    def __call__(self):
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        a = float(p["a"])
        return {"nse": 1.0 - (a - 1.0) ** 2, "pbias": 100.0 * (a - 1.5) ** 2, "r": 0.9, "kge": 0.8,
                "__kdt__": {"applied_params": dict(p), "case_id": "SITE:fixture"}}


def test_wiring_the_incumbent_is_the_weighted_minimum():
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 60)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["targets"] = [{"var": "streamflow", "weight": 1.0,
                         "family_weights": {"temporal_pattern_match": 3.0, "magnitude_accuracy": 1.0}}]
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = FAMS
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_TwoPull(wd),
                              budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        search = [h for h in hist if h["phase"] == "search"]
        def ok(h):
            l = h.get("losses")
            return l and all(x is not None and math.isfinite(x) for x in l)
        W = lambda h: (3.0 * h["losses"][0] + h["losses"][1]) / 4.0 if ok(h) else math.inf     # noqa: E731
        M = lambda h: (h["losses"][0] + h["losses"][1]) / 2.0 if ok(h) else math.inf           # noqa: E731
        best = min(range(len(search)), key=lambda k: (W(search[k]), k))
        plain = min(range(len(search)), key=lambda k: (M(search[k]), k))
        assert best != plain, "fixture must separate the weighted and the plain best"
        assert rep["convergence"]["rule"]["incumbent_final"] == best
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── review round 2 (Opus 5.5) ─────────────────────────────────────────────────────────────────────
def test_a_variable_settles_on_its_own_objectives_only():
    objs = (("Q:f", "Q"), ("S:f", "S"))
    r = _rule(W=3, objectives=objs, kinds={"Q": "categorical", "S": "categorical"}, tol={}, trade_off=True)
    for k in range(10):
        r.add([1.0, 1.0 - 0.1 * min(k, 6)], None, {})           # Q flat; S improves until call 6
    assert r.first_settle["Q"] == 3 and r.first_settle["S"] == 9


def test_no_stop_point_while_the_front_test_is_not_possible():
    r = _front_rule(W=1)
    for _ in range(3):
        r.add([0.5, 0.5], None, {})                              # copies: t0 never, cond4 None
    assert r.t0 is None and r.stop_point is None


def test_the_incumbents_own_value_missing_now_is_unknown_even_with_a_full_window():
    r = _rule(W=2)
    for k in range(3):
        r.add([1.0 - 1e-6 * k], None, {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}})
    rec = r.add([0.5], None, {"Q": {"r": 0.9, "alpha": 1.0}})   # new incumbent without beta
    assert rec["cond2"]["Q"]["beta"] == "unknown" and not rec["stop"]


def test_a_failing_add_records_nothing_and_later_calls_stay_in_step():
    r = _rule(W=1)
    r.add([1.0], None, PAN)
    with pytest.raises(ValueError):
        r.add([1.0, 2.0], None, PAN)                             # wrong length
    assert len(r.F) == len(r.cond) == len(r.inc) == 1
    orig = r._conditions
    r._conditions = lambda i: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        r.add([0.9], None, PAN)                                  # fails AFTER the per-call appends
    r._conditions = orig
    assert len(r.F) == len(r.S) == len(r.X) == len(r.cond) == 1 and r._best_single == 0
    rec = r.add([0.9], None, PAN)
    assert rec["i"] == 1 and r.inc[-1] == 1


class _HalfNaN:
    """nse peaks at a = 1; pbias is NaN at the default (a = 0.8) and in a thin band around a = 1, so
    the best nse calls are partial vectors (the pilot stays below its 30 % failure limit)."""
    def __init__(self, wd):
        self.wd = Path(wd)

    def __call__(self):
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        a = float(p["a"])
        nan = abs(a - 0.8) < 1e-9 or 0.97 < a < 1.03
        return {"nse": 1.0 - (a - 1.0) ** 2, "pbias": (float("nan") if nan else 100.0 * (a - 1.5) ** 2),
                "r": 0.9, "kge": 0.8, "__kdt__": {"applied_params": dict(p), "case_id": "SITE:fixture"}}


def test_wiring_partial_vectors_reach_b_k_and_a_partly_finite_default_run_anchors_protection():
    pytest.importorskip("pymoo")
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 60)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["strategy"].update(dict(TRADE, protect={"streamflow": ["r"]}))
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = FAMS
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_HalfNaN(wd),
                              budget=None, seed=0)
        ru = rep["convergence"]["rule"]
        # the default run (a = 0.8) has one finite loss: the anchor is kept, protection active
        assert ru["protect_active"] == {"streamflow": ["r"]} and ru["protect_dropped"] == []
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        search = [h for h in hist if h["phase"] == "search"]
        def fin(x):                                     # the history writes inf as the string 'inf'
            return x is not None and math.isfinite(float(x))
        b0 = min(float(h["losses"][0]) for h in search if h.get("losses") and fin(h["losses"][0]))
        partial = [h for h in search if h.get("losses") and fin(h["losses"][0]) and not fin(h["losses"][1])]
        assert partial, "fixture must produce partial calls"
        assert ru["best_losses"][0] == pytest.approx(b0)
        assert min(float(h["losses"][0]) for h in partial) == pytest.approx(b0), "best nse must be on a partial call"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_drop_reason_when_the_panel_setup_fails(monkeypatch):
    pytest.importorskip("pymoo")
    from calibration_kit import pilot as PL
    def boom(*a, **k):
        raise RuntimeError("tolerance setup broke")
    monkeypatch.setattr(PL, "pilot_tolerances", boom)
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, *_ = _run(tmp, strategy=dict(TRADE, protect={"streamflow": ["r"]}), families=FAMS)
        assert rep["convergence"]["tolerance_source"].startswith("unavailable")    # it really failed
        ru = rep["convergence"]["rule"]
        # the anchor was read BEFORE the failure, so protection stays active
        assert ru["protect_active"] == {"streamflow": ["r"]}, ru["protect_dropped"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_drop_reason_when_resuming_a_plan_from_an_older_kit():
    pytest.importorskip("pymoo")
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_step1_panel as S1
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, hist, plan, ki, wd = _run(tmp, strategy=dict(TRADE, protect={"streamflow": ["r"]}), families=FAMS)
        plan.pop("panel_setup", None)
        plan["pilot_summary"].pop("default_metrics", None)
        Path(wd, "kdt_budget_plan.json").write_text(json.dumps(plan))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep2 = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                               budget=None, seed=0)
        reasons = [d["reason"] for d in rep2["convergence"]["rule"]["protect_dropped"]]
        assert reasons and all("older kit" in r for r in reasons)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_bad_window_is_refused_at_load_before_any_model_run():
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_step1_panel as S1
    for bad in (0, -3, "fast", 2.5, True):
        tmp = tempfile.mkdtemp(prefix="kdt_s2_")
        try:
            ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1},
                                {"window": bad}, 60)
            runner = S1._Runner(wd)
            calls = {"n": 0}
            def counted():
                calls["n"] += 1
                return runner()
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=counted,
                                  budget=None, seed=0)
            assert rep["status"] == "invalid_contract" and "window" in rep["reason"], bad
            assert calls["n"] == 0, f"{bad!r}: the model ran {calls['n']} times before the refusal"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_wiring_drop_reason_when_the_panel_setup_fails_before_the_anchor(monkeypatch):
    pytest.importorskip("pymoo")
    from calibration_kit import panel as PN
    def boom(*a, **k):
        raise RuntimeError("pilot decisions broke")
    monkeypatch.setattr(PN, "pilot_decisions", boom)
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, *_ = _run(tmp, strategy=dict(TRADE, protect={"streamflow": ["r"]}), families=FAMS)
        reasons = [d["reason"] for d in rep["convergence"]["rule"]["protect_dropped"]]
        assert reasons and all("panel setup failed (RuntimeError)" in r for r in reasons)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── review round 3 (K3 + Opus 5.5) ────────────────────────────────────────────────────────────────
def test_wiring_drop_reason_when_the_saved_panel_setup_has_no_default_panel():
    pytest.importorskip("pymoo")
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_step1_panel as S1
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, hist, plan, ki, wd = _run(tmp, strategy=dict(TRADE, protect={"streamflow": ["r"]}), families=FAMS)
        plan["panel_setup"].pop("default_panel")                  # a plan written by the step-1 kit
        Path(wd, "kdt_budget_plan.json").write_text(json.dumps(plan))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep2 = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                               budget=None, seed=0)
        reasons = [d["reason"] for d in rep2["convergence"]["rule"]["protect_dropped"]]
        assert reasons and all(r == "resumed from a plan saved without a default panel" for r in reasons)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _EmptyAtDefault:
    """Returns no metrics at the defaults (a = 0.8, b = 2.0): one pilot failure in 10."""
    def __init__(self, wd):
        from calibration_kit import test_step1_panel as S1
        self.inner = S1._Runner(wd)

    def __call__(self):
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        if abs(float(p["a"]) - 0.8) < 1e-9 and abs(float(p["b"]) - 2.0) < 1e-9:
            return {}
        return self.inner()


def test_wiring_drop_reason_when_the_default_run_gave_no_metrics():
    pytest.importorskip("pymoo")
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 60)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["strategy"].update(dict(TRADE, protect={"streamflow": ["r"]}, probe_objectives=False))
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = FAMS
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_EmptyAtDefault(wd),
                              budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        # build step 8 (design §1 triage): a default run with no finite objective metric is `no_baseline`,
        # before any search — it no longer reaches the protection anchor
        assert rep["status"] == "no_baseline" and rep["triage"]["route"] == "no_baseline", rep.get("reason")
        assert "no finite objective metric" in rep["reason"]
        assert not [h for h in hist if h["phase"] == "search"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_frozen_offset_is_the_archive_minimum_not_the_best_partial_loss():
    r = _front_rule(W=2)
    for f in ([0.5, None], [1.0, 2.0], [2.0, 1.0], [1.1, 1.5], [1.45, 1.45]):
        r.add(f, None, {})
    assert r.V[-1][0] == 0.5                         # B_k counts the partial call
    assert r.z == [1.0, 1.0] and r.inc[-1] == 4      # z from A(t0) only (design §2.6)


def test_all_or_nothing_add_in_a_trade_off_rule_restores_every_piece_of_state():
    r = _front_rule(W=1)
    r.add([0.0, 1.0], None, {})
    before = (list(r._best), list(r._arch), len(r._arch_hist), r.t0, r.z, r.s)
    orig = r._conditions
    r._conditions = lambda i: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        r.add([1.0, 0.0], None, {})                 # would freeze t0 and change B_k and the archive
    r._conditions = orig
    assert (list(r._best), list(r._arch), len(r._arch_hist), r.t0, r.z, r.s) == before
    r.add([1.0, 0.0], None, {})
    assert r.t0 == 1 and r._arch == [0, 1]


def test_wiring_a_rule_fault_keeps_the_call_index_aligned(monkeypatch):
    from calibration_kit import rule as RU
    real = RU.ConvergenceRule._add
    state = {"n": 0}
    def flaky(self, *a, **k):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("one bad call")
        return real(self, *a, **k)
    monkeypatch.setattr(RU.ConvergenceRule, "_add", flaky)
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, hist, *_ = _run(tmp)
        ru = rep["convergence"]["rule"]
        search = [h for h in hist if h["phase"] == "search"]
        assert ru["errors"] and ru["calls"] == len(search)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── review round 4 (K3; Opus note) ────────────────────────────────────────────────────────────────
def test_wiring_the_true_reason_survives_a_later_setup_failure(monkeypatch):
    """The default run gave no metrics, THEN the tolerance work fails: the triage still says
    `no_baseline` (build step 8) — the setup failure does not hide it."""
    pytest.importorskip("pymoo")
    from calibration_kit import pilot as PL
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    def boom(*a, **k):
        raise RuntimeError("tolerance setup broke")
    monkeypatch.setattr(PL, "pilot_tolerances", boom)
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 60)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["strategy"].update(dict(TRADE, protect={"streamflow": ["r"]}, probe_objectives=False))
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = FAMS
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_EmptyAtDefault(wd),
                              budget=None, seed=0)
        assert rep["status"] == "no_baseline" and "no finite objective metric" in rep["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_a_convergence_block_that_is_not_a_mapping_is_refused_at_load():
    tmp = tempfile.mkdtemp(prefix="kdt_s2_")
    try:
        rep, *_ = _run(tmp, strategy={"convergence": True})
        assert rep["status"] == "invalid_contract" and "mapping" in rep["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
