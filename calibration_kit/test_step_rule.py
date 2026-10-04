"""Part A2 (rev 30): the step-end convergence rule (rule_steps.py). Each test checks one rule."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit import panel as P                                              # noqa: E402
from calibration_kit.rule_steps import (H, M_MAX, M_MIN, Q_MIN, StepRule, cp_lower,  # noqa: E402
                                        lookback)

TOL = {"r": 0.01, "alpha": 0.01, "beta": 0.01}
PAN = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}}


def _rule(objectives=(("Q:f", "Q"),), kinds=None, tol=None, **kw):
    kinds = kinds or {"Q": "series"}
    return StepRule(list(objectives), P.Panel(list(kinds), kinds=dict(kinds)),
                    tol or {v: dict(TOL) for v in kinds}, **kw)


def _feed(r, step_losses, per_step=3, panel=PAN, start=0):
    """One call per loss; each step = `per_step` calls of the same loss list entry."""
    for s, loss in enumerate(step_losses, start=start):
        for _ in range(per_step):
            r.add([loss], None, panel, step=s)


# ── the bound and the look-back ──────────────────────────────────────────────────────────────────
def test_cp_lower_matches_beta_quantile():
    # values from scipy.stats.beta.ppf(0.2, y, n - y + 1)
    for (y, n), want in {(1, 20): 0.011095, (3, 20): 0.07771, (10, 20): 0.383968,
                         (20, 20): 0.922681, (5, 7): 0.483242}.items():
        assert cp_lower(y, n) == pytest.approx(want, abs=1e-6)
    assert cp_lower(0, 20) == 0.0
    with pytest.raises(ValueError):
        cp_lower(21, 20)


def test_lookback_schedule():
    assert [lookback([True] * q) for q in (0, Q_MIN - 1)] == [None, None]
    assert lookback([True] * Q_MIN) == M_MIN and lookback([True] * (H - 1)) == M_MIN
    assert lookback([True] * H) == M_MAX                       # p_L = 0 -> the ceiling
    assert lookback([False] * H) == M_MIN                      # always moving -> the floor
    # learned from the transitions up to the last move: [F,T]*10 -> the 19 up to the last F, y = 10
    assert lookback([False, True] * (H // 2)) == math.ceil(math.log(.05) / math.log(1 - cp_lower(10, 19)))
    assert lookback([False] + [True] * (H - 1)) == 14          # 1 of 1 -> p_L = 0.2 -> 14
    # only the H transitions ending at the last move count (y = 10 of 20 -> p_L = 0.384 -> 7)
    assert lookback([True] * 30 + [False, True] * 10) == 7
    # the quiet run under test does not raise m (it did before the 2026-10-03 fix)
    base = [(q % 5 != 0) for q in range(1, 21)]
    assert {lookback(base + [True] * L) for L in range(0, 30)} == {lookback(base)}


# ── the tracer: a search that settles ────────────────────────────────────────────────────────────
def test_tracer_converges_after_five_quiet_steps():
    r = _rule()
    _feed(r, [10, 5, 3, 2, 2, 2, 2, 2, 2, 2])                # steps 0..9; step 9 is the open last step
    s = r.finish()
    # step ends: 0..8 (9 is partial) -> transitions 1..8; quiet from 3->4 (q=4) on
    assert s["steps_after_start"] == 8
    assert [t["quiet"] for t in r.transitions] == [False, False, False, True, True, True, True, True]
    assert s["verdict"] == "converged"
    assert s["converged_at"]["q"] == 8 and s["converged_at"]["to_step"] == 8
    assert s["converged_at"]["calls"] == 27 and s["converged_at"]["m"] == 5
    assert s["last_change"]["step"] == 3 and s["start_step"] == 0 and s["start_calls"] == 3


def test_no_test_before_five_transitions():
    r = _rule()
    _feed(r, [2, 2, 2, 2, 2])                                  # 4 step ends -> 3 transitions
    s = r.finish(last_step_complete=True)                      # 5 step ends -> 4 transitions
    assert s["steps_after_start"] == 4 and s["verdict"] == "too short to judge"
    assert all(t["m"] is None and not t["fired"] for t in r.transitions)


def test_never_quiet_is_not_converged():
    r = _rule()
    _feed(r, [100 * 0.9 ** k for k in range(12)])
    s = r.finish()
    assert s["verdict"] == "not converged within the budget" and s["converged_at"] is None


def test_partial_last_step_counts_only_when_complete():
    a = _rule(); _feed(a, [2] * 7)
    b = _rule(); _feed(b, [2] * 7)
    assert a.finish()["steps_after_start"] == 5
    assert b.finish(last_step_complete=True)["steps_after_start"] == 6


# ── the loss test ────────────────────────────────────────────────────────────────────────────────
def test_loss_threshold_and_floor():
    r = _rule(); _feed(r, [1.0, 0.996]); r.finish(True)
    assert r.transitions[0]["quiet"] is True                   # 0.4 % < 0.5 %
    r = _rule(); _feed(r, [1.0, 0.995]); r.finish(True)
    assert r.transitions[0]["quiet"] is False                  # 0.5 % is not < 0.5 %
    r = _rule(); _feed(r, [2e-6, 0.0]); r.finish(True)
    assert r.transitions[0]["quiet"] is True                   # floored denominator 1e-3


def test_best_point_not_last_call():
    """A worse call later in the step does not change the best point."""
    r = _rule()
    for s in range(3):
        r.add([1.0], None, PAN, step=s)
        r.add([9.0], None, {"Q": {"r": 0.1, "alpha": 3.0, "beta": 3.0}}, step=s)
        r.add(None, None, None, step=s)                       # a failed call
    r.finish(True)
    assert all(t["quiet"] for t in r.transitions)
    assert r.ends[-1]["incumbent"] == 0


# ── the score test ───────────────────────────────────────────────────────────────────────────────
def test_score_move_beyond_tolerance_is_not_quiet():
    r = _rule()
    r.add([1.0], None, PAN, step=0)
    r.add([0.999], None, {"Q": {"r": 0.92, "alpha": 1.0, "beta": 1.0}}, step=1)   # loss quiet, r +0.02
    r.finish(True)
    t = r.transitions[0]
    assert t["quiet"] is False and t["reasons"] == [["score", "Q", "r", 0.9, 0.92, 0.01]]


def test_score_within_tolerance_is_quiet():
    r = _rule()
    r.add([1.0], None, PAN, step=0)
    r.add([0.999], None, {"Q": {"r": 0.909, "alpha": 1.0, "beta": 1.0}}, step=1)
    r.finish(True)
    assert r.transitions[0]["quiet"] is True


def test_missing_score_fails_closed():
    r = _rule()
    r.add([1.0], None, PAN, step=0)
    r.add([0.999], None, {"Q": {"r": 0.9, "alpha": float("nan"), "beta": 1.0}}, step=1)
    r.finish(True)
    t = r.transitions[0]
    assert t["quiet"] is False and ["score_missing", "Q", "alpha"] in t["reasons"]
    r = _rule(tol={"Q": {"r": 0.01, "alpha": 0.01}})           # no tolerance for beta
    _feed(r, [1.0, 1.0]); r.finish(True)
    assert ["score_missing", "Q", "beta"] in r.transitions[0]["reasons"]


# ── step tags ────────────────────────────────────────────────────────────────────────────────────
def test_bad_step_tags_refused():
    r = _rule()
    r.add([1.0], None, PAN, step=2)
    with pytest.raises(ValueError):
        r.add([1.0], None, PAN, step=1)
    for bad in (None, 1.0, True, "3"):
        with pytest.raises(ValueError):
            r.add([1.0], None, PAN, step=bad)
    r.finish()
    with pytest.raises(ValueError):
        r.add([1.0], None, PAN, step=3)


def test_refused_call_leaves_state_unchanged():
    """A bad first call of a later step must not close the previous step (codex A2 r1 #1)."""
    r = _rule()
    r.add([1.0], None, PAN, step=0)
    with pytest.raises(ValueError):
        r.add([1.0, 2.0], None, PAN, step=1)                   # two losses for one objective
    assert r.ends == [] and r._cur_step == 0 and len(r.F) == 1
    r.add([1.0], None, PAN, step=0)                            # step 0 can still go on
    r.add([1.0], None, PAN, step=1)
    assert [e["step"] for e in r.ends] == [0] and r.ends[0]["calls"] == 2


def test_gap_in_step_tags_is_one_transition():
    r = _rule()
    r.add([1.0], None, PAN, step=1)                            # NSGA-II starts at generation 1
    r.add([1.0], None, PAN, step=3)
    r.finish(True)
    assert r.transitions[0]["from_step"] == 1 and r.transitions[0]["to_step"] == 3


# ── adaptive look-back in a long search ──────────────────────────────────────────────────────────
def test_adaptive_lookback_after_twenty():
    """Quiet runs of length 4 broken by a move never reach 5 before q = 20; afterwards m adapts
    from the 20 transitions up to the last move (4 moves in 20 -> p_L = 0.114 -> m = 25), and the
    rule fires once 25 quiet transitions follow the last move (q = 45)."""
    losses, v = [100.0], 100.0
    for q in range(1, 51):
        if q % 5 == 0 and q <= 20:
            v *= 0.9
        losses.append(v)
    r = _rule(); _feed(r, losses); s = r.finish(True)
    flags = [t["quiet"] for t in r.transitions]
    assert flags[:20] == [(q % 5 != 0) for q in range(1, 21)]
    assert r.transitions[19]["m"] == lookback(flags[:20]) == math.ceil(math.log(.05) / math.log(1 - cp_lower(4, 20)))
    assert not any(t["fired"] for t in r.transitions[:20])
    fired = s["converged_at"]
    assert fired is not None and fired["fired"]
    assert all(flags[fired["q"] - fired["m"]:fired["q"]])
    m4 = math.ceil(math.log(.05) / math.log(1 - cp_lower(4, 20)))
    assert m4 == 25 and fired["m"] == m4 and fired["q"] == 20 + m4


# ── trade-off (NSGA-II) ──────────────────────────────────────────────────────────────────────────
def _tradeoff():
    return _rule(objectives=(("Q:a", "Q"), ("Q:b", "Q")), trade_off=True)


def test_front_move_is_not_quiet_and_scale_frozen_at_start():
    r = _tradeoff()
    for f in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]):
        r.add(f, None, PAN, step=1)
    r.add([0.3, 0.3], None, PAN, step=2)                       # dominates [0.5,0.5]: front moves
    r.add([0.3, 0.3], None, PAN, step=3)
    r.finish(True)
    assert r.scale == ([0.0, 0.0], [1.0, 1.0])
    t1, t2 = r.transitions
    assert t1["quiet"] is False and t1["front_eps"] == pytest.approx(0.2)
    assert t2["front_eps"] == pytest.approx(0.0) and t2["quiet"] is True


def test_tradeoff_best_loss_per_objective():
    r = _tradeoff()
    r.add([1.0, 0.0], None, PAN, step=1); r.add([0.0, 1.0], None, PAN, step=1)
    r.add([-0.5, 2.0], None, PAN, step=2)                      # new best for objective a
    r.finish(True)
    assert any(x[0] == "loss" and x[1] == "Q:a" for x in r.transitions[0]["reasons"])


def test_tradeoff_compromise_scores_are_watched():
    r = _tradeoff()
    r.add([1.0, 0.0], None, PAN, step=1); r.add([0.0, 1.0], None, PAN, step=1)
    r.add([0.4, 0.4], None, {"Q": {"r": 0.5, "alpha": 1.0, "beta": 1.0}}, step=2)  # new compromise
    r.finish(True)
    rs = r.transitions[0]["reasons"]
    assert ["score", "Q", "r", 0.9, 0.5, 0.01] in rs


def test_loss_exact_boundary_is_not_quiet():
    """A move of exactly rel_gain is NOT quiet (strict <). Binary-exact numbers."""
    r = _rule(rel_gain=0.25); _feed(r, [1.0, 0.75]); r.finish(True)
    assert r.transitions[0]["quiet"] is False
    r = _rule(rel_gain=0.25); _feed(r, [1.0, 0.875]); r.finish(True)
    assert r.transitions[0]["quiet"] is True


def test_scale_frozen_at_start_up_end_not_later():
    """A wider front in step 2 must not change the scale frozen at the end of step 1."""
    r = _tradeoff()
    for f in ([1.0, 0.0], [0.0, 1.0]):
        r.add(f, None, PAN, step=1)
    r.add([-1.0, 2.0], None, PAN, step=2)
    r.add([-1.0, 2.0], None, PAN, step=3)
    r.finish(True)
    assert r.scale == ([0.0, 0.0], [1.0, 1.0])
    assert r.transitions[0]["front_eps"] == pytest.approx(1.0)    # (0 - (-1)) / 1, not / 2
