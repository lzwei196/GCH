"""Tests for the optimizer-side convergence reporting (handoff §5.14 tests 7 and 8).

7. The pymoo front observer NEVER terminates a run (the search reaches the cap) and records a
   firing generation on a toy 2-objective problem when given a loose tolerance. With pymoo's own
   defaults it may not fire at all at our budgets — that is recorded, not treated as a failure.
8. The SCE-UA classifier reads the three SPOTPY endings out of captured output, including real
   output produced by this server's spotpy 1.6.7.
Plus: DDS reports that it has no convergence test, and accepts a start vector.
"""
from __future__ import annotations
import contextlib
import io
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.backends.base import Problem                                    # noqa: E402
from calibration_kit.backends.pymoo_backend import PymooBackend                      # noqa: E402
from calibration_kit.backends.spotpy_backend import (SpotpyBackend,                  # noqa: E402
                                                     classify_sceua_output)


def _sphere(n=3):
    def ev(x):
        return [sum((xi - 0.3) ** 2 for xi in x)]
    return Problem(names=[f"p{i}" for i in range(n)], lower=[0.0] * n, upper=[1.0] * n,
                   objective_names=["loss"], evaluate=ev)


def _two_objective(n=6):
    """A ZDT1-like front: objective 1 pulls x0 down, objective 2 pulls the rest up."""
    def ev(x):
        f1 = x[0]
        g = 1.0 + 9.0 * sum(x[1:]) / max(1, len(x) - 1)
        f2 = g * (1.0 - (f1 / g) ** 0.5)
        return [f1, f2]
    return Problem(names=[f"p{i}" for i in range(n)], lower=[0.0] * n, upper=[1.0] * n,
                   objective_names=["f1", "f2"], evaluate=ev, is_multi_objective=True)


# ── test 7: the front observer observes ──────────────────────────────────────────────
@pytest.mark.skipif(not PymooBackend.available(), reason="pymoo not installed")
def test_front_observer_never_stops_the_run():
    budget = 2000
    # a LOOSE tolerance so the toy problem fires inside a short test; pymoo's own default
    # (ftol=0.005) does not fire at these budgets — that case is the next test.
    loose = PymooBackend("nsga2", pop_size=40).optimize(_two_objective(), budget=budget, seed=1,
                                                        front_ftol=0.1, front_period=10)
    assert loose.n_evaluations == budget, "an observer must never end the search"
    assert not loose.stopped_early
    t = loose.termination
    assert t["status"] == "max_evals"
    nv = t["native_verdict"]                       # pymoo's own test, recorded only (design §1.6)
    assert nv["fired_gen"] is not None, "a loose tolerance should fire"
    assert nv["fired_eval"] <= budget and nv["error"] is None
    # pymoo's own DefaultMultiObjectiveTermination via native_rules.PymooTest (2026-10-04): n_skip as shipped (5)
    assert nv["settings"]["period"] == 10 and nv["settings"]["n_skip"] == 5
    assert nv["fired_gen"] <= t["generations_done"] and len(nv["per_generation"]) == t["generations_done"]
    # every call carries its generation (gap 2h)
    gens = [h["gen"] for h in loose.history]
    assert gens[0] == 1 and gens == sorted(gens) and gens[-1] <= t["generations_done"] + 1
    assert gens.count(1) == 40                     # the first population is generation 1
    assert t["generations_available"] == pytest.approx(budget / 40)
    assert not t["warnings"]


@pytest.mark.skipif(not PymooBackend.available(), reason="pymoo not installed")
def test_front_observer_may_not_fire_at_the_default_tolerance():
    """Recorded honestly: pymoo's default ftol did not fire on this budget (checked on 0.6.1.6).
    A null firing generation means 'not confirmed at this tolerance', not 'did not converge'."""
    r = PymooBackend("nsga2", pop_size=40).optimize(_two_objective(), budget=1200, seed=1)
    assert r.n_evaluations == 1200
    fo = r.termination["native_verdict"]
    assert fo["settings"]["ftol"] == 0.005 and fo["settings"]["period"] == 50 and fo["settings"]["n_skip"] == 5
    assert fo["fired_gen"] is None or fo["fired_eval"] <= 1200


@pytest.mark.skipif(not PymooBackend.available(), reason="pymoo not installed")
def test_small_budget_is_flagged_as_unable_to_confirm():
    r = PymooBackend("nsga2", pop_size=40).optimize(_two_objective(), budget=400, seed=1)
    assert r.termination["generations_available"] == 10.0     # < period + n_skip + 1 (56)
    assert any("budget_too_small_to_confirm" in w for w in r.termination["warnings"])


# ── test 8: the SCE-UA classifier ────────────────────────────────────────────────────
MAX_TRIALS = """ComplexEvo loop #14 in progress...
*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT
ON THE MAXIMUM NUMBER OF TRIALS
2000
HAS BEEN EXCEEDED.
SEARCH WAS STOPPED AT TRIAL NUMBER: 2059
NUMBER OF DISCARDED TRIALS: 35
NORMALIZED GEOMETRIC RANGE = 0.185221
THE BEST POINT HAS IMPROVED IN LAST 100 LOOPS BY 12.500000 PERCENT
"""

POP_CONVERGED = """ComplexEvo loop #15 in progress...
THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE
SEARCH WAS STOPPED AT TRIAL NUMBER: 1302
NUMBER OF DISCARDED TRIALS: 0
NORMALIZED GEOMETRIC RANGE = 0.000041
THE BEST POINT HAS IMPROVED IN LAST 100 LOOPS BY 0.004000 PERCENT
"""

PCENTO = """ComplexEvo loop #5 in progress...
Objective function convergence criteria is now being updated and assessed...
Updated convergence criteria: 0.000043
THE BEST POINT HAS IMPROVED IN LAST 5 LOOPS BY LESS THAN THE USER-SPECIFIED THRESHOLD 0.000100
CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!
SEARCH WAS STOPPED AT TRIAL NUMBER: 1968
NUMBER OF DISCARDED TRIALS: 0
NORMALIZED GEOMETRIC RANGE = 0.293725
THE BEST POINT HAS IMPROVED IN LAST 5 LOOPS BY 0.000043 PERCENT
"""


def test_classifier_reads_the_three_endings():
    a = classify_sceua_output(MAX_TRIALS, kstop=100, pcento=1e-7, peps=1e-7, ngs=20)
    assert a["status"] == "max_trials" and a["stopped_at_trial"] == 2059
    assert a["geometric_range"] == 0.185221 and a["discarded_trials"] == 35
    assert a["settings"]["pcento"] == 1e-7 and a["loops"] == 14

    b = classify_sceua_output(POP_CONVERGED)
    assert b["status"] == "population_converged" and b["stopped_at_trial"] == 1302
    assert b["geometric_range"] == pytest.approx(4.1e-05)

    c = classify_sceua_output(PCENTO)
    assert c["status"] == "improvement_below_pcento"
    assert c["convergence_criteria"] == pytest.approx(4.3e-05)
    assert c["last_improvement_pct"] == pytest.approx(4.3e-05)


def test_classifier_decides_on_the_last_loop_not_an_earlier_one():
    """The per-loop lines repeat: an early 'population converged' message must not outrank the
    cap that actually ended the run."""
    mixed = ("ComplexEvo loop #1 in progress...\n"
             "THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE\n"
             + MAX_TRIALS)
    assert classify_sceua_output(mixed)["status"] == "max_trials"


def test_classifier_never_guesses():
    assert classify_sceua_output("")["status"] == "unknown"
    assert classify_sceua_output("Starting burn-in sampling...")["status"] == "unknown"


@pytest.mark.skipif(not SpotpyBackend.available(), reason="spotpy not installed")
def test_real_sceua_runs_are_classified():
    """Live runs on this server's spotpy: the cap, and the parameter-space collapse."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        capped = SpotpyBackend("sceua").optimize(_sphere(), budget=400, seed=1)
        converged = SpotpyBackend("sceua").optimize(_sphere(), budget=6000, seed=1,
                                                    kstop=1000, pcento=1e-12, peps=0.05)
    assert capped.termination["status"] == "max_trials"
    assert capped.termination["settings"]["pcento"] == 1e-7      # the defaults are recorded too
    assert converged.termination["status"] == "population_converged"
    assert converged.termination["geometric_range"] < 0.05
    assert converged.n_evaluations < 6000, "it stopped before the cap, and says why"


# ── DDS has no convergence test, and takes a start point ─────────────────────────────
@pytest.mark.skipif(not SpotpyBackend.available(), reason="spotpy not installed")
def test_dds_reports_its_schedule_and_accepts_x_initial():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = SpotpyBackend("dds").optimize(_sphere(), budget=120, seed=1, x_initial=[0.5, 0.5, 0.5])
        plain = SpotpyBackend("dds").optimize(_sphere(), budget=120, seed=1)
    assert r.termination["status"] == "budget_schedule_complete"
    assert "no convergence test" in r.termination["note"]
    assert r.termination["x_initial"] == "default_vector"
    assert plain.termination["x_initial"] is None
    assert r.n_evaluations == plain.n_evaluations == 120
    assert r.history[0]["x"] == [0.5, 0.5, 0.5], "the start point is the one we handed it"


# ── the C-test finding: SCE-UA's own test must be REACHABLE at the budget ─────────────
@pytest.mark.skipif(not SpotpyBackend.available(), reason="spotpy not installed")
def test_unreachable_kstop_is_warned_and_cap_is_not_called_convergence():
    """Found in the GR4J C-test logs (2026-09-27): every SCE-UA cell ended on MAXIMUM NUMBER OF
    TRIALS because spotpy's default kstop=100 needs ~100 evolution loops and a 3,000-evaluation
    budget buys a handful. The run must say so instead of leaving it to be mis-read as convergence.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = SpotpyBackend("sceua").optimize(_sphere(4), budget=1200, seed=1)
    t = r.termination
    assert t["status"] == "max_trials"
    assert t["converged_by_own_rule"] is False
    assert t["loops_available"] < t["settings"]["kstop"]
    # 2026-10-04: the warning is about the kit's rule (needs 10 loops), not SPOTPY's internal kstop
    assert t["warnings"] and "fewer than the 10 loops the kit's convergence rule needs" in t["warnings"][0]
    assert "WARNING sceua" in buf.getvalue()


@pytest.mark.skipif(not SpotpyBackend.available(), reason="spotpy not installed")
def test_a_real_convergence_sets_converged_by_own_rule():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = SpotpyBackend("sceua").optimize(_sphere(3), budget=6000, seed=1, kstop=1000,
                                            pcento=1e-12, peps=0.05)
    assert r.termination["status"] == "population_converged"
    assert r.termination["converged_by_own_rule"] is True


@pytest.mark.skipif(not PymooBackend.available(), reason="pymoo not installed")
def test_pymoos_own_termination_never_ends_the_search():
    """Design §1.6 / gap 2a: the old front_mode="enforce" path is removed. pymoo's test is recorded as
    the native verdict; only the cap (and the kit's stop mode, via on_eval) end the search."""
    r = PymooBackend("nsga2", pop_size=40).optimize(_two_objective(), budget=4000, seed=1,
                                                    front_mode="enforce", front_ftol=0.1,
                                                    front_period=10)
    assert r.n_evaluations == 4000 and r.termination["status"] == "max_evals"
    assert r.termination["native_verdict"]["fired_gen"] is not None      # it fired, and was only recorded
    assert "mode" not in r.termination


@pytest.mark.skipif(not PymooBackend.available(), reason="pymoo not installed")
def test_an_old_front_mode_argument_still_honours_the_cap():
    r = PymooBackend("nsga2", pop_size=40).optimize(_two_objective(), budget=400, seed=1,
                                                    front_mode="enforce")
    assert r.n_evaluations == 400 and r.termination["status"] == "max_evals"


@pytest.mark.skipif(not PymooBackend.available(), reason="pymoo not installed")
@pytest.mark.parametrize("budget", [41, 25, 137])
def test_the_cap_is_never_exceeded_even_mid_generation(budget):
    """pymoo checks termination between generations and evaluates a whole population in between, so
    a cap that is not a multiple of pop_size used to overshoot (41 -> 80 model runs)."""
    r = PymooBackend("nsga2", pop_size=40).optimize(_two_objective(), budget=budget, seed=1)
    assert r.n_evaluations == budget, r.n_evaluations
    if budget % 40:
        assert r.termination["cap_reached_mid_generation"] > 0
    assert r.best_x and all(math.isfinite(v) for v in r.best_loss)


@pytest.mark.skipif(not PymooBackend.available(), reason="pymoo not installed")
def test_a_cap_smaller_than_one_population_is_respected():
    r = PymooBackend("nsga2", pop_size=40).optimize(_two_objective(), budget=17, seed=1,
                                                    front_mode="enforce")
    assert r.n_evaluations == 17 and r.best_x
