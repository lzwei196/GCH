"""A3a: the optimizers' own stopping rules on our scores (native_rules.py)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.native_rules import SpotpyTest, pct_change          # noqa: E402

S = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0, "lnnse": 0.5}}


def test_pct_change_is_spotpys_formula():
    c = [1.0] * 9 + [0.999]
    # |0.999 - 1.0| * 100 / mean(...) ; mean = (9 + 0.999) / 10
    assert pct_change(c, 10) == pytest.approx(0.1 / 0.9999)
    assert pct_change([0.0] * 10, 10) == 0.0                       # mean 0 -> 0, as SPOTPY
    assert pct_change([1.0] * 9 + [None], 10) is None


def test_fires_only_from_kstop_and_on_threshold():
    t = SpotpyTest(watch_scores=False)
    for n in range(1, 10):
        assert t.add_loop(n, 1.0) is False                          # n < kstop: no objective test
    assert t.add_loop(10, 1.0) is True and t.fired_at["loop"] == 10
    t = SpotpyTest(watch_scores=False)
    for n in range(1, 10):
        t.add_loop(n, 1.0)
    assert t.add_loop(10, 0.99) is False                            # ~1 % change > 0.1 %


def test_exact_threshold_passes():
    """SPOTPY: criter_change_pcent <= pcento converges (<=, not <). Binary-exact numbers."""
    t = SpotpyTest(watch_scores=False, kstop=2, pcento=50.0)
    t.add_loop(1, 1.0)
    assert t.add_loop(2, 0.5) is False          # |0.5-1|*100/0.75 = 66.7 > 50
    t = SpotpyTest(watch_scores=False, kstop=2, pcento=100.0 * 0.5 / 0.75)
    t.add_loop(1, 1.0)
    assert t.add_loop(2, 0.5) is True           # equal -> converged


def test_gnrng_fires_any_time():
    t = SpotpyTest()
    assert t.add_loop(1, 1.0, gnrng=0.0005) is True and "gnrng" in t.fired_at["why"]
    t = SpotpyTest()
    assert t.add_loop(1, 1.0, gnrng=0.001) is False                 # not < peps


def test_scores_must_also_pass_and_missing_fails_closed():
    t = SpotpyTest()
    for n in range(1, 10):
        t.add_loop(n, 1.0, scores=S)
    moved = {"Q": dict(S["Q"], alpha=1.05)}
    assert t.add_loop(10, 1.0, scores=moved) is False               # alpha moved ~0.5 %
    t = SpotpyTest()
    for n in range(1, 10):
        t.add_loop(n, 1.0, scores=S)
    assert t.add_loop(10, 1.0, scores={"Q": dict(S["Q"], beta=float("nan"))}) is False
    t = SpotpyTest()
    for n in range(1, 11):
        fired = t.add_loop(n, 1.0, scores=S)
    assert fired is True
    t = SpotpyTest()                                                # watching scores but none given
    for n in range(1, 11):
        fired = t.add_loop(n, 1.0)
    assert fired is False


def test_loops_in_order_and_summary_words():
    t = SpotpyTest()
    with pytest.raises(ValueError):
        t.add_loop(2, 1.0)
    t = SpotpyTest(watch_scores=False)
    for n in range(1, 6):
        t.add_loop(n, 1.0)
    assert t.summary()["verdict"] == "too short to judge"
    for n in range(6, 12):
        t.add_loop(n, 1.0 - n * 0.01)
    assert t.summary()["verdict"] == "not converged within the budget"
    assert "not SPOTPY's shipped defaults" in t.summary()["settings"]["note"]


def test_pymoo_observer_on_a_tiny_problem():
    pytest.importorskip("pymoo")
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem
    from calibration_kit.native_rules import PymooTest
    from pymoo.termination.default import DefaultMultiObjectiveTermination
    kw = dict(period=5, xtol=0.05, ftol=0.05)        # loose, so pymoo's own rule really fires on this toy problem
    obs = PymooTest(**kw)

    class CB(Callback):
        def notify(self, algorithm):
            obs.update(algorithm)
    # the same rule as pymoo's live termination: the observer must fire at the generation where pymoo itself stops
    live = minimize(get_problem("zdt1"), NSGA2(pop_size=20), DefaultMultiObjectiveTermination(**kw),
                    seed=1, callback=CB(), verbose=False)
    t = live.algorithm.termination
    assert max(t.x.perc, t.f.perc) >= 1.0 and t.max_gen.perc < 1.0      # pymoo stopped by convergence, not a cap
    assert obs.fired_at is not None
    assert obs.fired_at["n_gen"] == obs.records[-1]["n_gen"]           # it fired on the last generation pymoo ran
    assert sum(r["fired"] for r in obs.records) == 1


def test_pymoo_cap_is_never_convergence():
    """Shipped settings on zdt1: pymoo's own run ends at its 1,000-generation cap without converging; the observer's
    own max-gen part also reaches 1 then, and must NOT count as convergence."""
    pytest.importorskip("pymoo")
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem
    from calibration_kit.native_rules import PymooTest
    obs = PymooTest()

    class CB(Callback):
        def notify(self, algorithm):
            obs.update(algorithm)
    minimize(get_problem("zdt1"), NSGA2(pop_size=20), ("n_gen", 1000), seed=1, callback=CB(), verbose=False)
    assert obs.t.max_gen.perc >= 1.0
    assert obs.fired_at is None and obs.summary()["verdict"] == "not converged within the budget"


def test_disappearing_variable_fails_closed():
    """codex A3a r1 #1: Q and SWE watched for 9 loops, SWE missing at loop 10 -> must not fire."""
    two = {"Q": dict(S["Q"]), "SWE": dict(S["Q"])}
    t = SpotpyTest()
    for n in range(1, 10):
        t.add_loop(n, 1.0, scores=two)
    assert t.add_loop(10, 1.0, scores={"Q": dict(S["Q"])}) is False


def test_pymoo_too_short_threshold():
    """codex A3a r1 #2: earliest x/f firing is after period + n_skip + 1 updates."""
    pytest.importorskip("pymoo")
    from calibration_kit.native_rules import PymooTest
    t = PymooTest(period=5)
    t.gens = 5 + 5                     # period + n_skip
    assert t.summary()["verdict"] == "too short to judge"
    t.gens = 5 + 5 + 1
    assert t.summary()["verdict"] == "not converged within the budget"
