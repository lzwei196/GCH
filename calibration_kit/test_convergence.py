"""Tests for the convergence marker and the metric panel (handoff §5.14 tests 1, 2, 3, 5).

1. The marker does NOT fire while the bias ratio keeps drifting, even though NSE is flat —
   and fires once the whole panel is flat. (The point of a panel: NSE alone would have said
   "converged" a long time earlier.)
2. Observe mode never stops a search: the backend reaches the cap and the best parameters are
   IDENTICAL to a run with no marker at the same seed.
3. `enforce` on DDS is downgraded to observe and says why.
5. Bootstrap tolerances shrink with record length and are deterministic under a seed.
Plus: the multi-objective incumbent follows the engine's own minimax rule, and the marker's
post-marker report flags a premature marker instead of calling it converged.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.convergence import (ConvergenceMarker, Incumbent, auto_window,   # noqa: E402
                                         make_hook, marker_evidence, resolve_mode)
from calibration_kit.panel import (Panel, bootstrap_tolerances, derive_panel,          # noqa: E402
                                   panel_from_series)
from calibration_kit.backends.base import Problem                                      # noqa: E402
from calibration_kit.backends.spotpy_backend import SpotpyBackend                      # noqa: E402

TOL = {"Q": {"nse": 0.01, "beta": 0.01}}


# ── test 1: a flat headline metric is not convergence ────────────────────────────────
def test_marker_waits_for_the_whole_panel():
    """NSE is flat from the start; beta drifts 0.02 per evaluation (twice its tolerance) for the
    first 200 evaluations, then stops. A marker that looked at NSE alone would fire at ~W."""
    W = 20
    mk = ConvergenceMarker(TOL, window=W, rel_gain=0.005, mode="observe")
    for i in range(400):
        beta = 1.0 + 0.02 * i if i < 200 else 1.0 + 0.02 * 199
        mk.update(0.1, {"Q": {"nse": 0.9, "beta": beta}})
    assert mk.marker is not None, "must fire once the panel goes flat"
    assert mk.marker >= 200, f"fired at {mk.marker} while beta was still drifting"
    assert mk.marker <= 200 + W + 1, f"fired at {mk.marker}, far later than the flat point"
    s = mk.summary()
    assert s["status"] == "converged" and not s["post_marker_exceeds_tol"]
    assert s["last_change"]["Q"]["beta"] == 199 and s["last_change"]["Q"]["nse"] is None
    # the same series judged on NSE alone: fires as soon as the window is full
    nse_only = ConvergenceMarker({"Q": {"nse": 0.01}}, window=W, rel_gain=0.005)
    for i in range(400):
        nse_only.update(0.1, {"Q": {"nse": 0.9, "beta": 1.0 + 0.02 * min(i, 199)}})
    assert nse_only.marker == W, nse_only.marker
    assert nse_only.marker < mk.marker


def test_marker_never_fires_while_loss_still_gains():
    mk = ConvergenceMarker(TOL, window=10, rel_gain=0.005)
    for i in range(200):
        mk.update(1.0 / (i + 1), {"Q": {"nse": 0.9, "beta": 1.0}})   # steady relative gain
    assert mk.marker is None
    s = mk.summary()
    assert s["status"] == "not_converged_at_cap" and s["marker_eval"] is None


def test_premature_marker_is_named():
    """Panel flat for a stretch, then it moves again past the marker: the report says premature."""
    mk = ConvergenceMarker(TOL, window=10, rel_gain=0.005)
    for i in range(60):
        mk.update(0.1, {"Q": {"nse": 0.9, "beta": 1.0}})
    assert mk.marker is not None
    for i in range(40):
        mk.update(0.05, {"Q": {"nse": 0.9, "beta": 1.0 + 0.01 * i}})
    s = mk.summary()
    assert s["status"] == "premature_marker"
    assert ["Q", "beta"] in s["post_marker_exceeds_tol"]
    assert 0.0 < s["budget_saved_fraction"] < 1.0


# ── test 2: observe mode changes nothing about the search ────────────────────────────
def _synthetic(n=3, best_nse=0.9):
    stats = {}
    def evaluate(x):
        nse = best_nse - sum((xi - 0.3) ** 2 for xi in x)
        stats["last"] = nse
        return [1.0 - nse]
    prob = Problem(names=[f"p{i}" for i in range(n)], lower=[0.0] * n, upper=[1.0] * n,
                   objective_names=["loss"], evaluate=evaluate)
    return prob, stats


def test_observe_mode_is_bit_identical_to_no_marker():
    budget, seed = 300, 7
    prob_a, stats_a = _synthetic()
    plain = SpotpyBackend("dds").optimize(prob_a, budget=budget, seed=seed)

    prob_b, stats_b = _synthetic()
    mk = ConvergenceMarker(TOL, window=auto_window(3), rel_gain=0.005, mode="observe")
    inc = Incumbent(multi=False)
    hook = make_hook(mk, inc, panel_of=lambda i: {"Q": {"nse": stats_b.get("last"), "beta": 1.0}})
    observed = SpotpyBackend("dds").optimize(prob_b, budget=budget, seed=seed, on_eval=hook)

    assert observed.n_evaluations == plain.n_evaluations == budget
    assert not observed.stopped_early
    assert observed.best_x == plain.best_x and observed.best_loss == plain.best_loss
    assert [h["loss"] for h in observed.history] == [h["loss"] for h in plain.history]
    assert mk.marker is not None, "a smooth 3-parameter problem should reach a marker in 300 evals"
    ev = marker_evidence(observed.history, mk.marker, lower=prob_b.lower, upper=prob_b.upper,
                         names=prob_b.names)
    assert set(ev) >= {"max_normalized_range", "wide_params", "identifiable"}


# ── test 3: DDS may not be stopped early ─────────────────────────────────────────────
def test_enforce_is_refused_for_dds():
    mode, warns = resolve_mode("enforce", "dds")
    assert mode == "observe" and warns and "dds" in warns[0]
    assert resolve_mode("enforce", "sceua") == ("enforce", [])
    assert resolve_mode("enforce", "nsga2") == ("enforce", [])
    assert resolve_mode(None, "sceua") == ("observe", [])
    m, w = resolve_mode("hard-stop", "sceua")
    assert m == "observe" and w


def test_enforce_stops_at_the_marker():
    mk = ConvergenceMarker(TOL, window=10, rel_gain=0.005, mode="enforce")
    fired = [mk.update(0.1, {"Q": {"nse": 0.9, "beta": 1.0}}) for _ in range(40)]
    assert fired.count(True) == 1 and fired.index(True) == 10


# ── test 5: tolerances are measured, and reproducible ────────────────────────────────
def test_bootstrap_tolerances_shrink_with_length_and_are_deterministic():
    import numpy as np
    rng = np.random.default_rng(0)
    obs = np.abs(rng.gamma(2.0, 2.0, 2000))
    sim = np.clip(obs * 1.1 + rng.normal(0, 0.5, 2000), 0, None)
    long_ = bootstrap_tolerances(sim, obs, n_boot=200, min_ok=100)
    short = bootstrap_tolerances(sim[:300], obs[:300], n_boot=200, min_ok=100)
    assert set(long_["tol"]) >= {"r", "alpha", "beta", "pbias", "nse", "kge"}
    assert all(long_["tol"][m] < short["tol"][m] for m in long_["tol"] if m in short["tol"])
    assert bootstrap_tolerances(sim, obs, n_boot=200, min_ok=100) == long_
    assert all(v > 0 for v in long_["tol"].values())
    tiny = bootstrap_tolerances(sim[:5], obs[:5])                # too short to measure anything
    assert set(tiny["source"].values()) == {"fixed_fallback"}


def test_panel_derivation_matches_the_series_panel():
    """A runner that reports only nse/kge/r/pbias gets r, beta, pbias, nse, kge — and NO alpha:
    alpha from KGE alone has an unknown sign and is not used (design §1.4)."""
    import numpy as np
    rng = np.random.default_rng(3)
    obs = np.abs(rng.gamma(2.0, 2.0, 1500))
    sim = np.clip(obs * 1.1 + rng.normal(0, 0.4, 1500), 0, None)
    truth = panel_from_series(sim, obs)
    got, flags = derive_panel({k: truth[k] for k in ("nse", "kge", "r", "pbias")})
    assert abs(got["beta"] - truth["beta"]) < 1e-9
    assert "alpha" not in got and "beta_from_pbias" in flags
    # one declared variable: var-scoped and flat payloads resolve to the same panel
    p = Panel(["Q"])
    flat = p.extract({k: truth[k] for k in ("nse", "kge", "r", "pbias")})
    scoped = p.extract({"Q": {k: truth[k] for k in ("nse", "kge", "r", "pbias")}})
    assert flat == scoped and set(flat["Q"]) >= {"r", "beta", "pbias", "nse", "kge"}


def test_panel_leaves_out_what_it_cannot_compute():
    import numpy as np
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([-1.0, 2.0, 3.0, 5.0])
    p = panel_from_series(sim, obs, "flow")
    assert "lnnse" not in p                        # -1 <= -0.01*mean(obs): the log is undefined
    assert "lnnse" in panel_from_series(np.array([0.5, 2.0, 3.0, 5.0]), obs, "flow")
    flat = panel_from_series(np.ones(50), np.ones(50))
    assert flat.get("nse") is None and flat.get("r") is None    # zero-variance obs: not inf, None


# ── the multi-objective incumbent ────────────────────────────────────────────────────
def test_multi_objective_incumbent_uses_the_minimax_rule():
    inc = Incumbent(multi=True)
    panels = [{"Q": {"nse": 0.1}}, {"Q": {"nse": 0.2}}, {"Q": {"nse": 0.3}}]
    inc.add([0.0, 1.0], panels[0])                 # extreme
    inc.add([1.0, 0.0], panels[1])                 # the other extreme
    loss, panel = inc.add([0.4, 0.4], panels[2])   # balanced -> smallest worst objective
    assert panel == panels[2]
    assert loss == 0.0                             # ideal point = (0, 0) -> sum 0
    dominated = inc.add([0.9, 0.9], {"Q": {"nse": 0.4}})[1]
    assert dominated == panels[2], "a dominated point must never become the incumbent"


def test_single_objective_incumbent_is_best_so_far():
    inc = Incumbent()
    assert inc.add([1.0], {"Q": {"nse": 0.1}}) == (1.0, {"Q": {"nse": 0.1}})
    assert inc.add([2.0], {"Q": {"nse": 0.2}}) == (1.0, {"Q": {"nse": 0.1}})
    assert inc.add([0.5], {"Q": {"nse": 0.3}}) == (0.5, {"Q": {"nse": 0.3}})
    assert inc.add([float("inf")], {})[0] == 0.5   # a failed eval never becomes the incumbent


def test_auto_window():
    assert auto_window(1) == 50 and auto_window(4) == 50 and auto_window(10) == 110


def test_no_records_is_not_observed_not_unconverged():
    """A backend that cannot report per evaluation leaves the marker empty. That is a missing
    observation, not a verdict about the search."""
    mk = ConvergenceMarker(TOL, window=10)
    s = mk.summary()
    assert s["status"] == "not_observed" and s["evals"] == 0
    mk.update(0.5, {"Q": {"nse": 0.5, "beta": 1.0}})
    assert mk.summary()["status"] == "not_converged_at_cap"


# ── the rules the first version of the marker got wrong (codex review 2026-09-27) ─────
def test_a_missing_panel_can_never_fire_the_marker():
    """No panel evidence is not flatness. The first version skipped unreported metrics, so a run
    with no panel at all fired on an empty check — in enforce mode that would stop a search."""
    mk = ConvergenceMarker(TOL, window=10, rel_gain=0.005, mode="enforce")
    fired = [mk.update(0.1, {}) for _ in range(50)]
    assert mk.marker is None and not any(fired)
    s = mk.summary()
    assert s["status"] == "not_converged_at_cap"
    assert s["panel_coverage"]["covered"] == 0 and s["panel_coverage"]["wanted"] == 2


def test_nan_metrics_are_missing_evidence_not_flat_values():
    mk = ConvergenceMarker(TOL, window=10, rel_gain=0.005)
    for _ in range(50):
        mk.update(0.1, {"Q": {"nse": float("nan"), "beta": float("inf")}})
    assert mk.marker is None


def test_partial_coverage_is_refused_by_default():
    """beta declared in the tolerances but never reported: with the default min_coverage=1.0 the
    marker waits, because 'flat' cannot be claimed for a metric nobody measured."""
    mk = ConvergenceMarker(TOL, window=10)
    for _ in range(40):
        mk.update(0.1, {"Q": {"nse": 0.9}})
    assert mk.marker is None
    loose = ConvergenceMarker(TOL, window=10, min_coverage=0.5)
    for _ in range(40):
        loose.update(0.1, {"Q": {"nse": 0.9}})
    assert loose.marker == 10 and loose.summary()["panel_coverage"]["covered"] == 1


def test_an_oscillating_metric_does_not_pass_the_window():
    """beta swings 1.0 <-> 1.1 every step. Comparing only the window's two endpoints called that
    flat whenever the period divided the window; the whole window is checked now."""
    mk = ConvergenceMarker({"Q": {"beta": 0.01}}, window=4, rel_gain=0.005)
    for i in range(60):
        mk.update(0.1, {"Q": {"beta": 1.0 + 0.1 * (i % 2)}})
    assert mk.marker is None


def test_a_marker_the_search_improved_past_is_premature():
    """The panel can sit inside tolerance while the LOSS keeps falling — that is not convergence."""
    mk = ConvergenceMarker({"Q": {"nse": 0.05}}, window=10, rel_gain=0.005)
    for _ in range(40):
        mk.update(1.0, {"Q": {"nse": 0.90}})
    assert mk.marker == 10
    for i in range(40):
        mk.update(1.0 - 0.02 * (i + 1), {"Q": {"nse": 0.91}})   # loss falls, nse inside tolerance
    s = mk.summary()
    assert s["post_marker_loss_moved"] is True and s["status"] == "premature_marker"
    assert s["post_marker_loss_gain"] > 0.005


def test_savings_are_measured_against_the_cap_not_the_shortened_run():
    """Enforce mode stops AT the marker, so the run is only marker+1 long. Measuring savings
    against that length reported 0% saved — the cap is the right denominator."""
    mk = ConvergenceMarker({"Q": {"nse": 0.01}}, window=10, rel_gain=0.005, mode="enforce")
    for _ in range(11):
        mk.update(0.1, {"Q": {"nse": 0.9}})
    assert mk.marker == 10
    short = mk.summary()
    assert short["budget_saved_fraction"] == 0.0 and short["budget_saved_against"] == "observed_evals"
    against_cap = mk.summary(cap=100)
    assert against_cap["budget_saved_fraction"] == pytest.approx(0.89)
    assert against_cap["budget_saved_against"] == "cap"


def test_a_run_that_stopped_early_is_not_called_unconverged_at_the_cap():
    mk = ConvergenceMarker({"Q": {"nse": 0.001}}, window=10, rel_gain=0.0)
    for i in range(20):
        mk.update(1.0 / (i + 1), {"Q": {"nse": 0.5 + 0.01 * i}})
    assert mk.marker is None
    assert mk.summary(at_cap=True)["status"] == "not_converged_at_cap"
    assert mk.summary(at_cap=False)["status"] == "not_converged_stopped_early"


def test_post_marker_change_reports_the_worst_excursion():
    """A metric that wanders and comes back must not report a zero change."""
    mk = ConvergenceMarker({"Q": {"nse": 0.01}}, window=5, rel_gain=0.005)
    for _ in range(20):
        mk.update(0.1, {"Q": {"nse": 0.90}})
    assert mk.marker is not None
    for v in (0.95, 0.90):
        mk.update(0.1, {"Q": {"nse": v}})
    s = mk.summary()
    assert s["post_marker_change"]["Q"]["nse"] == pytest.approx(0.05)
    assert s["post_marker_change_at_end"]["Q"]["nse"] == pytest.approx(0.0)
    assert s["status"] == "premature_marker"


def test_tolerances_can_be_learned_from_the_first_scored_evaluation():
    """An applicator-mode contract has no pre-search evaluation to read metrics from. Rather than
    fire on the loss alone (the NSE-only claim) or never fire at all, the marker takes its
    tolerances from the first panel it sees and says that it did."""
    from calibration_kit.panel import FIXED_FALLBACK_TOL
    mk = ConvergenceMarker({}, window=10, rel_gain=0.005,
                           fallback_table={m: FIXED_FALLBACK_TOL[m] for m in ("nse", "pbias")})
    for _ in range(40):
        mk.update(0.1, {"Q": {"nse": 0.9, "pbias": 2.0}})
    assert mk.tolerances_learned and set(mk.tol["Q"]) == {"nse", "pbias"}
    assert mk.marker == 10
    assert mk.summary()["tolerances_learned_from_first_eval"] is True
    # ... and a drifting metric still holds it back
    mk2 = ConvergenceMarker({}, window=10, rel_gain=0.005,
                            fallback_table={m: FIXED_FALLBACK_TOL[m] for m in ("nse", "pbias")})
    for i in range(60):
        mk2.update(0.1, {"Q": {"nse": 0.9, "pbias": 2.0 + 0.5 * i}})
    assert mk2.marker is None


def test_without_a_fallback_table_no_tolerances_means_no_claim():
    mk = ConvergenceMarker({}, window=5, rel_gain=0.005, mode="enforce")
    fired = [mk.update(0.1, {"Q": {"nse": 0.9}}) for _ in range(30)]
    assert not any(fired) and mk.marker is None
    assert mk.summary()["status"] == "not_converged_at_cap"


def test_a_variable_with_no_panel_evidence_blocks_the_marker():
    """Two calibrated variables, one of them never reported: the fit cannot be called settled for
    a variable nobody measured."""
    tol = {"Q": {"nse": 0.01}, "SWE": {"nse": 0.01}}
    mk = ConvergenceMarker(tol, window=10)
    for _ in range(40):
        mk.update(0.1, {"Q": {"nse": 0.9}})
    assert mk.marker is None
    assert mk._coverage_short["variables_without_evidence"] == ["SWE"]
