"""Tests for the measured budget and the pilot (handoff §5.14 tests 4 and 6).

4. plan_budget arithmetic: waves, efficiency, the κ warning, hard_max, fixed mode.
6. run_pilot returns n records, p90 >= median, and `pilot_unstable` when failures exceed 30%.
Plus the machine probe / lane arithmetic, which the cap depends on.
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.budget import (KAPPA_WARN, estimate_overhead_evals,          # noqa: E402
                                    parse_allowance, plan_budget)
from calibration_kit.compute import choose_lanes, lane_ceiling, probe_machine     # noqa: E402
from calibration_kit.evaluator import Evaluator                                   # noqa: E402
from calibration_kit.objectives import Objective                                  # noqa: E402
from calibration_kit.pilot import lhs_points, run_pilot                           # noqa: E402

PILOT = {"median_s": 8.0, "p90_s": 10.0, "fail_rate": 0.0, "threads_per_run": 1.0,
         "peak_rss_mb": 400.0}


# ── test 4: the cap arithmetic ───────────────────────────────────────────────────────
def test_cap_is_allowance_over_run_time():
    """1 lane, 1 seed, t_slow = 1.2 x p90 = 12 s, 1 h allowance, no overhead -> 300 calls (design §1.3)."""
    p = plan_budget("1h", PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=1, final_overhead_s=0)
    assert p["t_slow_s"] == pytest.approx(12.0)
    assert p["cap_per_seed"] == 300 and p["waves"] == 1
    assert p["kappa"] == pytest.approx(50.0)
    assert p["expected_finish_s"] == pytest.approx(3600.0)
    assert p["mode"] == "measured" and not p["warnings"] and not p["refuse"]


def test_seeds_that_do_not_fit_in_lanes_become_waves():
    """3 seeds in 1 lane = 3 waves, so each seed gets a third of the allowance."""
    one = plan_budget("3h", PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=3, final_overhead_s=0)
    assert one["waves"] == 3 and one["seeds_parallel"] == 1
    assert one["cap_per_seed"] == 300           # 3 h / 3 waves = 1 h per seed at 12 s per call
    three = plan_budget("3h", PILOT, lanes=3, efficiency=1.0, n_params=5, n_seeds=3, final_overhead_s=0)
    assert three["waves"] == 1 and three["cap_per_seed"] == 900    # all three seeds get the 3 h
    # more lanes than seeds buys nothing: the extra lanes have no seed to run
    four = plan_budget("3h", PILOT, lanes=4, efficiency=1.0, n_params=5, n_seeds=3, final_overhead_s=0)
    assert four["seeds_parallel"] == 3 and four["cap_per_seed"] == three["cap_per_seed"]


def test_efficiency_and_overhead_reduce_the_cap():
    base = plan_budget("1h", PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=1, final_overhead_s=0)
    slow = plan_budget("1h", PILOT, lanes=2, efficiency=0.8, n_params=5, n_seeds=1, final_overhead_s=0)
    assert slow["cap_per_seed"] == int(base["cap_per_seed"] * 0.8)
    # the older per-run overhead form is charged as shared time: 60 runs x 12 s
    with_overhead = plan_budget("1h", PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=1,
                                overhead_evals=60, final_overhead_s=0)
    assert with_overhead["cap_per_seed"] == base["cap_per_seed"] - 60
    # the expected finish time includes the overhead, not just the search
    assert with_overhead["expected_finish_s"] == pytest.approx(3600.0)


def test_kappa_warning_fires_on_a_short_allowance():
    p = plan_budget("1h", PILOT, lanes=1, efficiency=1.0, n_params=100, n_seeds=1, final_overhead_s=0)
    assert p["kappa"] == pytest.approx(300 / 101, abs=0.01)
    assert p["kappa"] < KAPPA_WARN
    assert any("kappa" in w for w in p["warnings"])
    assert p["cap_per_seed"] == 300, "the warning reports; it never silently changes the cap"


def test_hard_max_and_contract_ceiling_trim_the_cap():
    p = plan_budget("10h", PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=1, hard_max=500)
    assert p["cap_per_seed"] == 500 and any("hard_max" in w for w in p["warnings"])
    q = plan_budget("10h", PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=1,
                    max_evaluations=800)
    assert q["cap_per_seed"] == 800 and any("max_evaluations" in w for w in q["warnings"])


def test_fixed_mode_keeps_the_declared_budget_but_still_reports_kappa():
    p = plan_budget(None, PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=1,
                    mode="fixed", max_evaluations=480)
    assert p["mode"] == "fixed" and p["cap_per_seed"] == 480
    assert p["kappa"] == pytest.approx(80.0)
    assert p["expected_finish_s"] == pytest.approx(5 * 12.0 + 480 * 12.0)   # final overhead (5 runs) + search


def test_no_allowance_and_no_budget_is_reported_not_guessed():
    p = plan_budget(None, PILOT, lanes=1, efficiency=1.0, n_params=5, n_seeds=1)
    assert p["mode"] == "unplanned" and p["cap_per_seed"] == 0
    assert any("max_evaluations" in w for w in p["warnings"])
    q = plan_budget("1h", {"p90_s": 0.0}, lanes=1, efficiency=1.0, n_params=5, n_seeds=1,
                   max_evaluations=200)
    assert q["mode"] == "fixed" and q["cap_per_seed"] == 200     # no timing -> cannot measure


def test_a_tiny_cap_is_reported_as_tiny_not_raised():
    """The cap stays as measured — raising it would spend time the user never granted — and a cap
    under 20 is marked for refusal with the reason (design §1.3)."""
    p = plan_budget("60s", {"p90_s": 30.0, "median_s": 30.0}, lanes=1, efficiency=1.0,
                    n_params=5, n_seeds=1, final_overhead_s=0)
    assert p["cap_per_seed"] == 1                        # floor(60 / 36)
    assert p["refuse"] and "under 20" in p["refuse_reason"]
    assert any("kappa" in w for w in p["warnings"])


def test_pilot_failures_are_carried_into_the_plan():
    p = plan_budget("1h", dict(PILOT, fail_rate=0.2), lanes=1, efficiency=1.0, n_params=5, n_seeds=1)
    assert any("fail rate" in w for w in p["warnings"])


def test_parse_allowance():
    assert parse_allowance("72h") == 72 * 3600
    assert parse_allowance("90m") == 5400 and parse_allowance("3d") == 259200
    assert parse_allowance("1.5h") == 5400 and parse_allowance(5400) == 5400
    assert parse_allowance("5400") == 5400 and parse_allowance("nonsense") == 0.0
    assert parse_allowance(None) == 0.0


def test_overhead_estimate_names_its_parts():
    o = estimate_overhead_evals(8, runner_mode="runner", front_size=5)       # (no Morris screen since step 8)
    assert o["commission"] == 9 and o["consumption_proof"] == 9 and "morris_screen" not in o
    assert o["holdout"] == 5 and o["front_select"] == 10
    assert o["total"] == 10 + 9 + 9 + 5 + 10
    a = estimate_overhead_evals(8, runner_mode="applicator")
    assert "commission" not in a and a["total"] == 10 + 5


# ── lanes ────────────────────────────────────────────────────────────────────────────
def test_lane_ceiling_binds_on_cpu_or_memory():
    m = {"free_cores": 16, "mem_available_gb": 64.0}
    k, why = lane_ceiling(m, {"threads_per_run": 4.0, "peak_rss_mb": 500.0})
    assert k == 4 and why["binding"] == "cpu"
    k2, why2 = lane_ceiling(m, {"threads_per_run": 1.0, "peak_rss_mb": 8000.0})
    assert k2 == 6 and why2["binding"] == "memory"       # 0.8 * 64000 / 8000


def test_choose_lanes_needs_measured_efficiency():
    assert choose_lanes(8, None)[0] == 1
    assert choose_lanes(8, {"efficiency": {1: 1.0, 2: 0.95, 4: 0.9}})[0] == 4
    assert choose_lanes(8, {"efficiency": {1: 1.0, 2: 0.95, 4: 0.5}})[0] == 2
    assert choose_lanes(2, {"efficiency": {1: 1.0, 2: 0.95, 4: 0.9}})[0] == 2   # never over the ceiling


def test_probe_machine_is_sane():
    m = probe_machine()
    assert m["cores"] >= 1 and 1 <= m["free_cores"] <= m["cores"]
    if Path("/proc/meminfo").is_file():
        assert m["mem_available_gb"] > 0
    else:
        # The current probe reports unavailable memory as zero off Linux.
        assert m["mem_available_gb"] == 0


# ── test 6: the pilot ────────────────────────────────────────────────────────────────
def _evaluator(tmp_path, *, fail_fraction=0.0, sleep_s=0.0, emit_series=None):
    wd = str(tmp_path)
    params = [{"name": n, "address": "none", "default": 1.0, "range": [0.0, 4.0]}
              for n in ("a", "b")]
    state = {"i": -1}

    def run():
        state["i"] += 1
        if sleep_s:
            import time as _t
            _t.sleep(sleep_s * (1 + state["i"]))     # rising run time -> p90 > median
        named = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        if fail_fraction and (state["i"] % max(1, round(1 / fail_fraction))) == 1:
            return {}
        kdt = {"applied_params": dict(named)}
        if emit_series is not None and os.environ.get("KDT_CALIB_EMIT_SERIES") == "1":
            import numpy as _np
            f = Path(wd, "kdt_series_Q.npz")
            _np.savez(f, sim=emit_series[0], obs=emit_series[1])
            kdt["series"] = {"Q": str(f)}
        return {"nse": 1.0 - (named["a"] - 2.0) ** 2 / 10.0, "pbias": 3.0, "r": 0.9, "kge": 0.8,
                "__kdt__": kdt}

    obj = Objective(name="Q:temporal_pattern_match", var="Q",
                    family="temporal_pattern_match", metric_key="nse")
    return Evaluator(ki_path=wd, workdir=wd, parameters=params, objectives=[obj],
                     transform_inv={}, run_model=run, injection_mode="runner")


def test_pilot_returns_n_records_and_timing(tmp_path):
    ev = _evaluator(tmp_path, sleep_s=0.002)
    p = run_pilot(ev, [0.0, 0.0], [4.0, 4.0], [1.0, 1.0], n=10, seed=0)
    assert p["status"] == "ok" and p["n"] == 10 and len(p["records"]) == 10
    assert p["p90_s"] >= p["median_s"] >= p["min_s"] > 0
    assert p["max_s"] >= p["p90_s"]
    assert p["records"][0]["default"] and p["records"][0]["x"] == [1.0, 1.0]
    assert p["fail_rate"] == 0.0 and p["default_ok"]
    assert p["threads_per_run"] > 0 and p["peak_rss_mb"] > 0
    # every pilot evaluation is in the history, tagged `pilot`
    recs = [json.loads(l) for l in Path(tmp_path, "eval_history.jsonl").read_text().splitlines()]
    assert len(recs) == 10 and {r["phase"] for r in recs} == {"pilot"}
    assert all(r["cache_hit"] is False for r in recs), "the pilot must TIME the model, not the cache"
    assert ev.phase == "untagged"            # the tag is restored (the default since build step 7)


def test_pilot_points_cover_each_range(tmp_path):
    xs = lhs_points([0.0, 10.0], [1.0, 20.0], 9, seed=3)
    assert len(xs) == 9
    for dim, (lo, hi) in enumerate([(0.0, 1.0), (10.0, 20.0)]):
        col = sorted(x[dim] for x in xs)
        assert lo <= col[0] < col[-1] <= hi
        assert col[0] < lo + 0.2 * (hi - lo) and col[-1] > hi - 0.2 * (hi - lo)
    assert lhs_points([0.0], [1.0], 9, seed=3) == lhs_points([0.0], [1.0], 9, seed=3)


def test_pilot_unstable_when_more_than_a_third_fails(tmp_path):
    ev = _evaluator(tmp_path, fail_fraction=0.5)
    p = run_pilot(ev, [0.0, 0.0], [4.0, 4.0], [1.0, 1.0], n=10, seed=0)
    assert p["fail_rate"] > 0.30
    assert p["status"] == "pilot_unstable" and "ranges" in p["reason"]


def test_pilot_tolerances_fall_back_when_no_series(tmp_path):
    from calibration_kit.pilot import pilot_tolerances
    ev = _evaluator(tmp_path)
    p = run_pilot(ev, [0.0, 0.0], [4.0, 4.0], [1.0, 1.0], n=4, seed=0)
    tol, src = pilot_tolerances(p, ["Q"])
    # design §2.11: no series -> every recorded metric gets the fixed fallback, LABELLED
    assert src == "fixed_fallback" and set(tol) == {"Q"}
    assert set(tol["Q"]["source"].values()) == {"fixed_fallback"}


def test_pilot_tolerances_from_a_runner_written_series(tmp_path):
    """When the runner writes kdt_series_<var>.npz DURING the pilot's default evaluation, the
    tolerances are MEASURED from it. A file that was merely lying in the workdir is not evidence
    from this run, so the runner has to write it under KDT_CALIB_EMIT_SERIES."""
    import numpy as np
    from calibration_kit.pilot import pilot_tolerances
    rng = np.random.default_rng(0)
    obs = np.abs(rng.gamma(2.0, 2.0, 800))
    sim = np.clip(obs * 1.05 + rng.normal(0, 0.3, 800), 0, None)
    stale = Path(tmp_path, "kdt_series_STALE.npz")
    np.savez(stale, sim=sim, obs=obs)            # pre-existing: must NOT be picked up
    ev = _evaluator(tmp_path, emit_series=(sim, obs))
    p = run_pilot(ev, [0.0, 0.0], [4.0, 4.0], [1.0, 1.0], n=3, seed=0)
    assert set(p["series"]) == {"Q"}, p["series"]
    tol, src = pilot_tolerances(p, ["Q"], {"Q": "series"})
    assert src == "bootstrap" and set(tol) == {"Q"}
    assert all(v > 0 for v in tol["Q"]["tol"].values())
    assert {"r", "alpha", "beta", "pbias", "nse", "kge", "nrmse"} <= set(tol["Q"]["tol"])


def test_a_stale_series_file_is_not_accepted_as_measurement(tmp_path):
    """Evidence has to come from THIS run: a series left in the workdir by an earlier candidate or
    another split must not become a measured tolerance."""
    import numpy as np
    from calibration_kit.pilot import pilot_tolerances
    rng = np.random.default_rng(0)
    obs = np.abs(rng.gamma(2.0, 2.0, 400))
    np.savez(Path(tmp_path, "kdt_series_Q.npz"), sim=obs * 1.1, obs=obs)   # nobody wrote this now
    ev = _evaluator(tmp_path)
    p = run_pilot(ev, [0.0, 0.0], [4.0, 4.0], [1.0, 1.0], n=3, seed=0)
    assert p["series"] == {}
    tol, src = pilot_tolerances(p, ["Q"])
    assert src == "fixed_fallback"          # never "measured" from a file this run did not write
