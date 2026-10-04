"""End-to-end acceptance for the 2026-09-27 convergence work (handoff §5.16).

Runs the REAL calibrate() entry point on a tiny two-parameter fixture with a fake runner that
behaves like a real one: it injects through KDT_CALIB_PARAMS, echoes applied_params, honours
KDT_CALIB_SPLIT, writes a paired series when asked, and reports the whole metric panel. The run
must come back with the three new report blocks — budget_plan, convergence, phase_counts — and
observe mode must leave the search itself alone (it reaches the cap, it is not stopped early).
"""
from __future__ import annotations
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit import calib as C                        # noqa: E402
from calibration_kit.panel import panel_block, panel_from_series   # noqa: E402

N = 400
RNG = np.random.default_rng(11)
#: a stand-in hydrograph: seasonal signal + noise, positive, 400 daily steps
OBS = np.clip(20.0 + 12.0 * np.sin(np.linspace(0, 12.6, N)) + RNG.normal(0, 1.5, N), 0.05, None)
CAL = slice(0, 300)
HOLD = slice(300, N)


class Runner:
    """sim = a * OBS + b; the optimum is a=1, b=0, and both knobs really reach the output."""

    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        a, b = float(p["a"]), float(p["b"])
        sim_all = a * OBS + b
        sl = HOLD if (os.environ.get("KDT_CALIB_SPLIT") == "holdout") else CAL
        sim, obs = sim_all[sl], OBS[sl]
        # plain keys = the KI's own metrics (what the objectives read); the kit's panel goes in
        # the kit-owned block, never under the plain names (design §1.4)
        full = panel_from_series(sim, obs, "flow")
        m = {k: full[k] for k in ("nse", "kge", "r", "pbias") if k in full}
        m.update(panel_block({"streamflow": (sim, obs)}, {}))
        kdt = m["__kdt__"]
        kdt.update({"applied_params": dict(p), "case_id": "SITE:fixture",
                    # the split it scored, echoed (design §1 "Data within a search")
                    "split": "holdout" if os.environ.get("KDT_CALIB_SPLIT") == "holdout" else "calibration"})
        if os.environ.get("KDT_CALIB_EMIT_SERIES") == "1":
            f = self.workdir / "kdt_series_streamflow.npz"
            np.savez(f, sim=sim, obs=obs)
            kdt["series"] = {"streamflow": str(f)}
        return m


def _fixture(tmp, budget_block, convergence_block, max_evaluations=60):
    ki, wd = Path(tmp, "ki"), Path(tmp, "wd")
    ki.mkdir(parents=True); wd.mkdir(parents=True)
    strategy = {"max_evaluations": max_evaluations, "default_algorithm": "dds",
                "probe_objectives": True,
                "holdout": {"max_degradation": 0.5, "abs_tolerance": 0.2}}
    if budget_block is not None:
        strategy["budget"] = budget_block
    if convergence_block is not None:
        strategy["convergence"] = convergence_block
    yaml.safe_dump({"injection": {"mode": "runner"},
                    "parameters": [{"name": "a", "range": [0.5, 1.5], "default": 0.8,
                                    "type": "continuous"},
                                   {"name": "b", "range": [-5.0, 5.0], "default": 2.0,
                                    "type": "continuous"}],
                    "targets": [{"var": "streamflow", "weight": 1.0}],
                    "strategy": strategy,
                    "runner": {"command": ["python", "tools/calib_run.py"]}},
                   open(ki / "calibration.yaml", "w"))
    yaml.safe_dump({"outputs": [{"var": "streamflow", "observability": {"comparable_obs_shapes": [
        {"obs_shape": "point_time_series", "metric_families": ["temporal_pattern_match"]}]}}]},
        open(ki / "dag.yaml", "w"))
    return str(ki), str(wd)


@contextlib.contextmanager
def _clean_env():
    keys = ("KDT_CALIB_CONTRACT", "KDT_CALIB_SPLIT", "KDT_CALIB_PARAMS", "KDT_CALIB_EMIT_SERIES",
            "KDT_CALIB_CASE_TAG")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    os.environ["KDT_CALIB_C8"] = "enforce"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(budget_block=None, convergence_block=None, budget=None, max_evaluations=60, seed=0):
    tmp = tempfile.mkdtemp(prefix="kdt_conv_e2e_")
    try:
        ki, wd = _fixture(tmp, budget_block, convergence_block, max_evaluations)
        buf = io.StringIO()
        with _clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=Runner(wd), budget=budget, seed=seed)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        return rep, hist, buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── the acceptance run ───────────────────────────────────────────────────────────────
def test_measured_budget_and_marker_end_to_end():
    rep, hist, log = _run(budget_block={"mode": "measured", "allowance": "2m", "seeds": 1,
                                        "pilot_runs": 10},
                          convergence_block={"mode": "observe", "window": 20, "rel_gain": 0.005},
                          max_evaluations=80)
    assert rep["status"] == "completed", rep.get("reason")

    # 1) the budget was DERIVED, and says from what
    bp = rep["budget_plan"]
    assert bp["mode"] == "measured" and bp["cap_per_seed"] > 0
    assert bp["t_p90_s"] > 0 and bp["pilot"]["n"] == 10
    assert bp["machine"]["cores"] >= 1 and bp["lanes"] >= 1
    assert bp["overhead"]["total"] > 0 and bp["kappa"] is not None
    # a 2-minute allowance on a millisecond model is a huge cap -> trimmed to the declared ceiling
    assert rep["budget_used"]["cap"] == 80 and any("max_evaluations" in w for w in bp["warnings"])

    # 2) the rule observed (the contract's old mode name "observe" = keep going) and did NOT interfere
    cv = rep["convergence"]
    assert cv["mode"] == "keep_going" and cv["rule"]["window"] == 20
    assert any("old name" in w for w in cv["warnings"])
    # the fixture's sim is an exact straight line of obs, so r and alpha never vary between
    # resamples: those two fall back ("SD is 0"), the rest are measured (design §2.11)
    assert cv["tolerance_source"] == "bootstrap+fixed_fallback", cv["tolerance_source"]
    _src = cv["tolerance_records"]["streamflow"]["source"]
    assert _src["r"] == "fixed_fallback" and _src["beta"] == "bootstrap"
    assert set(cv["tolerances"]) == {"streamflow"}
    # only REQUIRED metrics are watched (r, alpha, beta for a series); NSE is recorded
    assert set(cv["tolerances"]["streamflow"]) == {"r", "alpha", "beta"}
    assert cv["tolerance_records"]["streamflow"]["tol"]["nse"] > 0
    assert set(cv["panel"]) >= {"r", "alpha", "beta", "pbias", "nse", "kge"}
    assert cv["verdict"]["verdict"] in ("converged", "not_converged", "unknown")
    assert rep["budget_used"]["n_evaluations"] == 80 and not rep["budget_used"]["stopped_early"]
    assert cv["ended"]["reason"] == "cap" and cv["ended"]["ended_at_call"] == 79
    # this contract asks for ONE seed: its own verdict, "seeds: not checked" (build step 6)
    sd = cv["seeds"]
    assert sd["n_slots"] == 1 and len(cv["per_seed"]) == 1 and cv["seeds_agree"] is None
    assert "not checked" in sd["how"] and sd["run_verdict"] == cv["verdict"]["verdict"]
    # DDS reports that it never had a convergence test of its own
    assert cv["optimizer_termination"]["status"] == "budget_schedule_complete"
    # starting DDS at the defaults changes its trajectory, so it is opt-in and OFF here
    assert cv["optimizer_termination"]["x_initial"] is None
    if cv["rule"]["stop_point"] is not None:
        assert 0 <= cv["rule"]["stop_point"] < 80
        assert "param_width_at_stop_point" in cv

    # 3) where the evaluations went
    pc = rep["phase_counts"]
    assert pc["pilot"]["n"] == 10
    assert pc["search"]["n"] == 80
    assert {r.get("seed") for r in hist if r["phase"] == "search"} == {0}
    assert pc["commission"]["n"] >= 1 and pc["consume_proof"]["n"] >= 1
    assert pc["holdout"]["n"] >= 2
    assert sum(v["n"] for v in pc.values()) == len(hist)
    # the history is the evidence: ordered, phase-tagged, with decoded parameters and panels
    assert [r["i"] for r in hist] == list(range(len(hist)))
    assert all(set(r["x"]) == {"a", "b"} for r in hist if not r.get("outside_evaluator"))
    assert [r["phase"] for r in hist if r.get("outside_evaluator")] == ["objective_probe"]
    searched = [r for r in hist if r["phase"] == "search" and r["ok"]]
    assert searched and all("nse" in r["panel"]["streamflow"] for r in searched)

    # 4) the calibration itself still works: it found a ~ 1, b ~ 0
    assert rep["best_params"]["a"] == pytest.approx(1.0, abs=0.15)
    assert rep["holdout"] is not None and rep["objectives"] == ["streamflow:temporal_pattern_match"]


def test_legacy_contract_keeps_its_declared_budget():
    """No strategy.budget block: fixed mode, the contract's max_evaluations is the cap — exactly
    what the 29 live System-1 contracts expect — and the new blocks are still reported."""
    rep, hist, log = _run(max_evaluations=40)
    assert rep["status"] == "completed"
    assert rep["budget_plan"]["mode"] == "fixed"
    assert rep["budget_used"]["cap"] == 40 and rep["budget_used"]["n_evaluations"] == 40
    assert rep["budget_plan"]["kappa"] == pytest.approx(40 / 3, abs=0.01)
    assert rep["convergence"]["mode"] == "keep_going"   # the default when nobody answers (§1.2)
    assert rep["phase_counts"]["search"]["n"] == 3 * 40     # 3 seeds by default, each with the cap
    assert [x["calls"] for x in rep["convergence"]["seeds"]["slots"]] == [40, 40, 40]


def test_a_legacy_contract_now_runs_the_pilot_like_every_contract():
    """Design gap 2k (pilot always on): a contract with neither a budget nor a convergence block runs
    the 10-run pilot too, so it anchors protection and tolerances. (This deliberately changes the old
    promise that a legacy contract ran no pilot; System 1 is pinned to the old kit until its switch.)
    Its cap stays the declared max_evaluations and DDS starts the way it always did."""
    legacy, hist_l, _ = _run(max_evaluations=30, seed=5)
    assert legacy["phase_counts"]["pilot"]["n"] == 10
    assert legacy["budget_plan"]["mode"] == "fixed" and legacy["budget_used"]["cap"] == 30
    assert legacy["convergence"]["optimizer_termination"]["x_initial"] is None
    searched = [r["losses"] for r in hist_l if r["phase"] == "search"]
    assert len(searched) == 3 * 30                         # 3 seeds by default (build step 6)
    assert legacy["convergence"]["verdict"]["verdict"] in ("converged", "not_converged", "unknown")


def rep_phases(rep):
    return set(rep["phase_counts"])


def test_caller_budget_still_wins():
    rep, _, _ = _run(budget_block={"mode": "measured", "allowance": "2m"}, budget=25,
                     max_evaluations=999)
    assert rep["budget_used"]["cap"] == 25 and rep["budget_plan"]["overridden_by_caller"] == 25
    assert rep["phase_counts"]["search"]["n"] == 3 * 25     # the caller's cap is per seed


def test_pilot_runs_zero_still_runs_the_default_run():
    """Design §1.3 / gap 2k: the pilot is always on — its default run anchors protection and
    tolerances. pilot_runs: 0 runs the default run only, and says so."""
    rep, hist, _ = _run(budget_block={"pilot_runs": 0}, max_evaluations=20)
    assert rep["status"] == "completed"
    assert rep["phase_counts"]["pilot"]["n"] == 1
    assert any("always on" in w for w in rep["budget_plan"]["warnings"])


def test_unstable_pilot_stops_a_measured_run_before_the_search():
    """A range that makes most runs fail must not be searched on a measured budget: the cap is
    derived from the pilot's timings, so a failing pilot makes the cap meaningless."""
    tmp = tempfile.mkdtemp(prefix="kdt_conv_e2e_bad_")
    try:
        ki, wd = _fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10}, None, 40)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["parameters"][0]["range"] = [0.5, 1.5]
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))

        r = Runner(wd)
        orig = r.__call__

        def flaky():
            m = orig()
            a = float(json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())["a"])
            return {} if a > 0.7 else m        # most of the box fails
        buf = io.StringIO()
        with _clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=flaky, budget=None, seed=0)
        assert rep["status"] == "pilot_unstable"
        assert rep["pilot"]["fail_rate"] > 0.30 and "ranges" in rep["reason"]
        assert rep["phase_counts"]["pilot"]["n"] == 10
        assert "search" not in rep["phase_counts"], "nothing was searched"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_resumed_run_reuses_its_plan_and_does_not_re_pay_the_pilot():
    """The cap must not move on a restart: DDS's perturbation schedule is written against the
    budget it was given, so a re-measured cap mid-campaign would be a different algorithm."""
    tmp = tempfile.mkdtemp(prefix="kdt_conv_e2e_resume_")
    try:
        ki, wd = _fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10}, None, 30)
        buf = io.StringIO()
        with _clean_env(), contextlib.redirect_stdout(buf):
            first = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                run_model=Runner(wd), budget=None, seed=0)
        assert first["phase_counts"]["pilot"]["n"] == 10
        plan = json.loads(Path(wd, "kdt_budget_plan.json").read_text())
        assert plan["cap_per_seed"] == first["budget_plan"]["cap_per_seed"]

        with _clean_env(), contextlib.redirect_stdout(buf):
            second = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                 run_model=Runner(wd), budget=None, seed=0)
        assert second["budget_plan"]["reused_from_workdir"].endswith("kdt_budget_plan.json")
        assert second["budget_plan"]["cap_per_seed"] == plan["cap_per_seed"]
        assert second["budget_used"]["cap"] == first["budget_used"]["cap"]
        # the pilot ran ONCE, in the first run: the resume's history has no new pilot records
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        assert sum(1 for r in hist if r["phase"] == "pilot") == 10
        assert "REUSED" in buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _flaky_runner(wd, fail_above=1.05):
    """Fails in the upper part of the range — enough of the Latin-hypercube pilot to trip the 30%
    rate, while the optimum (a ~ 1) is still reachable."""
    r = Runner(wd)

    def flaky():
        a = float(json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())["a"])
        return {} if a > fail_above else r()
    return flaky


def test_unstable_pilot_only_warns_on_a_legacy_fixed_budget():
    """A contract that has searched fine for months must not start refusing because a
    Latin-hypercube corner is infeasible: in fixed mode the pilot warns and the run continues."""
    tmp = tempfile.mkdtemp(prefix="kdt_conv_e2e_legacy_bad_")
    try:
        ki, wd = _fixture(tmp, {"pilot_runs": 10}, None, 20)
        buf = io.StringIO()
        with _clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=_flaky_runner(wd), budget=None, seed=0)
        assert rep["status"] == "completed"
        assert "WARNING pilot" in buf.getvalue()
        assert rep["budget_plan"]["pilot_summary"]["status"] == "pilot_unstable"
        assert rep["phase_counts"]["search"]["n"] == 3 * 20  # it did search (3 seeds by default)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _slow_runner(wd, sleep_s=0.2):
    r = Runner(wd)

    def slow():
        import time as _t
        _t.sleep(sleep_s)
        return r()
    return slow


def test_an_allowance_too_short_to_search_is_refused_not_silently_replaced():
    """A measured run whose allowance cannot cover the search must not fall back to the contract's
    hand-written max_evaluations — that would spend time the user never granted."""
    tmp = tempfile.mkdtemp(prefix="kdt_conv_e2e_short_")
    try:
        ki, wd = _fixture(tmp, {"mode": "measured", "allowance": "1s", "pilot_runs": 4}, None, 500)
        buf = io.StringIO()
        with _clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=_slow_runner(wd), budget=None, seed=0)
        assert rep["status"] == "budget_exhausted"
        assert "under 20" in rep["reason"] and "shared" in rep["reason"]      # design §1.3
        assert "search" not in rep["phase_counts"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_corrupt_or_foreign_plan_file_re_plans_instead_of_crashing():
    """The plan file is ordinary JSON in a workdir people poke at. A hand-edited, truncated or
    foreign plan must make the run re-plan loudly, never crash it and never reuse a stale cap."""
    tmp = tempfile.mkdtemp(prefix="kdt_conv_e2e_corrupt_")
    try:
        ki, wd = _fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 4}, None, 20)
        Path(wd, "kdt_budget_plan.json").write_text(json.dumps(
            {"setup_id": "not-this-setup", "cap_per_seed": "80", "pilot_summary": "nonsense"}))
        buf = io.StringIO()
        with _clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=Runner(wd), budget=None, seed=0)
        assert rep["status"] == "completed"
        assert "DIFFERENT setup" in buf.getvalue()
        assert "reused_from_workdir" not in rep["budget_plan"]
        assert rep["budget_used"]["cap"] == 20            # the contract's ceiling, not the stale 80
        assert rep["phase_counts"]["pilot"]["n"] == 4     # it measured for itself

        # same setup, but the numbers are the wrong type
        plan = json.loads(Path(wd, "kdt_budget_plan.json").read_text())
        plan["cap_per_seed"] = "15"
        plan["pilot_summary"] = "nonsense"
        plan["warnings"] = "nope"
        Path(wd, "kdt_budget_plan.json").write_text(json.dumps(plan))
        with _clean_env(), contextlib.redirect_stdout(buf):
            again = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                run_model=Runner(wd), budget=None, seed=0)
        assert again["status"] == "completed"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


