"""Build step 8a (gaps 2x, 2w, 2s): staged calibration removed, triage before the search, StopRule removed,
backends that do not report each call report "convergence not tracked".

Design §1 "Triage before the search" (Leo 2026-09-28): the kit always calibrates unless the run cannot be
calibrated. Routes, first match wins: `no_baseline` (the pilot's default run gives no finite objective
metric), `not_calibratable` (the proof gave a verdict for EVERY searched parameter and every verdict is
OBJECTIVE_MASKED / DORMANT / WINDOW_MASKED), `calibrate` (everything else; "sensitivity not proven" when no
parameter is ALIVE). Fit quality at the defaults is reported, never a reason to skip."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402
from calibration_kit import test_step1_panel as S1                                        # noqa: E402
from calibration_kit.calib import triage_route                                            # noqa: E402


def _run(budget_block=None, runner=None, strategy=None, max_evaluations=30, budget=None, backend=None,
         monkeypatch=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        bb = dict(budget_block or {})
        bb.setdefault("seeds", 1)
        ki, wd = E._fixture(tmp, bb, None, max_evaluations)
        if strategy:
            c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strategy)
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        if backend is not None:
            monkeypatch.setattr(C, "_make_backend", lambda algo: backend(algo))
        buf = io.StringIO()
        with E._clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=(runner or S1._Runner)(wd), budget=budget, seed=0)
        hp = Path(wd, "eval_history.jsonl")
        hist = [json.loads(l) for l in hp.read_text().splitlines()] if hp.exists() else []
        return rep, hist, buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── the triage function (design §1) ─────────────────────────────────────────────────────────────
def _proof(**verdicts):
    return {"params": [{"param": k, "verdict": v} for k, v in verdicts.items()], "summary": "x"}


def test_no_baseline_comes_first():
    t = triage_route({"default_n_finite": 0, "reason": "the default parameters break the constraints"},
                     _proof(a="OBJECTIVE_MASKED", b="OBJECTIVE_MASKED"), ["a", "b"])
    assert t["route"] == "no_baseline" and "no finite objective metric" in t["reason"]
    assert "break the constraints" in t["reason"]


def test_not_calibratable_needs_a_masking_verdict_for_every_searched_parameter():
    for v in ("OBJECTIVE_MASKED", "DORMANT", "WINDOW_MASKED"):
        t = triage_route({"default_n_finite": 1}, _proof(a=v, b="OBJECTIVE_MASKED"), ["a", "b"])
        assert t["route"] == "not_calibratable", v
        assert "each parameter moved alone from the defaults" in t["reason"] and f"a: {v}" in t["reason"]
    # one parameter without a verdict, or with another verdict -> calibrate
    assert triage_route({"default_n_finite": 1}, _proof(a="OBJECTIVE_MASKED"), ["a", "b"])["route"] == "calibrate"
    for other in ("ALIVE", "UNPROVEN", "INSENSITIVE"):
        t = triage_route({"default_n_finite": 1}, _proof(a="OBJECTIVE_MASKED", b=other), ["a", "b"])
        assert t["route"] == "calibrate", other
    # an empty proof, or none at all -> calibrate
    assert triage_route({"default_n_finite": 1}, {"params": []}, ["a", "b"])["route"] == "calibrate"
    assert triage_route({"default_n_finite": 1}, None, ["a", "b"])["route"] == "calibrate"


def test_sensitivity_not_proven_whenever_no_parameter_is_alive():
    t = triage_route({"default_n_finite": 2}, _proof(a="UNPROVEN", b="INSENSITIVE"), ["a", "b"])
    assert t["route"] == "calibrate" and t["sensitivity_not_proven"] is True
    assert "sensitivity not proven" in t["reason"]
    t2 = triage_route({"default_n_finite": 2}, _proof(a="ALIVE", b="INSENSITIVE"), ["a", "b"])
    assert t2["sensitivity_not_proven"] is False and "sensitivity not proven" not in t2["reason"]
    assert t2["proof"]["alive"] == ["a"]


def test_a_partly_finite_default_is_a_baseline_and_an_old_summary_is_not_checked():
    assert triage_route({"default_n_finite": 1}, None, ["a"])["route"] == "calibrate"
    old = triage_route({"default_ok": False}, None, ["a"])             # an older pilot summary
    assert old["route"] == "calibrate" and "baseline not checked" in old["notes"][0]
    assert triage_route({"default_n_finite": True}, None, ["a"])["route"] == "calibrate"   # not a count


# ── through calibrate() ────────────────────────────────────────────────────────────────────────
class _NoMetricsAtDefault(S1._Runner):
    def __call__(self):
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        if abs(float(p["a"]) - 0.8) < 1e-12 and abs(float(p["b"]) - 2.0) < 1e-12:
            return {}
        return super().__call__()


def test_no_baseline_stops_before_the_search_and_restores_the_inputs(monkeypatch):
    from calibration_kit.evaluator import Evaluator as _Ev
    calls = []
    orig = _Ev.restore_originals
    monkeypatch.setattr(_Ev, "restore_originals", lambda self: (calls.append(1), orig(self))[1])
    rep, hist, log = _run(runner=_NoMetricsAtDefault)
    assert rep["status"] == "no_baseline" and rep["route"] == "no_baseline" and calls
    assert not [h for h in hist if h["phase"] == "search"]
    assert "triage: no_baseline" in log and rep["triage"]["per_target"]["streamflow"]["fit_verdict"]


def test_a_poor_fit_at_the_defaults_is_reported_never_a_reason_to_skip():
    class _Poor(S1._Runner):
        def __init__(self, wd):
            super().__init__(wd, shift=40.0)              # the defaults fit very badly
    rep, _, _ = _run(runner=_Poor)
    tr = rep["triage"]
    assert rep["status"] == "completed" and tr["route"] == "calibrate"
    pt = tr["per_target"]["streamflow"]
    assert pt["fit_verdict"] in ("structural", "calibrate", "undetermined", "adequate")
    assert "default_standards" in pt


def test_the_completed_report_carries_the_triage_and_the_proof_summary():
    # the fixture runner emits no target proofs: the proof returns an empty list -> calibrate, and the
    # report says "sensitivity not proven" (design §1)
    rep, _, _ = _run()
    tr = rep["triage"]
    assert tr["route"] == "calibrate" and set(tr["proof"]) >= {"ran", "verdicts", "alive", "summary"}
    assert tr["proof"]["ran"] is False and tr["sensitivity_not_proven"] is True
    assert "sensitivity not proven" in tr["reason"]


def test_a_staged_contract_runs_one_search_with_a_warning():
    rep, hist, log = _run(strategy={"staged": True, "staged_start": 1, "staged_step": 1})
    assert rep["status"] == "completed" and "strategy.staged is removed" in rep["staged_warning"]
    assert "screen" not in {h["phase"] for h in hist}                # no Morris screen
    assert set(rep["best_params"]) == {"a", "b"}
    assert not hasattr(C, "calibrate_staged")


# ── 2w: StopRule removed ────────────────────────────────────────────────────────────────────────
def test_stoprule_is_gone_and_the_convention_floor_stays():
    from calibration_kit import stop
    assert not hasattr(stop, "StopRule") and callable(stop.convention_floor)
    rep, _, _ = _run()
    assert "stop" not in rep


# ── 2s: backends that do not report each call ──────────────────────────────────────────────────
class _Untracked:
    """A backend that scores a few fixed points and never calls the kit's per-call hook."""
    def __init__(self, algo):
        self.algo = algo

    def available(self):
        return True

    def optimize(self, problem, budget, seed, **kw):
        from calibration_kit.backends.base import CalibResult
        pts = [[0.8 + 0.05 * k, 2.0 - 0.5 * k] for k in range(4)]
        hist = [{"x": x, "loss": float(problem.evaluate(x)[0])} for x in pts]
        best = min(hist, key=lambda h: h["loss"])
        return CalibResult(best_x=best["x"], best_loss=[best["loss"]], n_evaluations=len(hist), history=hist,
                           backend=self.algo)


@pytest.mark.parametrize("algo", ["surrogate", "pestpp_ies", "madr"])
def test_backends_that_do_not_report_each_call_report_convergence_not_tracked(algo, monkeypatch):
    rep, _, _ = _run(budget_block={"seeds": 3}, strategy={"default_algorithm": algo}, backend=_Untracked,
                     monkeypatch=monkeypatch)
    assert rep["status"] == "completed", rep.get("reason")
    cv = rep["convergence"]
    assert cv["verdict"]["how"].startswith("convergence not tracked") and cv["verdict"]["verdict"] == "unknown"
    assert cv["seeds"]["n_slots"] == 1 and "not compared" in cv["seeds"]["note"]
    assert cv["ended"]["text"].startswith("convergence not tracked")


# ── round-1 review (Opus): the triage placement and the remaining rules ────────────────────────
def test_a_no_baseline_run_is_not_saved_so_a_fixed_runner_re_plans():
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        for block in ({"mode": "measured", "allowance": "1h", "seeds": 1}, {"seeds": 1}):
            ki, wd = E._fixture(tempfile.mkdtemp(dir=tmp), block, None, 20)
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                bad = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_NoMetricsAtDefault(wd),
                                  budget=None, seed=0)
            assert bad["status"] == "no_baseline" and not Path(wd, "kdt_budget_plan.json").exists()
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()) as buf:
                good = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                   budget=None, seed=0)
            assert good["status"] == "completed", (block, good.get("reason"))
            assert "REUSED" not in buf.getvalue() and good["phase_counts"]["pilot"]["n"] >= 10
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_saved_plan_whose_default_failed_is_never_reused():
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "1h", "seeds": 1}, None, 20)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
        pf = Path(wd, "kdt_budget_plan.json")
        pl = json.loads(pf.read_text())
        pl["pilot_summary"]["default_n_finite"] = 0                  # as if the default had failed
        pf.write_text(json.dumps(pl))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()) as buf:
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        assert "resumed_cap" not in rep["budget_plan"] and "not usable" in buf.getvalue()
        assert "gave no finite objective metric" in buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _Slow(S1._Runner):
    def __call__(self):
        import time as _t
        _t.sleep(0.05)
        return super().__call__()


class _SlowNoMetricsAtDefault(_NoMetricsAtDefault):
    def __call__(self):
        import time as _t
        _t.sleep(0.05)
        return super().__call__()


def test_the_triage_comes_before_the_budget_refusal_and_the_machine_probe(monkeypatch):
    """(Opus r2) a slow runner so a 1 s allowance IS refused for a working runner (control), and a recorder on
    compute.probe_lanes (always called at the machine-probe point)."""
    from calibration_kit import compute as _cmp
    called = []
    real = _cmp.probe_lanes
    monkeypatch.setattr(_cmp, "probe_lanes", lambda *a, **k: (called.append(1), real(*a, **k))[1])
    blk = {"mode": "measured", "allowance": "1s", "pilot_runs": 3}
    ok, _, _ = _run(budget_block=blk, runner=_Slow)
    assert ok["status"] == "budget_exhausted" and called            # control: this allowance IS refused
    called.clear()
    rep, _, _ = _run(budget_block=blk, runner=_SlowNoMetricsAtDefault)
    assert rep["status"] == "no_baseline", rep["status"]            # not budget_exhausted
    assert not called                                               # returned before the machine probe


def test_a_three_run_pilot_leaves_a_failed_default_to_the_triage():
    rep, _, _ = _run(budget_block={"mode": "measured", "allowance": "1h", "run_time": "1s"},
                     runner=_NoMetricsAtDefault)
    assert rep["status"] == "no_baseline" and rep["triage"]["route"] == "no_baseline"

    class _HalfAtDefault(S1._Runner):
        """nse finite, pbias not, at the defaults: a baseline (one finite objective)."""
        def __call__(self):
            m = super().__call__()
            p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
            if abs(float(p["a"]) - 0.8) < 1e-12 and abs(float(p["b"]) - 2.0) < 1e-12:
                m["pbias"] = float("nan")
            return m
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "1h", "run_time": "1s", "seeds": 1}, None, 20)
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep2 = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_HalfAtDefault(wd),
                               budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep2["status"] == "completed" and rep2["triage"]["route"] == "calibrate", rep2.get("reason")
    assert rep2["budget_plan"]["pilot_summary"]["default_n_finite"] == 1


CONV = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
    {"metric": "nse", "direction": "maximize", "bands": {"satisfactory": 0.5, "good": 0.99}}]}]}


def _run_conv(runner, strategy=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 20)
        Path(ki, "docs").mkdir(exist_ok=True)
        yaml.safe_dump(CONV, open(Path(ki, "docs", "validation_convention.yaml"), "w"))
        if strategy:
            c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strategy)
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            return C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=runner(wd), budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_per_target_block_reports_the_exact_fit_and_standards_as_information():
    class _Poor(S1._Runner):
        def __init__(self, wd):
            super().__init__(wd, shift=40.0)
    ok = _run_conv(S1._Runner)["triage"]["per_target"]["streamflow"]
    poor = _run_conv(_Poor)
    pt = poor["triage"]["per_target"]["streamflow"]
    assert ok["fit_verdict"] == "adequate" and pt["fit_verdict"] == "structural"
    assert poor["status"] == "completed"                               # structural never skips
    assert pt["fit_why"].startswith("information only")
    assert ok["default_standards"]["outcome"] == "meets" and pt["default_standards"]["outcome"] == "fails"


def test_the_default_standards_use_an_old_stop_floor_band_like_the_search_does():
    rep = _run_conv(S1._Runner, strategy={"stop": {"main_stat": "nse", "floor": {"band": "good"}}})
    st = rep["triage"]["per_target"]["streamflow"]["default_standards"]
    assert st["outcome"] == "fails"                                    # good = 0.99 (satisfactory 0.5 would meet)
    assert st["metrics"]["nse"]["band"] == "good" and st["metrics"]["nse"]["threshold"] == 0.99
    # the same band the search's own standards check uses
    assert rep["convergence"]["verdict"]["standards"]["streamflow"]["calibration"]["metrics"]["nse"]["band"] == "good"


def test_the_early_report_carries_the_proof_and_the_phase_counts():
    rep, _, _ = _run(runner=_NoMetricsAtDefault)
    for k in ("phase_counts", "consumption_proof", "consumption_coverage", "objective_smoke_test",
              "split_provenance", "pilot"):
        assert isinstance(rep.get(k), dict), k
    assert rep["phase_counts"]["pilot"]["n"] >= 1 and "pilot" in rep["split_provenance"]["by_phase"]
    assert rep["pilot"]["default_n_finite"] == 0
    assert "budget_plan" not in rep and "records" not in rep["pilot"]           # no plan yet; a summary only


def test_an_applicator_no_baseline_puts_the_inputs_back():
    class _App:
        def __init__(self, wd):
            self.wd, self.inner = Path(wd), S1._Runner(wd)

        def __call__(self):
            p = json.loads((self.wd / "params.json").read_text())
            if abs(float(p["a"]) - 0.8) < 1e-12 and abs(float(p["b"]) - 2.0) < 1e-12:
                return {}
            pf = self.wd / "_p.json"
            pf.write_text(json.dumps(p))
            os.environ["KDT_CALIB_PARAMS"] = str(pf)
            try:
                return self.inner()
            finally:
                os.environ.pop("KDT_CALIB_PARAMS", None)
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 20)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["injection"] = {"mode": "applicator"}
        for p in c["parameters"]:
            p["address"] = {"kind": "json_path", "file": "params.json", "path": p["name"]}
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        Path(wd, "params.json").write_text(json.dumps({"a": 0.8, "b": 2.0}))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_App(wd), budget=None, seed=0)
        back = json.loads(Path(wd, "params.json").read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["status"] == "no_baseline", rep.get("reason")
    assert back == {"a": 0.8, "b": 2.0}                                  # the pilot's values are gone


def test_a_plan_saved_by_an_older_kit_without_the_default_count_is_still_resumed():
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "1h", "seeds": 1}, None, 20)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
        pf = Path(wd, "kdt_budget_plan.json")
        pl = json.loads(pf.read_text())
        pl["pilot_summary"].pop("default_n_finite")                   # as an older kit wrote it
        pf.write_text(json.dumps(pl))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["budget_plan"].get("resumed_cap") and rep["triage"]["route"] == "calibrate"
    assert "baseline not checked" in rep["triage"]["notes"][0]



def test_an_old_stop_floor_band_on_kge_reaches_both_standards_checks():
    conv = {"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series", "headline_metrics": [
        {"metric": "kge", "direction": "maximize", "bands": {"satisfactory": 0.5, "good": 0.999}}]}]}
    tmp = tempfile.mkdtemp(prefix="kdt_s8_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 20)
        Path(ki, "docs").mkdir(exist_ok=True)
        yaml.safe_dump(conv, open(Path(ki, "docs", "validation_convention.yaml"), "w"))
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["strategy"]["stop"] = {"main_stat": "kge", "floor": {"band": "good"}}
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    st = rep["triage"]["per_target"]["streamflow"]["default_standards"]
    assert st["metrics"]["kge"]["band"] == "good"
    assert rep["convergence"]["verdict"]["standards"]["streamflow"]["calibration"]["metrics"]["kge"]["band"] == "good"
