"""Build step 9 (gap 2n): the runner/kit cross-check at the pilot's default run.

When the runner writes its default-run series, the kit recomputes the panel from it and compares the runner's
own values: the kit block (every metric) or, without a block, only the headline metrics whose name and definition
are the kit's (NSE, KGE, r). A difference above half a unit of the runner's last recorded decimal + 1e-6 makes that
metric missing for the whole search, with the reason; every other plain KI metric is listed "not checked"."""
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

from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import panel as P                                                    # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402
from calibration_kit import test_step1_panel as S1                                        # noqa: E402

RNG = np.random.default_rng(7)
OBS = 20 + 10 * np.sin(np.linspace(0, 20, 300)) + RNG.normal(0, 1, 300)
SIM = 0.9 * OBS + 1.5 + RNG.normal(0, 2, 300)


def _block(**over):
    b = P.panel_block({"Q": (SIM, OBS)}, {})
    b["__kdt__"]["panel"]["Q"].update(over)
    return b


def test_recorded_precision_is_half_the_last_decimal():
    assert P.recorded_precision(0.8123) == pytest.approx(5e-5)
    assert P.recorded_precision(12.5) == pytest.approx(0.05)
    assert P.recorded_precision(3.0) == pytest.approx(0.05)
    assert P.recorded_precision(1e-7) == pytest.approx(5e-8)
    assert P.recorded_precision(0.1 + 0.2) < 1e-16


def test_a_block_that_agrees_sets_nothing_missing():
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, _block(), {"Q": "flow"}, ["Q"])
    assert out["kit_missing"] == {} and out["notes"] == []
    rows = out["checks"]["Q"]["metrics"]
    assert out["checks"]["Q"]["source"] == "block"
    assert set(rows) == set(P.BLOCK_KEYS) and all(r["status"] == "agrees" for r in rows.values())


def test_a_block_rounded_to_its_written_decimals_still_agrees():
    true = P.panel_block({"Q": (SIM, OBS)}, {})["__kdt__"]["panel"]["Q"]
    for nd in (2, 4, 6):
        blk = {k: (round(v, nd) if isinstance(v, float) else v) for k, v in true.items()}
        out = P.cross_check_2n({"Q": (SIM, OBS, None)}, {"__kdt__": {"panel": {"Q": blk}}}, {"Q": "flow"}, ["Q"])
        assert out["kit_missing"] == {}, nd


def test_the_bound_is_each_metrics_finest_decimal_over_the_pilot_runs():
    """Leo 2026-09-30 (Opus 8b/9 r1 #5, r2 #2a): a lone beta = 1.0 at the default run, where the pilot's other
    runs show beta written to 4 decimals, is judged at 4 decimals and caught."""
    true = P.panel_block({"Q": (SIM, OBS)}, {})["__kdt__"]["panel"]["Q"]
    blk = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in true.items()}
    assert abs(true["beta"] - 1.0) < 0.05                   # inside a lone 1.0's own +-0.05
    blk["beta"] = 1.0
    pilot = {"Q": {"beta": [0.9751, 0.9612, 1.0213]}}
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, {"__kdt__": {"panel": {"Q": blk}}}, {"Q": "flow"}, ["Q"],
                           precision_from=pilot)
    assert out["kit_missing"]["Q"]["beta"].startswith("2n mismatch")
    assert out["checks"]["Q"]["metrics"]["beta"]["bound"] == pytest.approx(5e-5 + 1e-6)


def test_an_honest_runner_with_mixed_precision_agrees():
    """Opus 8b/9 r2 #2a: PBIAS written to 1 decimal beside ratios written to 4 is honest rounding."""
    kit = P.panel_from_series(SIM, OBS, "flow", mean_rule=False)
    dm = {"Q": {"r": round(kit["r"], 4), "alpha": round(kit["alpha"], 4), "nse": round(kit["nse"], 4),
                "kge": round(kit["kge"], 4), "lnnse": round(kit["lnnse"], 4), "pbias": round(kit["pbias"], 1)}}
    pilot = {"Q": {"pbias": [-11.1, -8.4, 3.0], "r": [0.8921, 0.8804]}}
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, dm, {"Q": "flow"}, ["Q"], precision_from=pilot)
    assert out["kit_missing"] == {}
    assert out["checks"]["Q"]["metrics"]["pbias"]["bound"] == pytest.approx(0.05 + 1e-6)


def test_a_block_value_off_by_more_than_the_bound_is_missing_with_the_reason():
    true = P.panel_from_series(SIM, OBS, "flow")["beta"]
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, _block(beta=round(true + 0.003, 4)), {"Q": "flow"}, ["Q"])
    why = out["kit_missing"]["Q"]["beta"]
    assert why.startswith("2n mismatch: runner ") and ", kit " in why
    assert out["checks"]["Q"]["metrics"]["beta"]["status"] == "mismatch"
    assert any(n.startswith("Q:beta 2n mismatch") for n in out["notes"])


def test_without_a_block_every_plain_value_the_panel_reads_is_checked_and_the_rest_listed():
    """Opus 8b/9 r1 #4: the panel reads plain r, alpha, beta, nse, kge, lnnse (and pbias in percent), so every
    one of them is checked; plain nrmse (never read) and other KI metrics are listed "not checked"."""
    kit = P.panel_from_series(SIM, OBS, "flow", mean_rule=False)
    dm = {"Q": {"nse": kit["nse"], "kge": kit["kge"] + 0.1, "r": kit["r"], "pbias": -kit["pbias"],   # Moriasi sign
                "nrmse": 1.0, "alpha": 5.0, "rmse": 0.3}}
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, dm, {"Q": "flow"}, ["Q"])
    # beta: the runner gives none and its PBIAS failed, so the beta the panel would work out goes too (Leo)
    assert set(out["kit_missing"]["Q"]) == {"kge", "pbias", "alpha", "beta"}
    assert out["kit_missing"]["Q"]["beta"].startswith("worked out from PBIAS, which failed the 2n check")
    assert any("beta missing for the search" in n for n in out["notes"])
    rows = out["checks"]["Q"]["metrics"]
    assert out["checks"]["Q"]["source"] == "plain"
    assert rows["nse"]["status"] == "agrees" and rows["r"]["status"] == "agrees"
    for m in ("nrmse", "rmse"):
        assert out["checks"]["Q"]["not_checked"][m]["status"] == "not checked (the KI may define it differently)"


def test_a_pbias_in_another_unit_is_not_read_so_not_checked():
    kit = P.panel_from_series(SIM, OBS, "flow", mean_rule=False)
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, {"Q": {"r": kit["r"], "pbias": 12.0}}, {"Q": "flow"}, ["Q"],
                           pbias_percent={"Q": False})
    assert out["kit_missing"] == {} and "pbias" in out["checks"]["Q"]["not_checked"]


def test_a_flat_single_variable_payload_is_checked():
    kit = P.panel_from_series(SIM, OBS, "flow", mean_rule=False)
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, {"nse": kit["nse"] - 0.2, "r": kit["r"]}, {"Q": "flow"}, ["Q"])
    assert set(out["kit_missing"]["Q"]) == {"nse"}


def test_with_a_block_the_kis_own_headline_values_are_checked_and_reported_only():
    kit = P.panel_from_series(SIM, OBS, "flow", mean_rule=False)
    dm = _block()
    dm.update({"nse": kit["nse"] + 0.3, "kge": kit["kge"], "pbias": 7.0})
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, dm, {"Q": "flow"}, ["Q"])
    assert out["kit_missing"] == {}                          # the panel reads the block, which agrees
    head = out["checks"]["Q"]["ki_headline"]
    assert head["nse"]["status"].startswith("mismatch (the KI's headline value") and head["kge"]["status"] == "agrees"
    assert out["checks"]["Q"]["not_checked"]["pbias"]["status"].startswith("not checked")
    assert any("reported only" in n for n in out["notes"])


def test_a_variable_without_a_series_is_not_checked():
    out = P.cross_check_2n({}, _block(beta=9.0), {"Q": "flow"}, ["Q"])
    assert out["kit_missing"] == {} and out["checks"]["Q"]["status"] == "not checked (no series)"


def test_a_mismatch_leaves_the_required_set_for_the_whole_search():
    pan = P.Panel(["Q"], kinds={"Q": "flow"})
    true = P.panel_from_series(SIM, OBS, "flow")["alpha"]
    pan.apply_pilot(P.cross_check_2n({"Q": (SIM, OBS, None)}, _block(alpha=true * 1.05), {"Q": "flow"}, ["Q"]))
    assert "alpha" not in pan.required("Q") and pan.capped("Q")
    assert pan.kit_missing["Q"]["alpha"].startswith("2n mismatch")


# ── wired into calibrate() ──────────────────────────────────────────────────────────────────────
class _BadBlock(S1._Runner):
    """The fixture runner, but its kit block's beta is off by 2 % (a runner using its own formula)."""
    def __call__(self):
        m = super().__call__()
        m["__kdt__"]["panel"]["streamflow"]["beta"] *= 1.02
        return m


def _cal(runner_cls):
    tmp = tempfile.mkdtemp(prefix="kdt_s9x_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 30)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=runner_cls(wd),
                              budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep, hist


def test_calibrate_reports_the_cross_check_and_drops_a_mismatched_metric():
    rep, hist = _cal(_BadBlock)
    ps = rep["convergence"]
    assert ps["kit_missing"]["streamflow"]["beta"].startswith("2n mismatch")
    assert ps["cross_check_2n"]["streamflow"]["metrics"]["beta"]["status"] == "mismatch"
    assert ps["cross_check_2n"]["streamflow"]["metrics"]["r"]["status"] == "agrees"
    srch = [h for h in hist if h["phase"] == "search" and h.get("panel", {}).get("streamflow")]
    assert srch and all("beta" not in h["panel"]["streamflow"] for h in srch)


def test_calibrate_with_an_honest_runner_checks_and_sets_nothing_missing():
    rep, _ = _cal(S1._Runner)
    ps = rep["convergence"]
    assert not any(str(w).startswith("2n") for w in (ps.get("kit_missing") or {}).get("streamflow", {}).values())
    rows = ps["cross_check_2n"]["streamflow"]["metrics"]
    assert {"r", "alpha", "beta", "nse", "kge"} <= set(rows)
    assert all(r["status"] == "agrees" for r in rows.values())


def test_a_resume_reuses_the_cross_check_it_does_not_redo_it():
    tmp = tempfile.mkdtemp(prefix="kdt_s9x_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 30)
        reps = []
        for _ in range(2):
            for f in Path(wd).glob("kdt_series_*.npz"):
                f.unlink()                           # the second call has no series: it must reuse, not redo
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                reps.append(C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_BadBlock(wd),
                                        budget=None, seed=0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    a, b = (r["convergence"] for r in reps)
    assert a["cross_check_2n"]["streamflow"]["metrics"]["beta"]["status"] == "mismatch"
    assert b["cross_check_2n"] == a["cross_check_2n"] and b["kit_missing"] == a["kit_missing"]


# ── review round 1 (Opus 8b/9) fixes that live in calib.py / pilot.py ─────────────────────────────
def test_the_2n_notes_reach_the_pilot_notes():
    rep, _ = _cal(_BadBlock)
    assert any(n.startswith("streamflow:beta 2n mismatch") for n in rep["convergence"]["pilot_notes"])


class _AlwaysSeries(S1._Runner):
    """An honest runner that writes its series file on EVERY call (it ignores KDT_CALIB_EMIT_SERIES)."""
    def __call__(self):
        import os as _os
        _prev = _os.environ.get("KDT_CALIB_EMIT_SERIES")
        _os.environ["KDT_CALIB_EMIT_SERIES"] = "1"
        try:
            return super().__call__()
        finally:
            if _prev is None:
                _os.environ.pop("KDT_CALIB_EMIT_SERIES", None)
            else:
                _os.environ["KDT_CALIB_EMIT_SERIES"] = _prev


def test_a_series_file_rewritten_by_later_pilot_runs_is_not_what_2n_reads():
    """Opus 8b/9 r1 #3: the default run's series is kept at once; later pilot runs overwrite the runner's file."""
    rep, _ = _cal(_AlwaysSeries)
    cv = rep["convergence"]
    assert not any(str(w).startswith("2n") for w in (cv.get("kit_missing") or {}).get("streamflow", {}).values())
    assert all(r["status"] == "agrees" for r in cv["cross_check_2n"]["streamflow"]["metrics"].values())


def test_a_resume_from_a_plan_saved_before_the_2n_check_says_it_was_not_checked():
    tmp = tempfile.mkdtemp(prefix="kdt_s9x_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 30)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_BadBlock(wd), budget=None, seed=0)
        pf = Path(wd, "kdt_budget_plan.json")
        plan = json.loads(pf.read_text())
        plan["panel_setup"].pop("cross_check_2n")
        plan["panel_setup"]["kit_missing"] = {}
        pf.write_text(json.dumps(plan))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_BadBlock(wd), budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cv = rep["convergence"]
    assert cv["cross_check_2n"]["streamflow"]["status"].startswith("not checked (the saved plan predates")
    assert any("2n cross-check not run" in n for n in cv["pilot_notes"])


def test_every_report_says_whether_the_runner_was_certified():
    rep, _ = _cal(S1._Runner)
    assert rep["certified"] is False                     # the fixture has no reference run


def test_a_runner_that_gives_its_own_beta_keeps_it_when_only_pbias_fails():
    kit = P.panel_from_series(SIM, OBS, "flow", mean_rule=False)
    dm = {"Q": {"r": kit["r"], "beta": kit["beta"], "pbias": -kit["pbias"]}}
    out = P.cross_check_2n({"Q": (SIM, OBS, None)}, dm, {"Q": "flow"}, ["Q"])
    assert set(out["kit_missing"]["Q"]) == {"pbias"}


def test_calibrate_with_a_moriasi_sign_runner_says_beta_is_missing_and_why():
    """End to end (Opus 8b/9 r2 #2b): the verdict must not blame the search when beta was never available."""
    class _Moriasi(S1._Runner):
        def __call__(self):
            m = super().__call__()
            blk = m["__kdt__"].pop("panel", {}).get("streamflow", {})
            m.update({k: blk[k] for k in ("alpha", "lnnse") if k in blk})   # every other required metric
            m["pbias"] = -m["pbias"]                   # PBIAS with the opposite sign, no beta of its own
            return m
    rep, _ = _cal(_Moriasi)
    cv = rep["convergence"]
    assert cv["kit_missing"]["streamflow"]["beta"].startswith("worked out from PBIAS")
    assert "beta" not in cv["required_panel"]["streamflow"]                     # beta no longer blocks the rule
    v = cv["step_rule_verdict"]["variables"]["streamflow"]      # the per-call step rule (recorded)
    assert "beta" not in v["reason"].split("capped")[0]                       # never "not recorded" for beta
    assert v["verdict"] != "converged"                                         # beta missing: never converged
    if v.get("capped"):
        assert "worked out from PBIAS" in v["cap_reason"]


def test_calibrate_with_a_kis_pbias_in_days_does_not_read_or_check_it():
    """Opus 8b/9 r2: a KI convention giving pbias in DAYS — the panel does not read plain pbias, so 2n lists it
    'not checked' and flags nothing."""
    class Days(S1._Runner):
        def __call__(self):
            m = super().__call__()
            blk = m["__kdt__"].pop("panel", {}).get("streamflow", {})
            m.update({k: blk[k] for k in ("alpha", "beta") if k in blk})
            m["pbias"] = 3.0                                   # the KI's pbias: a timing bias in days
            return m
    tmp = tempfile.mkdtemp(prefix="kdt_s9x_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 30)
        (Path(ki) / "docs").mkdir()
        yaml.safe_dump({"validation": [{"dag_variable": "streamflow", "obs_shape": "point_time_series",
                                        "headline_metrics": [
                                            {"metric": "nse", "direction": "maximize", "pass_band": "satisfactory",
                                             "bands": {"satisfactory": 0.5}},
                                            {"metric": "pbias", "unit": "days", "direction": "zero_centered",
                                             "pass_band": "satisfactory", "bands": {"satisfactory": 5}}]}]},
                       open(Path(ki) / "docs" / "validation_convention.yaml", "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=Days(wd), budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cv = rep["convergence"]
    row = cv["cross_check_2n"]["streamflow"]
    assert cv["pbias_percent"]["streamflow"] is False and cv["kit_missing"] == {}
    assert "pbias" not in row["metrics"] and row["not_checked"]["pbias"]["status"].startswith("not checked")
    assert row["metrics"]["beta"]["status"] == "agrees"


def test_calibrate_judges_each_metric_by_its_decimals_over_the_pilot_runs():
    """Leo 2026-09-30, end to end: the runner writes its block to 4 decimals, but at the DEFAULT run its beta to
    1 decimal. Judged alone that 1-decimal value would get +-0.05 and pass; over the pilot's runs beta is written
    to 4 decimals, so the rounded default beta is a 2n mismatch."""
    class _ShortBetaAtDefault(S1._Runner):
        def __call__(self):
            m = super().__call__()
            p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
            blk = m["__kdt__"]["panel"]["streamflow"]
            for k, v in list(blk.items()):
                if isinstance(v, float):
                    blk[k] = round(v, 4)
            if abs(float(p["a"]) - 0.8) < 1e-12 and abs(float(p["b"]) - 2.0) < 1e-12:     # the default run
                blk["beta"] = round(blk["beta"], 1)
            return m
    rep, _ = _cal(_ShortBetaAtDefault)
    cv = rep["convergence"]
    assert cv["kit_missing"]["streamflow"]["beta"].startswith("2n mismatch")
    assert cv["cross_check_2n"]["streamflow"]["metrics"]["beta"]["bound"] == pytest.approx(5e-5 + 1e-6)


def test_the_precision_is_read_from_this_pilot_not_an_earlier_setups():
    """Opus 8b/9 r3 #3: a workdir where an earlier setup's pilot wrote full-precision values must not tighten the
    bound for a new setup whose runner rounds PBIAS to 1 decimal."""
    class _Rounded(S1._Runner):
        def __call__(self):
            m = super().__call__()
            blk = m["__kdt__"]["panel"]["streamflow"]
            for k, v in list(blk.items()):
                if isinstance(v, float):
                    blk[k] = round(v, 1 if k in ("pbias", "nrmse") else 4)
            return m
    tmp = tempfile.mkdtemp(prefix="kdt_s9x_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, None, 30)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"]["max_evaluations"] = 25   # a new setup
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_Rounded(wd), budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cv = rep["convergence"]
    assert "cross_check_2n" in cv and cv["cross_check_2n"]["streamflow"].get("metrics")
    assert not any(str(w).startswith("2n") for w in (cv.get("kit_missing") or {}).get("streamflow", {}).values())
    assert cv["cross_check_2n"]["streamflow"]["metrics"]["pbias"]["bound"] == pytest.approx(0.05 + 1e-6)


def test_a_declared_series_the_kit_cannot_read_is_reported_not_hidden():
    """Opus 8b/9 r4 #2: dates saved the usual pandas way are an OBJECT array; np.savez pickles it and the kit loads
    with allow_pickle=False. The kit must say why the series was not used, not "no series"."""
    import numpy as _np
    class _ObjectDates(S1._Runner):
        def __call__(self):
            m = super().__call__()
            if os.environ.get("KDT_CALIB_EMIT_SERIES") == "1":
                f = Path(self.workdir) / "kdt_series_streamflow.npz"
                z = dict(_np.load(f))
                _np.savez(f, sim=z["sim"], obs=z["obs"],
                          date=_np.array([f"1981-01-{(i % 28) + 1:02d}" for i in range(len(z["sim"]))], dtype=object))
            return m
    rep, _ = _cal(_ObjectDates)
    cv = rep["convergence"]
    assert cv["cross_check_2n"]["streamflow"]["status"].startswith("not checked (series unreadable")
    assert any("could not be read" in n and "object arrays" in n for n in cv["pilot_notes"])
    assert cv["tolerance_source"] == "fixed_fallback"


def test_a_score_the_kit_cannot_compute_on_the_series_is_missing_not_trusted():
    """codex step 9 r1 #2: constant obs — r and alpha are undefined; a runner that still reports them is wrong."""
    obs = np.full(100, 5.0)
    sim = 5.0 + np.random.default_rng(1).normal(0, 0.5, 100)
    reply = {"__kdt__": {"panel": {"Q": {"r": 0.9, "alpha": 1.1, "beta": float(sim.mean() / obs.mean())}}}}
    out = P.cross_check_2n({"Q": (sim, obs, None)}, reply, {"Q": "series"}, ["Q"])
    for m in ("r", "alpha"):
        assert out["kit_missing"]["Q"][m].startswith("2n mismatch") and "undefined on the series" in out["kit_missing"]["Q"][m]
    assert "beta" not in out["kit_missing"]["Q"]


def test_an_undefined_score_also_in_the_kis_headline_does_not_crash():
    """codex step 9 r2 #2: constant obs, r in the section AND as the KI's plain headline."""
    obs = np.full(100, 5.0)
    sim = 5.0 + np.random.default_rng(2).normal(0, 0.5, 100)
    reply = {"r": 0.8, "__kdt__": {"panel": {"Q": {"r": 0.9, "beta": float(sim.mean() / obs.mean())}}}}
    out = P.cross_check_2n({"Q": (sim, obs, None)}, reply, {"Q": "series"}, ["Q"])
    assert out["kit_missing"]["Q"]["r"].startswith("2n mismatch")
    assert out["checks"]["Q"]["ki_headline"]["r"]["status"].startswith("mismatch (the KI's headline value")
    assert any("undefined (on the series)" in n for n in out["notes"])
