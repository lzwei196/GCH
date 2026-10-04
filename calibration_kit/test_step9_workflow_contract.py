"""A workflow written to the letter of the contract prompt (auto_dissect_multi_agent/stage_calibrate.py, "Scores the
kit watches") gives the engine everything its convergence check needs (Leo 2026-09-30; Opus 8b/9 r4 #1, #5, #8).

The runner below computes the watched scores with the prompt's own formulas (NOT the kit's code), writes them per
target in `__kdt__.panel`, keeps its own KI scores as plain keys (here deliberately with the opposite PBIAS sign and
another NRMSE, as some KIs define them), and saves its scored series once when asked."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import panel as P                                                    # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402


def prompt_scores(sim, obs):
    """The watched scores exactly as the prompt words them (all eight)."""
    s, o = np.asarray(sim, "float64"), np.asarray(obs, "float64")
    m = np.isfinite(s) & np.isfinite(o)
    s, o = s[m], o[m]
    out = {}
    if len(s) >= 3 and s.std() > 0 and o.std() > 0:
        out["r"] = float(np.corrcoef(s, o)[0, 1])
    if o.std() > 0:
        out["alpha"] = float(s.std() / o.std())
    if o.mean() != 0:
        out["beta"] = float(s.mean() / o.mean())
        out["pbias"] = float(100 * (s.mean() - o.mean()) / o.mean())
        out["nrmse"] = float(100 * np.sqrt(np.mean((s - o) ** 2)) / abs(o.mean()))
    den0 = np.sum((o - o.mean()) ** 2)
    if den0 > 0:
        out["nse"] = float(1 - np.sum((s - o) ** 2) / den0)
    if all(k in out for k in ("r", "alpha", "beta")):
        out["kge"] = float(1 - np.sqrt((out["r"] - 1) ** 2 + (out["alpha"] - 1) ** 2 + (out["beta"] - 1) ** 2))
    eps = 0.01 * o.mean()
    if o.mean() > 0 and s.min() > -eps and o.min() > -eps:
        ls, lo = np.log(s + eps), np.log(o + eps)
        den = np.sum((lo - lo.mean()) ** 2)
        if den > 0:
            out["lnnse"] = float(1 - np.sum((ls - lo) ** 2) / den)
    return out


class PromptWorkflow:
    """A runner that follows the prompt literally. `single_value`: the target is one regional value."""

    def __init__(self, wd, single_value=False):
        self.wd, self.single = Path(wd), single_value

    def __call__(self):
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        split = "holdout" if os.environ.get("KDT_CALIB_SPLIT") == "holdout" else "calibration"
        sl = E.HOLD if split == "holdout" else E.CAL
        sim_all = float(p["a"]) * E.OBS + float(p["b"])
        sim, obs = sim_all[sl], E.OBS[sl]
        if self.single:
            sim, obs = np.array([sim.mean()]), np.array([obs.mean()])
        # the KI's OWN scores, as some KIs define them (opposite PBIAS sign, NRMSE by sd): the objectives read these
        own = {"pbias": float(100 * (obs.sum() - sim.sum()) / obs.sum()),
               "nrmse": float(np.sqrt(np.mean((sim - obs) ** 2)) / (obs.std() or 1.0))}
        if not self.single:
            f = P.panel_from_series(sim, obs, "flow", mean_rule=False)
            own.update({"nse": f["nse"], "kge": f["kge"], "r": f["r"]})
        out = dict(own)
        out["__kdt__"] = {"applied_params": dict(p), "case_id": "SITE:fixture", "split": split}
        out["__kdt__"]["panel"] = {"streamflow": prompt_scores(sim, obs)}          # the watched scores, per target
        if os.environ.get("KDT_CALIB_EMIT_SERIES") == "1":
            try:
                path = self.wd / "kdt_series_streamflow.npz"
                dates = np.asarray((np.datetime64("1981-01-01") + np.arange(len(sim))).astype(str), dtype="U16")
                np.savez(path, sim=np.asarray(sim, "float64"), obs=np.asarray(obs, "float64"), date=dates)
                out["__kdt__"]["series"] = {"streamflow": str(path)}                # ADDED, the block kept
            except Exception:
                pass
        return out


def _run(single_value=False):
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "snapshot" if single_value else "flow"}}}
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, conv, 40)
        if single_value:
            d = yaml.safe_load(open(Path(ki, "dag.yaml")))
            d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = ["magnitude_accuracy"]
            yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=PromptWorkflow(wd, single_value), budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep, hist


def test_a_flow_workflow_written_from_the_prompt_gives_the_engine_everything():
    rep, hist = _run()
    cv = rep["convergence"]
    assert rep["status"] == "completed", rep.get("reason")
    assert cv["required_panel"]["streamflow"] == ["r", "alpha", "beta", "lnnse"] and cv["kit_missing"] == {}
    search = [h for h in hist if h["phase"] == "search" and h.get("ok")]
    assert search and all({"r", "alpha", "beta", "lnnse"} <= set(h["panel"]["streamflow"]) for h in search)
    # measured from the saved series (a score with no sampling spread in this noise-free fixture — r = 1 on every
    # resample — keeps its labelled fixed value)
    assert cv["tolerance_source"].startswith("bootstrap")
    srcs = cv["tolerance_records"]["streamflow"]["source"]
    assert srcs["beta"] == "bootstrap" and srcs["lnnse"] == "bootstrap"
    rows = cv["cross_check_2n"]["streamflow"]["metrics"]
    assert rows and all(r["status"] == "agrees" for r in rows.values())            # the prompt's formulas = the kit's
    assert "not recorded" not in str(cv["verdict"].get("how", ""))                 # judged on full evidence


def test_a_single_value_workflow_written_from_the_prompt_is_judged():
    """A snapshot target (one regional value) needs pbias and nrmse: the KI's own, differently defined pbias and
    nrmse as plain keys must not stand in for them, and the watched ones in the block must reach the engine."""
    rep, hist = _run(single_value=True)
    cv = rep["convergence"]
    assert rep["status"] == "completed", rep.get("reason")
    assert cv["required_panel"]["streamflow"] == ["pbias", "nrmse"] and cv["kit_missing"] == {}
    search = [h for h in hist if h["phase"] == "search" and h.get("ok")]
    assert search and all({"pbias", "nrmse"} <= set(h["panel"]["streamflow"]) for h in search)
    # the watched pbias is the prompt's (sim - obs), not the KI's opposite-sign plain key
    h = search[-1]
    assert abs(h["panel"]["streamflow"]["pbias"] + _own_pbias_sign_check(h)) < 1e-9
    assert "not recorded" not in str(cv["verdict"].get("how", ""))


def _own_pbias_sign_check(h):
    """The KI's own plain pbias for that call (opposite sign), recomputed from the call's x."""
    a, b = h["x"]["a"], h["x"]["b"]
    sim, obs = (a * E.OBS + b)[E.CAL], E.OBS[E.CAL]
    return float(100 * (obs.sum() - sim.sum()) / obs.sum())


def test_two_targets_each_get_their_own_watched_scores():
    """Opus 8b/9 r4 #5: with two targets the scores go under each var name; flat keys would be read by neither."""
    pan = P.Panel(["Q", "ET"], kinds={"Q": "flow", "ET": "series"})
    rng = np.random.default_rng(3)
    oq, oe = 20 + rng.normal(0, 3, 200), 3 + rng.normal(0, 0.5, 200)
    reply = {"__kdt__": {"panel": {"Q": prompt_scores(oq * 1.1, oq), "ET": prompt_scores(oe * 0.9, oe)}}}
    got = pan.extract(reply)
    assert pan.missing_required("Q", got["Q"]) == [] and pan.missing_required("ET", got["ET"]) == []
    assert got["Q"]["beta"] != got["ET"]["beta"]
    flat = {k: v for k, v in prompt_scores(oq * 1.1, oq).items()}
    assert P.Panel(["Q", "ET"], kinds={"Q": "flow", "ET": "series"}).extract(flat) == {}


# ── review round 5 (Opus 8b/9, extended metrics r2) ──────────────────────────────────────────────
class _OlderSection(PromptWorkflow):
    """HBV v2's shape: six keys in the section, NSE and KGE only as the runner's plain keys (Opus r8 #1)."""
    def __call__(self):
        out = super().__call__()
        out["__kdt__"]["panel"]["streamflow"].pop("nse", None)
        out["__kdt__"]["panel"]["streamflow"].pop("kge", None)
        return out


def test_a_protected_nse_stays_active_when_the_reply_uses_the_reserved_section():
    """The section may lack NSE / KGE (older wording, or a runner that leaves them out): the panel takes the runner's
    own plain NSE / KGE / r, so a contract protecting NSE never silently loses it."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "flow"}}}
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, conv, 40)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["strategy"]["protect"] = {"streamflow": ["nse", "beta"]}
        c["strategy"]["multi_objective"] = True
        c["strategy"]["default_algorithm"] = "nsga2"
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_OlderSection(wd),
                              budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ru = rep["convergence"]["rule"]
    assert "headline_from_plain" in rep["convergence"]["panel_flags"]["streamflow"]     # the fallback was used
    assert "nse" in (ru["protect_active"] or {}).get("streamflow", []), ru.get("protect_dropped")
    assert not any(str(x).startswith("streamflow:nse") for x in (ru.get("protect_dropped") or []))


def test_the_section_is_read_whatever_the_case_of_its_keys():
    pan = P.Panel(["Q"], kinds={"Q": "flow"})
    rng = np.random.default_rng(5)
    o = 20 + rng.normal(0, 3, 200)
    sc = prompt_scores(o * 1.1, o)
    upper = {"r": sc["r"], "alpha": sc["alpha"], "beta": sc["beta"], "lnNSE": sc["lnnse"], "PBIAS": sc["pbias"]}
    got = pan.extract({"__kdt__": {"panel": {"Q": upper}}})["Q"]
    assert pan.missing_required("Q", got) == []


def _series_file(tmp, name, dates):
    p = Path(tmp) / name
    np.savez(p, sim=np.arange(20.0) + 1, obs=np.arange(20.0) + 1.5, date=np.asarray(dates, dtype="U16"))
    return str(p)


def test_non_iso_dates_are_refused_with_the_reason():
    from calibration_kit.pilot import pilot_series
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        iso = _series_file(tmp, "a.npz", [f"1981-01-{i + 1:02d}" for i in range(20)])
        dmy = _series_file(tmp, "b.npz", [f"{i + 1:02d}/01/1981" for i in range(20)])
        why = {}
        got = pilot_series({"series": {"Q": iso, "ET": dmy}}, ["Q", "ET"], why)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert "Q" in got and "ET" not in got and "dates are not ISO" in why["ET"]


def test_both_targets_series_are_found_and_a_runners_save_error_is_reported():
    from calibration_kit import pilot as PL

    class _Ev:
        def __init__(self, wd, kdt):
            self.workdir, self._last_metrics = wd, {"calibration": {"__kdt__": kdt}}
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        q = _series_file(tmp, "kdt_series_Q.npz", [f"1981-01-{i + 1:02d}" for i in range(20)])
        et = _series_file(tmp, "kdt_series_ET.npz", [f"1981-01-{i + 1:02d}" for i in range(20)])
        paths, err = PL._series_from_run(_Ev(tmp, {"series": {"ET": et}}), before={}, with_error=True)
        assert set(paths) == {"Q", "ET"} and err is None               # Q was not declared but is this run's file
        paths2, err2 = PL._series_from_run(_Ev(tmp, {"series_error": "OSError: disk full"}), before={}, with_error=True)
        assert paths2 == {} and err2 == "OSError: disk full"           # no half-written file is picked up
        why = {}
        PL.pilot_series({"series": {}, "series_error": err2}, ["Q"], why)
        assert why["Q"] == "no series (the runner could not save it: OSError: disk full)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── review round 6 (Opus 8b/9) ──────────────────────────────────────────────────────────────────
class _MonthlyNSE(PromptWorkflow):
    """The section lacks nse; the KI's own plain nse is computed another way (e.g. on monthly means)."""
    series = True

    def __call__(self):
        out = super().__call__()
        out["__kdt__"]["panel"]["streamflow"].pop("nse", None)
        out["__kdt__"]["panel"]["streamflow"].pop("kge", None)
        out["nse"] = round(out["nse"] - 0.02, 6)                        # a different, KI-defined NSE
        if not self.series:
            out["__kdt__"].pop("series", None)
            p = self.wd / "kdt_series_streamflow.npz"
            if p.exists():
                p.unlink()
        return out


def _protect_run(runner_cls):
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "flow"}}}
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, conv, 40)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["strategy"].update({"protect": {"streamflow": ["nse", "beta"]}, "multi_objective": True,
                              "default_algorithm": "nsga2"})
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=runner_cls(wd),
                              budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


def test_a_ki_defined_plain_nse_beside_a_section_is_caught_by_2n_once():
    rep = _protect_run(_MonthlyNSE)
    cv = rep["convergence"]
    assert cv["kit_missing"]["streamflow"]["nse"].startswith("2n mismatch")
    assert "nse" not in (cv["rule"]["protect_active"] or {}).get("streamflow", [])
    row = cv["cross_check_2n"]["streamflow"]
    assert row["metrics"]["nse"]["from"].startswith("the runner's plain key")
    assert "nse" not in (row.get("ki_headline") or {})                          # not reported a second time
    assert "nse" not in (row.get("not_checked") or {})                          # nor listed as "not checked"
    assert sum("streamflow:nse" in n for n in cv["pilot_notes"]) == 1
    assert not any("not checked against the kit's formula" in n for n in cv["pilot_notes"])   # a series was saved


def test_a_plain_nse_taken_without_a_series_is_said_to_be_unchecked():
    class _NoSeries(_MonthlyNSE):
        series = False
    rep = _protect_run(_NoSeries)
    notes = rep["convergence"]["pilot_notes"]
    assert any("nse, kge, r" in n or ("nse" in n and "taken from the runner's plain keys" in n) for n in notes), notes
    assert any("not checked against the kit's formula (no series)" in n for n in notes)


def test_an_upper_case_section_value_that_disagrees_with_the_series_is_a_2n_mismatch():
    rng = np.random.default_rng(9)
    o = 20 + rng.normal(0, 3, 300)
    s = o * 1.05 + rng.normal(0, 1, 300)
    sc = prompt_scores(s, o)
    blk = {"r": sc["r"], "alpha": sc["alpha"], "BETA": sc["beta"] * 1.03, "lnNSE": sc["lnnse"]}
    out = P.cross_check_2n({"Q": (s, o, None)}, {"__kdt__": {"panel": {"Q": blk}}}, {"Q": "flow"}, ["Q"])
    assert out["kit_missing"]["Q"]["beta"].startswith("2n mismatch")
    assert out["checks"]["Q"]["metrics"]["lnnse"]["status"] == "agrees"


def test_a_runners_series_error_gets_a_pilot_note():
    class _CannotSave(PromptWorkflow):
        def __call__(self):
            out = super().__call__()
            if "series" in out["__kdt__"]:
                Path(out["__kdt__"].pop("series")["streamflow"]).unlink()
                out["__kdt__"]["series_error"] = "OSError: disk full (planted)"
            return out
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "flow"}}}
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, conv, 40)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_CannotSave(wd),
                              budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    notes = rep["convergence"]["pilot_notes"]
    assert any("the runner could not save it: OSError: disk full (planted)" in n and "fixed values" in n for n in notes)
    # the section is full (nse, kge, r in it): nothing was taken from the plain keys, so no such note
    assert not any("taken from the runner's plain keys" in n for n in notes)


def test_a_series_file_from_before_the_call_is_not_this_runs():
    """A file copied in with an older timestamp (copy2, rsync -a) is not picked up as this run's series."""
    import time as _t
    from calibration_kit import pilot as PL

    class _Ev:
        def __init__(self, wd):
            self.workdir, self._last_metrics = wd, {"calibration": {"__kdt__": {}}}
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        old = _series_file(tmp, "kdt_series_ET.npz", [f"1981-01-{i + 1:02d}" for i in range(20)])
        os.utime(old, (_t.time() - 86400, _t.time() - 86400))          # a day old, but not in `before`
        mk = Path(tmp) / ".kdt_series_call_start"
        mk.touch()
        paths = PL._series_from_run(_Ev(tmp), before={}, started=mk.stat().st_mtime)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert paths == {}


def test_an_exact_lower_case_key_wins_over_another_spelling():
    rng = np.random.default_rng(7)
    o = 20 + rng.normal(0, 3, 300)
    s = o * 1.05 + rng.normal(0, 1, 300)
    sc = prompt_scores(s, o)
    # the exact key FIRST, the other spelling after it: "last one wins" would take the wrong value
    blk = {"r": sc["r"], "alpha": sc["alpha"], "lnnse": sc["lnnse"], "beta": sc["beta"], "BETA": 9.9}
    pan = P.Panel(["Q"], kinds={"Q": "flow"})
    assert pan.extract({"__kdt__": {"panel": {"Q": blk}}})["Q"]["beta"] == sc["beta"]
    out = P.cross_check_2n({"Q": (s, o, None)}, {"__kdt__": {"panel": {"Q": blk}}}, {"Q": "flow"}, ["Q"])
    assert out["checks"]["Q"]["metrics"]["beta"]["status"] == "agrees"


# ── review round 8 (Opus 8b/9) ──────────────────────────────────────────────────────────────────
def test_a_snapshot_targets_plain_nse_is_the_kis_headline_not_a_panel_value():
    """E17u: a snapshot target never records nse / kge / r, so a plain NSE beside its section is the KI's headline
    value — compared and reported, never 'taken' or made missing."""
    rng = np.random.default_rng(4)
    o = 20 + rng.normal(0, 3, 200)
    s = o * 1.05 + rng.normal(0, 1, 200)
    sc = prompt_scores(s, o)
    reply = {"nse": sc["nse"] - 0.02, "kge": sc["kge"], "r": sc["r"],
             "__kdt__": {"panel": {"Y": {"pbias": sc["pbias"], "nrmse": sc["nrmse"]}}}}
    out = P.cross_check_2n({"Y": (s, o, None)}, reply, {"Y": "snapshot"}, ["Y"])
    assert out["kit_missing"] == {}
    row = out["checks"]["Y"]
    assert "nse" not in row["metrics"] and row["ki_headline"]["nse"]["status"].startswith("mismatch (the KI's headline")
    pan = P.Panel(["Y"], kinds={"Y": "snapshot"})
    assert set(pan.extract(reply)["Y"]) == {"pbias", "nrmse"}                       # M20: nothing taken
    assert "headline_from_plain" not in pan.flags.get("Y", [])                       # … and not flagged as taken


def _single_value_run(runner_cls):
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "snapshot"}}}
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, conv, 40)
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = ["magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=runner_cls(wd, True),
                              budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


def test_no_taken_note_for_scores_a_snapshot_target_never_records():
    """M18: the 'taken from the runner's plain keys' note names only what the panel records."""
    class _PlainHeadlines(PromptWorkflow):
        def __call__(self):
            out = super().__call__()
            out.update({"nse": 0.5, "kge": 0.4, "r": 0.9})
            decl = out["__kdt__"].pop("series", None)
            if decl:
                Path(decl["streamflow"]).unlink()
            return out
    rep = _single_value_run(_PlainHeadlines)
    assert not any("taken from the runner's plain keys" in n for n in rep["convergence"]["pilot_notes"])


def test_a_nan_in_the_section_counts_as_missing_for_the_note():
    """M21: a section key that holds no number does not count as present: the plain value taken is noted."""
    class _NanNse(_OlderSection):
        def __call__(self):
            out = super().__call__()
            out["__kdt__"]["panel"]["streamflow"]["nse"] = float("nan")
            decl = out["__kdt__"].pop("series", None)
            if decl:
                Path(decl["streamflow"]).unlink()                     # no series at all: the 2n check cannot run
            return out
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "flow"}}}
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, conv, 40)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_NanNse(wd), budget=None, seed=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert any(n.startswith("streamflow: nse") and "taken from the runner's plain keys" in n
               for n in rep["convergence"]["pilot_notes"])


class _UndeclaredSeries(PromptWorkflow):
    """Writes its series without declaring it: `stale` copies an OLD file in (copy2 keeps the old timestamp)."""
    stale = None

    def __call__(self):
        out = super().__call__()
        decl = out["__kdt__"].pop("series", None)
        if decl and self.stale:
            Path(decl["streamflow"]).unlink()
            shutil.copy2(self.stale, self.wd / "kdt_series_streamflow.npz")
        return out


def _undeclared_run(stale=None, clock_ahead=0.0, monkeypatch=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s9w_")
    try:
        if stale:
            src = Path(tmp) / "old_series.npz"
            np.savez(src, sim=np.arange(30.0) + 1, obs=np.arange(30.0) + 2,
                     date=np.asarray([f"1981-02-{i % 28 + 1:02d}" for i in range(30)], dtype="U16"))
            import time as _t
            os.utime(src, (_t.time() - 86400, _t.time() - 86400))
            _UndeclaredSeries.stale = str(src)
        else:
            _UndeclaredSeries.stale = None
        if clock_ahead and monkeypatch is not None:
            import time as _t
            from calibration_kit import pilot as PL

            class _Shifted:
                def __getattr__(self, n):
                    return getattr(_t, n)

                def time(self):
                    return _t.time() + clock_ahead
            monkeypatch.setattr(PL, "time", _Shifted())
        conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "flow"}}}
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10, "seeds": 1}, conv, 40)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_UndeclaredSeries(wd),
                              budget=None, seed=0)
    finally:
        _UndeclaredSeries.stale = None
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


def test_an_old_file_copied_in_during_the_call_is_not_used():
    """E16a / M23: the stale-file check is really applied in the pilot."""
    rep = _undeclared_run(stale=True)
    assert rep["convergence"]["cross_check_2n"]["streamflow"]["status"].startswith("not checked")


def test_the_stale_file_check_uses_the_file_systems_clock(monkeypatch):
    """E16d / M24: with the kit's own clock 5 s ahead, a fresh undeclared series is still this run's."""
    rep = _undeclared_run(clock_ahead=5.0, monkeypatch=monkeypatch)
    assert rep["convergence"]["cross_check_2n"]["streamflow"].get("metrics")      # the series was used


def test_a_mismatch_on_a_score_the_kind_never_uses_is_reported_not_missing():
    """Opus r9 nit (P1, P2): a snapshot target never uses nse; a wrong nse in its section or as a plain key is
    reported, not made 'missing for the search'."""
    rng = np.random.default_rng(6)
    o = 20 + rng.normal(0, 3, 200)
    s = o * 1.05 + rng.normal(0, 1, 200)
    sc = prompt_scores(s, o)
    p1 = {"__kdt__": {"panel": {"Y": dict(sc, nse=sc["nse"] - 0.02)}}}                 # all eight, nse wrong
    p2 = {"nse": sc["nse"] - 0.02, "pbias": sc["pbias"]}                                # no section, plain
    for reply in (p1, p2):
        out = P.cross_check_2n({"Y": (s, o, None)}, reply, {"Y": "snapshot"}, ["Y"])
        assert "nse" not in (out["kit_missing"].get("Y") or {})
        assert out["checks"]["Y"]["metrics"]["nse"]["status"] == "mismatch (not used for this kind)"
        assert not any("missing for the search" in n and ":nse" in n for n in out["notes"])
