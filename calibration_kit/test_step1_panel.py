"""Build step 1 (design HANDOFF_CONVERGENCE_2026-09-27_v2.md, gaps 2c, 2o, 2t): each test checks one
design rule. Section numbers refer to the design document.

2c  required panel per variable, from its kind; kit block vs plain keys; no copying of unscoped
    metrics; alpha from KGE alone never used; kit-set missing metrics; pilot decisions.
2o  tolerance estimator (§2.11): block length, fixed generator seed, fallbacks, ddof 0, date order,
    persisted.
2t  alpha reconstruction (§2.13), verified against measured alpha.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit import panel as P                                       # noqa: E402
from calibration_kit.alpha_reconstruct import reconstruct_alpha, verify_against_measured  # noqa: E402


def _series(n=1500, a=1.1, b=1.05, noise=0.4, seed=3, base=5.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    obs = base + 3 * np.sin(t / 30.0) + rng.gamma(2.0, 1.0, n)
    sim = (obs - obs.mean()) * a + obs.mean() * b + rng.normal(0, noise, n)
    return sim, obs


# ── 2c: the one table (§1.4) ──────────────────────────────────────────────────────────────────
def test_kind_table_is_the_design_table():
    T = P.KIND_TABLE
    assert set(T) == {"flow", "series", "snapshot", "categorical"}
    assert set(T["flow"]["required"]) == {"r", "alpha", "beta", "lnnse"}
    assert set(T["flow"]["recorded"]) == {"r", "alpha", "beta", "pbias", "lnnse", "nse", "kge", "nrmse"}
    assert set(T["series"]["required"]) == {"r", "alpha", "beta"}
    assert set(T["series"]["recorded"]) == {"r", "alpha", "beta", "pbias", "nse", "kge", "nrmse"}
    assert set(T["snapshot"]["required"]) == {"pbias", "nrmse"}
    assert set(T["snapshot"]["recorded"]) == {"pbias", "nrmse"}
    assert T["categorical"]["required"] == () and T["categorical"]["recorded"] == ()
    for k in T:                                   # required is always a subset of recorded
        assert set(T[k]["required"]) <= set(T[k]["recorded"])


def _dag(var, shape, mode=None):
    s = {"obs_shape": shape, "metric_families": ["temporal_pattern_match"]}
    if mode:
        s["comparison_mode"] = mode
    return {"outputs": [{"var": var, "observability": {"comparable_obs_shapes": [s]}}]}


def test_kind_defaults_from_the_dag_entry():
    assert P.resolve_kind("Q", _dag("Q", "point_time_series"), "point_time_series") == ("series", [])
    assert P.resolve_kind("Y", _dag("Y", "point_snapshot"), "point_snapshot") == ("snapshot", [])
    # a categorical comparison makes the kind categorical WHATEVER its obs_shape (flood extent)
    k, _ = P.resolve_kind("fld", _dag("fld", "spatial_snapshot", "categorical_event_comparison"),
                          "spatial_snapshot")
    assert k == "categorical"
    assert P.resolve_kind("ev", _dag("ev", "categorical_event"), "categorical_event")[0] == "categorical"
    # discharge is `series` unless the contract says `flow` (the dag does not mark flow outputs)
    assert P.resolve_kind("Q", _dag("Q", "point_time_series"), "point_time_series", "flow") == ("flow", [])


def test_a_stated_kind_that_does_not_fit_is_warned_and_an_unknown_kind_is_refused():
    k, w = P.resolve_kind("Y", _dag("Y", "point_snapshot"), "point_snapshot", "series")
    assert k == "series" and w and "does not fit" in w[0]
    k, w = P.resolve_kind("fld", _dag("fld", "spatial_snapshot", "categorical_event_comparison"),
                          "spatial_snapshot", "snapshot")
    assert k == "snapshot" and w
    with pytest.raises(ValueError):
        P.resolve_kind("Q", _dag("Q", "point_time_series"), "point_time_series", "discharge")


def test_setup_kinds_reads_the_contract_and_the_convention():
    contract = {"strategy": {"convergence": {"variables": {"Q": {"kind": "flow"}}}}}
    conv = {"validation": [{"dag_variable": "SWE", "headline_metrics": [{"metric": "pbias", "unit": "mm"}]},
                           {"dag_variable": "Q", "headline_metrics": [{"metric": "pbias", "unit": "percent"}]}]}
    dag = {"outputs": _dag("Q", "point_time_series")["outputs"] + _dag("SWE", "point_time_series")["outputs"]}
    s = P.setup_kinds(["Q", "SWE"], dag, {"Q": "point_time_series", "SWE": "point_time_series"}, contract, conv)
    assert s["kinds"] == {"Q": "flow", "SWE": "series"}
    assert s["pbias_percent"] == {"Q": True, "SWE": False}


def test_pbias_unit_rule_reads_every_entry_of_the_variable():
    conv = {"validation": [{"dag_variable": "DOM", "headline_metrics": [{"metric": "pbias", "unit": "days"}]},
                           {"dag_variable": "Q", "headline_metrics": [{"metric": "pbias"}]},       # no unit
                           {"dag_variable": "T", "secondary_metrics": [{"metric": "PBIAS", "unit": "K"}]}]}
    assert P.pbias_is_percent(conv, "DOM") is False
    assert P.pbias_is_percent(conv, "Q") is True          # a missing unit is not "other than percent"
    assert P.pbias_is_percent(conv, "T") is False
    assert P.pbias_is_percent(None, "anything") is True


# ── 2c: kit formulas ──────────────────────────────────────────────────────────────────────────
def test_kit_formulas():
    sim, obs = _series()
    p = P.panel_from_series(sim, obs, "flow")
    assert p["alpha"] == pytest.approx(sim.std() / obs.std(), rel=1e-12)          # population sd
    assert p["beta"] == pytest.approx(sim.mean() / obs.mean(), rel=1e-12)
    assert p["pbias"] == pytest.approx(100 * (p["beta"] - 1), rel=1e-12)
    rmse = math.sqrt(np.mean((sim - obs) ** 2))
    assert p["nrmse"] == pytest.approx(100 * rmse / abs(obs.mean()), rel=1e-12)
    eps = 0.01 * obs.mean()
    ls, lo = np.log(sim + eps), np.log(obs + eps)
    assert p["lnnse"] == pytest.approx(1 - np.sum((ls - lo) ** 2) / np.sum((lo - lo.mean()) ** 2), rel=1e-12)
    kge = 1 - math.sqrt((p["r"] - 1) ** 2 + (p["alpha"] - 1) ** 2 + (p["beta"] - 1) ** 2)
    assert p["kge"] == pytest.approx(kge, rel=1e-12)
    # lnnse only for flow; snapshot records only pbias and nrmse; categorical records nothing
    assert "lnnse" not in P.panel_from_series(sim, obs, "series")
    assert set(P.panel_from_series(sim, obs, "snapshot")) - {"n"} == {"pbias", "nrmse"}
    assert set(P.panel_from_series(sim, obs, "categorical")) == {"n"}


def test_nrmse_uses_the_absolute_mean_and_is_missing_at_zero_mean():
    obs = np.array([-600.0, -550.0, -650.0, -620.0, -580.0])
    good = P.panel_from_series(obs + 20, obs, "snapshot")
    bad = P.panel_from_series(obs + 800, obs, "snapshot")
    assert 0 < good["nrmse"] < bad["nrmse"]                      # a worse fit is a LARGER NRMSE
    z = np.array([-1.0, 1.0, -2.0, 2.0, 0.0])
    assert "nrmse" not in P.panel_from_series(z + 0.1, z, "snapshot", mean_rule=False)


def test_mean_near_zero_drops_the_mean_divided_metrics_with_the_reason():
    rng = np.random.default_rng(0)
    obs = rng.normal(0.05, 1.0, 400)                   # |mean| << 0.1 sd
    sim = obs + rng.normal(0, 0.2, 400)
    p = P.panel_from_series(sim, obs, "series")
    assert not ({"beta", "pbias", "nrmse"} & set(p))
    assert p["_why"] == {m: "mean near zero" for m in ("beta", "pbias", "nrmse")}
    assert {"r", "alpha", "nse"} <= set(p)
    # the bootstrap never applies the rule per resample
    assert "beta" in P.panel_from_series(sim, obs, "series", mean_rule=False)


# ── 2c: reading a runner payload ──────────────────────────────────────────────────────────────
def test_the_kit_block_is_read_and_plain_nrmse_never_is():
    sim, obs = _series()
    blk = P.panel_block({"Q": (sim, obs)}, {"Q": "series"})
    payload = {"nrmse": 0.45, "pbias": 3.0, **blk}                 # KI-defined plain keys beside it
    pn = P.Panel(["Q"], kinds={"Q": "series"})
    got = pn.extract(payload)["Q"]
    truth = P.panel_from_series(sim, obs, "series")
    assert got["nrmse"] == pytest.approx(truth["nrmse"]) and got["pbias"] == pytest.approx(truth["pbias"])
    assert "from_block" in pn.flags["Q"]
    # the helper writes ONLY the block, never plain keys
    assert set(blk) == {"__kdt__"} and set(blk["__kdt__"]) == {"panel"}
    # without a block, plain nrmse is never read
    pn2 = P.Panel(["Q"], kinds={"Q": "series"})
    got2 = pn2.extract({"Q": {"r": 0.9, "nse": 0.8, "nrmse": 12.0, "pbias": 5.0}})["Q"]
    assert "nrmse" not in got2 and got2["beta"] == pytest.approx(1.05)


def test_plain_pbias_in_another_unit_is_not_read():
    pn = P.Panel(["DOM"], kinds={"DOM": "series"}, pbias_percent={"DOM": False})
    got = pn.extract({"DOM": {"r": 0.9, "pbias": 4.0}})["DOM"]
    assert "pbias" not in got and "beta" not in got
    assert "pbias_not_percent_ignored" in pn.flags["DOM"]
    pn_u = P.Panel(["DOM"], kinds={"DOM": "series"}, pbias_percent={"DOM": False}, pbias_unit={"DOM": "days"})
    pn_u.extract({"DOM": {"r": 0.9, "pbias": 4.0}})
    assert "runner's pbias is in days; no beta given" in pn_u.flags["DOM"]
    got = pn.extract({"DOM": {"r": 0.9, "pbias": 4.0, "beta": 1.02}})["DOM"]   # beta only from a plain beta
    assert got["beta"] == 1.02 and got["pbias"] == pytest.approx(2.0)
    assert pn.missing_required("DOM", {"r": 0.9}) == ["alpha", "beta"]


def test_alpha_is_never_derived_from_kge_alone():
    got, flags = P.derive_panel({"nse": 0.7, "kge": 0.8, "r": 0.9, "pbias": 5.0})
    assert "alpha" not in got


def test_unscoped_metrics_are_not_copied_to_several_variables():
    pn = P.Panel(["Q", "SWE"], kinds={"Q": "series", "SWE": "series"})
    assert pn.extract({"r": 0.9, "nse": 0.8, "pbias": 2.0}) == {}
    one = P.Panel(["Q"])
    assert one.extract({"r": 0.9})["Q"] == {"r": 0.9}
    assert "flat_single_variable" in one.flags["Q"]
    # var-scoped values go to their own variable only
    got = pn.extract({"Q": {"r": 0.9}, "SWE": {"r": 0.5}})
    assert got == {"Q": {"r": 0.9}, "SWE": {"r": 0.5}}


def test_a_categorical_variable_records_nothing():
    pn = P.Panel(["fld"], kinds={"fld": "categorical"})
    assert pn.extract({"fld": {"csi": 0.5, "r": 0.9}}) == {}
    assert pn.required("fld") == () and pn.recorded("fld") == ()


# ── 2c: kit-set missing and pilot decisions ───────────────────────────────────────────────────
def test_kit_set_missing_leaves_the_required_set_and_triggers_the_cap():
    pn = P.Panel(["Q"], kinds={"Q": "series"})
    pn.set_kit_missing("Q", "beta", "mean near zero")
    assert pn.required("Q") == ("r", "alpha") and pn.capped("Q")
    pn.set_kit_missing("Q", "beta", "2n mismatch")               # "mean near zero" wins
    assert pn.kit_missing["Q"]["beta"] == "mean near zero"
    assert "beta" not in pn.extract({"Q": {"r": 0.9, "beta": 1.1}})["Q"]
    # a recorded-only metric does not trigger the cap
    pn2 = P.Panel(["Q"], kinds={"Q": "series"})
    pn2.set_kit_missing("Q", "nrmse", "2n mismatch")
    assert not pn2.capped("Q")


def test_pilot_decisions_flow_to_series_and_mean_near_zero():
    sim, obs = _series()
    d = P.pilot_decisions({"Q": (sim, obs)}, {"Q": "flow"})
    assert d["kinds"]["Q"] == "flow" and d["kit_missing"] == {}
    # a value at or below -0.01*mean(obs) makes the log undefined -> judged as series, run goes on
    bad = sim.copy(); bad[10] = -0.02 * obs.mean()
    d = P.pilot_decisions({"Q": (bad, obs)}, {"Q": "flow"})
    assert d["kinds"]["Q"] == "series" and d["notes"]
    # a small negative above the limit keeps the log defined -> stays flow
    ok = sim.copy(); ok[10] = -0.005 * obs.mean()
    assert P.pilot_decisions({"Q": (ok, obs)}, {"Q": "flow"})["kinds"]["Q"] == "flow"
    # mean near zero -> kit-set missing for the mean-divided metrics of the kind
    rng = np.random.default_rng(1)
    o = rng.normal(0.0, 1.0, 300); o = o - o.mean() + 0.02; s = o + rng.normal(0, 0.1, 300)
    d = P.pilot_decisions({"T": (s, o)}, {"T": "series"})
    assert d["kit_missing"]["T"] == {m: "mean near zero" for m in ("beta", "pbias", "nrmse")}
    pn = P.Panel(["T"], kinds={"T": "series"})
    pn.apply_pilot(d)
    assert pn.required("T") == ("r", "alpha") and pn.capped("T")


# ── 2o: tolerances (§2.11) ────────────────────────────────────────────────────────────────────
def test_block_length_rule_and_short_record_fallback():
    sim, obs = _series(200)
    assert P.bootstrap_tolerances(sim, obs)["block"] == 30
    assert P.bootstrap_tolerances(sim[:80], obs[:80])["block"] == round(math.sqrt(80))
    t = P.bootstrap_tolerances(sim[:9], obs[:9])
    assert set(t["source"].values()) == {"fixed_fallback"} and all("too short" in w for w in t["why"].values())
    assert t["tol"]["pbias"] == 1.0 and t["tol"]["r"] == 0.01


def test_tolerance_is_the_ddof0_sd_of_500_block_resamples_with_generator_seed_0():
    sim, obs = _series(400)
    t = P.bootstrap_tolerances(sim, obs, "series")
    # rebuild the estimator by hand for one metric
    rng = np.random.default_rng(0); n = 400; b = 30; vals = []
    for _ in range(500):
        st = rng.integers(0, n - b + 1, size=math.ceil(n / b))
        idx = np.concatenate([np.arange(s, s + b) for s in st])[:n]
        vals.append(np.corrcoef(sim[idx], obs[idx])[0, 1])
    assert t["tol"]["r"] == pytest.approx(float(np.std(vals)), rel=1e-12)
    assert t["source"]["r"] == "bootstrap"


def test_tolerances_do_not_depend_on_the_search_seed(tmp_path):
    from calibration_kit.pilot import pilot_tolerances
    sim, obs = _series(300)
    f = tmp_path / "kdt_series_Q.npz"; np.savez(f, sim=sim, obs=obs)
    a, _ = pilot_tolerances({"series": {"Q": str(f)}}, ["Q"], {"Q": "series"}, seed=1)
    b, _ = pilot_tolerances({"series": {"Q": str(f)}}, ["Q"], {"Q": "series"}, seed=7)
    assert a == b


def test_too_few_usable_resamples_or_sd_zero_falls_back_per_metric():
    sim, obs = _series(300)
    t = P.bootstrap_tolerances(sim, obs, "series", n_boot=50)          # 50 < 100 usable
    assert set(t["source"].values()) == {"fixed_fallback"}
    const = np.full(300, 5.0)
    t = P.bootstrap_tolerances(const, obs, "series")                   # r undefined, alpha SD 0
    assert t["source"]["r"] == "fixed_fallback" and t["source"]["alpha"] == "fixed_fallback"


def test_valid_pairs_are_taken_in_date_order():
    sim, obs = _series(300)
    dates = np.arange(300)
    perm = np.random.default_rng(5).permutation(300)
    a = P.bootstrap_tolerances(sim, obs, "series", dates=dates)
    b = P.bootstrap_tolerances(sim[perm], obs[perm], "series", dates=dates[perm])
    assert a["tol"] == b["tol"]
    # non-finite pairs are dropped before blocking
    s2 = sim.copy(); s2[[3, 50]] = np.nan
    assert P.bootstrap_tolerances(s2, obs, "series")["n"] == 298


def test_every_variable_gets_a_labelled_tolerance_record_and_it_persists(tmp_path):
    sim, obs = _series(300)
    recs = P.tolerances_for(["Q", "SWE", "fld"], {"Q": "flow", "SWE": "series", "fld": "categorical"},
                            {"Q": (sim, obs)})
    assert set(recs["Q"]["source"].values()) <= {"bootstrap", "fixed_fallback"}
    assert "lnnse" in recs["Q"]["tol"]
    assert set(recs["SWE"]["source"].values()) == {"fixed_fallback"}
    assert recs["fld"]["tol"] == {}
    f = tmp_path / "tol.json"
    P.write_tolerances(f, recs)
    assert P.read_tolerances(f) == json.loads(json.dumps(recs))


# ── 2t: alpha reconstruction (§2.13) ──────────────────────────────────────────────────────────
def _history(n_calls=60, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(1000)
    obs = 5 + 3 * np.sin(t / 30) + rng.gamma(2, 1, 1000)
    recs, alpha = [], []
    for _ in range(n_calls):
        a = rng.uniform(0.6, 1.4); b = rng.uniform(0.8, 1.25)
        sim = (obs - obs.mean()) * a + obs.mean() * b + rng.normal(0, 0.8, 1000)
        p = P.panel_from_series(sim, obs, "series")
        recs.append({k: p[k] for k in ("nse", "kge", "r", "pbias")})
        alpha.append(p["alpha"])
    return recs, alpha, obs


def test_full_precision_reconstruction_matches_measured_alpha_to_1e_6():
    recs, alpha, obs = _history()
    v = verify_against_measured(recs, alpha, preconditions_ok=True)
    assert v["ok"] and v["n_missing"] == 0 and v["max_abs_err"] <= 1e-6
    c2 = (obs.mean() / obs.std()) ** 2
    assert v["reconstruction"]["c2"] == pytest.approx(c2, rel=1e-6)
    assert any(a < 1 for a in alpha) and any(a > 1 for a in alpha)       # both roots exercised


def test_rounded_records_never_pick_the_wrong_root():
    recs, alpha, _ = _history()
    rd = [{"nse": round(r["nse"], 4), "kge": round(r["kge"], 4), "r": round(r["r"], 4),
           "pbias": round(r["pbias"], 2)} for r in recs]
    res = reconstruct_alpha(rd, decimals={"nse": 4, "kge": 4, "r": 4, "pbias": 2}, preconditions_ok=True)
    assert res["status"] == "c2_agreed"
    picked = [(a, t) for a, t in zip(res["alpha"], alpha) if a is not None]
    assert len(picked) >= 0.8 * len(alpha)
    for a, t in picked:                       # the picked root is the one on the measured side of 1
        assert (a - 1) * (t - 1) > 0 or abs(t - 1) < 1e-3


def test_preconditions_not_shown_means_every_alpha_is_missing():
    recs, alpha, _ = _history(10)
    for ok in (None, False):
        res = reconstruct_alpha(recs, preconditions_ok=ok)
        assert res["alpha"] == [None] * 10 and res["status"] == "preconditions_unknown"


def test_r_equal_one_makes_both_roots_match_so_alpha_is_missing():
    recs, alpha, obs = _history(30)
    sim = (obs - obs.mean()) * 1.2 + obs.mean() * 1.1                     # r = 1 exactly
    p = P.panel_from_series(sim, obs, "series")
    recs = recs + [{k: p[k] for k in ("nse", "kge", "r", "pbias")}]
    res = reconstruct_alpha(recs, preconditions_ok=True)
    assert res["alpha"][-1] is None and "both roots" in res["why"][-1]


def test_calls_with_beta_near_one_still_get_alpha_once_c2_is_fixed():
    recs, alpha, obs = _history(40)
    sim = (obs - obs.mean()) * 0.8 + obs.mean() * 1.004 + np.random.default_rng(9).normal(0, 0.8, 1000)
    p = P.panel_from_series(sim, obs, "series")
    recs = recs + [{k: p[k] for k in ("nse", "kge", "r", "pbias")}]
    res = reconstruct_alpha(recs, preconditions_ok=True)
    assert abs(p["beta"] - 1) <= 0.01
    assert res["alpha"][-1] == pytest.approx(p["alpha"], abs=1e-6)


def test_no_agreement_means_every_alpha_is_missing():
    rng = np.random.default_rng(2)
    recs = [{"nse": rng.uniform(0.2, 0.9), "kge": rng.uniform(0.2, 0.9), "r": rng.uniform(0.8, 0.99),
             "pbias": rng.uniform(-20, 20)} for _ in range(30)]
    res = reconstruct_alpha(recs, preconditions_ok=True)
    assert all(a is None for a in res["alpha"])
    assert res["status"] in ("c2_no_agreement", "c2_unestimable", "c2_not_unique")


def test_a_known_c2_skips_the_estimate():
    recs, alpha, obs = _history(20)
    res = reconstruct_alpha(recs, preconditions_ok=True, c2_known=(obs.mean() / obs.std()) ** 2)
    assert res["status"] == "c2_known"
    assert max(abs(a - t) for a, t in zip(res["alpha"], alpha)) <= 1e-6


# ── wiring: the calibration report carries the step-1 facts ───────────────────────────────────
def test_the_report_carries_kinds_required_panel_and_labelled_tolerances():
    from calibration_kit import test_calibrate_convergence_e2e as E
    rep, hist, log = E._run(budget_block={"mode": "measured", "allowance": "2m", "seeds": 1,
                                          "pilot_runs": 10},
                            convergence_block={"mode": "observe", "window": 20,
                                               "variables": {"streamflow": {"kind": "flow"}}},
                            max_evaluations=40)
    assert rep["status"] == "completed", rep.get("reason")
    cv = rep["convergence"]
    assert cv["kinds"] == {"streamflow": "flow"}
    assert cv["required_panel"] == {"streamflow": ["r", "alpha", "beta", "lnnse"]}
    srcs = cv["tolerance_records"]["streamflow"]["source"]
    assert set(srcs) >= {"r", "alpha", "beta", "lnnse"} and set(srcs.values()) <= {"bootstrap", "fixed_fallback"}
    assert cv["tolerance_source"] in ("bootstrap", "bootstrap+fixed_fallback")


# ── review round 1 fixes (K3 + Opus 5.5, 2026-09-29) ──────────────────────────────────────────
import contextlib, io, os, shutil, tempfile                                    # noqa: E402
import yaml                                                                    # noqa: E402


class _Runner:
    """Like the e2e fixture runner, with a shift on sim and optional extra variable."""
    def __init__(self, workdir, shift=0.0, only_var=None, series_obs_near_zero=False, bad_dates=False,
                 block_obs_near_zero=False, no_series=False):
        from calibration_kit import test_calibrate_convergence_e2e as E
        self.E, self.workdir, self.shift, self.only_var = E, Path(workdir), shift, only_var
        self.near0, self.bad_dates = series_obs_near_zero, bad_dates
        self.block_near0, self.no_series = block_obs_near_zero, no_series

    def __call__(self):
        E = self.E
        p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        sim_all = float(p["a"]) * E.OBS + float(p["b"]) + self.shift
        sl = E.HOLD if os.environ.get("KDT_CALIB_SPLIT") == "holdout" else E.CAL
        sim, obs = sim_all[sl], E.OBS[sl]
        full = P.panel_from_series(sim, obs, "flow", mean_rule=False)
        plain = {k: full[k] for k in ("nse", "kge", "r", "pbias") if k in full}
        m = {"streamflow": plain} if self.only_var else dict(plain)
        bo = obs - obs.mean() + 0.01 * obs.std() if self.block_near0 else obs
        P.merge_panel_block(m, P.panel_block({"streamflow": (sim - obs + bo, bo)}, {}))
        m["__kdt__"].update({"applied_params": dict(p), "case_id": "SITE:fixture",
                             "split": "holdout" if os.environ.get("KDT_CALIB_SPLIT") == "holdout" else "calibration"})
        if os.environ.get("KDT_CALIB_EMIT_SERIES") == "1" and not self.no_series:
            f = self.workdir / "kdt_series_streamflow.npz"
            so = obs - obs.mean() + 0.01 * obs.std() if self.near0 else obs
            if self.bad_dates:
                np.savez(f, sim=sim, obs=so, date=np.arange(len(sim) - 1))
            else:
                np.savez(f, sim=sim, obs=so)
            m["__kdt__"]["series"] = {"streamflow": str(f)}
        return m


def _calibrate(tmp, runner_kw=None, conv=None, targets=None, extra_outputs=(), shapes=None, keep_wd=None):
    from calibration_kit import calib as C
    from calibration_kit import test_calibrate_convergence_e2e as E
    ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10}, conv, 30)
    if targets or extra_outputs:
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        if targets:
            c["targets"] = targets
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        for v in extra_outputs:
            d["outputs"].append({"var": v, "observability": {"comparable_obs_shapes": [
                {"obs_shape": "point_time_series", "metric_families": ["temporal_pattern_match"]}]}})
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
    buf = io.StringIO()
    with E._clean_env(), contextlib.redirect_stdout(buf):
        rep = C.calibrate(ki, wd, shapes or {"streamflow": "point_time_series"},
                          run_model=_Runner(wd, **(runner_kw or {})), budget=None, seed=0)
    return rep, wd, buf.getvalue(), ki


def test_the_marker_watches_only_required_metrics_and_can_fire_again():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        rep, *_ = _calibrate(tmp, conv={"mode": "observe", "window": 10})
        cv = rep["convergence"]
        assert set(cv["tolerances"]["streamflow"]) == {"r", "alpha", "beta"}
        assert cv["rule"]["stop_point"] is not None, cv["ended"]   # the run can be seen to settle
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pilot_decisions_are_applied_saved_in_the_plan_and_reused_on_resume():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        conv = {"mode": "observe", "window": 10, "variables": {"streamflow": {"kind": "flow"}}}
        first, wd, log1, ki = _calibrate(tmp, runner_kw={"shift": -30.0}, conv=conv)
        cv1 = first["convergence"]
        # the default run's sim goes below -0.01*mean(obs): judged as series, the run goes on
        assert cv1["kinds"] == {"streamflow": "series"} and cv1["pilot_notes"]
        plan = json.loads(Path(wd, "kdt_budget_plan.json").read_text())
        assert plan["panel_setup"]["kinds"] == {"streamflow": "series"}
        assert plan["panel_setup"]["tolerance_records"]["streamflow"]["source"]["beta"] == "bootstrap"
        # resume with the series file gone: the frozen setup is reused, not re-derived
        for f in Path(wd).glob("kdt_series_*.npz"):
            f.unlink()
        from calibration_kit import calib as C
        from calibration_kit import test_calibrate_convergence_e2e as E
        buf = io.StringIO()
        with E._clean_env(), contextlib.redirect_stdout(buf):
            second = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                 run_model=_Runner(wd, shift=-30.0), budget=None, seed=0)
        cv2 = second["convergence"]
        assert "panel setup REUSED" in buf.getvalue()
        assert cv2["kinds"] == cv1["kinds"] and cv2["tolerance_records"] == cv1["tolerance_records"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_declared_target_whose_objectives_the_probe_drops_stays_monitored():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        rep, *_ = _calibrate(tmp, runner_kw={"only_var": True},
                             targets=[{"var": "streamflow", "weight": 1.0}, {"var": "swe", "weight": 1.0}],
                             extra_outputs=("swe",),
                             shapes={"streamflow": "point_time_series", "swe": "point_time_series"})
        assert rep["status"] == "completed", rep.get("reason")
        cv = rep["convergence"]
        assert "swe" in cv["kinds"] and cv["variables_without_objective"] == ["swe"]
        assert rep["objectives"] == ["streamflow:temporal_pattern_match"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_unknown_kind_is_a_clean_refusal():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        rep, *_ = _calibrate(tmp, conv={"variables": {"streamflow": {"kind": "discharge"}}})
        assert rep["status"] == "invalid_contract" and "discharge" in rep["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_panel_block_has_all_eight_keys_and_merging_keeps_the_runner_kdt_keys():
    sim, obs = _series()
    blk = P.panel_block({"Q": (sim, obs)}, {"Q": "series"})["__kdt__"]["panel"]["Q"]
    assert set(P.BLOCK_KEYS) <= set(blk)                      # lnnse too, whatever kind was passed
    m = {"nse": 0.5, "__kdt__": {"applied_params": {"a": 1}, "case_id": "X", "split": "calibration"}}
    P.merge_panel_block(m, P.panel_block({"Q": (sim, obs)}, {}))
    assert m["__kdt__"]["applied_params"] == {"a": 1} and m["__kdt__"]["split"] == "calibration"
    assert "Q" in m["__kdt__"]["panel"] and m["nse"] == 0.5
    # a flow variable reads lnnse from the block even if the runner called it a series
    pn = P.Panel(["Q"], kinds={"Q": "flow"})
    assert "lnnse" in pn.extract(m)["Q"]


def test_snapshots_with_one_or_two_values_have_pbias_and_nrmse():
    for n in (1, 2):
        obs = np.array([5.0, 6.0][:n]); sim = obs * 1.1
        p = P.panel_from_series(sim, obs, "snapshot")
        assert p["pbias"] == pytest.approx(10.0) and "nrmse" in p
    d = P.pilot_decisions({"Y": (np.array([0.1, -0.1]) + 0.01, np.array([1.0, -1.0]))}, {"Y": "snapshot"})
    assert d["kit_missing"]["Y"] == {"pbias": "mean near zero", "nrmse": "mean near zero"}


def test_the_kind_follows_the_one_obs_shape_rule_for_a_named_target():
    dag = {"outputs": [{"var": "yield", "observability": {"comparable_obs_shapes": [
        {"obs_shape": "point_snapshot", "metric_families": ["magnitude_accuracy"]}]}}]}
    s = P.setup_kinds(["yield"], dag, {"Yield_obs": "point_snapshot"},
                      {"targets": [{"var": "yield"}]})
    assert s["kinds"] == {"yield": "snapshot"} and not s["warnings"]


def test_lnnse_is_kept_for_small_negative_values_and_the_switch_looks_at_obs_too():
    sim, obs = _series()
    s2 = sim.copy(); s2[5] = -0.005 * obs.mean()                  # above -0.01*mean: log defined
    assert "lnnse" in P.panel_from_series(s2, obs, "flow")
    o2 = obs.copy(); o2[7] = -0.02 * obs.mean()                   # obs below the limit
    d = P.pilot_decisions({"Q": (sim, o2)}, {"Q": "flow"})
    assert d["kinds"]["Q"] == "series" and "obs has a value" in d["notes"][0]


def test_the_bootstrap_does_not_apply_the_mean_rule_per_resample():
    rng = np.random.default_rng(4)
    o = rng.normal(0.0, 1.0, 400); o = o - o.mean() + 0.02
    s = o + rng.normal(0, 0.3, 400)
    t = P.bootstrap_tolerances(s, o, "series")
    assert t["source"]["beta"] == "bootstrap" and t["tol"]["beta"] > 0


def test_a_variable_whose_metrics_are_all_kit_set_missing_is_capped_not_rule_0():
    pn = P.Panel(["Y"], kinds={"Y": "snapshot"})
    pn.apply_pilot({"kit_missing": {"Y": {"pbias": "mean near zero", "nrmse": "mean near zero"}}})
    assert pn.required("Y") == () and pn.recorded("Y") == ("pbias", "nrmse") and pn.capped("Y")


def test_verification_needs_every_measured_call_and_equal_lengths():
    recs, alpha, obs = _history(30)
    sim = (obs - obs.mean()) * 1.2 + obs.mean() * 1.1                     # r = 1: alpha missing
    p = P.panel_from_series(sim, obs, "series")
    v = verify_against_measured(recs + [{k: p[k] for k in ("nse", "kge", "r", "pbias")}],
                                alpha + [p["alpha"]], preconditions_ok=True)
    assert v["n_missing"] == 1 and v["ok"] is False
    with pytest.raises(ValueError):
        verify_against_measured(recs, alpha[:-1], preconditions_ok=True)


def test_two_c2_values_with_full_support_are_not_unique_so_alpha_is_missing():
    """Calls built so that the '+' roots all give c2 = A and the '-' roots all give c2 = B:
    4 D (1 - r) = (beta - 1)^2 (B - A). Both values have 100 % support -> never guess."""
    A, B = 6.0, 9.0
    rng = np.random.default_rng(11)
    recs = []
    for _ in range(12):
        r = rng.uniform(0.7, 0.95); beta = rng.uniform(1.05, 1.3)
        D = (beta - 1) ** 2 * (B - A) / (4 * (1 - r))
        ap = 1 + D
        nse = 2 * r * ap - ap * ap - (beta - 1) ** 2 * A
        kge = 1 - math.sqrt(D * D + (r - 1) ** 2 + (beta - 1) ** 2)
        recs.append({"nse": nse, "kge": kge, "r": r, "pbias": 100 * (beta - 1)})
    res = reconstruct_alpha(recs, preconditions_ok=True)
    assert res["status"] == "c2_not_unique" and all(a is None for a in res["alpha"])


# ── review round 2 fixes (Opus 5.5) ───────────────────────────────────────────────────────────
def test_targets_omitted_only_outputs_with_their_own_obs_entry_are_declared():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        from calibration_kit import calib as C
        from calibration_kit import test_calibrate_convergence_e2e as E
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10}, None, 30)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c.pop("targets", None)
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"].append({"var": "q_alias", "observability": {"comparable_obs_shapes": [
            {"obs_shape": "point_time_series", "metric_families": ["temporal_pattern_match"]}]}})
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        buf = io.StringIO()
        with E._clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_Runner(wd),
                              budget=None, seed=0)
        assert rep["status"] == "completed", rep.get("reason")
        assert list(rep["convergence"]["kinds"]) == ["streamflow"]
        assert rep["objectives"] == ["streamflow:temporal_pattern_match"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_one_obs_shape_rule_is_not_applied_to_unnamed_variables():
    dag = {"outputs": [{"var": "yield", "observability": {"comparable_obs_shapes": [
        {"obs_shape": "point_snapshot", "metric_families": ["magnitude_accuracy"]}]}}]}
    s = P.setup_kinds(["yield"], dag, {"Yield_obs": "point_snapshot"}, {})
    assert s["kinds"] == {"yield": "series"} and s["warnings"]


def test_mean_near_zero_from_the_helper_block_when_there_is_no_series():
    rng = np.random.default_rng(6)
    o = rng.normal(0.0, 1.0, 300); o = o - o.mean() + 0.02; sm = o + rng.normal(0, 0.2, 300)
    blk = P.panel_block({"T": (sm, o)}, {})
    assert not ({"beta", "pbias", "nrmse"} & set(blk["__kdt__"]["panel"]["T"]))      # helper applies the rule
    assert blk["__kdt__"]["panel"]["T"]["_why"]["beta"] == "mean near zero"
    d = P.pilot_decisions_from_payload(blk, {"T": "series"}, ["T"])
    assert d["kit_missing"]["T"] == {m: "mean near zero" for m in ("beta", "pbias", "nrmse")}
    pn = P.Panel(["T"], kinds={"T": "series"}); pn.apply_pilot(d)
    assert pn.required("T") == ("r", "alpha") and pn.capped("T")


def test_kit_set_missing_values_are_dropped_even_when_they_come_from_a_block():
    sim, obs = _series()
    pn = P.Panel(["Q"], kinds={"Q": "series"})
    pn.set_kit_missing("Q", "beta", "2n mismatch")
    got = pn.extract(P.panel_block({"Q": (sim, obs)}, {}))["Q"]
    assert "beta" not in got and "r" in got


def test_the_bootstrap_measures_lnnse_for_a_flow_variable():
    sim, obs = _series(400)
    t = P.bootstrap_tolerances(sim, obs, "flow")
    assert t["source"]["lnnse"] == "bootstrap" and t["tol"]["lnnse"] > 0


def test_switch_limit_is_inclusive_and_lnnse_is_absent_below_it():
    sim, obs = _series()
    at = sim.copy(); at[3] = -0.01 * obs.mean()                  # exactly at the limit: log(0)
    assert P.pilot_decisions({"Q": (at, obs)}, {"Q": "flow"})["kinds"]["Q"] == "series"
    assert "lnnse" not in P.panel_from_series(at, obs, "flow")
    below = sim.copy(); below[3] = -0.5 * obs.mean()
    assert "lnnse" not in P.panel_from_series(below, obs, "flow")


def test_block_length_and_short_record_limits_exactly():
    sim, obs = _series(200)
    assert P.bootstrap_tolerances(sim[:90], obs[:90])["block"] == 30
    assert P.bootstrap_tolerances(sim[:89], obs[:89])["block"] == 9
    assert P.bootstrap_tolerances(sim[:10], obs[:10])["block"] == 3          # n = 10 is measured
    assert P.bootstrap_tolerances(sim[:9], obs[:9])["block"] is None         # n = 9 falls back


def test_kit_missing_is_reused_on_resume_and_an_all_missing_variable_is_not_no_panel():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        conv = {"mode": "observe", "window": 10, "variables": {"streamflow": {"kind": "snapshot"}}}
        first, wd, log1, ki = _calibrate(tmp, runner_kw={"series_obs_near_zero": True}, conv=conv)
        cv1 = first["convergence"]
        km = cv1["kit_missing"]["streamflow"]
        assert {m for m, w in km.items() if w == "mean near zero"} == {"pbias", "nrmse"}
        # the fixture's kit block is computed from a normal-mean obs, its series from a near-zero one:
        # the 2n cross-check (step 9) finds the runner's block disagreeing with the kit on the rest
        assert all(w.startswith("2n mismatch") for w in km.values() if w != "mean near zero")
        assert cv1["required_panel"] == {"streamflow": []}
        assert cv1["variables_without_panel"] == []          # capped, not rule 0 (rev 27)
        for f in Path(wd).glob("kdt_series_*.npz"):
            f.unlink()
        from calibration_kit import calib as C
        from calibration_kit import test_calibrate_convergence_e2e as E
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            second = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                 run_model=_Runner(wd, series_obs_near_zero=True), budget=None, seed=0)
        assert second["convergence"]["kit_missing"] == cv1["kit_missing"]
        assert second["convergence"]["required_panel"] == cv1["required_panel"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_one_bad_series_file_affects_only_its_own_variable():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        rep, *_ = _calibrate(tmp, runner_kw={"bad_dates": True}, conv={"mode": "observe", "window": 10})
        cv = rep["convergence"]
        rec = cv["tolerance_records"]["streamflow"]
        assert set(rec["source"].values()) == {"fixed_fallback"}
        assert all("series unreadable" in w for w in rec["why"].values())
        assert cv["tolerance_source"] == "fixed_fallback"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kind_settings_for_unknown_names_or_bad_shapes_are_warned():
    dag = _dag("Q", "point_time_series")
    s = P.setup_kinds(["Q"], dag, {"Q": "point_time_series"},
                      {"strategy": {"convergence": {"variables": {"Q": "flow", "X": {"kind": "flow"}}}}})
    assert s["kinds"] == {"Q": "series"}
    assert any("mapping" in w for w in s["warnings"]) and any("not a declared variable" in w for w in s["warnings"])


def test_sim_and_obs_of_different_lengths_are_refused():
    with pytest.raises(ValueError):
        P.panel_from_series(np.ones(10), np.ones(9))


def test_without_a_series_the_helpers_mean_near_zero_is_decided_at_the_defaults():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        rep, *_ = _calibrate(tmp, runner_kw={"block_obs_near_zero": True, "no_series": True},
                             conv={"mode": "observe", "window": 10})
        cv = rep["convergence"]
        assert cv["kit_missing"] == {"streamflow": {m: "mean near zero" for m in ("beta", "pbias", "nrmse")}}
        assert cv["required_panel"] == {"streamflow": ["r", "alpha"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── review round 3 fixes (Opus 5.5) ───────────────────────────────────────────────────────────
def test_a_variables_list_instead_of_a_mapping_is_a_clean_refusal():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        rep, *_ = _calibrate(tmp, conv={"variables": [{"var": "streamflow", "kind": "flow"}]})
        assert rep["status"] == "invalid_contract" and "mapping" in rep["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_one_variables_bad_input_never_affects_another_variable(tmp_path):
    from calibration_kit.pilot import pilot_series, pilot_tolerances
    sim, obs = _series(300)
    good = tmp_path / "kdt_series_Q.npz"; np.savez(good, sim=sim, obs=obs)
    two_d = tmp_path / "kdt_series_SWE.npz"; np.savez(two_d, sim=np.c_[sim, sim], obs=np.c_[obs, obs])
    short = tmp_path / "kdt_series_ET.npz"; np.savez(short, sim=sim, obs=obs, date=np.arange(299))
    pilot = {"series": {"Q": str(good), "SWE": str(two_d), "ET": str(short)}}
    why = {}
    ser = pilot_series(pilot, ["Q", "SWE", "ET"], why)
    assert set(ser) == {"Q"} and why["SWE"].startswith("series unreadable") and why["ET"].startswith("series unreadable")
    recs, label = pilot_tolerances(pilot, ["Q", "SWE", "ET"], {"Q": "series", "SWE": "series", "ET": "series"})
    assert recs["Q"]["source"]["beta"] == "bootstrap"
    assert set(recs["SWE"]["source"].values()) == {"fixed_fallback"} and label == "bootstrap+fixed_fallback"
    # pilot decisions and tolerances per variable survive a bad tuple for another variable
    rng = np.random.default_rng(8)
    o0 = rng.normal(0, 1, 300); o0 = o0 - o0.mean() + 0.02; s0 = o0 + rng.normal(0, 0.1, 300)
    bad_sim = sim.copy(); bad_sim[3] = -obs.mean()
    # bad variable FIRST, then two good ones with real decisions: a flow->series switch and mean near zero
    d = P.pilot_decisions({"SWE": (np.c_[sim, sim], obs), "Q": (bad_sim, obs), "T": (s0, o0)},
                          {"SWE": "flow", "Q": "flow", "T": "series"})
    assert d["kinds"]["Q"] == "series" and d["kit_missing"]["T"]["beta"] == "mean near zero"
    assert any("SWE" in n for n in d["notes"])
    t = P.tolerances_for(["Q", "SWE"], {"Q": "series", "SWE": "series"},
                         {"Q": (sim, obs), "SWE": (sim, obs[:-1])})
    assert t["Q"]["source"]["beta"] == "bootstrap" and set(t["SWE"]["source"].values()) == {"fixed_fallback"}


def test_block_why_only_counts_the_kits_own_reason_for_mean_divided_metrics_of_the_kind():
    blk = {"__kdt__": {"panel": {"A": {"_why": "mean near zero"},                         # not a mapping
                                 "B": {"_why": {"beta": "runner says so", "r": "mean near zero"}},
                                 "C": {"_why": {"beta": "mean near zero", "pbias": "mean near zero"}}}}}
    d = P.pilot_decisions_from_payload(blk, {"A": "series", "B": "series", "C": "snapshot"}, ["A", "B", "C"])
    assert d["kit_missing"] == {"C": {"pbias": "mean near zero"}}     # snapshot records no beta


def test_block_why_is_used_only_for_variables_without_a_series():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        rep, *_ = _calibrate(tmp, runner_kw={"block_obs_near_zero": True}, conv={"mode": "observe", "window": 10})
        km = rep["convergence"]["kit_missing"].get("streamflow", {})
        assert "mean near zero" not in km.values()             # the series (normal mean) decides
        # (the block was computed from other data, so the 2n cross-check flags where it disagrees)
        assert km and all(w.startswith("2n mismatch") for w in km.values())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_without_a_series_and_without_the_probe_the_pilots_default_run_decides():
    tmp = tempfile.mkdtemp(prefix="kdt_s1_")
    try:
        from calibration_kit import calib as C
        from calibration_kit import test_calibrate_convergence_e2e as E
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "2m", "pilot_runs": 10},
                            {"mode": "observe", "window": 10}, 30)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"]["probe_objectives"] = False
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=_Runner(wd, block_obs_near_zero=True, no_series=True),
                              budget=None, seed=0)
        assert rep["convergence"]["kit_missing"] == {"streamflow": {m: "mean near zero" for m in ("beta", "pbias", "nrmse")}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_why_missing_names_the_right_call():
    recs, alpha, obs = _history(30)
    sim = (obs - obs.mean()) * 1.2 + obs.mean() * 1.1
    p = P.panel_from_series(sim, obs, "series")
    v = verify_against_measured(recs + [{k: p[k] for k in ("nse", "kge", "r", "pbias")}],
                                alpha + [p["alpha"]], preconditions_ok=True)
    assert list(v["why_missing"]) == [30] and "both roots" in v["why_missing"][30]
