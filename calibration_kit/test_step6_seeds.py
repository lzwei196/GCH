"""Build step 6 (design §1.7, §2.7; gap 2l): seeds. Three slots by default, every slot tracked, a crashed
seed replaced once, agreement on the seeds' final CALIBRATION incumbents (categorical "not compared" kept
apart from missing), the returned seed = the lowest calibration loss whatever the agreement, the reported
result = that seed's calibration incumbent, validation read once, run verdict and per-variable verdicts
across seeds, parameter spread reported."""
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

from calibration_kit.seeds import (across_seeds, param_spread, pick_returned_seed,       # noqa: E402
                                   seed_agreement)
from calibration_kit.verdict import CONVERGED, NOT_CONVERGED, UNKNOWN                     # noqa: E402
from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402
from calibration_kit import test_step1_panel as S1                                        # noqa: E402

REQ = {"Q": ["r", "alpha", "beta"], "EVT": []}
TOL = {"Q": {"r": 0.02, "alpha": 0.05, "beta": 0.05}}


def _p(r, a, b):
    return {"Q": {"r": r, "alpha": a, "beta": b}}


# ── agreement (§2.7) ─────────────────────────────────────────────────────────────────────────────
def test_seeds_agree_within_tolerance_and_categorical_is_not_compared():
    ag = seed_agreement([_p(0.90, 1.00, 1.00), _p(0.91, 1.02, 0.99), _p(0.905, 0.99, 1.01)], REQ, TOL)
    assert ag["agree"] is True and ag["not_compared"] == ["EVT"]
    assert ag["per_variable"]["EVT"].startswith("not compared") and not ag["missing"]
    assert ag["per_variable"]["Q"]["spread"]["r"] == pytest.approx(0.01)


def test_a_spread_beyond_its_tolerance_is_disagreement_even_with_something_missing():
    ag = seed_agreement([_p(0.90, 1.0, 1.0), _p(0.80, 1.0, 1.0), None], REQ, TOL)
    assert ag["agree"] is False and ["Q", "r", pytest.approx(0.1), 0.02] in ag["exceeds"]
    ag2 = seed_agreement([_p(0.90, 1.0, None), _p(0.80, 1.0, 1.0)], REQ, TOL)
    assert ag2["agree"] is False                                   # beta missing, r exceeds
    # the exceeding metric is itself missing at one seed: what WAS compared already exceeds
    ag3 = seed_agreement([_p(0.90, 1.0, 1.0), _p(0.80, 1.0, 1.0), _p(None, 1.0, 1.0)], REQ, TOL)
    assert ag3["agree"] is False and ag3["per_variable"]["Q"]["agree"] is False and "r" in ag3["per_variable"]["Q"]["missing"]


def test_missing_evidence_makes_agreement_unknown_never_true():
    miss = seed_agreement([_p(0.9, 1.0, 1.0), _p(0.9, 1.0, None), _p(0.9, 1.0, 1.0)], REQ, TOL)
    assert miss["agree"] is None and ["Q", "beta", "not recorded at every seed's incumbent"] in miss["missing"]
    assert miss["per_variable"]["Q"]["agree"] is None and miss["per_variable"]["Q"]["missing"] == ["beta"]
    slot = seed_agreement([_p(0.9, 1.0, 1.0), None, _p(0.9, 1.0, 1.0)], REQ, TOL)
    assert slot["agree"] is None and "no completed seed" in slot["reason"]
    notol = seed_agreement([_p(0.9, 1.0, 1.0), _p(0.9, 1.0, 1.0)], REQ, {"Q": {"r": 0.02, "alpha": 0.05}})
    assert notol["agree"] is None and ["Q", "beta", "no tolerance"] in notol["missing"]


def test_no_panel_at_all_means_seeds_not_compared_and_one_seed_is_not_checked():
    cat = seed_agreement([{}, {}, {}], {"EVT": []}, {})
    assert cat["agree"] is None and cat["reason"].startswith("seeds not compared")
    one = seed_agreement([_p(0.9, 1.0, 1.0)], REQ, TOL)
    assert one["agree"] is None and "not checked" in one["reason"]


# ── the returned seed (§1.7) ─────────────────────────────────────────────────────────────────────
def test_single_objective_returns_the_lowest_calibration_loss_ties_to_the_lowest_slot():
    cs = [{"slot": 0, "scalar": 0.30}, {"slot": 1, "scalar": 0.10}, {"slot": 2, "scalar": 0.10}]
    i, how = pick_returned_seed(cs, trade_off=False)
    assert i == 1 and "slot 1" in how
    assert pick_returned_seed([], trade_off=False)[0] is None


def test_trade_off_admissible_first_then_the_worst_normalized_objective_on_one_frozen_scale():
    # slot 0 never reached t0; slot 1's scale is used for everyone
    cs = [{"slot": 0, "losses": [0.1, 0.9], "admissible": True, "t0": None, "z": None, "s": None},
          {"slot": 1, "losses": [0.5, 0.5], "admissible": True, "t0": 30, "z": [0.0, 0.0], "s": [1.0, 1.0]},
          {"slot": 2, "losses": [0.4, 0.4], "admissible": False, "t0": 25, "z": [0.0, 0.0], "s": [10.0, 0.1]}]
    i, how = pick_returned_seed(cs, trade_off=True)
    assert i == 1 and "frozen scale of slot 1" in how
    assert how.startswith("admissible under protection, then") and "protection infeasible" not in how
    # none admissible: said so, never "admissible ..., then"
    none = [dict(c, admissible=False) for c in cs]
    assert pick_returned_seed(none, trade_off=True)[1].startswith("no seed is admissible under protection")
    # all admissible: neither prefix
    allok = pick_returned_seed([dict(c, admissible=True) for c in cs], trade_off=True)[1]
    assert "admissible" not in allok
    # without protection problems the smaller worst objective wins (0.4 < 0.5 on slot 1's scale)
    cs[2]["admissible"] = True
    assert pick_returned_seed(cs, trade_off=True)[0] == 2
    # the scale is the LOWEST slot's that reached t0, whatever the others froze
    cs2 = [{"slot": 0, "losses": [9.0, 9.0], "admissible": True, "t0": None, "z": None, "s": None},
           {"slot": 1, "losses": [0.3, 0.2], "admissible": True, "t0": 30, "z": [0.0, 0.0], "s": [1.0, 0.1]},
           {"slot": 2, "losses": [0.2, 0.3], "admissible": True, "t0": 25, "z": [0.0, 0.0], "s": [0.1, 1.0]}]
    assert pick_returned_seed(cs2, trade_off=True)[0] == 1        # on slot 2's scale slot 2 would win
    # no slot reached t0: the smallest sum of raw losses
    for c in cs:
        c["t0"] = None
    i, how = pick_returned_seed(cs, trade_off=True)
    assert i == 2 and "sum of raw losses" in how
    # ... and admissibility still comes first there (slot 2 had the smallest sum; slots 0 and 1 tie -> 0)
    cs[2]["admissible"] = False
    assert pick_returned_seed(cs, trade_off=True)[0] == 0


# ── verdicts across seeds (§2.7 aggregation) ─────────────────────────────────────────────────────
def test_run_verdict_across_seeds_uses_the_verdict_spelling():
    """The verdicts come from verdict.py (one spelling: NOT_CONVERGED = "not_converged")."""
    ok = {"agree": True, "per_variable": {"Q": {"agree": True}, "EVT": "not compared"}}
    c3 = [CONVERGED] * 3
    vv = [{"Q": CONVERGED, "EVT": CONVERGED}] * 3
    out = across_seeds(c3, vv, ok)
    assert out["run_verdict"] == CONVERGED and out["variables"] == {"Q": CONVERGED, "EVT": UNKNOWN}
    assert across_seeds(c3, vv, {"agree": None, "per_variable": {"Q": {"agree": None}}})["run_verdict"] == UNKNOWN
    assert across_seeds(c3, vv, {"agree": False, "per_variable": {"Q": {"agree": False}}})["run_verdict"] == \
        NOT_CONVERGED
    # one not-converged seed makes the run not converged (it used to read "unknown": two spellings)
    one_nc = across_seeds([CONVERGED, NOT_CONVERGED, CONVERGED],
                          [vv[0], {"Q": NOT_CONVERGED, "EVT": CONVERGED}, vv[0]], ok)
    assert one_nc["run_verdict"] == NOT_CONVERGED and one_nc["variables"]["Q"] == NOT_CONVERGED
    assert across_seeds([NOT_CONVERGED] * 3, [{"Q": NOT_CONVERGED}] * 3, ok)["run_verdict"] == NOT_CONVERGED
    # a missing slot does not hide a not-converged one
    assert across_seeds([NOT_CONVERGED, None, None], [{"Q": NOT_CONVERGED}, None, None],
                        {"agree": None, "per_variable": {"Q": {"agree": None}}})["run_verdict"] == NOT_CONVERGED
    assert across_seeds([CONVERGED, None, CONVERGED], [vv[0], None, vv[0]], ok)["run_verdict"] == UNKNOWN
    one = across_seeds([CONVERGED], [{"Q": CONVERGED}], {"agree": None})
    assert one["run_verdict"] == CONVERGED and "not checked" in one["how"]


def test_param_spread_is_reported():
    assert param_spread([{"a": 1.0, "b": 2.0}, {"a": 1.5, "b": 2.0}]) == {"a": 0.5, "b": 0.0}
    assert param_spread([{"a": 1.0}]) == {}


# ── through calibrate() ──────────────────────────────────────────────────────────────────────────
def _run(budget_block=None, strategy=None, max_evaluations=40, budget=None, env=None, backend_wrap=None,
         monkeypatch=None, runner=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, budget_block, None, max_evaluations)
        if strategy:
            c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strategy)
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        if backend_wrap is not None:
            real = C._make_backend
            monkeypatch.setattr(C, "_make_backend", lambda algo: backend_wrap(real(algo)))
        buf = io.StringIO()
        with E._clean_env(), contextlib.redirect_stdout(buf):
            for k, v in (env or {}).items():
                os.environ[k] = v
            try:
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                  run_model=(runner or S1._Runner)(wd), budget=budget, seed=0)
            finally:
                for k in (env or {}):
                    os.environ.pop(k, None)
        hp = Path(wd, "eval_history.jsonl")
        hist = [json.loads(l) for l in hp.read_text().splitlines()] if hp.exists() else []
        return rep, hist, buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_three_seeds_by_default_and_the_returned_seed_is_the_lowest_calibration_loss():
    rep, hist, _ = _run()
    sd = rep["convergence"]["seeds"]
    assert sd["n_slots"] == 3 and [s["seed"] for s in sd["slots"]] == [0, 1, 2]
    losses = [s["incumbent_losses"][0] for s in sd["slots"]]
    best = min(range(3), key=lambda k: (losses[k], k))
    assert sd["returned"]["slot"] == best + 1
    # the reported result IS the returned seed's calibration incumbent
    assert rep["best_loss"] == pytest.approx(sd["slots"][best]["incumbent_losses"])
    assert rep["best_params"] == pytest.approx(sd["slots"][best]["incumbent_params"])
    assert rep["convergence"]["seed"] == sd["slots"][best]["seed"]
    # every search call is tagged with its seed; the pilot and the proof ran ONCE (shared overhead)
    assert sorted({h.get("seed") for h in hist if h["phase"] == "search"}) == [0, 1, 2]
    assert rep["phase_counts"]["pilot"]["n"] == 10
    assert set(sd["param_spread"]) == {"a", "b"}
    assert sd["run_verdict"] == rep["convergence"]["run_verdict"]["verdict"]


def test_validation_is_read_once_on_the_returned_result():
    three, _, _ = _run()
    one, _, _ = _run(budget_block={"seeds": 1})
    assert three["phase_counts"]["holdout"]["n"] == one["phase_counts"]["holdout"]["n"]
    st = three["convergence"]["verdict"]["standards"]["streamflow"]
    assert st["validation_member"].startswith("the reported result")


def test_one_seed_reports_seeds_not_checked():
    rep, _, _ = _run(budget_block={"seeds": 1})
    sd = rep["convergence"]["seeds"]
    assert sd["n_slots"] == 1 and sd["agreement"]["agree"] is None and "not checked" in sd["how"]
    assert sd["run_verdict"] == rep["convergence"]["verdict"]["verdict"]


class _Crash:
    """Wraps a backend: optimize() raises for the seeds in `bad`."""
    bad: set = set()

    def __init__(self, inner):
        self.inner = inner

    def available(self):
        return self.inner.available()

    def optimize(self, problem, budget, seed, **kw):
        if seed in self.bad:
            raise RuntimeError(f"planted crash at seed {seed}")
        return self.inner.optimize(problem, budget=budget, seed=seed, **kw)


def test_a_crashed_seed_is_replaced_once_by_a_new_number(monkeypatch):
    class B(_Crash):
        bad = {1}
    rep, hist, log = _run(backend_wrap=B, monkeypatch=monkeypatch)
    sd = rep["convergence"]["seeds"]
    assert [s["seed"] for s in sd["slots"]] == [0, 4, 2]           # slot 2: seed 1 crashed -> seed 0+3+1
    assert sd["slots"][1]["replaced"]["seed"] == 1 and "planted crash" in sd["slots"][1]["replaced"]["error"]
    assert rep["status"] == "completed" and "re-run once with seed 4" in log
    assert sorted({h.get("seed") for h in hist if h["phase"] == "search"}) == [0, 2, 4]


def test_a_replacement_that_also_crashes_leaves_the_slot_missing(monkeypatch):
    class B(_Crash):
        bad = {1, 4}
    rep, _, _ = _run(backend_wrap=B, monkeypatch=monkeypatch)
    sd = rep["convergence"]["seeds"]
    assert sd["slots"][1]["crashed"] and sd["slots"][1]["replaced"]["seed"] == 1
    assert sd["agreement"]["agree"] is not True                    # a missing slot is never agreement
    assert sd["run_verdict"] in (UNKNOWN, NOT_CONVERGED)


def test_every_seed_crashing_is_a_status_and_the_inputs_are_restored(monkeypatch):
    class B(_Crash):
        bad = {0, 1, 2, 3, 4, 5}
    from calibration_kit.evaluator import Evaluator as _Ev
    calls = []
    orig = _Ev.restore_originals
    monkeypatch.setattr(_Ev, "restore_originals", lambda self: (calls.append(1), orig(self))[1])
    rep, _, _ = _run(backend_wrap=B, monkeypatch=monkeypatch)
    assert rep["status"] == "search_crashed" and calls
    assert [s["crashed"] for s in rep["seeds"]["slots"]] == [True, True, True]


def test_seeds_must_be_a_positive_whole_number():
    for bad in (0, -1, "3", True, 2.5):
        rep, _, _ = _run(budget_block={"seeds": bad})
        assert rep["status"] == "invalid_contract" and "seeds" in rep["reason"], bad


def test_seeds_run_one_after_another_so_the_plan_counts_one_lane():
    rep, _, _ = _run(budget_block={"mode": "measured", "allowance": "10m", "seeds": 3}, budget=25)
    bp = rep["budget_plan"]
    assert bp["lanes"] == 1 and bp["waves"] == 3 and rep["convergence"]["seeds"]["run_one_after_another"]


def test_a_trade_off_search_reports_the_rule_compromise_not_the_backend_pick():
    strat = {"multi_objective": True, "default_algorithm": "nsga2"}
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 2}, None, 60)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strat)
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            os.environ["KDT_CALIB_FRONT_SELECT"] = "1"
            try:
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=None, seed=0)
            finally:
                os.environ.pop("KDT_CALIB_FRONT_SELECT", None)
        sd = rep["convergence"]["seeds"]
        k = sd["returned"]["slot"] - 1
        assert rep["best_loss"] == pytest.approx(sd["slots"][k]["incumbent_losses"])
        assert rep["best_params"] == pytest.approx(sd["slots"][k]["incumbent_params"])
        # the rule's final incumbent (its compromise) is what was returned
        assert sd["slots"][k]["incumbent_call"] == rep["convergence"]["rule"]["incumbent_final"]
        assert any("KDT_CALIB_FRONT_SELECT=1 is ignored" in w for w in rep["convergence"]["warnings"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_resumed_run_replays_every_seed_and_returns_the_same_result():
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m"}, None, 30)
        reps = []
        for _ in range(2):
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                reps.append(C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                        budget=None, seed=0))
        a, b = reps
        assert b["budget_plan"].get("resumed_cap") == a["budget_used"]["cap"]
        assert b["best_params"] == pytest.approx(a["best_params"])
        assert b["convergence"]["seeds"]["returned"] == a["convergence"]["seeds"]["returned"]
        assert [s["incumbent_losses"][0] for s in b["convergence"]["seeds"]["slots"]] == pytest.approx(
            [s["incumbent_losses"][0] for s in a["convergence"]["seeds"]["slots"]])
        assert b["budget_used"]["total_search_calls_all_seeds"] == 3 * a["budget_used"]["cap"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def test_the_reported_result_is_the_incumbent_even_when_the_backend_picks_another_point(monkeypatch):
    """A backend whose own best_x is some other point (here: its first call). The report must carry the
    rule's incumbent of the returned seed, never the backend's pick (design §1.5 option a)."""
    class Other(_Crash):
        bad = set()

        def optimize(self, problem, budget, seed, **kw):
            res = self.inner.optimize(problem, budget=budget, seed=seed, **kw)
            res.best_x = list(res.history[0]["x"])
            res.best_loss = [float(res.history[0]["loss"])]
            return res
    rep, _, _ = _run(backend_wrap=Other, monkeypatch=monkeypatch)
    sd = rep["convergence"]["seeds"]
    k = sd["returned"]["slot"] - 1
    assert sd["slots"][k]["incumbent_call"] != 0
    assert rep["best_loss"] == pytest.approx(sd["slots"][k]["incumbent_losses"])
    assert rep["best_params"] == pytest.approx(sd["slots"][k]["incumbent_params"])



# ── the seed wiring checked against the call log (Opus step 6 r1, finding 3) ─────────────────────
def _seed_rows(hist, sd):
    return {s["seed"]: [h for h in hist if h["phase"] == "search" and h.get("seed") == s["seed"]]
            for s in sd["slots"] if not s["crashed"]}


def _pick_by_hand(cands, multi):
    """§1.7 written out again, independently of seeds.py: the returned slot."""
    if not multi:
        return min(cands, key=lambda c: (c["scalar"], c["slot"]))["slot"]
    scaled = sorted((c for c in cands if c["t0"] is not None), key=lambda c: c["slot"])
    if scaled:
        z, sc = scaled[0]["z"], scaled[0]["s"]
        norm = lambda c: [(f - a) / b for f, a, b in zip(c["losses"], z, sc)]      # noqa: E731
        key = lambda c: (0 if c["admissible"] else 1, max(norm(c)), sum(norm(c)), c["slot"])  # noqa: E731
    else:
        key = lambda c: (0 if c["admissible"] else 1, sum(c["losses"]), c["slot"])  # noqa: E731
    return min(cands, key=key)["slot"]


def _check_against_log(rep, hist):
    sd = rep["convergence"]["seeds"]
    rows = _seed_rows(hist, sd)
    for s in sd["slots"]:
        if s["crashed"]:
            continue
        r = rows[s["seed"]]
        assert len(r) == s["calls"]
        row = r[s["incumbent_call"]]
        assert [float(x) for x in row["losses"]] == pytest.approx(s["incumbent_losses"])
        assert {k: float(v) for k, v in row["x"].items()} == pytest.approx(s["incumbent_params"])
        for v, mm in (s["incumbent_panel"] or {}).items():
            for m, val in mm.items():
                assert val == pytest.approx(row["panel"][v][m])
        # the incumbent is the lowest scalar of that seed's own calls (single objective)
        if s["incumbent_scalar"] is not None:
            ok = [float(h["losses"][0]) for h in r if h["ok"]] if len(s["incumbent_losses"]) == 1 else None
            if ok:
                assert s["incumbent_scalar"] == pytest.approx(min(ok))
    # the agreement, the verdicts across seeds and the parameter spread follow from the slots themselves
    cv = rep["convergence"]
    tol = {v: dict((r or {}).get("tol") or {}) for v, r in (cv.get("tolerance_records") or {}).items()}
    live = [s for s in sd["slots"]]
    ag = seed_agreement([(s["incumbent_panel"] if (not s["crashed"] and s["incumbent_call"] is not None) else None)
                         for s in live], cv["required_panel"], tol)
    got = sd["agreement"]
    assert (ag["agree"], ag["exceeds"], ag["missing"]) == (got["agree"], got["exceeds"], got["missing"])
    for v, pv in ag["per_variable"].items():
        if isinstance(pv, dict):
            assert pv["spread"] == pytest.approx(got["per_variable"][v]["spread"])
    ac = across_seeds([s["verdict"] for s in live], [s["variables"] for s in live], got)
    assert (ac["run_verdict"], ac["how"], ac["variables"]) == (sd["run_verdict"], sd["how"],
                                                               sd["variables_across_seeds"])
    assert sd["param_spread"] == pytest.approx(param_spread([s["incumbent_params"] for s in live
                                                             if s["incumbent_params"] is not None]))
    # the returned seed can be picked again from the report alone (§1.7), with the same reason
    multi = bool(rep.get("multi_objective"))
    cands = [{"slot": s["slot"], "scalar": s["incumbent_scalar"], "losses": s["incumbent_losses"],
              "admissible": (s["incumbent_admissible"] if s["incumbent_admissible"] is not None else True),
              "t0": s.get("frozen_t0"), "z": s.get("frozen_offset"), "s": s.get("frozen_scale")}
             for s in live if not s["crashed"] and s["incumbent_call"] is not None]
    i, why = pick_returned_seed(cands, trade_off=multi)
    assert cands[i]["slot"] == sd["returned"]["slot"] and why == sd["returned"]["why"]
    assert _pick_by_hand(cands, multi) == sd["returned"]["slot"]      # an independent reading of §1.7
    if multi:                     # the reason's protection prefix follows the slots' admissibility
        adm = [c["admissible"] for c in cands]
        w = sd["returned"]["why"]
        if not any(adm):
            assert w.startswith("no seed is admissible under protection")
            assert cv["rule"]["protection_infeasible_final"] is True
        elif not all(adm):
            assert w.startswith("admissible under protection, then") and "protection infeasible" not in w
            assert sd["slots"][sd["returned"]["slot"] - 1]["incumbent_admissible"] is True
        else:
            assert "admissible" not in w
    if multi:                     # the returned slot's recorded scale is its rule's frozen scale
        k = sd["returned"]["slot"] - 1
        ru = cv["rule"]
        assert (sd["slots"][k]["frozen_t0"], sd["slots"][k]["frozen_offset"], sd["slots"][k]["frozen_scale"]) == \
            (ru["t0"], ru["frozen_offset"], ru["frozen_scale"])
    # the seed tag is off outside the search
    assert all("seed" not in h for h in hist if h["phase"] != "search")
    assert sd["total_search_calls"] == rep["phase_counts"]["search"]["n"]


def test_every_slot_matches_its_own_rows_in_the_call_log_and_a_single_seed_run():
    rep, hist, _ = _run(max_evaluations=40)
    _check_against_log(rep, hist)
    sd = rep["convergence"]["seeds"]
    # each slot is the same search a one-seed run with that seed makes: same incumbent, same verdicts
    for s in sd["slots"]:
        tmp = tempfile.mkdtemp(prefix="kdt_s6_")
        try:
            ki, wd = E._fixture(tmp, {"seeds": 1}, None, 40)
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                one = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=None, seed=s["seed"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        o = one["convergence"]["seeds"]["slots"][0]
        assert o["incumbent_call"] == s["incumbent_call"]
        assert o["incumbent_losses"] == pytest.approx(s["incumbent_losses"])
        assert isinstance(s["variables"], dict) and s["variables"]          # each seed keeps its own
        assert o["verdict"] == s["verdict"] and o["variables"] == s["variables"]


def test_a_weighted_two_family_scalar_is_what_the_pick_compares():
    strat = {"default_algorithm": "dds"}
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, None, None, 40)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strat)
        c["targets"] = [{"var": "streamflow", "weight": 1.0,
                         "family_weights": {"temporal_pattern_match": 1.0, "magnitude_accuracy": 0.25}}]
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not rep["multi_objective"] and len(rep["objectives"]) == 2
    sd = rep["convergence"]["seeds"]
    _check_against_log(rep, hist)
    sc = [s["incumbent_scalar"] for s in sd["slots"]]
    k = min(range(3), key=lambda i: (sc[i], i))
    assert sd["returned"]["slot"] == k + 1 and f"slot {k + 1}" in sd["returned"]["why"]
    assert rep["best_loss"] == pytest.approx([sc[k]])                  # the same form as before: [scalar]
    assert rep["best_losses_per_objective"] == pytest.approx(sd["slots"][k]["incumbent_losses"])
    w = [o.weight for o in __import__("calibration_kit.objectives", fromlist=["x"]).objectives_from_dag(
        {"outputs": [{"var": "streamflow", "observability": {"comparable_obs_shapes": [
            {"obs_shape": "point_time_series", "metric_families": ["temporal_pattern_match", "magnitude_accuracy"]}]}}]},
        [{"var": "streamflow", "weight": 1.0,
          "family_weights": {"temporal_pattern_match": 1.0, "magnitude_accuracy": 0.25}}],
        {"streamflow": "point_time_series"})]
    lo = sd["slots"][k]["incumbent_losses"]
    assert sc[k] == pytest.approx(sum(a * b for a, b in zip(w, lo)) / sum(w))


def test_a_mid_search_crash_is_counted_and_replaced(monkeypatch):
    class Mid(_Crash):
        bad = {1}

        def optimize(self, problem, budget, seed, **kw):
            if seed in self.bad:
                hook = kw.get("on_eval")

                def boom(history):
                    if len(history) >= 15:
                        raise RuntimeError("planted crash after 15 calls")
                    return hook(history) if hook else False
                kw = dict(kw, on_eval=boom)
            return self.inner.optimize(problem, budget=budget, seed=seed, **kw)
    rep, hist, log = _run(backend_wrap=Mid, monkeypatch=monkeypatch)
    sd = rep["convergence"]["seeds"]
    assert sd["slots"][1]["replaced"]["seed"] == 1 and sd["slots"][1]["replaced"]["calls"] == 15
    assert len([h for h in hist if h["phase"] == "search" and h.get("seed") == 1]) == 15
    assert sd["total_search_calls"] == rep["phase_counts"]["search"]["n"] == 3 * 40 + 15
    assert any("re-run once with seed 4" in w for w in rep["budget_plan"]["warnings"])
    _check_against_log(rep, hist)


def test_a_trade_off_search_reports_the_rule_compromise_when_the_backend_picks_another(monkeypatch):
    class Other(_Crash):
        bad = set()

        def optimize(self, problem, budget, seed, **kw):
            res = self.inner.optimize(problem, budget=budget, seed=seed, **kw)
            res.best_x = list(res.history[0]["x"])
            return res
    strat = {"multi_objective": True, "default_algorithm": "nsga2"}
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 2}, None, 60)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strat)
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        real = C._make_backend
        monkeypatch.setattr(C, "_make_backend", lambda algo: Other(real(algo)))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sd = rep["convergence"]["seeds"]
    k = sd["returned"]["slot"] - 1
    assert sd["slots"][k]["incumbent_call"] != 0
    assert rep["best_params"] == pytest.approx(sd["slots"][k]["incumbent_params"])
    assert rep["best_loss"] == pytest.approx(sd["slots"][k]["incumbent_losses"])   # per objective
    _check_against_log(rep, hist)


def test_a_backend_that_does_not_report_each_call_runs_one_seed():
    rep, _, log = _run(strategy={"default_algorithm": "surrogate"}, max_evaluations=14)
    if rep.get("status") == "backend_unavailable":
        pytest.skip("surrogate backend not installed")
    sd = rep["convergence"]["seeds"]
    assert sd["n_slots"] == 1 and "not compared" in sd["note"] and "ONE seed" in log


def test_a_plan_saved_for_another_number_of_seeds_is_not_resumed():
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "seeds": 1}, None, 30)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"]["budget"]["seeds"] = 3
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        buf = io.StringIO()
        with E._clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        assert "resumed_cap" not in rep["budget_plan"] and "DIFFERENT setup" in buf.getvalue()
        assert rep["budget_plan"]["seeds"] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_kit_missing_variable_is_labelled_as_such():
    ag = seed_agreement([{}, {}], {"Q": []}, {}, not_compared_why={"Q": "its required metrics are set missing"})
    assert ag["per_variable"]["Q"] == "not compared (its required metrics are set missing)"


def _fake_clock(monkeypatch):
    import types
    import time as _t
    clock = {"t": 1000.0}
    monkeypatch.setattr(C, "time", types.SimpleNamespace(perf_counter=lambda: clock["t"], time=_t.time,
                                                         sleep=_t.sleep))

    def factory(wd):
        inner = S1._Runner(wd)

        def run():
            clock["t"] += 10.0
            return inner()
        return run
    return factory


def test_the_seeds_total_search_time_is_checked_against_the_search_allowance(monkeypatch):
    """Fake clock: every model call takes 10 s on the kit's clock. Three seeds x 30 calls = 900 s of
    search against a search allowance of about 300 s: the plan says so once all seeds are done."""
    factory = _fake_clock(monkeypatch)
    rep, hist, _ = _run(budget_block={"mode": "measured", "allowance": "300s", "pilot_runs": 3}, budget=30,
                        runner=factory)
    ran = len([h for h in hist if h["phase"] == "search" and not h["cache_hit"]])   # cache hits run nothing
    assert ran > 60
    w = [x for x in rep["budget_plan"]["warnings"] if "the seeds' searches took" in x]
    assert len(w) == 1 and f"took {10 * ran} s" in w[0]
    assert rep["convergence"]["seeds"]["search_wall_s"] == pytest.approx(10.0 * ran)



def test_a_plan_saved_by_the_old_kit_for_one_seed_is_re_planned_with_the_same_contract():
    """The default went 1 -> 3 without the contract changing: a workdir planned by the pre-step-6 kit
    (whose plan identity had no seed count) must be re-planned, not resumed at a 1-seed cap."""
    import hashlib
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m"}, None, 30)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
        plan_f = Path(wd, "kdt_budget_plan.json")
        plan = json.loads(plan_f.read_text())
        contract = yaml.safe_load(Path(ki, "calibration.yaml").read_text())

        def ident(extra, with_runner_code=True):
            h = hashlib.sha1()
            h.update(json.dumps(contract, sort_keys=True, default=str).encode())
            for part in ("", "", "a,b", "runner") + extra:
                h.update(part.encode())
            if with_runner_code:                # 3c (2026-10-02): the runner's own code is part of the identity
                rs = contract.get("runner") or {}
                rcwd = str(rs.get("cwd") or "{ki_path}").replace("{ki_path}", str(ki))
                for tok in (rs.get("command") or []):
                    t = str(tok).replace("{ki_path}", str(ki))
                    if t.endswith(".py"):
                        f = Path(t) if os.path.isabs(t) else Path(rcwd) / t
                        h.update(b"runner:" + (f.read_bytes() if f.is_file() else b"<missing>"))
            return h.hexdigest()
        assert plan["setup_id"] == ident(("seeds=3",))                  # the formula is the kit's
        plan["setup_id"] = ident((), with_runner_code=False)            # ... as the old (pre-step-6) kit wrote it
        plan_f.write_text(json.dumps(plan))
        buf = io.StringIO()
        with E._clean_env(), contextlib.redirect_stdout(buf):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        assert "DIFFERENT setup" in buf.getvalue() and "resumed_cap" not in rep["budget_plan"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def test_a_crashed_attempts_time_counts_and_a_run_inside_its_allowance_is_not_warned(monkeypatch):
    """Fake clock, 10 s per model call, cap 30, a ~990 s search allowance for all seeds (a 330 s share each).
    Without a crash about 880 s run: inside the TOTAL allowance -> no warning (a per-seed comparison would
    wrongly warn). With seed 1 crashing after 26 calls, the crashed attempt's time pushes it past."""
    class Mid(_Crash):
        bad = {1}

        def optimize(self, problem, budget, seed, **kw):
            if seed in self.bad:
                hook = kw.get("on_eval")

                def boom(history):
                    if len(history) >= 26:
                        raise RuntimeError("planted crash after 26 calls")
                    return hook(history) if hook else False
                kw = dict(kw, on_eval=boom)
            return self.inner.optimize(problem, budget=budget, seed=seed, **kw)
    block = {"mode": "measured", "allowance": "990s", "pilot_runs": 3}
    factory = _fake_clock(monkeypatch)
    calm, hist0, _ = _run(budget_block=block, budget=30, runner=factory)
    ran0 = len([h for h in hist0 if h["phase"] == "search" and not h["cache_hit"]])
    assert 10 * ran0 < calm["budget_plan"]["ledger"]["search_s"]
    assert not any("the seeds' searches took" in w for w in calm["budget_plan"]["warnings"])
    factory = _fake_clock(monkeypatch)
    rep, hist, _ = _run(budget_block=block, budget=30, runner=factory, backend_wrap=Mid, monkeypatch=monkeypatch)
    ran = len([h for h in hist if h["phase"] == "search" and not h["cache_hit"]])
    assert 10 * ran > rep["budget_plan"]["ledger"]["search_s"]
    w = [x for x in rep["budget_plan"]["warnings"] if "the seeds' searches took" in x]
    assert len(w) == 1 and f"took {10 * ran} s" in w[0]


def test_protection_decides_the_returned_seed_in_a_trade_off_search():
    """NSGA-II, protect streamflow alpha at the default a = 1.0 (alpha = 1): the returned seed's compromise
    must be admissible whenever any seed's is, and the reason says so when one was passed over."""
    seen_pass_over = False
    for cap, base in ((60, 21), (40, 18)):
        tmp = tempfile.mkdtemp(prefix="kdt_s6_")
        try:
            ki, wd = E._fixture(tmp, None, None, cap)
            c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
            c["parameters"][0]["default"] = 1.0
            c["strategy"].update({"multi_objective": True, "default_algorithm": "nsga2",
                                  "protect": {"streamflow": ["alpha"]}})
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
            d = yaml.safe_load(open(Path(ki, "dag.yaml")))
            d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
                "temporal_pattern_match", "magnitude_accuracy"]
            yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=None, seed=base)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        sd = rep["convergence"]["seeds"]
        adm = [s["incumbent_admissible"] for s in sd["slots"]]
        assert all(a in (True, False) for a in adm)
        k = sd["returned"]["slot"] - 1
        if any(adm):
            assert adm[k] is True
        if any(adm) and not all(adm):
            assert sd["returned"]["why"].startswith("admissible under protection, then")
            assert "protection infeasible" not in sd["returned"]["why"]
            losses = [s["incumbent_losses"] for s in sd["slots"]]
            seen_pass_over = seen_pass_over or any(
                not adm[j] and max(losses[j]) < max(losses[k]) for j in range(3))
    assert seen_pass_over, "the fixture must pass over a lower-loss seed that breaks protection"


def test_a_hand_edited_saved_ledger_never_crashes_the_run():
    for bad in ("990s", "n/a", [990]):
        tmp = tempfile.mkdtemp(prefix="kdt_s6_")
        try:
            ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m"}, None, 20)
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
            pf = Path(wd, "kdt_budget_plan.json")
            pl = json.loads(pf.read_text())
            pl["ledger"]["search_s"] = bad
            pl["ledger"]["per_seed_search_s"] = bad
            pf.write_text(json.dumps(pl))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=None, seed=0)
            assert rep["status"] == "completed", bad
        finally:
            shutil.rmtree(tmp, ignore_errors=True)



def _nsga_protect_run(cap, base, protect=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s6_")
    try:
        ki, wd = E._fixture(tmp, None, None, cap)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["parameters"][0]["default"] = 1.0
        c["strategy"].update({"multi_objective": True, "default_algorithm": "nsga2"})
        if protect:
            c["strategy"]["protect"] = protect
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
            "temporal_pattern_match", "magnitude_accuracy"]
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=base)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        return rep, hist
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_trade_off_pick_uses_the_lowest_slots_frozen_scale_and_can_be_redone_from_the_report():
    """NSGA-II, 3 seeds, cap 60, default a = 1.0, base seeds 4 and 11: runs where the scale decides the
    pick. The report carries every slot's frozen scale; redoing the pick from it gives the same slot and
    reason (_check_against_log), and on at least one of them the returned seed's OWN scale would pick
    another slot."""
    decided_by_scale = False
    for base in (4, 11):
        rep, hist = _nsga_protect_run(60, base)
        _check_against_log(rep, hist)
        sd = rep["convergence"]["seeds"]
        low = min((s for s in sd["slots"] if s["frozen_t0"] is not None), key=lambda s: s["slot"])
        assert f"frozen scale of slot {low['slot']}" in sd["returned"]["why"]
        k = sd["returned"]["slot"] - 1
        own = sd["slots"][k]
        if own["frozen_t0"] is not None:
            alt = [{"slot": s["slot"], "losses": s["incumbent_losses"], "admissible": True,
                    "t0": own["frozen_t0"], "z": own["frozen_offset"], "s": own["frozen_scale"]}
                   for s in sd["slots"]]
            j, _ = pick_returned_seed(alt, trade_off=True)
            decided_by_scale = decided_by_scale or alt[j]["slot"] != sd["returned"]["slot"]
    assert decided_by_scale, "one of these runs must be decided by which slot's scale is used"


def test_when_no_seed_is_admissible_the_result_is_never_called_protected():
    rep, hist = _nsga_protect_run(60, 18, protect={"streamflow": ["alpha"]})
    sd = rep["convergence"]["seeds"]
    assert [s["incumbent_admissible"] for s in sd["slots"]] == [False, False, False]
    assert sd["returned"]["why"].startswith("no seed is admissible under protection")
    assert "admissible under protection, then" not in sd["returned"]["why"]
    assert rep["convergence"]["rule"]["protection_infeasible_final"] is True
    _check_against_log(rep, hist)


def test_the_protection_text_counts_only_seeds_the_pick_can_return():
    cs = [{"slot": 1, "losses": [0.2, None], "admissible": True, "t0": None, "z": None, "s": None},
          {"slot": 2, "losses": [0.3, 0.3], "admissible": False, "t0": None, "z": None, "s": None}]
    i, why = pick_returned_seed(cs, trade_off=True)
    assert cs[i]["slot"] == 2 and why.startswith("no seed is admissible under protection")


def test_every_trade_off_slots_frozen_scale_is_its_own_one_seed_rule():
    rep, _ = _nsga_protect_run(60, 4)
    for s in rep["convergence"]["seeds"]["slots"]:
        tmp = tempfile.mkdtemp(prefix="kdt_s6_")
        try:
            ki, wd = E._fixture(tmp, {"seeds": 1}, None, 60)
            c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
            c["parameters"][0]["default"] = 1.0
            c["strategy"].update({"multi_objective": True, "default_algorithm": "nsga2"})
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
            d = yaml.safe_load(open(Path(ki, "dag.yaml")))
            d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = [
                "temporal_pattern_match", "magnitude_accuracy"]
            yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                one = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=None, seed=s["seed"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ru = one["convergence"]["rule"]
        assert (s["frozen_t0"], s["frozen_offset"], s["frozen_scale"]) == (ru["t0"], ru["frozen_offset"],
                                                                           ru["frozen_scale"])
