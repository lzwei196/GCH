"""Test 9 (handoff §5.14): the phase-tagged evaluation history.

The eval CACHE is keyed by a hash and holds only successful evals, so it cannot say
where a search converged: no order, no phase, no parameter vector, no failures. The
history log must answer all four. These tests pin exactly that, with a fake runner —
no model binary, no network, milliseconds.
"""
import json
import math
import os
from pathlib import Path

import pytest

from calibration_kit.evaluator import Evaluator, split_authority
from calibration_kit.objectives import Objective


def _obj(metric="nse", family="temporal_pattern_match", var="Q"):
    return Objective(name=f"{var}:{family}", var=var, family=family, metric_key=metric)


def _make(tmp_path, *, fail_on=None):
    """Evaluator over a fake runner: loss = (a-2)^2/10, echoes applied_params like calib_run.py."""
    wd = str(tmp_path)
    params = [{"name": "a", "address": "none", "default": 1.0, "range": [0.0, 5.0]}]

    def run():
        named = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
        a = float(named["a"])
        if fail_on is not None and abs(a - fail_on) < 1e-9:
            return {}                                   # runner produced nothing
        return {"nse": 1.0 - (a - 2.0) ** 2 / 10.0,
                "pbias": 5.0 * (a - 2.0), "kge": 0.8, "r": 0.9,
                "__kdt__": {"applied_params": dict(named)}}

    ev = Evaluator(ki_path=wd, workdir=wd, parameters=params, objectives=[_obj()],
                   transform_inv={}, run_model=run, injection_mode="runner")
    ev.panel_extract = lambda m: {"Q": {k: m.get(k) for k in ("nse", "pbias", "kge", "r")}}
    return ev


def _records(ev):
    return [json.loads(l) for l in Path(ev.workdir, "eval_history.jsonl").read_text().splitlines()]


def test_history_records_phase_split_and_decoded_x(tmp_path):
    ev = _make(tmp_path)
    with ev.phase_as("pilot"):
        ev.evaluate([1.0])
        ev.evaluate([3.0])
    with ev.phase_as("search"):
        ev.evaluate([2.0])
    # the split must come from the real authority path, not a raw env var (an inherited
    # KDT_CALIB_SPLIT is deliberately NOT authoritative) — the log records what was scored.
    with split_authority("holdout"), ev.phase_as("holdout"):
        ev.evaluate([2.0])

    recs = _records(ev)
    assert [r["phase"] for r in recs] == ["pilot", "pilot", "search", "holdout"]
    assert [r["i"] for r in recs] == [0, 1, 2, 3]          # ORDER, which the cache does not have
    assert [r["x"]["a"] for r in recs] == [1.0, 3.0, 2.0, 2.0]   # the DECODED vector
    assert [r["split"] for r in recs] == ["calibration"] * 3 + ["holdout"]
    assert all(r["ok"] for r in recs)
    # the holdout eval is a different split -> a different cache key -> not a hit
    assert [r["cache_hit"] for r in recs] == [False, False, False, False]
    assert recs[2]["losses"] == [0.0]                       # a=2 is the optimum
    assert recs[0]["panel"]["Q"]["pbias"] == pytest.approx(-5.0)
    assert all(r["wall_s"] >= 0.0 and r["t"] > 0 for r in recs)


def test_history_records_cache_hits_and_failures(tmp_path):
    ev = _make(tmp_path, fail_on=4.0)
    with ev.phase_as("search"):
        ev.evaluate([1.5])
        ev.evaluate([1.5])          # same vector, same split -> cache hit
        ev.evaluate([4.0])          # runner returns no metrics -> +inf, NOT cached
        ev.evaluate([4.0])          # so this is a fresh failure, not a hit

    recs = _records(ev)
    assert len(recs) == 4                       # FAILURES are logged; the cache holds none of them
    assert [r["cache_hit"] for r in recs] == [False, True, False, False]
    assert [r["ok"] for r in recs] == [True, True, False, False]
    assert [r.get("reason") for r in recs][2:] == ["no_metrics", "no_metrics"]
    assert recs[2]["losses"] == ["inf"]         # non-finite losses stay JSON-safe
    assert recs[0]["losses"] == recs[1]["losses"]


def test_history_records_infeasible_before_any_run(tmp_path):
    ev = _make(tmp_path)
    ev.constraints_ok = lambda named: named["a"] <= 2.5
    with ev.phase_as("screen"):
        assert ev.evaluate([4.0]) == [float("inf")]
    r = _records(ev)[0]
    assert r["phase"] == "screen" and r["reason"] == "infeasible" and r["ok"] is False
    # the split the call was FOR is recorded even though no run happened (Opus 8b/9 r2 #1)
    assert r["x"]["a"] == 4.0 and r["split"] == "calibration"


def test_phase_as_nests_and_restores(tmp_path):
    ev = _make(tmp_path)
    assert ev.phase == "untagged"               # default (build step 7): never counted as search effort
    with ev.phase_as("pilot"):
        with ev.phase_as("commission"):
            ev.evaluate([1.0])
        ev.evaluate([1.1])
    assert ev.phase == "untagged"
    assert [r["phase"] for r in _records(ev)] == ["commission", "pilot"]


def test_history_never_breaks_a_run(tmp_path):
    """The log is a side effect: a logging fault must not change a loss or raise."""
    ev = _make(tmp_path)
    ev.panel_extract = lambda m: (_ for _ in ()).throw(RuntimeError("bad extractor"))
    assert ev.evaluate([2.0]) == [0.0]
    assert _records(ev)[0]["panel"] == {}
    ev._hist_file = Path(tmp_path, "no_such_dir", "x.jsonl")   # unwritable
    assert ev.evaluate([1.0])[0] == pytest.approx(0.1)
