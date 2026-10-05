"""Behavioural regressions through the REAL entry points (plan checkpoint 2026-09-16, codex+kimi).

The v2 acceptance suite exercises Evaluator.consumption_proof in isolation. Both seats showed that
was not enough: the wiring around it had three defects the unit fixtures could not see. These tests
run calibrate() end-to-end on a tiny fixture KI with a fake runner. (Staged calibration was removed in
build step 8: a `staged: true` contract runs ALL its parameters in one search, with a warning.)

  P1  objective-masked box: raw output moves, every ROUNDED objective bitwise flat. Before the fix the
      objective smoke test returned runner_unresponsive BEFORE the raw-series gate. Now: warned, never a
      veto; the raw-series proof decides — every parameter OBJECTIVE_MASKED, so the triage route is
      `not_calibratable` (design §1, build step 8). A staged contract behaves the same.
  P2  dead knob: applied-echo OK, raw output identical -> STOP (param_unreachable), proof in the result.
      A staged contract behaves the same.
  P3  a `staged: true` contract runs one search over all its declared parameters, with a warning.
  P4  force-live: with _force_live the evaluator must NOT serve a preloaded cache entry (replaces the
      v2 test_n, which only grepped the source for a substring).
  P5  the completed report carries consumption_proof + consumption_coverage + objective_smoke_test.

KNOWN GAPS (asserted as the INTENDED behaviour; reported as KNOWN-GAP until plan step C lands; the
runner FAILS if one of them unexpectedly passes so the marker is flipped, never forgotten):
  G1  endpoint-insensitive consumed knob (astra): y = 1 + x(1-x), default 0, range [0,1]. The far
      bound x=1 returns y(0) bitwise. Today: UNREACHABLE (a STOP that claims a wrong address). Intended
      after the interior probe (C11): not UNREACHABLE.
  G2  run-to-run noise (astra): a knob the model ignores, output jitters 1e-9 between runs. Today:
      INSENSITIVE (acquitted). Intended after the repeat-default noise floor (C10): not acquitted.
"""
import hashlib, json, os, shutil, sys, tempfile, io, contextlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np
import yaml

from calibration_kit import calib as C
from calibration_kit.evaluator import Evaluator, resolve_train_split
from calibration_kit.objectives import Objective

N = 60
IDX = [f"1981-01-{d+1:02d}" if d < 31 else f"1981-02-{d-30:02d}" for d in range(N)]
BASE = 100.0 + 30.0 * np.sin(np.linspace(0, 6.28, N))


def _proof(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"schema": "kdt-target-proof/1", "target": "streamflow", "unit": "m3/s", "split": "calibration",
            "window": [IDX[0], IDX[-1]], "n": N, "dtype": "float64",
            "index_hash": hashlib.sha1("\n".join(IDX).encode()).hexdigest(),
            "values_hash": hashlib.sha1(v.tobytes()).hexdigest(),
            "values": [float(x) for x in v], "index": IDX, "source": "fixture"}


def _read_params():
    return json.loads(open(os.environ["KDT_CALIB_PARAMS"]).read())


def _metrics(nse, series, params):
    return {"nse": nse, "n": N, "__kdt__": {"applied_params": dict(params), "case_id": "SITE:fixture",
                                            "target_proofs": [_proof(series)]}}


class Runner:
    """A fake KI runner. `mode` picks the model:
       masked : raw = BASE + zero-mean perturbation from a and b; nse ALWAYS 0.5 (rounded flat)
       dead_b : raw depends on a only; b echoed as applied but never read; nse from raw
       alive  : raw depends on a and b; nse from raw
       endpoint_insensitive : raw = BASE + a(1-a); a in [0,1], default 0
       noisy_dead : raw = BASE + 1e-9 * call-count; a ignored
    """
    def __init__(self, mode):
        self.mode = mode; self.calls = 0

    def __call__(self):
        self.calls += 1
        p = _read_params(); a = float(p["a"]); b = float(p.get("b", 0.0))
        if self.mode == "masked":
            z = np.zeros(N); z[0] = +5.0 * (a + b); z[1] = -5.0 * (a + b)     # raw moves, mean does not
            s = BASE + z; nse = 0.5
        elif self.mode == "dead_b":
            s = BASE * (1.0 + 0.1 * a); nse = round(float(1.0 - abs(a - 0.3)), 4)
        elif self.mode == "alive":
            s = BASE * (1.0 + 0.1 * a) + 2.0 * b; nse = round(float(0.1 + 0.6 * a + 0.2 * b), 4)   # poor at defaults
        elif self.mode == "endpoint_insensitive":
            s = BASE + a * (1.0 - a); nse = round(float(0.5 + a * (1.0 - a)), 4)
        elif self.mode == "noisy_dead":
            s = BASE + 1e-9 * self.calls; nse = 0.5
        else:
            raise ValueError(self.mode)
        return _metrics(nse, s, p)


def _fixture_ki(staged=False, params=None, budget=6):
    d = tempfile.mkdtemp(prefix="kdt_ckpt_")
    ki = os.path.join(d, "ki"); wd = os.path.join(d, "wd"); os.makedirs(ki); os.makedirs(wd)
    params = params or [
        {"name": "a", "range": [0.0, 1.0], "default": 0.3, "type": "continuous"},
        {"name": "b", "range": [0.0, 1.0], "default": 0.2, "type": "continuous"},
    ]
    contract = {
        "injection": {"mode": "runner"},
        "parameters": params,
        "targets": [{"var": "streamflow", "weight": 1.0}],
        "strategy": {"max_evaluations": budget, "default_algorithm": "dds", "probe_objectives": True,
                     "staged": bool(staged), "screen_trajectories": 2, "min_round_evaluations": 1, "staged_start": 1, "staged_step": 1},
        "runner": {"command": ["python", "tools/calib_run.py"]},
    }
    yaml.safe_dump(contract, open(os.path.join(ki, "calibration.yaml"), "w"))
    dag = {"outputs": [{"var": "streamflow", "observability": {"comparable_obs_shapes": [
        {"obs_shape": "point_time_series", "metric_families": ["temporal_pattern_match"]}]}}]}
    yaml.safe_dump(dag, open(os.path.join(ki, "dag.yaml"), "w"))
    return d, ki, wd


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _run(mode, staged=False, params=None, budget=6):
    d, ki, wd = _fixture_ki(staged=staged, params=params, budget=budget)
    r = Runner(mode); buf = io.StringIO()
    try:
        with _env(KDT_CALIB_CONTRACT=None, KDT_CALIB_C8="enforce", KDT_CALIB_SPLIT=None, KDT_CALIB_PARAMS=None):
            with contextlib.redirect_stdout(buf):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=r, budget=budget, seed=0)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return rep, r, buf.getvalue()


# ------------------------------------------------------------------------------------------ P1
def test_p1_objective_masked_box_is_not_killed_by_smoke_test_calibrate():
    rep, r, log = _run("masked")
    assert rep.get("status") != "runner_unresponsive", rep
    assert "WARNING objective smoke test" in log            # warned, never a veto
    # the raw-series proof decides: every searched parameter OBJECTIVE_MASKED -> not_calibratable
    assert rep.get("status") == "not_calibratable", (rep.get("status"), rep.get("reason"))
    tr = rep["triage"]
    assert tr["route"] == "not_calibratable" and tr["proof"]["verdicts"] == {"a": "OBJECTIVE_MASKED",
                                                                              "b": "OBJECTIVE_MASKED"}
    assert "each parameter moved alone from the defaults" in tr["reason"] and tr["sensitivity_not_proven"]
    assert "a: OBJECTIVE_MASKED" in tr["reason"] and "b: OBJECTIVE_MASKED" in tr["reason"]
    # the no-search report keeps the proof's evidence (Opus 8a r1 #7 / r2 #2)
    rows = {r["param"]: r for r in rep["consumption_proof"]["params"]}
    assert set(rows) == {"a", "b"} and all(r.get("why") and "targets" in r for r in rows.values())
    assert rep["consumption_coverage"]["by_verdict"] == {"OBJECTIVE_MASKED": ["a", "b"]}
    assert rep["objective_smoke_test"]["responsive"] is False


def test_p1_objective_masked_box_is_not_killed_by_smoke_test_staged():
    rep, r, log = _run("masked", staged=True)
    assert rep.get("status") == "not_calibratable", (rep.get("status"), rep.get("reason"))
    assert "strategy.staged is removed" in log and rep["staged_warning"]


# ------------------------------------------------------------------------------------------ P2
def test_p2_dead_knob_stops_calibrate_with_proof_in_result():
    rep, r, log = _run("dead_b")
    assert rep.get("status") == "param_unreachable", (rep.get("status"), rep.get("reason"))
    assert rep["unreachable"] == ["b"], rep["unreachable"]
    assert rep["consumption_proof"]["ok"] is False


def test_p2_dead_knob_stops_staged_before_any_round():
    rep, r, log = _run("dead_b", staged=True)
    assert rep.get("status") == "param_unreachable", (rep.get("status"), rep.get("reason"))
    assert rep["unreachable"] == ["b"] and "strategy.staged is removed" in log


# ------------------------------------------------------------------------------------------ P3
def test_p3_a_staged_contract_runs_one_search_over_all_its_parameters():
    rep, r, log = _run("alive", staged=True, budget=8)
    assert not hasattr(C, "calibrate_staged")                  # the escalation path is gone
    assert rep.get("status") == "completed", (rep.get("status"), rep.get("reason"))
    assert "strategy.staged is removed" in log and rep["staged_warning"]
    assert set(rep["best_params"]) == {"a", "b"}                # every declared parameter searched
    assert rep["triage"]["route"] == "calibrate"


# ------------------------------------------------------------------------------------------ P4
def test_p4_force_live_bypasses_preloaded_cache_behaviourally():
    d, ki, wd = _fixture_ki()
    r = Runner("alive")
    try:
        ev = Evaluator(ki_path=ki, workdir=wd,
                       parameters=[{"name": "a", "range": [0.0, 1.0], "default": 0.3, "type": "continuous"},
                                   {"name": "b", "range": [0.0, 1.0], "default": 0.2, "type": "continuous"}],
                       objectives=[Objective(name="nse", var="streamflow", family="temporal_pattern_match",
                                             metric_key="nse")],
                       transform_inv={}, run_model=r, injection_mode="runner")
        x = [0.3, 0.2]
        named = ev._named(x)
        stale = _metrics(0.1234, BASE, named)              # a stale-but-valid cached payload
        with _env(KDT_CALIB_SPLIT=None):
            ev._metrics_cache[ev._cache_key(named, resolve_train_split(None))] = stale
            l_hit = ev.evaluate(x)
            assert r.calls == 0, "cache should have served this"
            assert abs(l_hit[0] - (1.0 - 0.1234)) < 1e-12, l_hit
            ev._force_live = True
            l_live = ev.evaluate(x)
            assert r.calls == 1, "force-live must run the model"
            assert l_live != l_hit, (l_live, l_hit)
            ev._force_live = False
            ev.evaluate(x)
            assert r.calls == 1, "after force-live the (now refreshed) cache serves again"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------ P5
def test_p5_completed_report_carries_consumption_evidence():
    rep, r, log = _run("alive")
    assert rep.get("status") == "completed", (rep.get("status"), rep.get("reason"))
    for k in ("consumption_proof", "consumption_coverage", "objective_smoke_test"):
        assert k in rep, k
    cov = rep["consumption_coverage"]
    assert cov["n_params"] == 2 and cov["n_unproven"] == 0 and set(cov["proven"]) == {"a", "b"}, cov
    assert rep["objective_smoke_test"]["responsive"] is True


def test_p5_no_proof_driver_reports_nothing_proven_not_a_pass():
    class NoProof(Runner):
        def __call__(self):
            m = super().__call__(); m["__kdt__"].pop("target_proofs"); return m
    d, ki, wd = _fixture_ki()
    r = NoProof("alive")
    try:
        with _env(KDT_CALIB_CONTRACT=None, KDT_CALIB_C8="enforce", KDT_CALIB_SPLIT=None):
            with contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=r, budget=6, seed=0)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert rep.get("status") == "completed", (rep.get("status"), rep.get("reason"))
    cov = rep["consumption_coverage"]
    assert cov["ok"] is None and cov["n_proven"] == 0 and cov["n_unproven"] == 2, cov


# ------------------------------------------------------------------------------------ KNOWN GAPS
KNOWN_GAPS = {}


def known_gap(reason):
    def deco(fn):
        import pytest
        KNOWN_GAPS[fn.__name__] = reason
        return pytest.mark.xfail(reason=reason, raises=AssertionError, strict=True)(fn)
    return deco


@known_gap("until plan step C11 (interior probe) — a single far-bound displacement cannot tell "
           "'came back to the same value' from 'never read'")
def test_g1_endpoint_insensitive_consumed_knob_must_not_be_called_unreachable():
    params = [{"name": "a", "range": [0.0, 1.0], "default": 0.0, "type": "continuous"}]
    rep, r, log = _run("endpoint_insensitive", params=params)
    assert rep.get("status") != "param_unreachable", \
        "a consumed knob (y=1+x(1-x)) was STOPped as UNREACHABLE at its insensitive endpoint"


@known_gap("until plan step C10 (repeat-default noise floor) — 1e-9 run-to-run jitter must not "
           "acquit a knob the model ignores")
def test_g2_noise_only_response_must_not_acquit_a_dead_knob():
    params = [{"name": "a", "range": [0.0, 1.0], "default": 0.3, "type": "continuous"}]
    rep, r, log = _run("noisy_dead", params=params)
    cov = rep.get("consumption_coverage") or {}
    assert "a" not in (cov.get("by_verdict") or {}).get("INSENSITIVE", []), \
        f"dead knob acquitted as INSENSITIVE on 1e-9 noise: {cov.get('by_verdict')}"


if __name__ == "__main__":
    import traceback
    ok = True; gaps_open = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        if name in KNOWN_GAPS:
            try:
                fn(); ok = False
                print(f"  FAIL {name}: KNOWN-GAP test PASSED — the gap is closed; remove the marker")
            except AssertionError:
                gaps_open += 1; print(f"  KNOWN-GAP {name} ({KNOWN_GAPS[name]})")
            except Exception as e:
                ok = False; print(f"  FAIL {name}: {e}"); traceback.print_exc()
            continue
        try:
            fn(); print(f"  PASS {name}")
        except Exception as e:
            ok = False; print(f"  FAIL {name}: {e}"); traceback.print_exc()
    print(f"\n{'ALL PASS' if ok else '*** FAILURES ***'}  (known gaps open: {gaps_open})")
    sys.exit(0 if ok else 1)
