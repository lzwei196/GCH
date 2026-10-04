"""Tests for stop.py (now only `convention_floor`) and the backends' stop hook — canonical python_env.

`StopRule` was removed in build step 8 (gap 2w); its tests went with it. What stays:
1. convention_floor reads a REAL KI convention (VIC discharge: good = 0.65, |PBIAS| band).
2. pymoo (NSGA-II) honours the kit's per-call hook (the kit ends a search through it), runs to the cap
   without one, and never reads a failed evaluation as finite.
"""
from __future__ import annotations
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.stop import convention_floor                     # noqa: E402
from calibration_kit.backends.base import Problem                     # noqa: E402
from calibration_kit.backends.spotpy_backend import SpotpyBackend     # noqa: E402

VIC_KI = "/mnt/disk1/Hydrocraft_server/models/VIC/knowledge_infrastructure"


def _synthetic(n=3, best_nse=0.9):
    """loss = 1 - NSE; NSE = best_nse - sum((x - 0.3)^2) -> a single smooth optimum."""
    stats = {}
    def evaluate(x):
        nse = best_nse - sum((xi - 0.3) ** 2 for xi in x)
        stats["last"] = nse
        return [1.0 - nse]
    prob = Problem(names=[f"p{i}" for i in range(n)], lower=[0.0] * n, upper=[1.0] * n,
                   objective_names=["loss"], evaluate=evaluate)
    return prob, stats


def _ki_with_convention(tmp, good=0.65, sat=0.5, vgood=0.75):
    d = Path(tmp) / "docs"; d.mkdir(parents=True, exist_ok=True)
    (d / "validation_convention.yaml").write_text(f"""
- obs_shape: point_time_series
  dag_variable: OUT_DISCHARGE
  headline_metrics:
  - metric: nse
    direction: maximize
    bands: {{very_good: {vgood}, good: {good}, satisfactory: {sat}}}
    pass_band: satisfactory
    cites: [moriasi2007, moriasi2015]
  - metric: pbias
    direction: minimize_abs
    bands: {{very_good: 10, good: 15, satisfactory: 25}}
    pass_band: satisfactory
    cites: [moriasi2015]
""")
    return tmp


def test_convention_floor_real_vic():
    f = convention_floor(VIC_KI, "nse", "good", "OUT_DISCHARGE")
    assert f["value"] == 0.65 and f["band"] == "good" and f["cites"], f
    try:
        convention_floor(VIC_KI, "nse", "excellent", "OUT_DISCHARGE")
        raise AssertionError("an unknown band must be refused")
    except LookupError:
        pass
    print("T1 convention floor from the real VIC convention: PASS", f["value"], f["cites"])


def _synthetic_mo(n=3, best_nse=0.9):
    """Two-objective version: obj1 = 1 - NSE (magnitude), obj2 = the same squared error (a
    correlated second objective). Both minimize toward x=0.3; the main stat is NSE. Used to prove
    the pymoo (NSGA-II) backend honours the stop hook — the defect codex/kimi flagged."""
    stats = {}
    def evaluate(x):
        se = sum((xi - 0.3) ** 2 for xi in x)
        nse = best_nse - se
        stats["last"] = nse
        return [1.0 - nse, se]
    prob = Problem(names=[f"p{i}" for i in range(n)], lower=[0.0] * n, upper=[1.0] * n,
                   objective_names=["magnitude", "sqerr"], evaluate=evaluate, is_multi_objective=True)
    return prob, stats


def _run_pymoo(stop_after, budget, on_eval=True, seed=0):
    """NSGA-II with a plain per-call hook that asks to stop after `stop_after` calls (the kit's hook)."""
    from calibration_kit.backends.pymoo_backend import PymooBackend
    prob, stats = _synthetic_mo()
    hook = (lambda history: len(history) >= stop_after) if on_eval else None
    return PymooBackend("nsga2", pop_size=20).optimize(prob, budget=budget, seed=seed, on_eval=hook), None


def test_pymoo_honors_the_kit_hook():
    # NSGA-II must stop when the kit's per-call hook asks (it ignored on_eval before 2026-08-30)
    res, _ = _run_pymoo(300, budget=4000)
    assert res.stopped_early, f"pymoo did not stop early: {res.notes}"
    assert res.n_evaluations < 4000, f"ran the full budget ({res.n_evaluations}) — hook ignored"
    assert res.history and "x" in res.history[0] and "loss" in res.history[0], "history not in shared shape"
    assert res.pareto_x and res.pareto_f, "front must still be produced on early stop"


def test_pymoo_no_hook_runs_to_cap():
    # Regression: with no stop rule, NSGA-II runs the full budget and still returns a shared-shape
    # history + a front (stopped_early False).
    res, _ = _run_pymoo(10 ** 9, budget=120, on_eval=False)
    assert not res.stopped_early and res.n_evaluations >= 100, res.notes
    assert res.history and res.pareto_f, "history + front must be present without a stop hook"
    print(f"T6 pymoo no-hook runs to cap ({res.n_evaluations} evals), front intact: PASS")


def test_pymoo_failed_eval_not_finite():
    # codex/kimi 2026-08-30: an empty/failed evaluate() must NOT read as a finite solution. A row that
    # returns [] must get history loss=inf (excluded by stop.py's finite mask) and must not enter the
    # front. Half the box fails here; the front must contain only the succeeding points.
    from calibration_kit.backends.pymoo_backend import PymooBackend
    import math as _m
    def evaluate(x):
        if x[0] > 0.6:            # a "failed" region -> contract-valid empty result
            return []
        se = sum((xi - 0.3) ** 2 for xi in x)
        return [1.0 - (0.9 - se), se]
    prob = Problem(names=["p0", "p1"], lower=[0.0, 0.0], upper=[1.0, 1.0],
                   objective_names=["magnitude", "sqerr"], evaluate=evaluate, is_multi_objective=True)
    res = PymooBackend("nsga2", pop_size=20).optimize(prob, budget=120, seed=0)
    inf_rows = [h for h in res.history if not _m.isfinite(h["loss"])]
    assert inf_rows, "failed evals must appear as inf-loss history rows"
    # every returned front point must be from the succeeding region (x0 <= 0.6) and finite
    assert res.pareto_x and all(x[0] <= 0.6 + 1e-9 for x in res.pareto_x), res.pareto_x
    assert all(_m.isfinite(v) for row in res.pareto_f for v in row), "front carries a penalized point"
    print(f"T7 pymoo failed-eval guard: {len(inf_rows)} inf rows, front all-finite & in-region: PASS")


if __name__ == "__main__":
    test_convention_floor_real_vic(); test_pymoo_honors_the_kit_hook(); test_pymoo_no_hook_runs_to_cap()
    test_pymoo_failed_eval_not_finite()
    print("ALL STOP-LAYER TESTS PASS")
