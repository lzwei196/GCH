"""Part A4 (rev 30): the descriptive settle point for DDS / random / LLM arms (settle.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit import panel as P                                              # noqa: E402
from calibration_kit.settle import SettlePoint                                      # noqa: E402

TOL = {"Q": {"r": 0.01, "alpha": 0.01, "beta": 0.01}}
PAN = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}}


def _sp(**kw):
    return SettlePoint(P.Panel(["Q"], kinds={"Q": "series"}), TOL, **kw)


def test_last_real_improvement_is_the_settle_point():
    s = _sp()
    for v in (10, 5, 4.99, 3, 2.999, 2.998, 50):
        s.add(v, PAN)
    out = s.summary()
    assert out["settled_by_run"] == 4 and out["runs"] == 7 and out["wording"] == "settled by run 4 of 7"
    assert abs(out["share_after"] - 3 / 7) < 1e-12


def test_slow_drift_adds_up_against_the_last_change():
    """Each step 0.3 % (below 0.5 %), but compared with the last change it passes 0.5 % at the 2nd step."""
    s = _sp()
    for v in (1.0, 0.997, 0.994, 0.991):
        s.add(v, PAN)
    assert [c["run"] for c in s.changes] == [1, 3]
    assert s.summary()["settled_by_run"] == 3


def test_score_move_at_a_new_best_counts():
    s = _sp()
    s.add(1.0, PAN)
    s.add(0.999, {"Q": {"r": 0.95, "alpha": 1.0, "beta": 1.0}})
    assert s.summary()["settled_by_run"] == 2 and s.changes[-1]["why"][0][:3] == ["score", "Q", "r"]


def test_score_move_at_a_worse_point_does_not_count():
    s = _sp()
    s.add(1.0, PAN)
    s.add(2.0, {"Q": {"r": 0.1, "alpha": 9.0, "beta": 9.0}})
    assert s.summary()["settled_by_run"] == 1


def test_missing_score_at_new_best_fails_closed_and_failed_runs_are_counted():
    s = _sp()
    s.add(1.0, PAN)
    s.add(None, None)                                   # a failed run still counts as a run
    s.add(0.999, {"Q": {"r": 0.9, "alpha": float("nan"), "beta": 1.0}})
    out = s.summary()
    assert out["runs"] == 3 and out["settled_by_run"] == 3
    assert ["score_missing", "Q", "alpha"] in s.changes[-1]["why"]


def test_exact_boundary_and_ties():
    s = _sp(rel_gain=0.25)
    s.add(1.0, PAN); s.add(0.75, PAN)                   # exactly 25 % -> a change (>=)
    assert s.summary()["settled_by_run"] == 2
    s = _sp(rel_gain=0.25)
    s.add(1.0, PAN); s.add(0.875, PAN); s.add(0.875, PAN)   # 12.5 %: no change; a tie is not a new best
    assert s.summary()["settled_by_run"] == 1


def test_no_finite_run():
    s = _sp()
    s.add(None); s.add(float("inf"))
    assert s.summary()["settled_by_run"] is None and s.summary()["wording"] == "no finite run"


def test_tie_with_moved_scores_is_not_a_new_best():
    s = _sp()
    s.add(1.0, PAN)
    s.add(1.0, {"Q": {"r": 0.5, "alpha": 1.0, "beta": 1.0}})    # same loss, scores moved: the earlier point stays best
    assert s.summary()["settled_by_run"] == 1
