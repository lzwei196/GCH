"""Tests for the seed helpers kept for parallel seed lanes (lanes, workdirs, processes). The seed RULES
(agreement on calibration incumbents, the returned seed, verdicts across seeds) are tested in
test_step6_seeds.py (build step 6)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.seeds import run_seeds, seed_lanes, seed_workdirs   # noqa: E402

TOL = {"Q": {"nse": 0.02, "beta": 0.01}}


def test_seed_lanes():
    assert seed_lanes(3, 1) == (1, 3)
    assert seed_lanes(3, 3) == (3, 1)
    assert seed_lanes(3, 8) == (3, 1)
    assert seed_lanes(5, 2) == (2, 3)


def test_seed_workdirs_do_not_share_caches(tmp_path):
    src = tmp_path / "wd"
    (src / "inputs").mkdir(parents=True)
    (src / "inputs" / "forcing.txt").write_text("x")
    (src / "eval_metrics_cache_abc.jsonl").write_text("{}\n")
    (src / "eval_history.jsonl").write_text("{}\n")
    wds = seed_workdirs(src, [0, 1], parent=tmp_path / "lanes")
    assert set(wds) == {0, 1}
    for w in wds.values():
        assert Path(w, "inputs", "forcing.txt").read_text() == "x"
        assert not Path(w, "eval_history.jsonl").exists()
        assert not list(Path(w).glob("eval_metrics_cache_*"))


def _one(seed):          # module level: run_seeds pickles it
    return {"status": "completed", "train_metrics": {"nse": 0.9 + 0.001 * seed, "pbias": 1.0,
                                                     "r": 0.95, "kge": 0.88},
            "budget_used": {"n_evaluations": 100 + seed},
            "convergence": {"rule": {"stop_point": 40 + seed}, "verdict": {"verdict": "converged"},
                            "tolerances": TOL, "panel": ["nse", "beta"],
                            "optimizer_termination": {"status": "budget_schedule_complete"}},
            "best_loss": [0.1 - 0.001 * seed], "seed": seed}


def _boom(seed):
    raise RuntimeError(f"seed {seed} died")


def test_run_seeds_runs_each_seed_in_its_own_process():
    out = run_seeds(_one, [0, 1, 2], lanes=3)
    assert set(out["reports"]) == {0, 1, 2} and not out["errors"]
    assert out["seeds_parallel"] == 3 and out["waves"] == 1
    assert out["reports"][2]["seed"] == 2


def test_a_dead_seed_is_reported_not_swallowed():
    out = run_seeds(_boom, [0], lanes=1)
    assert out["reports"] == {} and "seed 0 died" in out["errors"][0]
