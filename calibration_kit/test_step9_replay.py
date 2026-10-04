"""Build step 9 (gap 2u, design §4 step 3): the replay of the rebuilt rule.

A replay feeds a recorded search through the same rule and verdicts the live kit uses, with an evidence manifest.
Level A (decided) only when the history is complete, every required metric is recorded and the manifest is complete;
otherwise Level B, labelled "on available evidence: …", never called converged / safe / premature. On a history the
new kit wrote, the replay must reproduce the run's own stop point and verdict per seed."""
import contextlib
import copy
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import replay as R                                                   # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402
from calibration_kit import test_step1_panel as S1                                        # noqa: E402


def _run(tmp, budget_block=None, strategy=None, families=None, cap=120, conv=None):
    ki, wd = E._fixture(tmp, budget_block or {"seeds": 3}, conv, cap)
    if strategy:
        c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"].update(strategy)
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
    if families:
        d = yaml.safe_load(open(Path(ki, "dag.yaml")))
        d["outputs"][0]["observability"]["comparable_obs_shapes"][0]["metric_families"] = families
        yaml.safe_dump(d, open(Path(ki, "dag.yaml"), "w"))
    with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
        rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
    return rep, wd


def _check_matches(rep, out):
    assert out["level"] == "A" and out["manifest"]["missing"] == []
    slots = {s["seed"]: s for s in rep["convergence"]["seeds"]["slots"]}
    assert set(out["seeds"]) == set(slots)
    for sd, row in out["seeds"].items():
        assert row["level"] == "A" and row["matches_the_runs_own_rule"] is True
        assert row["stop_point"] == slots[sd]["ended"]["stop_point"]
        # replay_workdir replays the per-call step rule: compare with its recorded verdicts (2026-10-04 names)
        assert row["verdict"] == slots[sd]["step_rule_verdict"]
        assert row["variables"] == slots[sd]["step_rule_variables"]


@pytest.mark.parametrize("algo", ["dds", "sceua"])
def test_a_new_kit_history_replays_to_the_runs_own_stop_point_and_verdicts(algo):
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, strategy={"default_algorithm": algo})
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _check_matches(rep, out)
    for row in out["seeds"].values():
        assert ("note" in row) == (algo == "dds")                        # DDS: a diagnostic only


def test_a_trade_off_history_replays_exactly():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 2}, strategy={"multi_objective": True, "default_algorithm": "nsga2"},
                       families=["temporal_pattern_match", "magnitude_accuracy"], cap=80)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out["manifest"]["trade_off"] is True
    _check_matches(rep, out)


def _hist_edit(wd, fn):
    p = Path(wd, "eval_history.jsonl")
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows = fn(rows)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_missing_required_metric_makes_it_level_b_and_never_a_verdict():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})

        def drop_beta(rows):
            k = 0
            for r in rows:
                if r["phase"] == "search" and r.get("panel", {}).get("streamflow"):
                    k += 1
                    if k == 5:
                        r["panel"]["streamflow"].pop("beta", None)
            return rows
        _hist_edit(wd, drop_beta)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    row = out["seeds"][0]
    assert out["level"] == "B" and row["level"] == "B" and row["verdict"] == "diagnostic only"
    assert row["safety"] is None and row["how"] is None
    assert row["label"].startswith("on available evidence:") and "streamflow:beta not recorded on 1 calls" in row["label"]
    assert "diagnostic" in row


def test_an_older_report_without_the_rules_setup_is_undecidable():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})
        old = copy.deepcopy(rep)
        for k in ("objective_vars", "weights", "window"):
            old["convergence"]["rule"].pop(k, None)
        out = R.replay_workdir(wd, old)
        rows = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out["level"] == "B" and "the window W" in out["manifest"]["missing"]
    recs = [{"losses": r["losses"], "panel": r["panel"]} for r in rows if r["phase"] == "search"]
    rr = R.replay_records(recs, old)
    assert rr["level"] == "B" and rr["status"].startswith("undecidable")


def test_a_call_count_that_disagrees_with_the_report_is_level_b():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})
        dropped = {"done": False}

        def drop_one(rows):
            out = []
            for r in rows:
                if r["phase"] == "search" and not dropped["done"] and r["i"] > 30:
                    dropped["done"] = True
                    continue
                out.append(r)
            return out
        _hist_edit(wd, drop_one)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    row = out["seeds"][0]
    assert row["level"] == "B" and "the report says 120, the log holds 119" in row["label"]


def test_records_from_an_older_kit_are_level_b_unless_complete_is_stated():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})
        rows = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    recs = [{"losses": r["losses"], "panel": r["panel"]} for r in rows if r["phase"] == "search"]
    b = R.replay_records(recs, rep)
    assert b["level"] == "B" and "a complete call history" in b["seeds"][None]["label"]
    assert b["seeds"][None]["diagnostic"]["stop_point"] == rep["convergence"]["ended"]["stop_point"]
    a = R.replay_records(recs, rep, complete=True)
    assert a["level"] == "A" and a["seeds"][None]["stop_point"] == rep["convergence"]["ended"]["stop_point"]


def test_alpha_is_rebuilt_only_when_the_preconditions_are_vouched_for():
    """Noisy series (r < 1, so the two roots differ): alpha dropped from the records is rebuilt to the true
    value when the caller vouches for §2.13's preconditions, and left missing otherwise."""
    import numpy as np
    from calibration_kit.panel import panel_from_series
    rng = np.random.default_rng(3)
    obs = 20 + 10 * np.sin(np.linspace(0, 20, 400)) + rng.normal(0, 1, 400)
    recs, truth = [], []
    for a, b in [(0.8, 2.0), (1.1, -1.0), (0.95, 0.5), (1.3, 3.0), (0.7, 1.0), (1.05, 0.2)]:
        sim = a * obs + b + rng.normal(0, 2.0, 400)
        p = panel_from_series(sim, obs, "flow")
        truth.append(p["alpha"])
        recs.append({"panel": {"Q": {k: p[k] for k in ("nse", "kge", "r", "pbias")}}})
    no = R.fill_alpha(copy.deepcopy(recs), ["Q"], preconditions_ok=False)
    assert no["Q"]["filled"] == 0
    got = copy.deepcopy(recs)
    yes = R.fill_alpha(got, ["Q"], preconditions_ok=True)
    assert yes["Q"]["filled"] == len(recs)
    assert [r["panel"]["Q"]["alpha"] for r in got] == pytest.approx(truth, rel=1e-6)
    assert all(r["panel"]["Q"]["_reconstructed"] == ["alpha"] for r in got)


def test_only_the_last_calibrate_call_of_a_log_is_replayed():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 60)
        for cap in (60, 40):                             # two calls with different call counts
            c = yaml.safe_load(open(Path(ki, "calibration.yaml"))); c["strategy"]["max_evaluations"] = cap
            yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=None, seed=0)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert any("2 calibrate() calls" in n for n in out["notes"])
    assert out["seeds"][0]["calls"] == 40 and out["level"] == "A"


def _native_replay_of(rep, wd):
    """The kit's convergence rule replayed (replay.replay_native) on the returned seed of a one-seed run."""
    cv = rep["convergence"]
    rows = [r for r in R.read_history(wd) if r.get("phase") == "search"]
    objs = list(zip(cv["rule"]["objectives"], cv["rule"]["objective_vars"]))
    tol = {v: dict((r or {}).get("tol") or {}) for v, r in (cv.get("tolerance_records") or {}).items()}
    return R.replay_native(rows, cv, rep["algorithm"], objs, cv["kinds"], tol, weights=cv["rule"].get("weights"))


@pytest.mark.parametrize("mode", ["stop", "keep_going"])
def test_the_kits_convergence_rule_replays_to_the_live_verdict(mode):
    """2026-10-04: SCE-UA's own test on our scores, live (stop or keep_going) and replayed from the saved history
    with the report's loop tags and SPOTPY's per-loop record, gives the same firing loop and verdict; the settle point
    too."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1}, strategy={"default_algorithm": "sceua"}, cap=8000,
                       conv={"mode": mode})
        out = _native_replay_of(rep, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    live = rep["convergence"]["native"]
    assert live["fired_at"] is not None                               # the fixture really fires
    assert out["native"]["fired_at"] == live["fired_at"] and out["native"]["verdict"] == live["verdict"]
    assert out["native"]["loops_seen"] == live["loops_seen"]          # a loop cut by the cap is never counted
    assert out["native"]["gnrng_recorded"] is True
    assert live["stopped_the_search"] is (mode == "stop")
    assert out["settle"]["settled_by_run"] == rep["convergence"]["settle"]["settled_by_run"]


def test_a_native_replay_without_spotpys_loop_record_says_so_and_never_fires_earlier():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1}, strategy={"default_algorithm": "sceua"}, cap=8000)
        full = _native_replay_of(rep, wd)
        old = copy.deepcopy(rep)
        old["convergence"]["optimizer_termination"].pop("loop_record", None)
        part = _native_replay_of(old, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert part["native"]["gnrng_recorded"] is False and "left out" in part["native"]["note"]
    if part["native"]["fired_at"] is not None:
        assert part["native"]["fired_at"]["loop"] >= full["native"]["fired_at"]["loop"]


def test_a_weighted_two_family_search_replays_exactly():
    """The scalar loss needs the objective WEIGHTS from the report (rule.summary)."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 2}, families=["temporal_pattern_match", "magnitude_accuracy"])
        out = R.replay_workdir(wd, rep)
        old = copy.deepcopy(rep)
        old["convergence"]["rule"].pop("weights", None)
        out2 = R.replay_workdir(wd, old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["convergence"]["rule"]["trade_off"] is False and len(rep["convergence"]["rule"]["objectives"]) == 2
    _check_matches(rep, out)
    assert out2["level"] == "B" and "objective weights" in out2["manifest"]["missing"]


def test_a_protected_trade_off_replays_exactly_and_needs_its_anchor():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 2}, cap=80,
                       strategy={"multi_objective": True, "default_algorithm": "nsga2",
                                 "protect": {"streamflow": ["r", "beta"]}},
                       families=["temporal_pattern_match", "magnitude_accuracy"])
        out = R.replay_workdir(wd, rep)
        old = copy.deepcopy(rep)
        old["convergence"]["rule"].pop("default_panel", None)
        out2 = R.replay_workdir(wd, old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert isinstance(rep["convergence"]["rule"]["protect_active"], dict) and rep["convergence"]["rule"]["protect_active"]
    _check_matches(rep, out)
    assert out2["level"] == "B" and any("protection anchor" in m for m in out2["manifest"]["missing"])


def test_a_manifest_gap_that_is_not_fatal_still_gives_level_b():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})
        old = copy.deepcopy(rep)
        old["convergence"].pop("tolerance_records", None)
        out = R.replay_workdir(wd, old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert "tolerances" in out["manifest"]["missing"] and out["level"] == "B" and out["seeds"][0]["level"] == "B"


def test_a_record_without_its_phase_tag_gives_level_b():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})

        def untag(rows):
            for r in rows:
                if r["phase"] == "pilot":
                    r.pop("phase")
                    break
            return rows
        _hist_edit(wd, untag)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out["seeds"][0]["level"] == "B" and "phase tags" in out["seeds"][0]["label"]


def test_search_calls_that_scored_another_split_give_level_b():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})

        def relabel(rows):
            for r in rows:
                if r["phase"] == "search" and "split_echo" in r:
                    r["split_echo"] = "holdout"
            return rows
        _hist_edit(wd, relabel)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    row = out["seeds"][0]
    assert row["level"] == "B" and "echoed a split other than calibration" in row["label"]
    assert "rule_verdict_if_decided" not in (row.get("diagnostic") or {})      # no verdict word at Level B


def test_a_crashed_attempt_is_not_replayed_as_a_search(monkeypatch):
    from calibration_kit.test_step6b_lanes import _Crash

    class B(_Crash):
        bad = {1}
    real = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algo: B(real(algo)))
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 3}, cap=60)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert [s["seed"] for s in rep["convergence"]["seeds"]["slots"]] == [0, 4, 2]
    assert set(out["seeds"]) == {0, 4, 2} and 1 not in out["seeds"]
    att = out["crashed_attempts"][1]
    assert att["verdict"] is None and att["replaced_by"] == 4 and att["label"].startswith("crashed")
    _check_matches(rep, out)


def test_a_dream_seed_is_replayed_with_its_recorded_r_hat():
    """Opus 8b/9 r1 #1: R-hat decides a DREAM seed, in the replay as in the run."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1}, strategy={"default_algorithm": "dream"}, cap=250)
        out = R.replay_workdir(wd, rep)
        old = copy.deepcopy(rep)
        for s_ in old["convergence"]["seeds"]["slots"]:
            s_.pop("dream", None)
        out2 = R.replay_workdir(wd, old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    slot = rep["convergence"]["seeds"]["slots"][0]
    assert isinstance(slot["dream"], dict)
    _check_matches(rep, out)
    assert out2["seeds"][0]["level"] == "B" and "R-hat" in out2["seeds"][0]["label"]


# ── review round 2 (Opus 8b/9) ──────────────────────────────────────────────────────────────────
def test_calls_that_never_ran_the_model_do_not_make_the_replay_level_b():
    """Opus 8b/9 r2 #1: constraint-rejected calls and calls whose runner raised carry the split they were for,
    and a complete honest run with them still replays at Level A."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 80)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["constraints"] = ["a + 0.1*b <= 1.3"]
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))

        class Raising(S1._Runner):
            n = 0

            def __call__(self):
                Raising.n += 1
                if Raising.n > 30 and Raising.n % 13 == 0:      # the runner fails now and then (not the pilot)
                    raise RuntimeError("planted runner failure")
                return super().__call__()
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=Raising(wd), budget=None, seed=0)
        rows = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    never = [r for r in rows if r["phase"] == "search" and r.get("reason") in ("infeasible", "exception")]
    assert {r.get("reason") for r in never} == {"infeasible", "exception"}   # both kinds really happened
    assert all(r.get("split") == "calibration" for r in never)
    _check_matches(rep, out)
    # a log written before the evaluator recorded the split on such calls replays the same way
    old = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        Path(old, "eval_history.jsonl").write_text("".join(
            json.dumps({k: v for k, v in r.items() if not (k == "split" and r.get("reason") in ("infeasible", "exception"))})
            + "\n" for r in rows))
        out_old = R.replay_workdir(old, rep)
    finally:
        shutil.rmtree(old, ignore_errors=True)
    _check_matches(rep, out_old)


def test_a_seed_the_report_lists_but_the_log_lacks_makes_the_top_level_b():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 2})
        _hist_edit(wd, lambda rows: [r for r in rows if not (r["phase"] == "search" and r.get("seed") == 1)])
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out["level"] == "B" and any("[1] are in the report but have no search records" in n for n in out["notes"])


def test_a_seed_the_report_does_not_list_is_level_b():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 2})
        old = copy.deepcopy(rep)
        old["convergence"]["seeds"]["slots"] = [s for s in old["convergence"]["seeds"]["slots"] if s["seed"] != 1]
        out = R.replay_workdir(wd, old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out["seeds"][1]["level"] == "B" and "this seed's slot in the report" in out["seeds"][1]["label"]


def test_the_self_check_compares_the_verdict_too():
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})
        old = copy.deepcopy(rep)
        s0 = old["convergence"]["seeds"]["slots"][0]
        s0["step_rule_verdict"] = "converged" if s0["step_rule_verdict"] != "converged" else "not_converged"
        out = R.replay_workdir(wd, old)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    row = out["seeds"][0]
    assert row["level"] == "A" and row["stop_point"] == s0["ended"]["stop_point"]
    assert row["matches_the_runs_own_rule"] is False


def test_a_log_with_a_gap_and_a_duplicate_of_the_same_count_is_level_b():
    """codex step 9 r1 #1: drop one call and duplicate another — the count still matches the report, but the log is
    not every call in order, so the replay must not decide."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})

        def gap_and_dup(rows):
            # drop search call 40 (a gap) and pad with a copy of the seed's last search call at the end of the
            # log, numbered on (no new session): the per-seed count is unchanged
            out = [r for r in rows if not (r["phase"] == "search" and r["i"] == 40)]
            last = [r for r in out if r["phase"] == "search"][-1]
            out.append(dict(last, i=out[-1]["i"] + 1))
            return out
        _hist_edit(wd, gap_and_dup)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    row = out["seeds"][0]
    assert row["calls"] == rep["convergence"]["seeds"]["slots"][0]["calls"]           # the count alone matches
    assert out["level"] == "B" and row["level"] == "B" and "unbroken, ordered call log" in row["label"]


def test_a_log_missing_its_first_call_is_level_b():
    """codex step 9 r2 #1: the call numbers must start at 0 — a log missing call 0, padded to the same count, is not
    every call."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1})

        def drop_first(rows):
            out = rows[1:]
            last = [r for r in out if r["phase"] == "search"][-1]
            out.append(dict(last, i=out[-1]["i"] + 1))
            return out
        _hist_edit(wd, drop_first)
        out = R.replay_workdir(wd, rep)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out["level"] == "B" and "unbroken, ordered call log" in out["seeds"][0]["label"]


def test_native_replay_never_counts_a_loop_cut_by_the_cap():
    """A made-up SCE-UA record: start-up (loop 0) and loops 1-2 complete, loop 3 cut by the cap. The replay must see
    two loops, and must see three when the record says the search ended at a loop end."""
    P = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}}
    rows = [{"losses": [1.0 - 0.01 * i], "panel": P} for i in range(12)]
    term = {"status": "ended_by_kit", "loop_starts": [[0, 0], [3, 1], [6, 2], [9, 3]],
            "loop_record": [{"loop": 1, "gnrng": 0.5, "bestf": 0.95}, {"loop": 2, "gnrng": 0.4, "bestf": 0.92}]}
    tol = {"Q": {"r": 0.01, "alpha": 0.01, "beta": 0.01}}
    out = R.replay_native(rows, {"optimizer_termination": term}, "sceua", [("Q:l", "Q")], {"Q": "series"}, tol)
    assert out["native"]["loops_seen"] == 2
    term2 = dict(term, status="stopped_by_kit_rule",
                 loop_record=term["loop_record"] + [{"loop": 3, "gnrng": 0.3, "bestf": 0.89}])
    out2 = R.replay_native(rows, {"optimizer_termination": term2}, "sceua", [("Q:l", "Q")], {"Q": "series"}, tol)
    assert out2["native"]["loops_seen"] == 3


def test_every_seed_can_be_replayed_with_the_kits_rule_from_its_slot():
    """codex A3d #4: each slot keeps its native result and the optimizer's step records, so a seed that was not
    returned replays to its own live verdict."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 2}, strategy={"default_algorithm": "sceua"}, cap=6000)
        rows = R.read_history(wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cv = rep["convergence"]
    objs = list(zip(cv["rule"]["objectives"], cv["rule"]["objective_vars"]))
    tol = {v: dict((r or {}).get("tol") or {}) for v, r in (cv.get("tolerance_records") or {}).items()}
    for slot in cv["seeds"]["slots"]:
        recs = [r for r in rows if r.get("phase") == "search" and r.get("seed") == slot["seed"]]
        block = dict(cv, optimizer_termination=slot["optimizer_termination"])
        out = R.replay_native(recs, block, "sceua", objs, cv["kinds"], tol, weights=cv["rule"].get("weights"))
        assert out["native"]["fired_at"] == slot["native"]["fired_at"]
        assert out["settle"]["settled_by_run"] == slot["settle"]["settled_by_run"]


@pytest.mark.parametrize("rg", [0.2, 0.0])
def test_native_replay_uses_the_runs_rel_gain(rg):
    """codex A3d r2/r3: a non-default strategy.convergence.rel_gain — 0.0 included — must reach the replayed settle
    point."""
    tmp = tempfile.mkdtemp(prefix="kdt_s9_")
    try:
        rep, wd = _run(tmp, budget_block={"seeds": 1}, strategy={"default_algorithm": "dds"}, cap=200,
                       conv={"rel_gain": rg})
        out = _native_replay_of(rep, wd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert rep["convergence"]["rule"]["rel_gain"] == rg and out["settle"]["rel_gain"] == rg
    assert out["settle"]["settled_by_run"] == rep["convergence"]["settle"]["settled_by_run"]
