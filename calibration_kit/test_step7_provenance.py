"""Build step 7 (gaps 2j, 2p): phase tags and split provenance.

2j — every model run is counted under what it was for: an evaluation made outside any phase block is
"untagged" (never search effort); the consumption proof's receipt check is `proof`; the certification
replay and the objective probe (runner calls outside the evaluator) are logged, timed and charged.
2p — design §1 "Data within a search": every call records the split the runner says it scored
(`__kdt__.split`) next to the split the kit asked for; the report says whether the pilot and search calls
can support a calibration-only claim. The agent prompt no longer tells a runner to "score both"."""
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

from calibration_kit import calib as C                                                    # noqa: E402
from calibration_kit import test_calibrate_convergence_e2e as E                           # noqa: E402
from calibration_kit import test_step1_panel as S1                                        # noqa: E402
from calibration_kit.evaluator import split_echo, split_provenance                        # noqa: E402


def _run(budget_block=None, runner=None, max_evaluations=30, ref=None, expected_case_id=None, seeds=1):
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        bb = dict(budget_block or {})
        bb.setdefault("seeds", seeds)
        ki, wd = E._fixture(tmp, bb, None, max_evaluations)
        if ref is not None:
            Path(ki, "calib").mkdir(exist_ok=True)
            Path(ki, "calib", "reference_run.json").write_text(json.dumps(ref))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                              run_model=(runner or S1._Runner)(wd), budget=None, seed=0,
                              expected_case_id=expected_case_id)
        hp = Path(wd, "eval_history.jsonl")
        hist = [json.loads(l) for l in hp.read_text().splitlines()] if hp.exists() else []
        return rep, hist
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── 2p: split provenance ─────────────────────────────────────────────────────────────────────────
def test_split_echo_reads_the_kit_block_only():
    assert split_echo({"__kdt__": {"split": "Calibration "}}) == "calibration"
    assert split_echo({"__kdt__": {"split": ""}}) is None
    assert split_echo({"scored_split": "calibration"}) is None          # only __kdt__.split counts
    assert split_echo({"__kdt__": "x"}) is None and split_echo(None) is None


def test_split_provenance_counts_missing_and_wrong_echoes():
    recs = [{"i": 0, "phase": "pilot", "split": "calibration", "split_echo": "calibration"},
            {"i": 1, "phase": "search", "split": "calibration", "split_echo": None},
            {"i": 2, "phase": "search", "split": "calibration", "split_echo": "full"},
            {"i": 3, "phase": "holdout", "split": "holdout", "split_echo": "holdout"},
            {"i": 4, "phase": "search", "split": "calibration"}]            # no metrics: not counted
    sp = split_provenance(recs)
    s = sp["by_phase"]["search"]
    assert (s["calls"], s["echo_missing"], s["echo_wrong"]) == (2, 1, 1)
    assert s["missing_at"] == [1] and s["wrong_at"] == [[2, "calibration", "full"]]
    assert sp["calibration_only_supported"] is False and "2 of 3" in sp["text"]
    ok = split_provenance([r for r in recs if r["i"] in (0, 3)])
    assert ok["calibration_only_supported"] is True
    assert split_provenance([])["calibration_only_supported"] is None


def test_an_echoing_runner_supports_the_calibration_only_claim_and_holdout_calls_echo_holdout():
    rep, hist = _run()
    sp = rep["split_provenance"]
    assert sp["calibration_only_supported"] is True
    assert sp["by_phase"]["search"]["echo_ok"] == sp["by_phase"]["search"]["calls"] > 0
    assert sp["by_phase"]["holdout"]["echo_ok"] >= 1 and not sp["by_phase"]["holdout"]["echo_wrong"]
    assert all(h.get("split_echo") == h.get("split") for h in hist if "split_echo" in h
               and not h.get("outside_evaluator"))


class _NoEcho(S1._Runner):
    def __call__(self):
        m = super().__call__()
        m["__kdt__"].pop("split", None)
        return m


class _Full(S1._Runner):
    """Scores the calibration subset but says it scored the full record on the search calls."""
    def __call__(self):
        m = super().__call__()
        if os.environ.get("KDT_CALIB_SPLIT") != "holdout":
            m["__kdt__"]["split"] = "full"
        return m


def test_a_runner_without_the_echo_or_with_another_split_cannot_support_the_claim():
    for runner, key in ((_NoEcho, "echo_missing"), (_Full, "echo_wrong")):
        rep, _ = _run(runner=runner)
        sp = rep["split_provenance"]
        assert sp["calibration_only_supported"] is False, runner
        assert sp["by_phase"]["search"][key] == sp["by_phase"]["search"]["calls"]
        assert "cannot support a calibration-only claim" in sp["text"]


def test_the_split_provenance_covers_this_call_only():
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, {"seeds": 1}, None, 20)
        with open(Path(wd, "eval_history.jsonl"), "w") as fh:          # an older attempt, no echo
            fh.write(json.dumps({"i": 0, "phase": "search", "split": "calibration", "split_echo": None}) + "\n")
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                              budget=None, seed=0)
        assert rep["split_provenance"]["calibration_only_supported"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_agent_prompt_no_longer_says_score_both():
    p = Path(__file__).resolve().parents[1] / "auto_dissect_multi_agent" / "stage_calibrate.py"
    if not p.is_file():
        pytest.skip("Requires the external KI host's agent prompt")
    txt = p.read_text()
    assert "score both" not in txt and 'metrics["__kdt__"]["split"]' in txt


# ── 2j: phase tags ───────────────────────────────────────────────────────────────────────────────
def test_an_evaluation_outside_any_phase_block_is_untagged_never_search():
    from calibration_kit.evaluator import Evaluator
    from calibration_kit import objectives as O
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, None, None, 20)
        objs = O.objectives_from_dag(yaml.safe_load(Path(ki, "dag.yaml").read_text()), [{"var": "streamflow"}],
                                     {"streamflow": "point_time_series"})
        ev = Evaluator(ki_path=ki, workdir=wd, parameters=[{"name": "a", "range": [0.5, 1.5]},
                                                           {"name": "b", "range": [-5, 5]}],
                       objectives=objs, transform_inv={"a": lambda v: v, "b": lambda v: v},
                       run_model=S1._Runner(wd), injection_mode="runner")
        with E._clean_env():
            ev.evaluate([1.0, 0.0])
            with ev.phase_as("search"):
                ev.evaluate([0.9, 0.1])
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
        assert [h["phase"] for h in hist] == ["untagged", "search"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_record_of_a_calibration_has_a_named_phase_and_one_numbering():
    rep, hist = _run()
    assert "untagged" not in {h["phase"] for h in hist}
    assert [h["i"] for h in hist] == list(range(len(hist)))
    # the objective probe is a model run too: logged, timed, charged
    probe = [h for h in hist if h["phase"] == "objective_probe"]
    assert len(probe) == 1 and probe[0]["outside_evaluator"] and hist[0]["phase"] == "objective_probe"
    assert rep["phase_counts"]["objective_probe"]["n"] == 1
    parts = rep["budget_plan"]["ledger"]["shared_parts_s"]
    assert parts["objective_probe"] == pytest.approx(rep["phase_counts"]["objective_probe"]["wall_s"], abs=1e-9)
    assert "certify" in parts


def test_the_certification_replay_is_logged_timed_and_charged():
    ref = {"case_id": "SITE:fixture", "validated_run_id": "fixture-ref", "params": {"a": 1.0, "b": 0.0},
           "headline_metric": "nse", "headline_value": 1.0, "tolerance": {"abs": 0.05},
           "reference_split": "calibration"}
    rep, hist = _run(ref=ref, expected_case_id="SITE:fixture")
    assert rep["status"] == "completed", rep.get("reason")
    cert = [h for h in hist if h["phase"] == "certify"]
    assert len(cert) == 1 and cert[0]["ok"] is True and hist[0]["phase"] == "certify"
    assert rep["phase_counts"]["certify"]["n"] == 1
    assert rep["budget_plan"]["ledger"]["shared_parts_s"]["certify"] == pytest.approx(
        rep["phase_counts"]["certify"]["wall_s"], abs=1e-9)
    assert [h["i"] for h in hist] == list(range(len(hist)))


def test_the_proof_receipt_check_is_tagged_proof():
    """With consumption_cache_reuse the gate runs ONE live default evaluation to validate a stored
    receipt. That run is proof work, never search effort."""
    seen = []

    class _Ev:                                           # the gate only needs these
        phase = "untagged"
        _force_live = False
        _last_metrics = {}

        @contextlib.contextmanager
        def phase_as(self, name):
            prev, self.phase = self.phase, name
            try:
                yield
            finally:
                self.phase = prev

        def evaluate(self, x):
            seen.append(self.phase)
            return [0.1]

        def _proofs_of(self, m):
            return []

        def consumption_proof(self, *a, **k):
            return {"ok": None, "summary": "stub"}
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, None, None, 20)
        cache = Path(wd, "kdt_consumption_proof")
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "old.json").write_text(json.dumps({"code_key": "x"}))
        contract = yaml.safe_load(Path(ki, "calibration.yaml").read_text())
        contract["strategy"]["consumption_cache_reuse"] = True
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C._consumption_gate(_Ev(), contract, ki, wd, [1.0, 0.0], [0.5, -5], [1.5, 5], ["a", "b"],
                                contract["parameters"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert seen and seen[0] == "proof"


# ── round-1 review (Opus): the rules pinned end to end ──────────────────────────────────────────
class _Holdout(S1._Runner):
    """Scores the HOLDOUT window on every call and says so: the leak §1 is about."""
    def __call__(self):
        m = super().__call__()
        m["__kdt__"]["split"] = "holdout"
        return m


class _FullOnPilotDefault(S1._Runner):
    """Echoes "full" only on the pilot's default run (the series run)."""
    def __call__(self):
        m = super().__call__()
        if os.environ.get("KDT_CALIB_EMIT_SERIES") == "1":
            m["__kdt__"]["split"] = "full"
        return m


class _AlwaysFull(S1._Runner):
    def __call__(self):
        m = super().__call__()
        m["__kdt__"]["split"] = "full"
        return m


def test_a_runner_that_scores_the_holdout_on_search_calls_is_flagged():
    rep, _ = _run(runner=_Holdout)
    sp = rep["split_provenance"]
    assert sp["calibration_only_supported"] is False
    assert sp["by_phase"]["search"]["echo_wrong"] == sp["by_phase"]["search"]["calls"] > 0


def _two_calls(runner, drop_saved=False):
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "seeds": 1}, None, 30)
        reps = []
        for k in range(2):
            if k == 1 and drop_saved:
                pf = Path(wd, "kdt_budget_plan.json")
                pl = json.loads(pf.read_text())
                pl.pop("pilot_split_provenance", None)
                pf.write_text(json.dumps(pl))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                reps.append(C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=runner(wd),
                                        budget=None, seed=0))
        return reps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_wrong_pilot_echo_is_flagged_and_still_counts_after_a_resume():
    first, again = _two_calls(_FullOnPilotDefault)
    for rep in (first, again):
        sp = rep["split_provenance"]
        assert sp["calibration_only_supported"] is False and sp["by_phase"]["pilot"]["echo_wrong"] == 1
    assert again["budget_plan"].get("resumed_cap")
    assert again["split_provenance"]["by_phase"]["pilot"]["source"].startswith("the saved budget plan")
    assert "every pilot" not in again["split_provenance"]["text"]


def test_a_resumed_search_counts_its_cache_hits_with_their_echo():
    first, again = _two_calls(S1._Runner)
    s = again["split_provenance"]["by_phase"]["search"]
    assert s["calls"] == 30 and s["echo_ok"] == 30
    assert again["split_provenance"]["calibration_only_supported"] is True
    assert again["split_provenance"]["by_phase"]["pilot"]["calls"] == 10         # the saved pilot


def test_a_resume_from_a_plan_without_the_pilots_record_is_not_claimed():
    _, again = _two_calls(S1._Runner, drop_saved=True)
    sp = again["split_provenance"]
    assert sp["calibration_only_supported"] is None and sp["text"].startswith("pilot not checked")


def test_the_certification_and_probe_echoes_are_judged():
    ref = {"case_id": "SITE:fixture", "validated_run_id": "fixture-ref", "params": {"a": 1.0, "b": 0.0},
           "headline_metric": "nse", "headline_value": 1.0, "tolerance": {"nse": 0.05},
           "reference_split": "calibration"}
    rep, hist = _run(runner=_AlwaysFull, ref=ref, expected_case_id="SITE:fixture")
    by = rep["split_provenance"]["by_phase"]
    assert by["certify"]["echo_wrong"] == 1 and by["objective_probe"]["echo_wrong"] == 1
    rec = {h["phase"]: h for h in hist if h.get("outside_evaluator")}
    assert rec["certify"]["split"] == "calibration" and rec["objective_probe"]["split"] == "calibration"


def test_a_probe_payload_that_set_pilot_decisions_counts_in_the_claim():
    recs = [{"i": 0, "phase": "objective_probe", "split": "calibration", "split_echo": "full"},
            {"i": 1, "phase": "search", "split": "calibration", "split_echo": "calibration"}]
    assert split_provenance(recs)["calibration_only_supported"] is True
    both = split_provenance(recs, claim_phases=("pilot", "search", "objective_probe"))
    assert both["calibration_only_supported"] is False and "objective-probe" in both["text"]
    # the text names only the phases it counted: no pilot call here, so it never says "every pilot"
    only = split_provenance([recs[1]])
    assert only["calibration_only_supported"] is True and only["text"].startswith("every search call")


def test_a_failed_certification_is_logged():
    ref = {"case_id": "SITE:fixture", "validated_run_id": "fixture-ref", "params": {"a": 1.0, "b": 0.0},
           "headline_metric": "nse", "headline_value": 0.2, "tolerance": {"nse": 0.01},
           "reference_split": "calibration"}
    rep, hist = _run(ref=ref, expected_case_id="SITE:fixture")
    assert rep["status"] == "runner_uncertified"
    cert = [h for h in hist if h["phase"] == "certify"]
    assert len(cert) == 1 and cert[0]["ok"] is False and cert[0]["reason"]


def test_the_certification_and_probe_runs_are_timed():
    from calibration_kit import test_step5_budget as S5
    ref = {"case_id": "SITE:fixture", "validated_run_id": "fixture-ref", "params": {"a": 1.0, "b": 0.0},
           "headline_metric": "nse", "headline_value": 1.0, "tolerance": {"nse": 0.05},
           "reference_split": "calibration"}
    rep, hist = _run(runner=S5._Sleepy, ref=ref, expected_case_id="SITE:fixture")
    w = {h["phase"]: h["wall_s"] for h in hist if h.get("outside_evaluator")}
    assert w["certify"] >= S5._Sleepy.SLEEP and w["objective_probe"] >= S5._Sleepy.SLEEP
    parts = rep["budget_plan"]["ledger"]["shared_parts_s"]
    assert parts["certify"] >= S5._Sleepy.SLEEP and parts["objective_probe"] >= S5._Sleepy.SLEEP


def test_the_machine_probe_lane_runs_are_logged(monkeypatch):
    from calibration_kit import test_step5_budget as S5
    from calibration_kit import compute
    monkeypatch.setattr(compute, "probe_machine", lambda: {"cores": 192, "load1": 0.0, "free_cores": 192,
                                                           "mem_available_gb": 64.0})
    S5.pin_pilot_memory(monkeypatch)
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "parallel_safe": True, "seeds": 2},
                            None, 20)
        S5.probe_contract(tmp, ki)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=None, budget=25, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    mp = [h for h in hist if h["phase"] == "machine_probe"]
    assert mp and all(h["outside_evaluator"] and h["wall_s"] >= 0.3 for h in mp)   # each lane run sleeps 0.3 s
    assert rep["phase_counts"]["machine_probe"]["n"] == len(mp)
    assert sorted(h["i"] for h in hist) == list(range(len(hist)))


# ── round-2 review (Opus): resume + probe, saved records, and the remaining rules ────────────────
class _ProbeDecides(S1._Runner):
    """The objective probe (the first runner call) carries the kit panel block with "mean near zero" and
    echoes "full"; every later call has NO panel block (so the pilot's default run decides nothing) and
    echoes correctly. The probe's payload therefore sets the pilot decisions."""
    def __init__(self, wd):
        super().__init__(wd, block_obs_near_zero=True, no_series=True)
        self.n = 0

    def __call__(self):
        self.n += 1
        m = super().__call__()
        if self.n == 1:
            m["__kdt__"]["split"] = "full"
        else:
            m["__kdt__"].pop("panel", None)
        return m


def test_a_probe_that_set_pilot_decisions_is_judged_on_the_first_call_and_on_the_resume():
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "seeds": 1}, None, 30)
        reps = []
        for _ in range(2):
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                reps.append(C.calibrate(ki, wd, {"streamflow": "point_time_series"},
                                        run_model=_ProbeDecides(wd), budget=None, seed=0))
        plan = json.loads(Path(wd, "kdt_budget_plan.json").read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert plan["probe_used_for_pilot"] is True and plan["probe_split_provenance"]["echo_wrong"] == 1
    assert plan["panel_setup"]["kit_missing"]                       # the probe's payload decided something
    for rep in reps:
        sp = rep["split_provenance"]
        assert "objective_probe" in sp["claim_phases"]
        assert sp["calibration_only_supported"] is False, sp["text"]
    assert reps[1]["split_provenance"]["by_phase"]["objective_probe"]["source"].startswith("the saved budget plan")
    assert "objective_probe (this call)" in reps[1]["split_provenance"]["by_phase"]   # this call's probe set aside


def test_a_probe_whose_payload_decided_nothing_is_not_in_the_claim():
    class _NoPanel(S1._Runner):
        def __init__(self, wd):
            super().__init__(wd, no_series=True)
            self.n = 0

        def __call__(self):
            self.n += 1
            m = super().__call__()
            m["__kdt__"].pop("panel", None)
            if self.n == 1:
                m["__kdt__"]["split"] = "full"
            return m
    rep, _ = _run(runner=_NoPanel)
    sp = rep["split_provenance"]
    assert "objective_probe" not in sp["claim_phases"] and sp["calibration_only_supported"] is True


def test_the_certification_split_follows_the_reference_split():
    for ref_split, want in (("holdout", "holdout"), (None, "full"), ("full", "full")):
        ref = {"case_id": "SITE:fixture", "validated_run_id": "fixture-ref", "params": {"a": 1.0, "b": 0.0},
               "headline_metric": "nse", "headline_value": 1.0, "tolerance": {"nse": 0.05}}
        if ref_split:
            ref["reference_split"] = ref_split
        rep, hist = _run(ref=ref, expected_case_id="SITE:fixture")
        cert = [h for h in hist if h["phase"] == "certify"]
        assert cert and cert[0]["split"] == want, (ref_split, cert)


FAIL_LANE_PY = """
import json, os, sys, time
time.sleep(0.2)
if "lane_1" in sys.argv[1]:
    sys.exit(1)
json.dump({"nse": 0.5}, open(sys.argv[1], "w"))
"""


def test_every_lane_run_is_logged_once_and_a_failed_one_is_ok_false(monkeypatch):
    from calibration_kit import compute
    monkeypatch.setattr(compute, "probe_machine", lambda: {"cores": 192, "load1": 0.0, "free_cores": 192,
                                                           "mem_available_gb": 64.0})
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        from calibration_kit import test_step5_budget as S5
        S5.pin_pilot_memory(monkeypatch)
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "parallel_safe": True, "seeds": 2},
                            None, 20)
        S5.probe_contract(tmp, ki, probe_body=("    time.sleep(0.2)\n    if 'lane_1' in out:\n        sys.exit(1)\n"
                                               "    json.dump({'nse': 0.5}, open(out, 'w'))"))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=None, budget=25, seed=0)
        hist = [json.loads(l) for l in Path(wd, "eval_history.jsonl").read_text().splitlines()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lp = rep["budget_plan"]["lane_probe"]
    made = sum(int(k) for k in (lp.get("efficiency") or {})) + sum(int(k) for k in (lp.get("errors") or {}))
    mp = [h for h in hist if h["phase"] == "machine_probe"]
    assert len(mp) == made == 3                                     # k = 1 (ok), k = 2 (one ok, one fails)
    assert sorted(h["ok"] for h in mp) == [False, True, True]
    assert sorted(h["i"] for h in hist) == list(range(len(hist)))


def test_a_wrong_search_echo_beats_an_unchecked_pilot():
    _, again = _two_calls(_Full, drop_saved=True)
    sp = again["split_provenance"]
    assert sp["calibration_only_supported"] is False and "pilot not checked" not in sp["text"]


def test_a_resume_with_wrong_search_echoes_stays_false():
    first, again = _two_calls(_Full)
    assert first["split_provenance"]["calibration_only_supported"] is False
    assert again["split_provenance"]["calibration_only_supported"] is False
    assert again["split_provenance"]["by_phase"]["search"]["echo_wrong"] == 30     # the cache hits too


def test_the_saved_pilot_record_is_this_calls_pilot_only():
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "seeds": 1}, None, 20)
        with open(Path(wd, "eval_history.jsonl"), "w") as fh:            # an older pilot with a wrong echo
            fh.write(json.dumps({"i": 0, "phase": "pilot", "split": "calibration", "split_echo": "full"}) + "\n")
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
        plan = json.loads(Path(wd, "kdt_budget_plan.json").read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert plan["pilot_split_provenance"]["calls"] == 10 and plan["pilot_split_provenance"]["echo_wrong"] == 0


def test_a_saved_pilot_with_no_calls_is_never_called_every_pilot_call():
    """(Since build step 8 a pilot whose default run gave no metrics stops at triage as no_baseline, so a
    saved pilot record with 0 calls only comes from an older plan: checked at the function.)"""
    empty = {"calls": 0, "echo_ok": 0, "echo_missing": 0, "echo_wrong": 0, "missing_at": [], "wrong_at": []}
    recs = [{"i": k, "phase": "search", "split": "calibration", "split_echo": "calibration"} for k in range(30)]
    sp = split_provenance(recs, saved={"pilot": empty})
    assert sp["calibration_only_supported"] is True and sp["text"].startswith("every search call")


def test_a_hand_edited_saved_record_never_crashes_the_report():
    for bad in ({}, {"calls": 3}, {"calls": "3", "echo_ok": 3, "echo_missing": 0, "echo_wrong": 0},
                "none", [1]):
        tmp = tempfile.mkdtemp(prefix="kdt_s7_")
        try:
            ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "seeds": 1}, None, 20)
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd), budget=None, seed=0)
            pf = Path(wd, "kdt_budget_plan.json")
            pl = json.loads(pf.read_text())
            pl["pilot_split_provenance"] = bad
            pf.write_text(json.dumps(pl))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                rep = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=S1._Runner(wd),
                                  budget=None, seed=0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        sp = rep["split_provenance"]
        assert rep["status"] == "completed" and sp["calibration_only_supported"] is None, (bad, sp)
        assert "not checked" in sp["text"]


def test_the_failed_and_crashed_reports_carry_the_split_provenance(monkeypatch):
    class _OnlyDefault(S1._Runner):
        """Works at the defaults only: a baseline exists (triage: calibrate), every search call fails."""
        def __call__(self):
            p = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
            if abs(float(p["a"]) - 0.8) < 1e-12 and abs(float(p["b"]) - 2.0) < 1e-12:
                return super().__call__()
            return {}
    rep, _ = _run(runner=_OnlyDefault)
    assert rep["status"] == "failed_no_finite_solution" and "split_provenance" in rep and "triage" in rep

    class _Nothing(S1._Runner):
        def __call__(self):
            return {}
    nb, _ = _run(runner=_Nothing)
    assert nb["status"] == "no_baseline" and nb["triage"]["route"] == "no_baseline"
    assert isinstance(nb["split_provenance"], dict) and "calibration_only_supported" in nb["split_provenance"]

    from calibration_kit import test_step6_seeds as S6

    class B(S6._Crash):
        bad = {0, 1}
    real = C._make_backend
    monkeypatch.setattr(C, "_make_backend", lambda algo: B(real(algo)))
    rep2, _ = _run()
    assert rep2["status"] == "search_crashed" and "split_provenance" in rep2 and "triage" in rep2
    assert rep2["split_provenance"]["by_phase"]["pilot"]["calls"] == 10



# ── round-3 review (Opus) ────────────────────────────────────────────────────────────────────────
class _PanelEverywhere(S1._Runner):
    """Every call carries the kit panel block with "mean near zero" (what the step-9 helper will write);
    the probe echoes "full". The pilot's default run then decides, not the probe."""
    def __init__(self, wd):
        super().__init__(wd, block_obs_near_zero=True, no_series=True)
        self.n = 0

    def __call__(self):
        self.n += 1
        m = super().__call__()
        if self.n == 1:
            m["__kdt__"]["split"] = "full"
        return m


def _two_calls_with(runner, edit=None):
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "seeds": 1}, None, 30)
        reps = []
        for k in range(2):
            if k == 1 and edit is not None:
                pf = Path(wd, "kdt_budget_plan.json")
                pl = json.loads(pf.read_text())
                edit(pl)
                pf.write_text(json.dumps(pl))
            with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
                reps.append(C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=runner(wd),
                                        budget=None, seed=0))
        plan = json.loads(Path(wd, "kdt_budget_plan.json").read_text())
        return reps, plan
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_probe_is_not_in_the_claim_when_the_pilots_default_run_decided():
    reps, plan = _two_calls_with(_PanelEverywhere)
    assert plan["probe_used_for_pilot"] is False and plan["panel_setup"]["kit_missing"]
    for rep in reps:
        sp = rep["split_provenance"]
        assert "objective_probe" not in sp["claim_phases"] and sp["calibration_only_supported"] is True


def test_a_saved_probe_flag_that_is_not_true_or_false_is_not_checked():
    for bad in ("true", 1, None, "drop"):
        def edit(pl, bad=bad):
            if bad == "drop":
                pl.pop("probe_used_for_pilot", None)
            else:
                pl["probe_used_for_pilot"] = bad
        reps, _ = _two_calls_with(_ProbeDecides, edit=edit)
        sp = reps[1]["split_provenance"]
        # the probe that set the decisions echoed "full": its record is gone, so the claim is not True
        assert sp["calibration_only_supported"] is None and "objective probe not checked" in sp["text"], (bad, sp)


def test_a_missing_saved_probe_record_sets_this_calls_probe_aside():
    reps, _ = _two_calls_with(_PanelEverywhere, edit=lambda pl: (pl.update(probe_used_for_pilot=True),
                                                                  pl.pop("probe_split_provenance", None)))
    sp = reps[1]["split_provenance"]
    assert "objective_probe (this call)" in sp["by_phase"] and "objective_probe" not in sp["by_phase"]
    assert sp["calibration_only_supported"] is None and "objective probe not checked" in sp["text"]


def test_a_saved_record_must_be_whole_non_negative_counts_that_add_up():
    from calibration_kit.evaluator import valid_prov_record
    ok = {"calls": 3, "echo_ok": 2, "echo_missing": 1, "echo_wrong": 0, "missing_at": [4], "wrong_at": []}
    assert valid_prov_record(ok)
    for k, v in (("calls", -1), ("echo_ok", True), ("echo_wrong", 1.0), ("missing_at", "4")):
        assert not valid_prov_record(dict(ok, **{k: v})), (k, v)
    assert not valid_prov_record(dict(ok, echo_missing=5))                     # 2 + 5 + 0 != 3
    assert not valid_prov_record(dict(ok, calls=5))                            # 5 != 2 + 1 + 0
    # counts that DO add up but are negative or true/false are still rejected
    assert not valid_prov_record({"calls": 1, "echo_ok": 2, "echo_missing": -1, "echo_wrong": 0})
    assert not valid_prov_record({"calls": 1, "echo_ok": True, "echo_missing": 0, "echo_wrong": 0})
    reps, _ = _two_calls_with(S1._Runner, edit=lambda pl: pl.update(pilot_split_provenance={
        "calls": 0, "echo_ok": 0, "echo_missing": 5, "echo_wrong": 0}))
    sp = reps[1]["split_provenance"]
    assert sp["calibration_only_supported"] is None and "not usable" in sp["text"]


def test_a_caller_can_no_longer_hand_in_a_pilot_or_a_parameter_subset():
    """Build step 8b: the staged-calibration arguments are removed; a caller that still passes them is
    told so at once (TypeError), never silently given a one-seed, no-plan search."""
    for kw in ({"pilot": {"n": 1}}, {"pilot": False}, {"active_idx": [0]}):
        with pytest.raises(TypeError):
            C.calibrate("ki", "wd", {}, **kw)


@pytest.mark.parametrize("saved_flag", [False, True])       # True: how a kit-written plan has a probe record
def test_a_resume_that_re_derives_the_pilot_decisions_judges_this_calls_probe(saved_flag):
    """Opus r4 E19: the saved panel setup cannot be reused (here: removed), so the decisions are derived
    again from the saved pilot and THIS call's probe — which sets "mean near zero" and echoes "full"."""
    tmp = tempfile.mkdtemp(prefix="kdt_s7_")
    try:
        ki, wd = E._fixture(tmp, {"mode": "measured", "allowance": "10m", "seeds": 1}, None, 30)
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            first = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_ProbeDecides(wd),
                                budget=None, seed=0)
        pf = Path(wd, "kdt_budget_plan.json")
        pl = json.loads(pf.read_text())
        assert pl["probe_used_for_pilot"] is True
        pl.pop("panel_setup")                    # the saved setup cannot be reused ...
        pl["probe_used_for_pilot"] = saved_flag  # ... whatever the saved flag says
        # a CLEAN saved probe record: only this call's probe (echo "full") can make the claim False
        pl["probe_split_provenance"] = {"calls": 1, "echo_ok": 1, "echo_missing": 0, "echo_wrong": 0,
                                        "missing_at": [], "wrong_at": []}
        pf.write_text(json.dumps(pl))
        with E._clean_env(), contextlib.redirect_stdout(io.StringIO()):
            again = C.calibrate(ki, wd, {"streamflow": "point_time_series"}, run_model=_ProbeDecides(wd),
                                budget=None, seed=0)
        plan = json.loads(pf.read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert first["split_provenance"]["calibration_only_supported"] is False
    sp = again["split_provenance"]
    assert again["budget_plan"].get("resumed_cap") and "objective_probe" in sp["claim_phases"]
    assert sp["calibration_only_supported"] is False
    assert plan["probe_used_for_pilot"] is True and plan["probe_split_provenance"]["echo_wrong"] == 1



def test_a_reuse_that_fails_part_way_is_judged_on_the_saved_flag():
    """Opus r6 note A: a saved setup the reuse branch cannot finish (hand edit) — the saved decisions may
    already be applied, so the saved flag (the probe decided) must still count."""
    reps, _ = _two_calls_with(_ProbeDecides, edit=lambda pl: pl["panel_setup"]["kit_missing"].update(zzz=[]))
    sp = reps[1]["split_provenance"]
    assert "objective_probe" in sp["claim_phases"] and sp["calibration_only_supported"] is False, sp["text"]
