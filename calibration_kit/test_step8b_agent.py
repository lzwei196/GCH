"""Build step 8b (gaps 2r, 2x agent side): the calibration agent and its readers follow the rebuilt engine.

- the contract-writing prompt (stage_calibrate.py) asks the three questions, explains DDS, sets the default,
  writes each target's `kind`, writes `parallel_safe` only when copies share nothing, and no longer offers `staged`;
- the agent acts on the engine's no-search routes (`no_baseline` -> driver repair, `not_calibratable` -> setup
  diagnosis; a certified module's no_baseline -> setup diagnosis) and never on fit quality;
- calibrate.py maps every engine status to an honest verdict and records the `triage` block;
- db_calibration.py's verdict list holds the new verdicts (for new databases)."""
import importlib.util
import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "auto_dissect_multi_agent"))

for _module in ("calibrate", "db_calibration", "stage_calibrate"):
    if importlib.util.find_spec(_module) is None:
        pytest.skip(f"Requires the external KI host module: {_module}", allow_module_level=True)
import calibrate as CAL
import db_calibration as DB
import stage_calibrate as SC


# ── the prompt (gap 2r) ─────────────────────────────────────────────────────────────────────────
def _prompt_text():
    return (ROOT / "auto_dissect_multi_agent" / "stage_calibrate.py").read_text()


def test_the_prompt_asks_the_three_questions_and_sets_the_default():
    t = _prompt_text()
    assert "Roughly how long does one model run take, if you know?" in t
    assert "How long can this run take? For example, 3 days on this server." in t
    assert "should it stop there to save time, or keep going to the budget" in t
    assert "DEFAULT when nobody answers: keep_going" in t
    assert "DDS cannot stop early" in t and "Tolson & Shoemaker 2007" in t


def test_the_prompt_writes_kind_and_parallel_safe_and_no_longer_offers_staged():
    t = _prompt_text()
    assert "`strategy.convergence.variables.<var>.kind` for EVERY target" in t
    for k in ("`flow`", "`series`", "`snapshot`", "`categorical`"):
        assert k in t
    assert "`parallel_safe: true`" in t and "share NOTHING" in t
    assert "`staged: true`" not in t and "do NOT write `staged`" in t
    assert "The cap is PER SEED" in t and "3-run" in t
    # the removed Morris pool / screen is not described any more (Opus 8b/9 r1 #10)
    for gone in ("Declare a POOL", "Morris screen confirms", "screen_failed` calibration", "staged rounds hand you",
                 "sensitivity-screen corner"):
        assert gone not in t, gone


# ── the routes (design §1) ──────────────────────────────────────────────────────────────────────
def test_the_triage_route_is_read_from_the_engine_report():
    assert SC.triage_route_of({"status": "no_baseline", "route": "no_baseline"}) == "no_baseline"
    assert SC.triage_route_of({"status": "completed", "triage": {"route": "calibrate"}}) == "calibrate"
    assert SC.triage_route_of({"status": "runner_emits_no_objective_metric"}) == "no_baseline"
    assert SC.triage_route_of(None) is None
    assert SC.NO_SEARCH_ROUTES == ("no_baseline", "not_calibratable")


def _fake_orchestrator(monkeypatch, calls):
    O = types.ModuleType("orchestrator")

    def run_claude_resilient(prompt, **kw):
        calls.append(prompt)
        return None, "AGENT OUTPUT"
    O.run_claude_resilient = run_claude_resilient
    monkeypatch.setitem(sys.modules, "orchestrator", O)
    tr = types.ModuleType("tool_reviewer")

    class _V:
        verdict, issues, reviewer = "REQUEST_CHANGES", [], "stub"
    tr.review_tool = lambda **kw: _V()
    monkeypatch.setitem(sys.modules, "tool_reviewer", tr)
    lp = types.ModuleType("learning_proposals")
    lp.propose_out_of_scope_finding = lambda **kw: None
    monkeypatch.setitem(sys.modules, "learning_proposals", lp)
    monkeypatch.setattr(SC, "_archive_capability", lambda *a, **k: "archived")


def _ki(tmp):
    ki = Path(tmp, "ki")
    (ki / "tools").mkdir(parents=True)
    (ki / "calibration.yaml").write_text("x: 1\n")
    (ki / "tools" / "calib_run.py").write_text("# runner\n")
    return ki


REPORTS = {
    "no_baseline": {"status": "no_baseline", "route": "no_baseline",
                    "reason": "the pilot's default run gives no finite objective metric",
                    "triage": {"route": "no_baseline", "proof": {"verdicts": {}}, "per_target": {"Q": {}}},
                    "pilot": {"default_metrics": {}}},
    "not_calibratable": {"status": "not_calibratable", "route": "not_calibratable",
                         "reason": "every searched parameter was proven not to move the objectives",
                         "triage": {"route": "not_calibratable",
                                    "proof": {"verdicts": {"par_zz_one": "OBJECTIVE_MASKED", "par_zz_two": "DORMANT"}},
                                    "per_target": {"Q": {"fit_verdict": "structural"}}},
                         "pilot": {"default_metrics": {"nse": 0.2345}}},
}


@pytest.mark.parametrize("route, action, prompt_head", [
    ("no_baseline", "driver_repair", "# Calibration-Driver Repair Agent"),
    ("not_calibratable", "setup_diagnosis", "# Calibration Setup-Diagnosis Agent")])
def test_each_no_search_route_starts_its_agent_with_the_triage_detail(monkeypatch, route, action, prompt_head):
    calls = []
    _fake_orchestrator(monkeypatch, calls)
    with tempfile.TemporaryDirectory() as tmp:
        out = SC.act_on_calibration_blocker("M", _ki(tmp), tmp, dict(REPORTS[route]), {"nse": 0.7})
    assert out["action"] == action and out["status"] == "ran"
    assert len(calls) == 1 and calls[0].startswith(prompt_head)
    assert f"route={route}" in calls[0]
    if route == "not_calibratable":                   # the triage DETAIL reaches the agent (not only the route text)
        assert '"par_zz_one": "OBJECTIVE_MASKED"' in calls[0] and '"par_zz_two": "DORMANT"' in calls[0]
        assert '"nse": 0.2345' in calls[0]


def test_a_calibrate_route_or_an_old_route_starts_no_agent(monkeypatch):
    calls = []
    _fake_orchestrator(monkeypatch, calls)
    with tempfile.TemporaryDirectory() as tmp:
        ki = _ki(tmp)
        for rep in ({"status": "completed", "triage": {"route": "calibrate"}}, {"route": "already_adequate"},
                    {"route": "diagnose_setup"}):
            out = SC.act_on_calibration_blocker("M", ki, tmp, rep, None)
            assert out["action"] == "none"
    assert calls == []


def test_run_stage_acts_on_the_new_routes_and_sends_a_certified_no_baseline_to_setup_diagnosis(monkeypatch):
    calls = []
    _fake_orchestrator(monkeypatch, calls)
    monkeypatch.setenv("KDT_CALIBRATE", "1")
    monkeypatch.setattr(SC, "_has_contract", lambda ki: True)
    monkeypatch.setattr(SC, "_capability_matches_case", lambda ki, case: True)
    from calibration_kit import calib
    monkeypatch.setattr(calib, "calibrate", lambda **kw: dict(REPORTS["no_baseline"]))
    with tempfile.TemporaryDirectory() as tmp:
        ki = _ki(tmp)
        rep = SC.run_stage("M", ki, tmp, {"Q": "point_time_series"})
        assert rep["triage_action"]["action"] == "driver_repair"
        (ki / "calib").mkdir()
        (ki / "calib" / "reference_run.json").write_text("{}")         # a CERTIFIED module
        rep2 = SC.run_stage("M", ki, tmp, {"Q": "point_time_series"})
        assert rep2["triage_action"]["action"] == "setup_diagnosis"
    assert calls[0].startswith("# Calibration-Driver Repair Agent")
    assert calls[1].startswith("# Calibration Setup-Diagnosis Agent")


# ── the report -> verdict mapping and the recorded keys ───────────────────────────────────────
@pytest.mark.parametrize("report, verdict", [
    ({"status": "no_baseline", "route": "no_baseline"}, "no_baseline"),
    ({"status": "runner_emits_no_objective_metric"}, "no_baseline"),
    ({"status": "not_calibratable", "route": "not_calibratable"}, "not_calibratable"),
    ({"status": "pilot_unstable"}, "pilot_unstable"),
    ({"status": "budget_exhausted"}, "budget_exhausted"),
    ({"status": "search_crashed"}, "search_crashed"),
    ({"status": "workdir_busy"}, "workdir_busy"),
    ({"status": "param_unreachable"}, "param_unreachable"),
    ({"status": "completed", "triage": {"route": "calibrate"}, "holdout": {"passed": True}}, "holdout_pass"),
    ({"status": "completed", "triage": {"route": "calibrate"}, "holdout": {"passed": False}}, "holdout_fail"),
    ({"status": "completed", "triage": {"route": "calibrate"}}, "holdout_inconclusive"),
])
def test_every_engine_status_maps_to_an_honest_verdict(report, verdict):
    assert CAL._map_report_to_verdict(report)[1] == verdict


def test_every_verdict_the_mapping_can_give_is_allowed_by_the_database(tmp_path):
    db = str(tmp_path / "calibration.db")
    DB.ensure_table(db)
    con = sqlite3.connect(db)
    try:
        for v in ("no_baseline", "not_calibratable", "pilot_unstable", "budget_exhausted", "search_crashed",
                  "workdir_busy", "param_unreachable"):
            con.execute("INSERT INTO calibration_runs (run_id, model_id, case_id, engine_status, scientific_verdict) "
                        "VALUES (?, 'M', 'C', 'completed', ?)", (f"r_{v}", v))
        con.commit()
    finally:
        con.close()


@pytest.mark.parametrize("report, dm, route", [
    # a completed run: the pilot summary lives in the budget plan
    ({"status": "completed", "triage": {"route": "calibrate", "reason": "the run can be calibrated",
                                        "proof": {"verdicts": {"k": "ALIVE"}}},
      "budget_plan": {"pilot_summary": {"default_metrics": {"nse": 0.61}}}, "holdout": {"passed": True}},
     {"nse": 0.61}, "calibrate"),
    # a no-search run: the pilot summary is the report's own `pilot`
    (dict(REPORTS["not_calibratable"]), {"nse": 0.2345}, "not_calibratable")])
def test_the_recorded_row_reads_the_triage_block_and_the_pilot(monkeypatch, tmp_path, report, dm, route):
    rows = []
    O = types.ModuleType("orchestrator")
    O.resolve_ki_path = lambda m: (str(tmp_path / "ki"), None)
    O.WORK_DIR = str(tmp_path / "work")
    monkeypatch.setitem(sys.modules, "orchestrator", O)
    (tmp_path / "ki").mkdir()
    monkeypatch.setattr(DB, "record_calibration_run", lambda row, *a, **k: rows.append(row))
    monkeypatch.setattr(CAL, "readiness", lambda *a, **k: {
        "case_id": "C", "validated_run_id": "v", "determining_metric": "nse", "obs_shape_by_var": {},
        "prior_metric": None})
    monkeypatch.setattr(SC, "run_stage", lambda *a, **k: dict(report))
    CAL.calibrate_one("M", db_path=str(tmp_path / "c.db"))
    r = rows[-1]
    assert r["baseline_metrics_json"] == dm
    assert r["sensitivity_summary_json"] == report["triage"]["proof"]
    assert r["triage_route"] == route and r["triage_reason"]


def test_a_no_baseline_the_engine_reports_as_certified_goes_to_setup_diagnosis(monkeypatch):
    """Opus 8b/9 r1 #11: System 1's site contracts certify through runner.reference_run (no KI file); the agent
    reads the engine's `certified`."""
    calls = []
    _fake_orchestrator(monkeypatch, calls)
    monkeypatch.setenv("KDT_CALIBRATE", "1")
    monkeypatch.setattr(SC, "_has_contract", lambda ki: True)
    monkeypatch.setattr(SC, "_capability_matches_case", lambda ki, case: True)
    from calibration_kit import calib
    monkeypatch.setattr(calib, "calibrate", lambda **kw: dict(REPORTS["no_baseline"], certified=True))
    with tempfile.TemporaryDirectory() as tmp:
        ki = _ki(tmp)                                   # no calib/reference_run.json in the KI
        rep = SC.run_stage("M", ki, tmp, {"Q": "point_time_series"})
    assert rep["triage_action"]["action"] == "setup_diagnosis"
    assert calls[0].startswith("# Calibration Setup-Diagnosis Agent")


# ── review round 2 (Opus 8b/9): the job each route's agent is given ─────────────────────────────
def _prompt_for(monkeypatch, report, certified_file=False):
    calls = []
    _fake_orchestrator(monkeypatch, calls)
    with tempfile.TemporaryDirectory() as tmp:
        ki = _ki(tmp)
        if certified_file:
            report = dict(report, _action_route="not_calibratable")
        SC.act_on_calibration_blocker("M", ki, tmp, dict(report), None)
    return calls[0]


def test_the_driver_repair_job_is_about_the_default_run():
    t = _prompt_for(pytest.MonkeyPatch(), REPORTS["no_baseline"])
    assert "make the DEFAULT run run and score" in t and "at the defaults AND at 3 random in-range vectors" in t
    assert "Find WHY a parameter change doesn't change the scored series" not in t


def test_the_not_calibratable_job_walks_every_proof_verdict_and_the_decoupled_scorer():
    t = _prompt_for(pytest.MonkeyPatch(), REPORTS["not_calibratable"])
    for v in ("DORMANT:", "OBJECTIVE_MASKED:", "WINDOW_MASKED:"):
        assert v in t
    assert "the scorer\n     does not read the output this run produced" in t or "scorer does not read the output" in t.replace("\n     ", " ")
    assert "This module is CERTIFIED" not in t


def test_a_certified_no_baseline_gets_the_setup_job_written_for_it():
    t = _prompt_for(pytest.MonkeyPatch(), REPORTS["no_baseline"], certified_file=True)
    assert t.startswith("# Calibration Setup-Diagnosis Agent")
    assert "This module is CERTIFIED" in t and "the contract's parameter DEFAULTS vs the reference run" in t
    assert "cannot start until" not in t and "although\nthis module's driver reproduced" in t
    assert "For EACH parameter, use its consumption-proof verdict" not in t


def test_the_contract_prompt_makes_workflows_report_the_watched_scores():
    t = _prompt_text()
    for line in ("alpha  = std(sim) / std(obs)", "beta   = mean(sim) / mean(obs)",
                 "pbias  = 100 * (mean(sim) - mean(obs)) / mean(obs)",
                 "lnnse  = NSE of log(x + 0.01*mean(obs))", "nrmse  = 100 * RMSE / |mean(obs)|"):
        assert line in t, line
    # the watched scores go in the kit's reserved section, per target (Leo, 2026-09-30)
    assert 'out["__kdt__"]["panel"] = {{"<var>":' in t and "under the target's `var` name" in t
    assert "kdt_panel" not in t and "extended=True" not in t          # nothing that is not installed
    # the scored series, once, when the kit asks — so the tolerances are measured (Leo, 2026-09-30)
    assert "KDT_CALIB_EMIT_SERIES=1" in t and "kdt_series_<target var>.npz" in t and "Never let this step fail the run" in t
    # a date array the kit can read (not an object array), ADDED to the existing __kdt__ block (Opus 8b/9 r4 #2, #3)
    assert 'dtype="U16"' in t and "[z[k] for k in z.files]" in t and "ADD the paths to your existing `__kdt__` block" in t
    assert '"YYYY-MM-DD" or "YYYY-MM-DDTHH:MM"' in t and '"<second var>": "<its path>"' in t and "series_error" in t
    assert "nse    = 1 - sum((sim - obs)^2)" in t and "kge    = 1 - sqrt(" in t and "exact lower-case keys" in t
    # commissioning reproduces the pilot's default run exactly
    assert "ALSO run ONCE exactly as the kit's pilot runs the default vector" in t
