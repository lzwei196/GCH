"""Option 1 (Leo, 2026-10-02): the engine's own pilot checks the runner's reply (`reply_gate`); when the watched
scores or the series are missing an agent upgrades the reply inside a guarded transaction — kept only on a reviewer's
APPROVE, a complete reply in the engine's pilot, and unchanged own scores; anything else restores the package."""
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "auto_dissect_multi_agent"))

from calibration_kit import test_calibrate_convergence_e2e as E          # noqa: E402

OLD_RUN = """
import json, os, sys
sys.path.insert(0, {root!r})
import numpy as np
from calibration_kit import test_calibrate_convergence_e2e as E
from calibration_kit import panel as P
out = sys.argv[sys.argv.index("--out") + 1]
p = json.load(open(os.environ["KDT_CALIB_PARAMS"]))
sim_all = float(p["a"]) * E.OBS + float(p["b"])
sl = E.HOLD if os.environ.get("KDT_CALIB_SPLIT") == "holdout" else E.CAL
f = P.panel_from_series(sim_all[sl], E.OBS[sl], "flow", mean_rule=False)
reply = {{"nse": f["nse"], "kge": f["kge"], "pbias": f["pbias"], "r": f["r"],
         "__kdt__": {{"applied_params": p, "case_id": "SITE:fixture",
                     "split": os.environ.get("KDT_CALIB_SPLIT", "calibration")}}}}
json.dump(reply, open(out, "w"))
"""

NEW_TAIL = """
wd = sys.argv[sys.argv.index("--workdir") + 1]
s, o = sim_all[sl], E.OBS[sl]
reply["__kdt__"]["panel"] = {{"streamflow": {{k: f[k] for k in ("r", "alpha", "beta", "pbias", "lnnse", "nrmse", "nse", "kge") if k in f}}}}
if os.environ.get("KDT_CALIB_EMIT_SERIES") == "1":
    sp = os.path.join(wd, "kdt_series_streamflow.npz")
    np.savez(sp, sim=s, obs=o, date=np.asarray([str(np.datetime64("1981-01-01") + i) for i in range(len(s))], dtype="U16"))
    reply["__kdt__"]["series"] = {{"streamflow": sp}}
json.dump(reply, open(out, "w"))
"""


def _ki(tmp, new=False):
    conv = {"mode": "keep_going", "variables": {"streamflow": {"kind": "flow"}}}
    ki, wd = E._fixture(tmp, {"seeds": 1}, conv, 30)
    (Path(ki) / "tools").mkdir(exist_ok=True)
    src = OLD_RUN.format(root=str(ROOT)) + (NEW_TAIL.format() if new else "")
    (Path(ki) / "tools" / "calib_run.py").write_text(src)
    c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
    c["runner"] = {"kind": "subprocess", "cwd": "{ki_path}",
                   "command": [sys.executable, "tools/calib_run.py", "--out", "{metrics_json}", "--workdir", "{workdir}"]}
    yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
    return ki, wd


OBS_SHAPE = {"streamflow": "point_time_series"}
KW = dict(obs_shape_by_var=OBS_SHAPE)


def _gate(ki, wd, gate=True):
    from calibration_kit import calib
    return calib.calibrate(ki_path=str(ki), workdir=str(wd), reply_gate=gate, **KW)


def _runner(ki):
    return Path(ki) / "tools" / "calib_run.py"


class _Tmp:
    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="kdt_ru_")
        self.env = E._clean_env()
        self.env.__enter__()
        return self.tmp

    def __exit__(self, *a):
        self.env.__exit__(*a)
        shutil.rmtree(self.tmp, ignore_errors=True)


# ── the engine's gate ──────────────────────────────────────────────────────────────────────────
def test_the_gate_stops_an_old_runner_after_the_pilot_and_lets_a_new_one_search():
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        old = _gate(ki, wd)
        left = sorted(p.name for p in Path(wd).iterdir() if p.name.startswith("calib_plan"))
        _runner(ki).write_text(OLD_RUN.format(root=str(ROOT)) + NEW_TAIL.format())
        chk = _gate(ki, wd, "check")
        new = _gate(ki, wd)
    assert old["status"] == "reply_incomplete" and old["reply"]["complete"] is False
    assert set(old["reply"]["missing"]["streamflow"]) == {"alpha", "lnnse"}
    assert old["reply"]["series"]["streamflow"] is False
    assert not left                                                   # nothing saved for a resume
    assert isinstance(old["default_reply"], dict) and "nse" in old["default_reply"]
    assert chk["status"] == "reply_checked" and chk["reply"]["complete"] is True
    assert new["status"] not in ("reply_incomplete", "reply_checked")
    assert new["convergence"]["reply"]["complete"] is True


def test_no_gate_means_an_old_runner_is_searched_as_before():
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _gate(ki, wd, None)
    assert rep["status"] not in ("reply_incomplete", "reply_checked")
    assert rep["convergence"]["reply"]["complete"] is False            # reported, never a stop


def test_a_reply_whose_scores_disagree_with_its_series_is_incomplete():
    """the gate uses the 2n check: watched scores not computed on the saved pairs are a mismatch."""
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        bad = NEW_TAIL.format().replace('reply["__kdt__"]["panel"] = ', 'f["alpha"] = f["alpha"] * 1.5\nreply["__kdt__"]["panel"] = ')
        _runner(ki).write_text(OLD_RUN.format(root=str(ROOT)) + bad)
        rep = _gate(ki, wd)
    assert rep["status"] == "reply_incomplete" and "alpha" in str(rep["reply"]["mismatch"].get("streamflow"))


# ── the upgrade of the PROJECT's runner ─────────────────────────────────────────────────────────────────────────
def _fake_tools(monkeypatch, verdict, tail=NEW_TAIL, also=None, boom=None, agent_reply='{"status": "upgraded"}'):
    import stage_calibrate as SC
    calls, reviews = [], []
    O = types.ModuleType("orchestrator")

    def run_claude_resilient(prompt, **kw):
        calls.append(prompt)
        view = Path(prompt.split("Edit `", 1)[1].split("/tools/calib_run.py", 1)[0])
        if tail is not None:
            p = view / "tools" / "calib_run.py"
            p.write_text(p.read_text() + tail.format())
        if also:
            also(view)
        if boom:
            raise boom
        return (None if agent_reply is None else 0), (agent_reply or "")
    O.run_claude_resilient = run_claude_resilient
    monkeypatch.setitem(sys.modules, "orchestrator", O)
    tr = types.ModuleType("tool_reviewer")

    class _V:
        issues, reviewer = [], "stub"
    _V.verdict = verdict

    def review_tool(**kw):
        reviews.append(kw)
        if verdict == "RAISE":
            raise RuntimeError("reviewer down")
        return _V()
    tr.review_tool = review_tool
    monkeypatch.setitem(sys.modules, "tool_reviewer", tr)
    caps = []
    monkeypatch.setattr(SC, "_write_capability_case", lambda *a, **k: caps.append(a))
    monkeypatch.delenv("KDT_CALIB_REPLY_UPGRADE", raising=False)
    return SC, calls, reviews, caps


def _tree(d):
    """every entry, links not followed, byte-code included"""
    out = {}
    for p in sorted(Path(d).rglob("*")):
        rel = str(p.relative_to(d))
        out[rel] = ("L", os.readlink(p)) if p.is_symlink() else (p.read_bytes() if p.is_file() else "D")
    return out


def _go(SC, ki, tmp):
    return SC.calibrate_with_reply_upgrade("M", ki, Path(tmp) / "project", case=None, **KW)


def _proj(tmp):
    p = Path(tmp) / "project"
    return p, p / "ki_view", p / "reply_upgrade"


def _own(view):
    return {r: ((view / r).read_bytes() if (view / r).is_file() else None)
            for r in ("tools/calib_run.py", "calibration.yaml", "calib/capability_case.json")}


@pytest.mark.parametrize("verdict, kept, remembered", [("APPROVE", True, False), ("REQUEST_CHANGES", False, True),
                                                       ("RAISE", False, False), ("WAIT", False, False)])
def test_the_projects_runner_is_upgraded_only_on_approve_and_the_ki_is_never_written(monkeypatch, verdict, kept,
                                                                                    remembered):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, verdict)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        ki_before = _tree(ki)
        rep = _go(SC, ki, tmp)
        proj, view, d = _proj(tmp)
        ki_after = _tree(ki)
        runner = (view / "tools" / "calib_run.py").read_text()
        marker, failed = (d / "IN_PROGRESS.json").exists(), (d / "failed.json").exists()
        scratch = (d / "agent_scratch").exists() or (d / "ki_before.json").exists()
        cap_target = caps[0][0] if caps else None
    up = rep["reply_upgrade"]
    assert ki_after == ki_before                                      # the KI: not one byte, not one byte-code file
    assert len(calls) == 1 and "Calibration-Runner Reply Upgrade Agent" in calls[0]
    assert f"Edit `{view}/tools/calib_run.py`" in calls[0] and str(d / "agent_scratch") in calls[0]
    assert "alpha  = std(sim) / std(obs)" in calls[0] and '"alpha"' in calls[0]      # the rules + what is missing
    assert "alpha  = std(sim) / std(obs)" in reviews[0]["tool_summary"]["review_focus"]
    assert "holdout" in reviews[0]["tool_summary"]["review_focus"]
    assert "import json, os, sys" in reviews[0]["diff"]                # the whole file, not only the hunk
    assert up["status"] == ("upgraded" if kept else "restored") and up["reply_before"]["complete"] is False
    assert ('["panel"]' in runner) is kept
    assert not marker and not scratch and failed is remembered and bool(caps) is kept
    assert up.get("transient", False) is (verdict in ("RAISE", "WAIT"))
    if kept:
        assert cap_target == str(view)                                # the approval record is the project's
        assert rep["convergence"]["reply"]["complete"] is True         # the search ran on the upgraded reply
    else:
        assert rep["convergence"]["reply"]["complete"] is False        # ... the old runner still calibrates
    assert rep["status"] not in ("reply_incomplete", "reply_checked")
    assert rep["project_workflow"]["used"] is True and rep["project_workflow"]["view"] == str(view)


def test_an_agent_that_changes_nothing_ends_without_a_review_and_may_try_again(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=None)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        failed = (_proj(tmp)[2] / "failed.json").exists()
    assert rep["reply_upgrade"]["status"] == "unchanged" and not reviews and not caps and not failed


def test_an_agent_that_was_not_available_is_not_held_against_the_runner(monkeypatch):
    """the usage limit: the helper returns (None, 'ENV_ABORT...') and the runner is as before."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=None, agent_reply=None)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        failed = (_proj(tmp)[2] / "failed.json").exists()
    assert rep["reply_upgrade"]["status"] == "unchanged" and rep["reply_upgrade"]["transient"] is True and not failed


def test_a_runner_whose_upgrade_was_rejected_is_not_tried_again_in_a_fresh_run_folder(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "REQUEST_CHANGES")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        _go(SC, ki, tmp)
        shutil.rmtree(_proj(tmp)[0] / "run")                             # a fresh search, the same project runner
        rep2 = _go(SC, ki, tmp)
    assert len(calls) == 1 and rep2["reply_upgrade"] == {"status": "skipped",
                                                         "reason": "an upgrade of this runner failed before"}


def test_an_upgrade_whose_reply_is_still_incomplete_is_put_back(monkeypatch):
    half = NEW_TAIL.replace('"r", "alpha", "beta", "pbias", "lnnse"', '"r", "beta", "pbias", "lnnse"')
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=half)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        runner = (_proj(tmp)[1] / "tools" / "calib_run.py").read_bytes()
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
    up = rep["reply_upgrade"]
    assert up["status"] == "restored" and "still incomplete" in up["reason"] and runner == orig and not caps
    assert up["check_after"]["reply"]["missing"]["streamflow"] == ["alpha"]


@pytest.mark.parametrize("edit, word", [
    ('reply["nse"] = reply["nse"] - 0.01', "own default-run scores"),                # a score moved
    ('reply["bias_new"] = 5.0', "bias_new: new"),                                     # a NEW plain score
    ('reply.pop("kge")', "kge: gone"),                                                # a score dropped
])
def test_an_upgrade_that_touches_the_models_own_scores_is_put_back(monkeypatch, edit, word):
    moved = NEW_TAIL.replace('json.dump(reply, open(out, "w"))', edit + '\njson.dump(reply, open(out, "w"))')
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=moved)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        runner = (_proj(tmp)[1] / "tools" / "calib_run.py").read_bytes()
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
    up = rep["reply_upgrade"]
    assert up["status"] == "restored" and word in up["reason"] and runner == orig and not caps


def test_the_projects_contract_and_approval_record_are_put_back_when_the_agent_edits_them(monkeypatch):
    def meddle(view):
        (view / "calibration.yaml").write_text((view / "calibration.yaml").read_text() + "\n# meddled\n")
        (view / "calib").mkdir(exist_ok=True)
        (view / "calib" / "capability_case.json").write_text('{"forged": true}')
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", also=meddle)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        view = _proj(tmp)[1]
        contract = (view / "calibration.yaml").read_text()
        cap = (view / "calib" / "capability_case.json").exists()
    up = rep["reply_upgrade"]
    assert up["status"] == "upgraded" and "# meddled" not in contract and not cap
    assert set(up["other_edits_put_back"]) == {"calibration.yaml", "calib/capability_case.json"}


@pytest.mark.parametrize("stray", ["edit_helper_through_link", "add_file_in_linked_folder", "edit_ki_runner",
                                   "edit_dag"])
def test_an_agent_that_changes_the_ki_stops_everything(monkeypatch, stray):
    """the links let the agent reach the KI: any change there fails the upgrade, the project's runner is put back,
    and NO search runs. The KI is not rolled back — the changed entries are reported."""
    box = {}

    def meddle(view):
        ki = Path(box["ki"])
        if stray == "edit_helper_through_link":
            (view / "tools" / "helper_x.py").write_text("X = 2  # edited through the link\n")
        elif stray == "add_file_in_linked_folder":
            (view / "docs" / "added_by_agent.md").write_text("x")
        elif stray == "edit_ki_runner":
            (ki / "tools" / "calib_run.py").write_text((ki / "tools" / "calib_run.py").read_text() + "\n# x\n")
        else:
            (view / "dag.yaml").write_text((view / "dag.yaml").read_text() + "\n# x\n")
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", also=meddle)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        box["ki"] = ki
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")
        (Path(ki) / "docs").mkdir(exist_ok=True)
        (Path(ki) / "docs" / "note.md").write_text("n")
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
        rep = _go(SC, ki, tmp)
        proj, view, d = _proj(tmp)
        runner = (view / "tools" / "calib_run.py").read_bytes()
        rep2 = _go(SC, ki, tmp)                                # run again without putting the KI right: stops again
    assert rep["status"] == "reply_upgrade_stopped" and "the agent changed the KI" in rep["reason"]
    assert rep["reply_upgrade"]["ki_changed"] and not reviews and not caps
    assert runner == orig and "convergence" not in rep
    assert rep2["status"] == "reply_upgrade_unfinished" and "ki_changed" in rep2["reason"] and len(calls) == 1


@pytest.mark.parametrize("stray", ["sed_i_on_a_helper", "new_helper_file", "tools_swapped_for_a_link_into_the_ki"])
def test_an_agent_that_breaks_the_view_is_put_right_before_any_run(monkeypatch, stray):
    """`sed -i` swaps a LINK for a real file (the KI is untouched): the view is no longer a view of the KI."""
    def meddle(view):
        if stray == "sed_i_on_a_helper":
            f = view / "tools" / "helper_x.py"
            text = f.read_text()
            f.unlink()
            f.write_text(text.replace("1", "2"))
        elif stray == "new_helper_file":
            (view / "tools" / "my_new_helper.py").write_text("Y = 1\n")
        else:                                  # codex 3b #2: the put-back must not write THROUGH this link
            shutil.rmtree(view / "tools")
            os.symlink(str(Path(box["ki"]) / "tools"), view / "tools")
    box = {}
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", also=meddle,
                                           tail=None if stray.startswith("tools_swapped") else NEW_TAIL)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        box["ki"] = ki
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")
        ki_before = _tree(ki)
        rep = _go(SC, ki, tmp)
        ki_after = _tree(ki)
        proj, view, d = _proj(tmp)
        assert not (view / "tools").is_symlink() and not (view / "tools" / "calib_run.py").is_symlink()
        runner = (view / "tools" / "calib_run.py").read_bytes()
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
        failed, marker = (d / "failed.json").exists(), (d / "IN_PROGRESS.json").exists()
    assert ki_after == ki_before and not reviews and not caps
    assert rep["status"] == "reply_upgrade_stopped"                                           # codex 3b #1: no search
    assert ("no longer a view" in rep["reason"]) or ("replaced the project's tools/" in rep["reason"])
    assert "convergence" not in rep and runner == orig and not failed and not marker


def test_an_error_in_the_agent_puts_the_runner_back(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", boom=RuntimeError("agent died"))
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        proj, view, d = _proj(tmp)
        runner = (view / "tools" / "calib_run.py").read_bytes()
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
        marker, failed = (d / "IN_PROGRESS.json").exists(), (d / "failed.json").exists()
    up = rep["reply_upgrade"]
    assert up["status"] == "restored" and "agent died" in up["reason"] and runner == orig
    assert not marker and not failed                                   # not the runner's failure: may try again


def test_an_interrupt_puts_the_runner_back_and_is_passed_on(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", boom=KeyboardInterrupt())
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        with pytest.raises(KeyboardInterrupt):
            _go(SC, ki, tmp)
        proj, view, d = _proj(tmp)
        runner = (view / "tools" / "calib_run.py").read_bytes()
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
        marker, failed = (d / "IN_PROGRESS.json").exists(), (d / "failed.json").exists()
    assert runner == orig and not marker and not failed


def test_a_killed_upgrade_is_recovered_at_the_next_call_and_only_the_projects_three_files_are_touched(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        orig = _own(view)
        SC._backup_own(view, d / "backup")
        (d / "IN_PROGRESS.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": os.uname().nodename}))
        (view / "tools" / "calib_run.py").write_text("half written")       # ... and then the process was killed
        (view / "calibration.yaml").write_text("broken: [")
        (proj / "notes.txt").write_text("someone's later work in the project")
        (Path(ki) / "docs_later.md").write_text("someone's later work in the KI")
        assert SC.recover_interrupted_reply_upgrade(Path(tmp) / "nothing_here") is None
        rep = _go(SC, ki, tmp)                                             # the next calibration call
        after = _own(view)
        marker = (d / "IN_PROGRESS.json").exists()
        kept_work = (proj / "notes.txt").exists() and (Path(ki) / "docs_later.md").exists()
    assert after == orig and not marker and kept_work
    assert rep["status"] not in ("reply_upgrade_unfinished", "reply_upgrade_stopped")


def test_a_marker_with_a_live_process_number_under_our_own_lock_is_stale_and_recovered(monkeypatch):
    """codex 3b r9: the upgrade always runs under the project's lock; a call that holds the lock knows the marker's
    owner is gone, even if its process number now belongs to some other live program."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        orig = _own(view)
        SC._backup_own(view, d / "backup")
        (d / "IN_PROGRESS.json").write_text(json.dumps({"pid": os.getppid(), "host": os.uname().nodename}))
        (view / "tools" / "calib_run.py").write_text("half written by the killed upgrade")
        direct = SC.recover_interrupted_reply_upgrade(proj)               # without the lock: taken for the owner
        untouched = (view / "tools" / "calib_run.py").read_text() == "half written by the killed upgrade"
        rep = _go(SC, ki, tmp)                                             # with the lock: stale, recovered
        back, marker = _own(view) == orig, (d / "IN_PROGRESS.json").exists()
    assert direct["status"] == "owner_alive" and "delete" in direct["to_go_on"] and untouched
    assert "convergence" in rep and back and not marker


def test_a_failed_put_back_stops_the_search_and_keeps_the_marker(monkeypatch):
    """the one file could not be put back: never search on a half-edited runner, never say 'restored'."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "REQUEST_CHANGES")
    real = SC._put_back_own

    def failing(view, bdir, only=None):
        if only is None:
            raise OSError("read-only file system")
        return real(view, bdir, only=only)
    monkeypatch.setattr(SC, "_put_back_own", failing)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        marker = (_proj(tmp)[2] / "IN_PROGRESS.json").exists()
    assert rep["status"] == "reply_upgrade_stopped" and "could not be put back" in rep["reason"]
    assert marker and "convergence" not in rep


def test_a_second_calibration_of_the_same_project_is_refused_while_one_runs(monkeypatch):
    import fcntl
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        proj = _proj(tmp)[0]
        proj.mkdir(parents=True)
        with open(proj / ".kdt_project.lock", "a+") as other:
            fcntl.flock(other, fcntl.LOCK_EX)
            rep = _go(SC, ki, tmp)
    assert rep["status"] == "workdir_busy" and not calls


@pytest.mark.parametrize("value", ["0", "false", "No", "off"])
def test_the_switch_turns_the_upgrade_off_but_the_project_folder_is_still_used(monkeypatch, value):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", value)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
    assert not calls and rep["reply_upgrade"] == {"status": "skipped", "reason": "switched off"}
    assert rep["convergence"]["reply"]["complete"] is False and rep["project_workflow"]["used"] is True


def test_a_workflow_that_cannot_run_from_a_project_folder_is_calibrated_in_the_ki_as_before(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        c = yaml.safe_load(open(Path(ki, "calibration.yaml")))
        c["runner"]["command"] = ["bash", "-c", "exec " + sys.executable + " tools/calib_run.py --out $0 --workdir $1",
                                  "{metrics_json}", "{workdir}"]
        yaml.safe_dump(c, open(Path(ki, "calibration.yaml"), "w"))
        rep = _go(SC, ki, tmp)
    assert not calls and rep["project_workflow"]["used"] is False and "python interpreter" in rep["project_workflow"]["why"]
    assert rep["status"] not in ("reply_incomplete", "reply_checked") and "convergence" in rep


def test_the_environment_is_as_before_after_the_call(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_CONTRACT", "/some/old/value")
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        os.environ["KDT_CALIB_CONTRACT"] = "/some/old/value"               # _clean_env removed it
        try:
            rep = _go(SC, ki, tmp)
            seen = (os.environ.get("KDT_CALIB_CONTRACT"), os.environ.get("PYTHONPYCACHEPREFIX"))
        finally:
            os.environ.pop("KDT_CALIB_CONTRACT", None)
    assert seen == ("/some/old/value", None) and rep["reply_upgrade"]["status"] == "upgraded"


def test_run_stage_calibrates_through_the_project_folder(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIBRATE", "1")
    monkeypatch.setattr(SC, "_has_contract", lambda ki: True)
    monkeypatch.setattr(SC, "_capability_matches_case", lambda ki, case: True)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        ki_before = _tree(ki)
        rep = SC.run_stage("M", ki, Path(tmp) / "project", OBS_SHAPE)
        ki_after = _tree(ki)
        runner = (_proj(tmp)[1] / "tools" / "calib_run.py").read_text()
    assert rep["reply_upgrade"]["status"] == "upgraded" and '["panel"]' in runner and ki_after == ki_before


def _wrap_check(monkeypatch, change):
    """let the engine's check-mode call return something else (the rest of the engine runs for real)"""
    from calibration_kit import calib
    real = calib.calibrate

    def wrapped(*a, **k):
        rep = real(*a, **k)
        return change(rep) if k.get("reply_gate") == "check" else rep
    monkeypatch.setattr(calib, "calibrate", wrapped)


def test_an_upgrade_after_which_the_engine_would_optimise_something_else_is_put_back(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    _wrap_check(monkeypatch, lambda rep: dict(rep, objectives=list(rep["objectives"]) + ["streamflow:pbias"]))
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        runner = (_proj(tmp)[1] / "tools" / "calib_run.py").read_bytes()
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
        failed = (_proj(tmp)[2] / "failed.json").exists()
    up = rep["reply_upgrade"]
    assert up["status"] == "restored" and "what the engine optimises" in up["reason"] and runner == orig
    assert failed and not caps


def test_a_check_that_could_not_run_is_not_held_against_the_runner(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    _wrap_check(monkeypatch, lambda rep: {"status": "no_baseline"})
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
        runner = (_proj(tmp)[1] / "tools" / "calib_run.py").read_bytes()
        orig = (Path(ki) / "tools" / "calib_run.py").read_bytes()
        failed = (_proj(tmp)[2] / "failed.json").exists()
    up = rep["reply_upgrade"]
    assert up["status"] == "restored" and "could not run (no_baseline)" in up["reason"] and up["transient"] is True
    assert runner == orig and not failed and not caps


def test_recovery_never_writes_through_a_tools_folder_that_became_a_link(monkeypatch):
    """codex 3b #2, the recovery path: killed after the agent swapped the project's tools/ for a link into the KI."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        (view / "tools" / "calib_run.py").write_text("# the project's own, different from the KI's\n" +
                                                     (view / "tools" / "calib_run.py").read_text())
        want = (view / "tools" / "calib_run.py").read_bytes()
        SC._backup_own(view, d / "backup")
        (d / "IN_PROGRESS.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": os.uname().nodename}))
        shutil.rmtree(view / "tools")
        os.symlink(str(Path(ki) / "tools"), view / "tools")                 # ... and then the process was killed
        ki_before = _tree(ki)
        got = SC.recover_interrupted_reply_upgrade(proj)
        ki_after = _tree(ki)
        ok = (not (view / "tools").is_symlink()) and (view / "tools" / "calib_run.py").read_bytes() == want
    assert got["status"] == "recovered" and ki_after == ki_before and ok


def test_a_continued_search_is_not_upgraded_and_the_report_says_so(monkeypatch):
    """codex 3b #3: the gate fires only on a fresh pilot; a saved search keeps the runner it began with."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "off")
        _go(SC, ki, tmp)
        monkeypatch.delenv("KDT_CALIB_REPLY_UPGRADE")
        rep = _go(SC, ki, tmp)
    assert not calls and rep["reply_upgrade"]["status"] == "skipped" and "being continued" in rep["reply_upgrade"]["reason"]


def test_a_view_swapped_for_a_link_to_the_ki_is_never_written_through(monkeypatch):
    """codex 3b r2 #1: the agent replaces project/ki_view by a link to the KI. Nothing may be put back through it
    (the project's contract differs from the KI's here); the call stops, the marker stays; the next call stops too;
    recovery writes nothing either."""
    box = {}

    def meddle(view):
        shutil.rmtree(view)
        os.symlink(str(box["ki"]), view)
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=None, also=meddle)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        box["ki"] = ki
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        (view / "calibration.yaml").write_text((view / "calibration.yaml").read_text() + "\n# the project's own\n")
        ki_before = _tree(ki)
        rep = _go(SC, ki, tmp)
        marker = (d / "IN_PROGRESS.json").exists()
        rec = SC.recover_interrupted_reply_upgrade(proj)
        rep2 = _go(SC, ki, tmp)
        ki_after = _tree(ki)
    assert ki_after == ki_before
    assert rep["status"] == "reply_upgrade_stopped" and "convergence" not in rep and marker
    assert rec["status"] == "recovery_failed" and rep2["status"] == "reply_upgrade_unfinished" and len(calls) == 1


def test_a_damaged_project_folder_stops_the_next_call_instead_of_falling_back_to_the_ki(monkeypatch):
    """codex 3b r2 #2: after a stop for a broken view, a rerun must not quietly calibrate the KI's workflow."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        (view / "tools" / "helper_x.py").unlink()
        (view / "tools" / "helper_x.py").write_text("X = 2\n")             # a real file where a link belongs
        rep = _go(SC, ki, tmp)
        run_made = (proj / "run").exists()
    assert rep["status"] == "project_workflow_broken" and "real file" in rep["reason"] and not calls
    assert "convergence" not in rep and not run_made


def test_a_new_runner_that_litters_the_view_when_it_runs_stops_the_search(monkeypatch):
    """codex 3b r3: the upgraded runner itself writes a real helper file into tools/ when the check run executes
    it. The reply is complete and approved — but the folder is no longer a view, so no search runs."""
    litter = NEW_TAIL + '\nopen(os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_helper.py"), "w").write("G = 1")\n'
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=litter)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        ki_before = _tree(ki)
        rep = _go(SC, ki, tmp)
        ki_after = _tree(ki)
    assert ki_after == ki_before
    assert rep["status"] == "project_workflow_broken" and "generated_helper.py" in rep["reason"]
    assert "convergence" not in rep and rep["reply_upgrade"]["status"] == "upgraded"


def test_the_approval_record_is_never_written_through_a_calib_folder_that_became_a_link(monkeypatch):
    """codex 3b r4 #1: the new runner, when the check run executes it, swaps the project's calib/ for a link into
    the KI. The approval record must not be written through it."""
    swap = NEW_TAIL + ('\nimport shutil as _sh\n_v = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n'
                       'if not os.path.islink(os.path.join(_v, "calib")):\n'
                       '    _sh.rmtree(os.path.join(_v, "calib"))\n'
                       '    os.symlink(os.environ["KI_FOR_TEST"] + "/calib", os.path.join(_v, "calib"))\n')
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=swap)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "calib").mkdir(exist_ok=True)
        monkeypatch.setenv("KI_FOR_TEST", str(ki))
        ki_before = _tree(ki)
        rep = _go(SC, ki, tmp)
        ki_after = _tree(ki)
    assert ki_after == ki_before and not caps
    assert rep["status"] == "reply_upgrade_stopped" and "damaged" in rep["reason"] and "convergence" not in rep


@pytest.mark.parametrize("what", ["rewrites_itself", "deletes_itself", "edits_the_contract"])
def test_a_runner_that_changes_the_projects_files_when_it_runs_is_not_kept(monkeypatch, what):
    """codex 3b r5 #2: what is kept must be exactly the file that was reviewed and checked."""
    me = 'os.path.abspath(__file__)'
    line = {"rewrites_itself": f'open({me}, "a").write("\\n# changed after the review\\n")',
            "deletes_itself": f'os.unlink({me})',
            "edits_the_contract": f'open(os.path.join(os.path.dirname(os.path.dirname({me})), "calibration.yaml"), "a").write("\\n# x\\n")'}[what]
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", tail=NEW_TAIL + "\n" + line + "\n")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        ki_before = _tree(ki)
        rep = _go(SC, ki, tmp)
        ki_after = _tree(ki)
        view = _proj(tmp)[1]
        runner = (view / "tools" / "calib_run.py").read_bytes()
        contract = (view / "calibration.yaml").read_bytes()
        orig_r, orig_c = (Path(ki) / "tools" / "calib_run.py").read_bytes(), (Path(ki) / "calibration.yaml").read_bytes()
    print("REASON", rep.get("status"), (rep.get("reply_upgrade") or {}).get("status"), (rep.get("reply_upgrade") or {}).get("reason"))
    assert ki_after == ki_before and not caps
    up = rep["reply_upgrade"]
    assert up["status"] in ("stop", "restored")                           # never kept ...
    assert runner == orig_r and contract == orig_c                        # ... and the originals are back


def test_a_runner_that_writes_into_the_ki_during_the_search_is_named_in_the_report(monkeypatch):
    """codex 3b r6: the runner regenerates a file under `<its folder>/../outputs/` — through the view's link that
    is the KI's outputs folder. The search is as before project folders, but the report now says so."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "outputs").mkdir()
        r = Path(ki) / "tools" / "calib_run.py"
        r.write_text(r.read_text() + '\nopen(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '
                                     '"outputs", "run.nc"), "w").write(str(p))\n')
        rep = _go(SC, ki, tmp)
        clean_ki, _ = None, None
    assert rep["project_workflow"]["ki_changed_during_run"] == ["outputs/run.nc (added)"]


def test_a_clean_run_reports_no_ki_change(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
    assert "ki_changed_during_run" not in rep["project_workflow"] and rep["reply_upgrade"]["status"] == "upgraded"


def test_an_agent_that_changes_the_ki_and_then_dies_still_stops_everything(monkeypatch):
    """codex 3b r7 #1: the error path must not skip the look at the KI."""
    def meddle(view):
        (view / "tools" / "helper_x.py").write_text("X = 2  # edited through the link\n")
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", also=meddle, boom=RuntimeError("agent died"))
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")
        rep = _go(SC, ki, tmp)
        failed = (_proj(tmp)[2] / "failed.json").exists()
    assert rep["status"] == "reply_upgrade_stopped" and "the KI changed during the reply upgrade" in rep["reason"]
    assert rep["reply_upgrade"]["ki_changed"] == ["tools/helper_x.py (changed)"] and "agent died" in rep["reason"]
    assert "convergence" not in rep and not failed


def test_a_killed_upgrade_whose_agent_had_changed_the_ki_is_not_passed_over(monkeypatch):
    """codex 3b r7 #2: the KI's record from before the agent is kept on disk; recovery compares, and a change
    keeps the marker until a person has looked."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        orig = _own(view)
        SC._backup_own(view, d / "backup")
        (d / "ki_before.json").write_text(json.dumps({"ki_path": str(ki), "fingerprint": PW.ki_fingerprint(ki)}))
        (d / "IN_PROGRESS.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": os.uname().nodename}))
        (view / "tools" / "calib_run.py").write_text("half written")
        (view / "tools" / "helper_x.py").write_text("X = 2  # the agent, through the link\n")    # ... then killed
        rep = _go(SC, ki, tmp)
        marker, back = (d / "IN_PROGRESS.json").exists(), _own(view) == orig
        (d / "IN_PROGRESS.json").unlink()                                   # the person looked, and goes on
        rep2 = _go(SC, ki, tmp)
    assert rep["status"] == "reply_upgrade_unfinished" and "tools/helper_x.py (changed)" in rep["reason"]
    assert marker and back and "convergence" not in rep
    assert rep2["status"] not in ("reply_upgrade_unfinished", "reply_upgrade_stopped") and "convergence" in rep2


def test_a_killed_upgrade_with_an_untouched_ki_recovers_and_goes_on(monkeypatch):
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        SC._backup_own(view, d / "backup")
        (d / "ki_before.json").write_text(json.dumps({"ki_path": str(ki), "fingerprint": PW.ki_fingerprint(ki)}))
        (d / "IN_PROGRESS.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": os.uname().nodename}))
        (view / "tools" / "calib_run.py").write_text("half written")
        rep = _go(SC, ki, tmp)
        marker = (d / "IN_PROGRESS.json").exists()
    assert "convergence" in rep and not marker


def test_a_stopped_call_still_names_what_the_first_runs_wrote_into_the_ki(monkeypatch):
    """codex 3b r7 #3: the gate's runs write outputs/pilot.nc into the KI; then the agent breaks the view and the
    call stops — the stopped report still names the KI change."""
    def meddle(view):
        (view / "tools" / "my_new_helper.py").write_text("Y = 1\n")
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", also=meddle)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "outputs").mkdir()
        r = Path(ki) / "tools" / "calib_run.py"
        r.write_text(r.read_text() + '\nopen(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '
                                     '"outputs", "pilot.nc"), "w").write("x")\n')
        rep = _go(SC, ki, tmp)
    assert rep["status"] == "reply_upgrade_stopped" and "no longer a view" in rep["reason"]
    assert rep["project_workflow"]["ki_changed_during_run"] == ["outputs/pilot.nc (added)"]


def test_the_kis_record_and_the_marker_are_on_disk_while_the_agent_works(monkeypatch):
    """a kill during the agent's session must find both: the marker and the KI's record from before the session."""
    seen = {}

    def look(view):
        d = view.parent / "reply_upgrade"
        seen["marker"] = json.loads((d / "IN_PROGRESS.json").read_text())
        seen["record"] = json.loads((d / "ki_before.json").read_text())
        seen["backup"] = (d / "backup" / "tools" / "calib_run.py").is_file()
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", also=look)
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        _go(SC, ki, tmp)
    assert seen["marker"]["pid"] == os.getpid() and seen["backup"]
    assert seen["record"]["ki_path"] == str(ki) and "tools/calib_run.py" in seen["record"]["fingerprint"]


def test_an_interrupt_after_the_agent_changed_the_ki_leaves_the_marker_for_the_next_call(monkeypatch):
    """codex 3b r8 #1: the interrupt is passed on, so no report is returned — the marker and the KI's record must
    stay, and the next call stops and names the change. Once the KI is as before, the next call goes on."""
    def meddle(view):
        (view / "tools" / "helper_x.py").write_text("X = 2  # edited through the link\n")
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE", also=meddle, boom=KeyboardInterrupt())
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")
        st = os.stat(Path(ki) / "tools" / "helper_x.py")
        with pytest.raises(KeyboardInterrupt):
            _go(SC, ki, tmp)
        proj, view, d = _proj(tmp)
        marker, record = (d / "IN_PROGRESS.json").exists(), (d / "ki_before.json").exists()
        runner_back = (view / "tools" / "calib_run.py").read_bytes() == (Path(ki) / "tools" / "calib_run.py").read_bytes()
        monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
        rep = _go(SC, ki, tmp)
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")            # the person puts the KI right
        os.utime(Path(ki) / "tools" / "helper_x.py", ns=(st.st_atime_ns, st.st_mtime_ns))
        rep2 = _go(SC, ki, tmp)
        marker2 = (d / "IN_PROGRESS.json").exists()
    assert marker and record and runner_back
    assert rep["status"] == "reply_upgrade_unfinished" and "tools/helper_x.py (changed)" in rep["reason"]
    assert "convergence" in rep2 and not marker2


def test_a_recovery_that_cannot_put_the_files_back_still_names_the_ki_change(monkeypatch):
    """codex 3b r8 #2"""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 1\n")
        from calibration_kit import project_workflow as PW
        proj, view, d = _proj(tmp)
        PW.open_view(ki, proj)
        SC._backup_own(view, d / "backup")
        (d / "ki_before.json").write_text(json.dumps({"ki_path": str(ki), "fingerprint": PW.ki_fingerprint(ki)}))
        (d / "IN_PROGRESS.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": os.uname().nodename}))
        (Path(ki) / "tools" / "helper_x.py").write_text("X = 2  # the agent\n")
        shutil.rmtree(view)
        os.symlink(str(ki), view)                                          # ... and the view was swapped; then killed
        ki_before = _tree(ki)
        rec = SC.recover_interrupted_reply_upgrade(proj)
        ki_after = _tree(ki)
        marker = (d / "IN_PROGRESS.json").exists()
    assert rec["status"] == "recovery_failed" and rec["ki_changed"] == ["tools/helper_x.py (changed)"]
    assert marker and ki_after == ki_before


@pytest.mark.parametrize("upgrade", ["on", "off"])
def test_a_runner_that_rewrites_the_projects_contract_in_the_first_runs_stops_the_call(monkeypatch, upgrade):
    """codex 3b r10: the OLD runner itself rewrites <view>/calibration.yaml when it runs — that changed contract
    must not become the backup, nor what is searched next without a word."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    if upgrade == "off":
        monkeypatch.setenv("KDT_CALIB_REPLY_UPGRADE", "0")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        r = Path(ki) / "tools" / "calib_run.py"
        r.write_text(r.read_text() + '\n_c = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '
                                     '"calibration.yaml")\nif "# rewritten" not in open(_c).read():\n'
                                     '    open(_c, "a").write("\\n# rewritten by the runner\\n")\n')
        ki_before = _tree(ki)
        rep = _go(SC, ki, tmp)
        ki_after = _tree(ki)
    assert ki_after == ki_before and not calls
    assert rep["status"] == "project_workflow_broken" and "calibration.yaml" in rep["reason"]
    assert ("search_report" in rep) is (upgrade == "off")


# ── 3c: the first runs are not repeated ────────────────────────────────────────────────────────────────────────
def _phases(rep):
    return {k: v["n"] for k, v in rep["phase_counts"].items()}


def test_after_a_kept_upgrade_the_search_reuses_the_checks_pilot(monkeypatch):
    """the pilot runs twice (the gate with the old runner, the check with the new one), never a third time."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "APPROVE")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
    assert rep["reply_upgrade"]["status"] == "upgraded" and _phases(rep)["pilot"] == 20
    assert rep["budget_plan"].get("reused_from_workdir") and rep["convergence"]["reply"]["complete"] is True


def test_after_a_rejected_upgrade_the_search_reuses_the_gates_pilot(monkeypatch):
    """the old runner is back: its pilot from the gate call is the search's pilot — run once."""
    SC, calls, reviews, caps = _fake_tools(monkeypatch, "REQUEST_CHANGES")
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        rep = _go(SC, ki, tmp)
    assert rep["reply_upgrade"]["status"] == "restored" and _phases(rep)["pilot"] == 10
    assert rep["budget_plan"].get("reused_from_workdir") and rep["convergence"]["reply"]["complete"] is False


def test_a_pilot_made_with_one_runner_is_never_reused_for_another(monkeypatch):
    """the saved plan is tied to the runner's code: a changed runner plans afresh."""
    from calibration_kit import calib, project_workflow as PW
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        got = PW.open_view(ki, Path(tmp) / "project")
        for k, v in got["env"].items():
            monkeypatch.setenv(k, v)
        a = calib.calibrate(ki_path=got["view"], workdir=got["workdir"], reply_gate="check", **KW)
        r = Path(got["view"]) / "tools" / "calib_run.py"
        r.write_text(r.read_text() + NEW_TAIL.format())
        b = calib.calibrate(ki_path=got["view"], workdir=got["workdir"], reply_gate="check", **KW)
        c = calib.calibrate(ki_path=got["view"], workdir=got["workdir"], reply_gate="check", **KW)
    assert a["reply"]["complete"] is False and b["reply"]["complete"] is True      # b did NOT reuse a's pilot
    assert _phases(b)["pilot"] == 20 and _phases(c)["pilot"] == 20                 # c reused b's (same runner)
    assert c["status"] == "reply_checked" and c["reply"]["complete"] is True


def test_the_gate_does_not_act_once_a_search_has_begun(monkeypatch):
    from calibration_kit import calib, project_workflow as PW
    with _Tmp() as tmp:
        ki, wd = _ki(tmp)
        got = PW.open_view(ki, Path(tmp) / "project")
        for k, v in got["env"].items():
            monkeypatch.setenv(k, v)
        first = calib.calibrate(ki_path=got["view"], workdir=got["workdir"], **KW)          # a whole search
        again = calib.calibrate(ki_path=got["view"], workdir=got["workdir"], reply_gate=True, **KW)
        chk = calib.calibrate(ki_path=got["view"], workdir=got["workdir"], reply_gate="check", **KW)
    assert first["convergence"]["reply"]["complete"] is False
    assert again["status"] not in ("reply_incomplete", "reply_checked")
    assert chk["status"] not in ("reply_incomplete", "reply_checked")
