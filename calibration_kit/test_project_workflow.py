"""The project's view of a KI: the two workflow files are the project's own, the rest are links, the KI is only read."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from calibration_kit.project_workflow import ViewError, build_ki_view, view_env

RUNNER = '''
import json, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
KI = HERE.parent
import helper                                   # a sibling tool
from sub import deep                            # a tool folder that is a link inside the KI
out = sys.argv[sys.argv.index("--out") + 1]
json.dump({"tag": TAG, "helper": helper.VALUE, "deep": deep.VALUE, "data": (KI / "data" / "d.txt").read_text(),
           "dag": (KI / "dag.yaml").read_text(), "me": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "file": str(Path(__file__).resolve())}, open(out, "w"))
'''


def _ki(tmp_path):
    ki = tmp_path / "models" / "M" / "ki"
    (ki / "tools").mkdir(parents=True)
    (ki / "data").mkdir()
    (ki / "shared_sub").mkdir()
    (ki / "shared_sub" / "__init__.py").write_text("")
    (ki / "shared_sub" / "deep.py").write_text("VALUE = 7\n")
    os.symlink("../shared_sub", ki / "tools" / "sub")          # as VIC: tools/s2_forcing -> ../s2_forcing
    (ki / "tools" / "helper.py").write_text("VALUE = 42\n")
    (ki / "tools" / "calib_run.py").write_text('TAG = "ki"\n' + RUNNER)
    (ki / "data" / "d.txt").write_text("ki data")
    (ki / "dag.yaml").write_text("outputs: []\n")
    (ki / "calibration.yaml").write_text("model_id: M\n")
    (ki / "calib").mkdir()
    (ki / "calib" / "capability_case.json").write_text('{"case": "ki"}')
    (ki / "calib" / "forcing.nc").write_text("forcing")
    (ki / "tools" / "__pycache__").mkdir()                     # old byte-code in the KI: never linked
    (ki / "tools" / "__pycache__" / "helper.cpython-312.pyc").write_bytes(b"old")
    return ki


def _tree(d):
    """every entry, links not followed, byte-code folders INCLUDED"""
    out = {}
    for dp, dns, fns in os.walk(d, followlinks=False):
        for n in dns + fns:
            p = Path(dp) / n
            out[str(p.relative_to(d))] = ("L", os.readlink(p)) if p.is_symlink() else (p.read_bytes() if p.is_file() else "D")
    return out


def _run(view, out):
    env = dict(os.environ, **view_env(view))
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    r = subprocess.run([sys.executable, "tools/calib_run.py", "--out", str(out)], cwd=str(view), capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(Path(out).read_text())


def test_the_view_has_the_projects_two_files_and_links_for_the_rest(tmp_path):
    ki = _ki(tmp_path)
    before = _tree(ki)
    view = tmp_path / "project" / "calib" / "view"
    rep = build_ki_view(ki, view)
    assert sorted(rep["copied"]) == ["calib/capability_case.json", "calibration.yaml", "tools/calib_run.py"]
    assert rep["removed"] == [] and len(rep["linked"]) == 6      # the own files are never offered as links
    assert not (view / "calib").is_symlink()
    for rel in ("calibration.yaml", "tools/calib_run.py", "calib/capability_case.json"):
        assert (view / rel).is_file() and not (view / rel).is_symlink()
        assert (view / rel).read_bytes() == (ki / rel).read_bytes()
    assert not (view / "tools").is_symlink()
    links = {k: v[1] for k, v in _tree(view).items() if isinstance(v, tuple)}
    assert links == {"calib/forcing.nc": str(ki / "calib" / "forcing.nc"), "dag.yaml": str(ki / "dag.yaml"), "data": str(ki / "data"), "shared_sub": str(ki / "shared_sub"),
                     "tools/helper.py": str(ki / "tools" / "helper.py"), "tools/sub": str(ki / "tools" / "sub")}
    assert _tree(ki) == before


def test_the_projects_runner_runs_with_the_kis_helpers_and_the_ki_is_never_written(tmp_path):
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    (view / "tools" / "calib_run.py").write_text('TAG = "project"\n' + RUNNER)        # the project edits ITS runner
    before = _tree(ki)
    got = _run(view, tmp_path / "m.json")
    import hashlib
    assert got == {"tag": "project", "helper": 42, "deep": 7, "data": "ki data", "dag": "outputs: []\n",
                   "me": hashlib.sha256((view / "tools" / "calib_run.py").read_bytes()).hexdigest(),
                   "file": str((view / "tools" / "calib_run.py").resolve())}           # its identity is its own file
    assert _tree(ki) == before                                 # not even byte-code files appear in the KI
    assert list((view / ".pycache").rglob("deep*.pyc"))         # ... they are kept in the project
    assert (ki / "tools" / "calib_run.py").read_text().startswith('TAG = "ki"')


def test_the_same_reply_as_in_the_ki(tmp_path):
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    a = _run(view, tmp_path / "a.json")
    b = _run(ki, tmp_path / "b.json")
    for d in (a, b):
        d.pop("file")
    assert a == b


def test_a_refresh_follows_the_ki_and_never_touches_the_projects_files(tmp_path):
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    (view / "tools" / "calib_run.py").write_text("# the project's upgraded runner\n")
    (view / "calibration.yaml").write_text("model_id: M\n# project\n")
    (view / "notes.txt").write_text("the project's own note")
    (ki / "tools" / "new_tool.py").write_text("X = 1\n")
    (ki / "new_top.txt").write_text("x")
    (ki / "data" / "d.txt").write_text("changed in the KI")
    (ki / "dag.yaml").unlink()
    (ki / "tools" / "calib_run.py").write_text("# the KI's runner changed later\n")
    rep = build_ki_view(ki, view)
    assert rep["copied"] == []
    assert (view / "tools" / "calib_run.py").read_text() == "# the project's upgraded runner\n"
    assert (view / "calibration.yaml").read_text() == "model_id: M\n# project\n"
    assert (view / "notes.txt").read_text() == "the project's own note"
    assert os.readlink(view / "tools" / "new_tool.py") == str(ki / "tools" / "new_tool.py")
    assert os.readlink(view / "new_top.txt") == str(ki / "new_top.txt")
    assert (view / "data" / "d.txt").read_text() == "changed in the KI"
    assert not os.path.lexists(view / "dag.yaml") and str(view / "dag.yaml") in rep["removed"]


def test_a_view_of_a_moved_ki_is_re_pointed(tmp_path):
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    ki2 = tmp_path / "models" / "M2"
    os.rename(ki.parent, ki2)
    build_ki_view(ki2 / "ki", view)
    assert os.readlink(view / "data") == str(ki2 / "ki" / "data")
    assert os.readlink(view / "tools" / "helper.py") == str(ki2 / "ki" / "tools" / "helper.py")


def test_a_ki_whose_tools_folder_is_a_link(tmp_path):
    ki = _ki(tmp_path)
    shared = tmp_path / "shared_tools"
    os.rename(ki / "tools", shared)
    os.symlink(shared, ki / "tools")
    (shared / "sub").unlink()
    os.symlink(str(ki / "shared_sub"), shared / "sub")
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    assert not (view / "tools").is_symlink() and not (view / "tools" / "calib_run.py").is_symlink()
    assert _run(view, tmp_path / "m.json")["helper"] == 42


@pytest.mark.parametrize("where", ["inside", "around", "same"])
def test_a_view_inside_or_around_the_ki_is_refused(tmp_path, where):
    ki = _ki(tmp_path)
    before = _tree(ki)
    view = {"inside": ki / "view", "around": ki.parent, "same": ki}[where]
    with pytest.raises(ViewError):
        build_ki_view(ki, view)
    assert _tree(ki) == before


def test_missing_workflow_files_and_linked_own_files_are_refused(tmp_path):
    ki = _ki(tmp_path)
    (ki / "tools" / "calib_run.py").unlink()
    with pytest.raises(ViewError):
        build_ki_view(ki, tmp_path / "p1")
    (ki / "tools" / "calib_run.py").write_text("x = 1\n")
    v = tmp_path / "p2"
    (v / "tools").mkdir(parents=True)
    os.symlink(str(ki / "tools" / "calib_run.py"), v / "tools" / "calib_run.py")
    with pytest.raises(ViewError):
        build_ki_view(ki, v)
    assert (ki / "tools" / "calib_run.py").read_text() == "x = 1\n"


@pytest.mark.parametrize("rel", ["tools/helper.py", "data", "dag.yaml"])
def test_a_real_file_under_a_ki_name_is_refused_not_used(tmp_path, rel):
    """codex 1B #4: a stale real file in the view must never stand in for the KI's."""
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    (view / rel).unlink()
    if rel == "data":
        (view / rel).mkdir()
    else:
        (view / rel).write_text("stale")
    with pytest.raises(ViewError):
        build_ki_view(ki, view)


def test_a_view_reached_through_a_link_into_the_ki_is_refused(tmp_path):
    """codex 1B #2: the view does not exist yet and its parent is a link into the KI."""
    ki = _ki(tmp_path)
    before = _tree(ki)
    (tmp_path / "project").mkdir()
    os.symlink(str(ki), tmp_path / "project" / "link_to_ki")
    with pytest.raises(ViewError):
        build_ki_view(ki, tmp_path / "project" / "link_to_ki" / "view")
    assert _tree(ki) == before


@pytest.mark.parametrize("rel", ["tools/calib_run.py", "calibration.yaml", "calib/capability_case.json"])
def test_a_hard_link_to_the_kis_file_is_refused(tmp_path, rel):
    """codex 1B #3: an edit of a hard-linked "own" file would change the KI's file."""
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    (view / rel).parent.mkdir(parents=True)
    os.link(ki / rel, view / rel)
    with pytest.raises(ViewError):
        build_ki_view(ki, view)


def test_two_builders_at_once(tmp_path, monkeypatch):
    """codex 1B #5: the other process made the link first — fine when it is the right link, refused when not."""
    import calibration_kit.project_workflow as PW
    ki = _ki(tmp_path)
    real = os.symlink

    def racing(src, dst, *a, **k):
        real(src, dst)                                          # the other process got there first
        return real(src, dst)                                   # ... so ours raises FileExistsError
    monkeypatch.setattr(PW.os, "symlink", racing)
    PW.build_ki_view(ki, tmp_path / "p" / "view")
    assert os.readlink(tmp_path / "p" / "view" / "data") == str(ki / "data")

    def racing_wrong(src, dst, *a, **k):
        real("/somewhere/else", dst)
        return real(src, dst)
    monkeypatch.setattr(PW.os, "symlink", racing_wrong)
    with pytest.raises(ViewError):
        PW.build_ki_view(ki, tmp_path / "p2" / "view")


def test_the_approval_record_is_the_projects_own(tmp_path):
    """codex 1B #1: the stage writes calib/capability_case.json for THIS workflow — it must land in the project."""
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    before = _tree(ki)
    (view / "calib" / "capability_case.json").write_text('{"case": "project"}')
    build_ki_view(ki, view)
    assert _tree(ki) == before
    assert (view / "calib" / "capability_case.json").read_text() == '{"case": "project"}'
    assert (view / "calib" / "forcing.nc").read_text() == "forcing"            # KI data still read through a link


def test_a_ki_without_calib_or_approval_still_gets_an_own_calib_folder(tmp_path):
    ki = _ki(tmp_path)
    import shutil
    shutil.rmtree(ki / "calib")
    view = tmp_path / "project" / "view"
    rep = build_ki_view(ki, view)
    assert (view / "calib").is_dir() and not (view / "calib").is_symlink() and "calib/capability_case.json" not in rep["copied"]


def test_an_older_view_whose_calib_is_a_link_is_converted(tmp_path):
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    import shutil
    shutil.rmtree(view / "calib")
    os.symlink(str(ki / "calib"), view / "calib")
    before = _tree(ki)
    build_ki_view(ki, view)
    assert not (view / "calib").is_symlink() and (view / "calib" / "capability_case.json").is_file()
    assert not (view / "calib" / "capability_case.json").is_symlink() and _tree(ki) == before


@pytest.mark.parametrize("d, target", [("tools", "ki"), ("calib", "elsewhere")])
def test_an_own_folder_that_is_a_link_is_refused(tmp_path, d, target):
    """the project's tools/ may never be a link (a write would land in the KI); calib/ only when it is the old
    link to this KI's calib (then it is converted)."""
    ki = _ki(tmp_path)
    before = _tree(ki)
    view = tmp_path / "project" / "view"
    view.mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(str(ki / d) if target == "ki" else str(tmp_path / "elsewhere"), view / d)
    with pytest.raises(ViewError):
        build_ki_view(ki, view)
    assert _tree(ki) == before


@pytest.mark.parametrize("rel", ["tools/legacy_helper.py", "calib/old_record.json", "tools/olddir"])
def test_a_left_over_real_file_in_tools_or_calib_is_refused(tmp_path, rel):
    """codex 1B r2 #1: the KI has no such name, but the runner could import or list it."""
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    (view / rel).mkdir() if rel.endswith("dir") else (view / rel).write_text("X = 1\n")
    with pytest.raises(ViewError):
        build_ki_view(ki, view)


def test_left_over_files_are_refused_in_calib_even_when_the_ki_has_no_calib(tmp_path):
    ki = _ki(tmp_path)
    import shutil
    shutil.rmtree(ki / "calib")
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    (view / "calib" / "capability_case.json").write_text("{}")                  # the stage's own record: fine
    build_ki_view(ki, view)
    (view / "calib" / "stale.json").write_text("{}")
    with pytest.raises(ViewError):
        build_ki_view(ki, view)


def test_byte_code_and_top_level_project_files_are_allowed(tmp_path):
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    build_ki_view(ki, view)
    (view / "tools" / "__pycache__").mkdir()
    (view / ".pycache").mkdir()
    (view / "notes.txt").write_text("x")
    build_ki_view(ki, view)


@pytest.mark.parametrize("d", ["tools", "calib"])
def test_an_own_folder_path_that_is_a_file_is_refused_before_anything_is_made(tmp_path, d):
    """codex 1B r2 #2"""
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    view.mkdir(parents=True)
    (view / d).write_text("not a folder")
    with pytest.raises(ViewError):
        build_ki_view(ki, view)
    assert sorted(p.name for p in view.iterdir()) == [d]


def test_a_copy_cut_short_never_leaves_a_half_file_as_the_projects(tmp_path, monkeypatch):
    """codex 1B r3 #1: the copy fails half way — no own file is left, and the left-over is refused, not trusted."""
    import calibration_kit.project_workflow as PW
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    real = PW.shutil.copyfile

    def cut_short(src, dst, *a, **k):
        Path(dst).write_text("half")
        raise OSError("disk full")
    monkeypatch.setattr(PW.shutil, "copyfile", cut_short)
    with pytest.raises(OSError):
        PW.build_ki_view(ki, view)
    assert not (view / "calibration.yaml").exists() and not (view / "tools" / "calib_run.py").exists()
    assert not list(view.rglob("*.kdt_tmp_*"))
    monkeypatch.setattr(PW.shutil, "copyfile", real)
    (view / "tools" / "calib_run.py.kdt_tmp_999").write_text("half")              # as a hard kill would leave it
    with pytest.raises(ViewError):
        PW.build_ki_view(ki, view)
    (view / "tools" / "calib_run.py.kdt_tmp_999").unlink()
    rep = PW.build_ki_view(ki, view)
    assert (view / "tools" / "calib_run.py").read_bytes() == (ki / "tools" / "calib_run.py").read_bytes()
    assert (view / "tools" / "calib_run.py").stat().st_nlink == 1 and len(rep["copied"]) == 3


def test_another_builder_that_installed_the_file_first_wins(tmp_path, monkeypatch):
    import calibration_kit.project_workflow as PW
    ki = _ki(tmp_path)
    view = tmp_path / "project" / "view"
    real = PW.shutil.copyfile

    def other_first(src, dst, *a, **k):
        real(src, dst)
        final = Path(str(dst).split(".kdt_tmp_")[0])
        final.write_text("# installed by the other builder\n")
    monkeypatch.setattr(PW.shutil, "copyfile", other_first)
    PW.build_ki_view(ki, view)
    assert (view / "tools" / "calib_run.py").read_text() == "# installed by the other builder\n"
    assert not list(view.rglob("*.kdt_tmp_*"))


@pytest.mark.parametrize("d", ["calib"])
def test_a_ki_entry_named_like_an_own_folder_but_a_file_is_refused(tmp_path, d):
    """codex 1B r3 #3"""
    ki = _ki(tmp_path)
    import shutil
    shutil.rmtree(ki / d)
    (ki / d).write_text("a file")
    with pytest.raises(ViewError):
        build_ki_view(ki, tmp_path / "project" / "view")
    assert not (tmp_path / "project" / "view" / "tools").exists()


# ── snippet 2: open_view — the project folder the engine is pointed at ─────────────────────────────────────────
from calibration_kit.project_workflow import open_view            # noqa: E402

CONTRACT = """template_version: calib-0.1
model_id: M
identity:
  obs_file: {ki}/data/d.txt          # a data path in the KI: must stay as it is
injection:
  mode: {mode}
runner:
  kind: subprocess   # keep this comment
  command: ["python", "{script}", "--out", "{{metrics_json}}"]
{cwd}"""


def _contract(ki, mode="runner", script="tools/calib_run.py", cwd=""):
    (ki / "calibration.yaml").write_text(CONTRACT.format(ki=ki, mode=mode, script=script, cwd=cwd))


def test_open_view_gives_the_engine_a_folder_and_keeps_byte_code_in_the_project(tmp_path):
    ki = _ki(tmp_path)
    _contract(ki)
    before = _tree(ki)
    got = open_view(ki, tmp_path / "project")
    view = tmp_path / "project" / "ki_view"
    assert got["workdir"] == str(tmp_path / "project" / "run")
    assert got["view"] == str(view) and got["env"] == {"PYTHONPYCACHEPREFIX": str(view / ".pycache"),
                                                       "KDT_CALIB_CONTRACT": str(view / "calibration.yaml")}
    assert got["report"]["re_pointed"] == [] and (view / "calibration.yaml").read_text() == (ki / "calibration.yaml").read_text()
    assert _tree(ki) == before


@pytest.mark.parametrize("mode", ["applicator", None])
def test_a_workflow_whose_engine_writes_into_the_ki_is_refused(tmp_path, mode):
    """the default injection mode is `applicator` (real Daisy and MARRMoT name none)."""
    ki = _ki(tmp_path)
    _contract(ki, mode=mode or "runner")
    if mode is None:
        (ki / "calibration.yaml").write_text((ki / "calibration.yaml").read_text().replace("injection:\n  mode: runner\n", ""))
    with pytest.raises(ViewError, match="injection mode"):
        open_view(ki, tmp_path / "project")


def test_a_contract_that_names_the_kis_runner_by_full_path_is_re_pointed(tmp_path):
    """real DSSAT: command names /…/<KI>/tools/calib_run.py — that would start the KI's file, not the project's."""
    import yaml
    ki = _ki(tmp_path)
    _contract(ki, script=str(ki / "tools" / "calib_run.py"))
    before = _tree(ki)
    got = open_view(ki, tmp_path / "project")
    view = Path(got["view"])
    text = (view / "calibration.yaml").read_text()
    c = yaml.safe_load(text)
    assert c["runner"]["command"] == ["python", "{ki_path}/tools/calib_run.py", "--out", "{metrics_json}"]
    assert c["identity"]["obs_file"] == f"{ki}/data/d.txt" and "# keep this comment" in text
    assert got["report"]["re_pointed"] == [f"{ki}/tools/calib_run.py -> {{ki_path}}/tools/calib_run.py"]
    assert _tree(ki) == before
    assert open_view(ki, tmp_path / "project")["report"]["re_pointed"] == []        # a second call changes nothing


def test_a_contract_whose_cwd_is_the_ki_by_full_path_is_re_pointed(tmp_path):
    """real HYPE: cwd: /…/<KI>"""
    import yaml
    ki = _ki(tmp_path)
    _contract(ki, cwd=f"  cwd: {ki}\n")
    got = open_view(ki, tmp_path / "project")
    c = yaml.safe_load((Path(got["view"]) / "calibration.yaml").read_text())
    assert c["runner"]["cwd"] == "{ki_path}" and c["identity"]["obs_file"] == f"{ki}/data/d.txt"


def test_the_same_path_outside_the_runner_block_is_left_alone(tmp_path):
    import yaml
    ki = _ki(tmp_path)
    _contract(ki, script=f'tools/calib_run.py", "--obs", "{ki}/data/d.txt')    # the same full path is also the obs_file
    got = open_view(ki, tmp_path / "project")
    c = yaml.safe_load((Path(got["view"]) / "calibration.yaml").read_text())
    assert c["identity"]["obs_file"] == f"{ki}/data/d.txt" and c["runner"]["command"][3] == "{ki_path}/data/d.txt"


def test_a_re_point_that_cannot_be_checked_is_refused_and_the_file_is_left(tmp_path):
    ki = _ki(tmp_path)
    full = str(ki / "tools" / "calib_run.py")
    (ki / "calibration.yaml").write_text(                        # the runner written as one flow mapping on the key line
        'injection: {mode: runner}\nrunner: {kind: subprocess, command: [python, ' + full + ']}\n'
        'other:\n  runner_note: ' + full + '\n')
    try:
        got = open_view(ki, tmp_path / "project")
    except ViewError as e:
        assert "re-pointed" in str(e)
        got = None
    view = tmp_path / "project" / "ki_view"
    import yaml
    c = yaml.safe_load((view / "calibration.yaml").read_text())
    assert c["other"]["runner_note"] == full                     # never touched, whichever way it went
    if got is None:
        assert (view / "calibration.yaml").read_text() == (ki / "calibration.yaml").read_text()
    else:
        assert c["runner"]["command"][1] == "{ki_path}/tools/calib_run.py"
    assert not list(view.rglob("*.kdt_tmp_*"))


# ── snippet 2b: the ENGINE run from the project's view ─────────────────────────────────────────────────────────
def test_the_engine_gives_the_same_pilot_from_the_view_as_from_the_ki_and_never_writes_the_ki(tmp_path, monkeypatch):
    import copy
    from calibration_kit import calib
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_reply_upgrade as RU
    with E._clean_env():
        ki, _wd = RU._ki(str(tmp_path), new=True)
        ki = Path(ki)
        got = open_view(ki, tmp_path / "project")
        for k, v in got["env"].items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
        before = _tree(ki)
        kw = dict(obs_shape_by_var={"streamflow": "point_time_series"}, reply_gate="check")
        a = calib.calibrate(ki_path=got["view"], workdir=str(tmp_path / "project" / "run"), **kw)
        assert _tree(ki) == before                               # the engine + runner from the view: KI untouched
        b = calib.calibrate(ki_path=str(ki), workdir=str(tmp_path / "wd_ki"), **kw)

    def clean(rep):
        rep = copy.deepcopy(rep)
        d = rep["default_reply"]
        (d.get("__kdt__") or {}).pop("series", None)
        return rep["status"], rep["reply"], d, rep["algorithm"], rep["triage"].get("route"), rep["cross_check_2n"]
    assert a["status"] == "reply_checked" and a["reply"]["complete"] is True
    assert clean(a) == clean(b)


def test_the_engine_runs_the_projects_runner_not_the_kis(tmp_path, monkeypatch):
    from calibration_kit import calib
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_reply_upgrade as RU
    with E._clean_env():
        ki, _wd = RU._ki(str(tmp_path))                          # the KI's runner is an OLD one
        got = open_view(ki, tmp_path / "project")
        for k, v in got["env"].items():
            monkeypatch.setenv(k, v)
        pr = Path(got["view"]) / "tools" / "calib_run.py"
        pr.write_text(pr.read_text() + RU.NEW_TAIL.format())     # the PROJECT's runner is upgraded
        kw = dict(obs_shape_by_var={"streamflow": "point_time_series"}, reply_gate="check")
        a = calib.calibrate(ki_path=got["view"], workdir=str(tmp_path / "project" / "run"), **kw)
        b = calib.calibrate(ki_path=str(ki), workdir=str(tmp_path / "wd_ki"), **kw)
    assert a["reply"]["complete"] is True and b["reply"]["complete"] is False


def test_a_runner_block_that_names_the_ki_in_another_field_is_refused_and_the_file_is_left(tmp_path):
    """only command and cwd may be re-pointed; a metrics file inside the KI needs a human look."""
    ki = _ki(tmp_path)
    _contract(ki, script=str(ki / "tools" / "calib_run.py"), cwd=f"  metrics_file: {ki}/m.json\n")
    with pytest.raises(ViewError, match="re-pointed"):
        open_view(ki, tmp_path / "project")
    view = tmp_path / "project" / "ki_view"
    assert (view / "calibration.yaml").read_text() == (ki / "calibration.yaml").read_text()
    assert not list(view.rglob("*.kdt_tmp_*"))


# ── the runner section has ONE allowed shape (codex snippet 2, rounds 1-7) ─────────────────────────────────────
def _raw(ki, runner_yaml):
    (ki / "calibration.yaml").write_text("injection: {mode: runner}\nrunner:\n" + runner_yaml)


GOOD = '  kind: subprocess\n  command: ["python3", "tools/calib_run.py", "--out", "{metrics_json}"]\n'

REFUSED = {
    # every case a review round found, each now outside the one allowed shape
    "metrics_file_in_ki": GOOD + '  metrics_file: KI/data/m.json\n',
    "metrics_file_relative": GOOD + '  metrics_file: "ki_link/calib_metrics.json"\n',
    "metrics_file_up": GOOD + '  metrics_file: "{workdir}/../../models/M/ki/calib/capability_case.json"\n',
    "detached": '  kind: detached\n  runner: KI/tools/calib_run.py\n  command: ["python3", "tools/calib_run.py"]\n',
    "python_kind": '  kind: python\n  callable: "tools.calib_run:run"\n',
    "alias": '  kind: subprocess\n  command: ["python", "ALIAS/tools/calib_run.py"]\n',
    "shell": '  kind: subprocess\n  command: ["bash", "-lc", "python KI/tools/calib_run.py --out x"]\n',
    "shell_alias": '  kind: subprocess\n  command: ["bash", "-c", "python ALIAS/tools/calib_run.py #", "tools/calib_run.py"]\n',
    "env_launcher": '  kind: subprocess\n  command: ["/usr/bin/env", "python", "tools/calib_run.py"]\n',
    "python_c": '  kind: subprocess\n  command: ["python", "-c", "import runpy; runpy.run_path(\'tools/sub/x.py\')", '
                '"{ki_path}/tools/calib_run.py"]\n',
    "python_m": '  kind: subprocess\n  command: ["python", "-m", "tools.calib_run"]\n',
    "through_a_link": '  kind: subprocess\n  command: ["python", "{ki_path}/data/../tools/calib_run.py"]\n',
    "workdir_up": '  kind: subprocess\n  command: ["python", "{workdir}/../../models/M/ki/tools/calib_run.py", '
                  '"{ki_path}/tools/calib_run.py"]\n',
    "arg_up": '  kind: subprocess\n  command: ["python", "tools/calib_run.py", "--c", "{ki_path}/data/../calibration.yaml"]\n',
    "other_script": '  kind: subprocess\n  command: ["python", "tools/other.py"]\n',
    "cwd_relative": GOOD + '  cwd: ki_link\n',
    "cwd_sub": GOOD + '  cwd: "{ki_path}/data"\n',
    "empty": '  kind: subprocess\n  command: []\n',
    "one_token": '  kind: subprocess\n  command: ["tools/calib_run.py"]\n',
    "nested": '  kind: subprocess\n  command: ["python", "tools/calib_run.py", ["x"]]\n',
    "ki_python": '  kind: subprocess\n  command: ["{ki_path}/tools/python", "tools/calib_run.py"]\n',
    "ki_python_full": '  kind: subprocess\n  command: ["KI/tools/python", "tools/calib_run.py"]\n',
    "ki_python_alias": '  kind: subprocess\n  command: ["ALIAS/tools/python", "tools/calib_run.py"]\n',
    "relative_python": '  kind: subprocess\n  command: ["tools/python", "tools/calib_run.py"]\n',
    "number_arg": '  kind: subprocess\n  command: ["python3", "tools/calib_run.py", "--n", 4]\n',
    "not_python": '  kind: subprocess\n  command: ["pythonw.sh", "tools/calib_run.py"]\n',
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_runner_section_outside_the_one_allowed_shape_is_refused(tmp_path, case):
    ki = _ki(tmp_path)
    os.symlink(str(ki), tmp_path / "alias_ki")
    (ki / "tools" / "python").write_text("#!/bin/sh\n")          # a program the KI ships under the name python
    os.chmod(ki / "tools" / "python", 0o755)
    _raw(ki, REFUSED[case].replace("KI", str(ki)).replace("ALIAS", str(tmp_path / "alias_ki")))
    before = _tree(ki)
    with pytest.raises(ViewError):
        open_view(ki, tmp_path / "project")
    assert _tree(ki) == before


@pytest.mark.parametrize("body", [
    GOOD,
    GOOD + '  metrics_file: "{metrics_json}"\n  cwd: "{ki_path}"\n  timeout: 120\n  notes: anything\n',
    '  kind: subprocess\n  command: ["/usr/bin/python3.12", "{ki_path}/tools/calib_run.py", "--workdir", "{workdir}", '
    '"--cfg", "{ki_path}/data/d.txt", "--n", "4"]\n',
])
def test_the_real_shapes_are_accepted(tmp_path, body):
    ki = _ki(tmp_path)
    _raw(ki, body)
    open_view(ki, tmp_path / "project")


def test_a_contract_named_in_the_environment_cannot_swap_in_another_workflow(tmp_path, monkeypatch):
    """the engine reads KDT_CALIB_CONTRACT when set — the env open_view returns pins it to the project's contract,
    and the engine then really reads the project's."""
    from calibration_kit import calib
    from calibration_kit import test_calibrate_convergence_e2e as E
    from calibration_kit import test_reply_upgrade as RU
    with E._clean_env():
        ki, _wd = RU._ki(str(tmp_path), new=True)
        stale = tmp_path / "stale_contract.yaml"
        stale.write_text("injection: {mode: applicator}\nparameters: []\n")
        monkeypatch.setenv("KDT_CALIB_CONTRACT", str(stale))
        got = open_view(ki, tmp_path / "project")
        for k, v in got["env"].items():
            monkeypatch.setenv(k, v)
        assert os.environ["KDT_CALIB_CONTRACT"] == str(Path(got["view"]) / "calibration.yaml")
        rep = calib.calibrate(ki_path=got["view"], workdir=got["workdir"],
                              obs_shape_by_var={"streamflow": "point_time_series"}, reply_gate="check")
    assert rep["status"] == "reply_checked"


@pytest.mark.parametrize("target", ["ki", "ki_sub", "view", "elsewhere"])
def test_a_run_folder_that_is_a_link_is_refused(tmp_path, target):
    """`<project>/run -> <KI>`: the engine deletes and writes its metrics file in the run folder."""
    ki = _ki(tmp_path)
    _contract(ki)
    proj = tmp_path / "project"
    (proj / "ki_view").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    os.symlink({"ki": str(ki), "ki_sub": str(ki / "data"), "view": str(proj / "ki_view"),
                "elsewhere": str(tmp_path / "elsewhere")}[target], proj / "run")
    before = _tree(ki)
    with pytest.raises(ViewError, match="run folder"):
        open_view(ki, proj)
    assert _tree(ki) == before and not (proj / "ki_view" / "tools").exists()


def test_an_interpreter_in_the_folder_the_kis_tools_link_points_to_is_refused(tmp_path):
    """codex s2 r9 #1: ki/tools -> shared_tools; the command names shared_tools/python3 by its full path."""
    ki = _ki(tmp_path)
    shared = tmp_path / "shared_tools"
    os.rename(ki / "tools", shared)
    os.symlink(shared, ki / "tools")
    (shared / "sub").unlink()
    (shared / "python3").write_text("#!/bin/sh\n")
    _raw(ki, f'  kind: subprocess\n  command: ["{shared}/python3", "tools/calib_run.py"]\n')
    with pytest.raises(ViewError, match="program"):
        open_view(ki, tmp_path / "project")


@pytest.mark.parametrize("name", ["{ki_path}", "a}b"])
def test_a_project_folder_named_like_a_fill_in_is_refused(tmp_path, name):
    """codex s2 r9 #2"""
    ki = _ki(tmp_path)
    _contract(ki)
    with pytest.raises(ViewError, match="fill-ins"):
        open_view(ki, tmp_path / name)
    assert not (tmp_path / name).exists()


# ── snippet 3a: has the KI changed? ────────────────────────────────────────────────────────────────────────────
from calibration_kit.project_workflow import ki_changes, ki_fingerprint          # noqa: E402


def test_an_untouched_ki_shows_no_change_also_after_a_view_was_built_and_run(tmp_path):
    ki = _ki(tmp_path)
    _contract(ki)
    fp = ki_fingerprint(ki)
    got = open_view(ki, tmp_path / "project")
    _run(Path(got["view"]), tmp_path / "m.json")
    assert ki_changes(fp, ki_fingerprint(ki)) == []


CHANGES = {
    "edit a helper through the view's link": (lambda ki, v: (v / "tools" / "helper.py").write_text("VALUE = 43  # x\n"),
                                              ["tools/helper.py (changed)"]),
    "same size edit": (lambda ki, v: (v / "tools" / "helper.py").write_text("VALUE = 43\n"), ["tools/helper.py (changed)"]),
    "edit through a linked folder": (lambda ki, v: (v / "tools" / "sub" / "deep.py").write_text("VALUE = 8\n"),
                                     ["shared_sub/deep.py (changed)"]),
    "add a tool in a linked folder": (lambda ki, v: (v / "tools" / "sub" / "n.py").write_text("x"),
                                      ["shared_sub/n.py (added)"]),
    "edit the KI's runner": (lambda ki, v: (ki / "tools" / "calib_run.py").write_text("x = 1\n"),
                             ["tools/calib_run.py (changed)"]),
    "edit the KI's contract keeping its size": (lambda ki, v: (ki / "calibration.yaml").write_text(
        (ki / "calibration.yaml").read_text().replace("calib-0.1", "calib-0.2")), ["calibration.yaml (changed)"]),
    "remove a data file": (lambda ki, v: (v / "data" / "d.txt").unlink(), ["data/d.txt (removed)"]),
    "re-point a link": (lambda ki, v: ((ki / "tools" / "sub").unlink(), os.symlink("../data", ki / "tools" / "sub")),
                        ["tools/sub (changed)"]),
    "edit a nested file": (lambda ki, v: ((ki / "data" / "deep").mkdir(), (ki / "data" / "deep" / "f").write_text("x")),
                           ["data/deep (added)", "data/deep/f (added)"]),
    "make a helper not runnable": (lambda ki, v: os.chmod(v / "tools" / "helper.py", 0o400), ["tools/helper.py (changed)"]),
    "close a folder": (lambda ki, v: os.chmod(ki / "shared_sub", 0o500), ["shared_sub (changed)"]),
    "change the permissions of the KI's runner": (lambda ki, v: os.chmod(ki / "tools" / "calib_run.py", 0o755),
                                                  ["tools/calib_run.py (changed)"]),
    "add a folder": (lambda ki, v: (ki / "newdir").mkdir(), ["newdir (added)"]),
    "a folder named outputs deeper down is real content": (
        lambda ki, v: ((ki / "data" / "outputs").mkdir(), (ki / "data" / "outputs" / "baseline.csv").write_text("x")),
        ["data/outputs (added)", "data/outputs/baseline.csv (added)"]),
    "the top-level outputs folder is watched too": (
        lambda ki, v: ((ki / "outputs").mkdir(), (ki / "outputs" / "r.csv").write_text("x")),
        ["outputs (added)", "outputs/r.csv (added)"]),
    "a folder named like a run record deeper down is real content": (
        lambda ki, v: ((ki / "data" / ".kdt_runs").mkdir(), (ki / "data" / ".kdt_runs" / "b.csv").write_text("x")),
        ["data/.kdt_runs (added)", "data/.kdt_runs/b.csv (added)"]),
    "the harness's own run records are not watched": (
        lambda ki, v: ((ki / ".kdt_snapshots").mkdir(), (ki / ".kdt_snapshots" / "s").write_text("x")), []),
    "run folders and byte-code are not watched": (lambda ki, v: ((ki / ".kdt_runs").mkdir(), (ki / ".kdt_runs" / "r").write_text("x"),
                                                               (ki / "tools" / "__pycache__" / "z.pyc").write_text("x")), []),
}


@pytest.mark.parametrize("what", sorted(CHANGES))
def test_a_change_in_the_ki_is_seen(tmp_path, what):
    import time
    ki = _ki(tmp_path)
    _contract(ki)
    got = open_view(ki, tmp_path / "project")
    fp = ki_fingerprint(ki)
    time.sleep(0.01)
    do, want = CHANGES[what]
    do(ki, Path(got["view"]))
    assert ki_changes(fp, ki_fingerprint(ki)) == want


def test_an_edit_of_the_projects_own_files_is_not_a_ki_change(tmp_path):
    ki = _ki(tmp_path)
    _contract(ki)
    got = open_view(ki, tmp_path / "project")
    fp = ki_fingerprint(ki)
    v = Path(got["view"])
    (v / "tools" / "calib_run.py").write_text("# upgraded\n")
    (v / "calibration.yaml").write_text("x: 1\n")
    (v / "calib" / "capability_case.json").write_text("{}")
    assert ki_changes(fp, ki_fingerprint(ki)) == []


def test_a_ki_whose_tools_is_a_link_is_still_watched_inside(tmp_path):
    ki = _ki(tmp_path)
    shared = tmp_path / "shared_tools"
    os.rename(ki / "tools", shared)
    os.symlink(shared, ki / "tools")
    fp = ki_fingerprint(ki)
    (shared / "helper.py").write_text("VALUE = 0  # changed\n")
    assert ki_changes(fp, ki_fingerprint(ki)) == ["tools/->/helper.py (changed)"]


def test_edits_through_links_that_lead_out_of_the_ki_are_seen(tmp_path):
    """real VIC: tools/plot/x.py -> a shared skills folder; real CaMa-Flood: tools/downscale -> shared scripts."""
    ki = _ki(tmp_path)
    _contract(ki)
    shared = tmp_path / "shared_skills"
    (shared / "pkg").mkdir(parents=True)
    (shared / "pkg" / "a.py").write_text("A = 1\n")
    (shared / "one.py").write_text("B = 1\n")
    (shared / "pkg" / ".kdt_runs").mkdir()                        # top level of a LINKED folder: real content
    (shared / "pkg" / ".kdt_runs" / "t.py").write_text("T = 1\n")
    os.symlink(str(shared / "pkg"), ki / "tools" / "pkg")
    os.symlink(str(shared / "one.py"), ki / "tools" / "one.py")
    os.symlink(str(shared), ki / "loop_a")
    os.symlink(str(ki), shared / "back_to_ki")
    os.symlink(str(shared), shared / "pkg" / "self")              # a loop outside the KI: must end
    got = open_view(ki, tmp_path / "project")
    v = Path(got["view"])
    fp = ki_fingerprint(ki)
    assert ki_changes(fp, ki_fingerprint(ki)) == []
    (v / "tools" / "pkg" / "a.py").write_text("A = 2  # edited through the view\n")
    (v / "tools" / "one.py").write_text("B = 2  # edited through the view\n")
    ch = ki_changes(fp, ki_fingerprint(ki))
    assert "tools/pkg/->/a.py (changed)" in ch and "tools/one.py/-> (changed)" in ch
    fp3 = ki_fingerprint(ki)
    os.chmod(v / "tools" / "pkg", 0o555)                          # the linked outside folder itself
    assert "tools/pkg/-> (changed)" in ki_changes(fp3, ki_fingerprint(ki))
    os.chmod(v / "tools" / "pkg", 0o755)
    fp2 = ki_fingerprint(ki)
    (v / "tools" / "pkg" / ".kdt_runs" / "t.py").write_text("T = 2  # edited\n")
    assert "tools/pkg/->/.kdt_runs/t.py (changed)" in ki_changes(fp2, ki_fingerprint(ki))


def test_a_link_replaced_by_a_real_file_leaves_the_ki_alone_but_the_view_is_then_refused(tmp_path):
    """codex 3a r2 #1: `sed -i` on tools/helper.py in the view swaps the LINK for a real file. The KI is unchanged
    (nothing to report there) — the view is no longer a view of the KI, and building it again refuses it."""
    ki = _ki(tmp_path)
    _contract(ki)
    got = open_view(ki, tmp_path / "project")
    v = Path(got["view"])
    fp = ki_fingerprint(ki)
    text = (v / "tools" / "helper.py").read_text()
    (v / "tools" / "helper.py").unlink()
    (v / "tools" / "helper.py").write_text(text.replace("42", "43"))        # what sed -i does
    assert ki_changes(fp, ki_fingerprint(ki)) == []
    with pytest.raises(ViewError, match="real file"):
        open_view(ki, tmp_path / "project")


def test_a_folder_that_cannot_be_listed_fails_the_fingerprint(tmp_path):
    """codex 3a r4: a helper folder the walk cannot list would be left unwatched without a word."""
    if os.geteuid() == 0:
        pytest.skip("root can list everything")
    ki = _ki(tmp_path)
    (ki / "tools" / "private").mkdir()
    (ki / "tools" / "private" / "secret.py").write_text("S = 1\n")
    os.chmod(ki / "tools" / "private", 0o100)
    try:
        with pytest.raises(ViewError, match="cannot read"):
            ki_fingerprint(ki)
    finally:
        os.chmod(ki / "tools" / "private", 0o700)


def test_a_link_into_a_left_out_run_folder_is_watched_through_the_link(tmp_path):
    """codex 3a r5: tools/shared -> ../.kdt_runs/shared — the top-level .kdt_runs/ is left out of the walk, but what a
    link in the watched part points to there is real content."""
    ki = _ki(tmp_path)
    _contract(ki)
    (ki / ".kdt_runs" / "shared").mkdir(parents=True)
    (ki / ".kdt_runs" / "shared" / "helper2.py").write_text("H = 1\n")
    (ki / ".kdt_runs" / "run1.csv").write_text("x")
    os.symlink("../.kdt_runs/shared", ki / "tools" / "shared")
    got = open_view(ki, tmp_path / "project")
    fp = ki_fingerprint(ki)
    (ki / ".kdt_runs" / "run2.csv").write_text("y")                             # a harness record: not watched
    assert ki_changes(fp, ki_fingerprint(ki)) == []
    (Path(got["view"]) / "tools" / "shared" / "helper2.py").write_text("H = 2  # edited\n")
    assert ki_changes(fp, ki_fingerprint(ki)) == ["tools/shared/->/helper2.py (changed)"]


@pytest.mark.parametrize("text", ["broken: [", "- just\n- a list\n"])
def test_a_contract_that_cannot_be_read_is_a_plain_refusal(tmp_path, text):
    ki = _ki(tmp_path)
    (ki / "calibration.yaml").write_text(text)
    with pytest.raises(ViewError, match="contract"):
        open_view(ki, tmp_path / "project")
