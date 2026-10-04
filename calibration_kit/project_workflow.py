"""The calibration workflow belongs to the project, not to the KI (Leo, 2026-10-02).

A workflow is two files written for one project: `calibration.yaml` and `tools/calib_run.py`. They live in the
project's folder. Everything else a runner needs (helper tools, data, dag.yaml, docs) stays in the KI and is only read.

`build_ki_view(ki_path, view_dir)` makes a KI-shaped folder in the project:

    <view>/calibration.yaml        the project's own file (copied from the KI the first time, never overwritten)
    <view>/tools/                  a real folder
    <view>/tools/calib_run.py      the project's own file (copied from the KI the first time, never overwritten)
    <view>/tools/<other>      ->   link to <KI>/tools/<other>
    <view>/calib/                  a real folder
    <view>/calib/capability_case.json   the project's own (the approval of THIS workflow; written by the stage)
    <view>/calib/<other>      ->   link to <KI>/calib/<other>
    <view>/<other>            ->   link to <KI>/<other>

The engine is given the view as its `ki_path`, so nothing in the engine changes: the runner is started as
`python tools/calib_run.py` from the view, its `__file__` is the project's real file, its sibling helpers and
`Path(__file__).parent.parent / ...` resolve through the links into the KI. Calling it again refreshes the links
(new KI entries appear, removed ones go) and never touches a real file in the view. A real file or folder in the
view under a name the KI also has is refused (it would silently stand in for the KI's).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

OWN = ("calibration.yaml", "tools/calib_run.py")             # the project's own files (the workflow)
# written by the calibration stage for THIS workflow (its approval record): the project's own as well. Copied from
# the KI when the KI has one (the copied workflow is the approved one), else made later by the stage.
OWN_OPTIONAL = ("calib/capability_case.json",)
OWN_DIRS = ("tools", "calib")                                # real folders in the view, linked entry by entry
_SKIP = ("__pycache__", ".git")


class ViewError(RuntimeError):
    """the project's folder cannot be set up, or is no longer a proper view of the KI"""


class WorkflowNotSupported(ViewError):
    """the folder is fine, but this workflow's shape cannot run from a project folder (applicator mode, a runner
    section outside the one allowed shape) — the caller may calibrate it in the KI as before"""


def _inside(a: Path, b: Path) -> bool:
    try:
        a.relative_to(b)
        return True
    except ValueError:
        return False


def _link_level(src_dir, dst_dir: Path, leave_out: set, report: dict, strict: bool = False) -> None:
    """Make dst_dir hold one link per entry of src_dir (except leave_out); drop links whose entry is gone.
    strict (the tools/ and calib/ folders): any real file or folder other than `leave_out` is refused — a left-over
    helper there would be imported or listed by the runner though it is neither the project's workflow nor the KI's."""
    want = {n for n in (os.listdir(src_dir) if src_dir is not None else []) if n not in leave_out and n not in _SKIP}
    if strict:
        extra = sorted(n for n in os.listdir(dst_dir)
                       if not (dst_dir / n).is_symlink() and n not in leave_out and n not in _SKIP)
        if extra:
            raise ViewError(f"{dst_dir} holds real files or folders that are not the project's workflow files: "
                            f"{extra}; keep notes and other project files outside this folder")
    src_dir = Path(src_dir) if src_dir is not None else dst_dir
    for name in sorted(os.listdir(dst_dir)):
        p = dst_dir / name
        if p.is_symlink() and (name not in want or os.readlink(p) != str(src_dir / name)):
            p.unlink()                                       # a stale link (entry removed, or the KI moved)
            report["removed"].append(str(p))
    for name in sorted(want):
        p = dst_dir / name
        if not p.is_symlink() and os.path.lexists(p):
            # a real file or folder under a KI name would silently stand in for the KI's (stale data, an old
            # helper): only the two workflow files may be the project's own
            raise ViewError(f"{p} is a real file or folder in the view but {src_dir / name} exists in the KI; "
                            f"only {', '.join(OWN)} may be the project's own")
        if p.is_symlink():
            continue
        try:
            os.symlink(str(src_dir / name), p)
            report["linked"].append(str(p))
        except FileExistsError:                              # another process built the same view just now
            if not (p.is_symlink() and os.readlink(p) == str(src_dir / name)):
                raise ViewError(f"{p} appeared while the view was built and is not the expected link")


def build_ki_view(ki_path, view_dir) -> dict:
    """Build or refresh the project's view of a KI. Returns what it did. Writes nothing in the KI."""
    ki = Path(ki_path).resolve()
    view = Path(view_dir).resolve()                          # links in any parent folder are followed first
    if not ki.is_dir():
        raise ViewError(f"no KI folder at {ki}")
    if _inside(view, ki) or _inside(ki, view):
        raise ViewError(f"the project's view {view} and the KI {ki} must not be inside one another")
    for rel in OWN:
        if not (ki / rel).is_file() and not (view / rel).is_file():
            raise ViewError(f"neither the project nor the KI has {rel}")
    report = {"view": str(view), "ki": str(ki), "linked": [], "removed": [], "copied": []}
    for d in OWN_DIRS:
        if os.path.lexists(view / d) and not (view / d).is_symlink() and not (view / d).is_dir():
            raise ViewError(f"{view / d} exists and is not a folder")
        if os.path.lexists(ki / d) and not (ki / d).is_dir():
            raise ViewError(f"the KI's {d} is not a folder: the view cannot show it")
    left = sorted(str(p) for d in ("",) + OWN_DIRS if (view / d).is_dir() and not (view / d).is_symlink()
                  for p in (view / d).iterdir() if ".kdt_tmp_" in p.name)
    if left:
        raise ViewError(f"left-over files of a copy that was cut short: {left}; remove them and build again")
    for d in OWN_DIRS:
        if (view / d).is_symlink():
            if d == "tools" or not str(os.readlink(view / d)) == str(ki / d):
                raise ViewError(f"{view / d} is a link; it must be the project's own folder")
            (view / d).unlink()                              # a view built before this folder became the project's
            report["removed"].append(str(view / d))
        (view / d).mkdir(parents=True, exist_ok=True)
    for rel in OWN + OWN_OPTIONAL:
        dst = view / rel
        if dst.is_symlink():
            raise ViewError(f"{dst} is a link; it must be the project's own file")
        if dst.exists() and (not dst.is_file() or dst.stat().st_nlink > 1):
            # a hard link shares its content with another file (maybe the KI's): an edit would change both
            raise ViewError(f"{dst} is not a plain file of the project's own (a folder, or a hard link)")
        if not dst.exists() and (ki / rel).is_file():
            # whole or not at all: a copy cut short (disk full, a kill) must never be taken for the project's file
            tmp = dst.with_name(dst.name + f".kdt_tmp_{os.getpid()}")
            try:
                shutil.copyfile(ki / rel, tmp)               # content only: the project owns the copy
                os.link(tmp, dst)                            # fails if another builder installed it meanwhile
            except FileExistsError:
                pass
            finally:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
            report["copied"].append(rel)
    own = OWN + OWN_OPTIONAL
    _link_level(ki, view, set(OWN_DIRS) | {r for r in own if "/" not in r}, report)
    for d in OWN_DIRS:
        _link_level((ki / d).resolve() if (ki / d).is_dir() else None, view / d,
                    {r.split("/", 1)[1] for r in own if r.startswith(d + "/")}, report, strict=True)
    return report


def view_env(view_dir) -> dict:
    """Environment for every run from a view: Python keeps its byte-code files in the project
    (`<view>/.pycache/`), so importing a KI helper through a link writes nothing into the KI."""
    return {"PYTHONPYCACHEPREFIX": str(Path(view_dir).resolve() / ".pycache")}


def _point_runner_at_view(view: Path, ki: Path) -> list:
    """The project's contract must start the PROJECT's runner. A contract that names the KI by its full path
    (`/…/<KI>/tools/calib_run.py`, or `cwd: /…/<KI>`) would start the KI's file instead: such a path is rewritten to
    `{ki_path}` (which the engine fills with the view). Only the runner's command and cwd may change; the rewrite is
    checked by reading the file back. Returns the rewritten strings."""
    import yaml
    f = view / "calibration.yaml"
    text = f.read_text()
    before = yaml.safe_load(text) or {}
    runner = before.get("runner") or {}
    toks = [t for t in list(runner.get("command") or []) + [runner.get("cwd")] if isinstance(t, str)]
    kis = str(ki)
    changed = [(t, "{ki_path}" + t[len(kis):]) for t in dict.fromkeys(toks) if t == kis or t.startswith(kis + "/")]
    if not changed:
        return []
    # edit only the lines of the top-level `runner:` block; a bare path becomes a quoted one (`{…}` unquoted is not
    # a string in YAML); comments and every other line stay as they are
    import re
    pat = re.compile(r"""(?P<q>["']?)""" + re.escape(kis) + r"""(?P<rest>(?:/[^\s"',\]#]*)?)(?P=q)""")
    lines, inside, out = text.split("\n"), False, []
    for ln in lines:
        if re.match(r"^\S", ln):
            inside = bool(re.match(r"^runner\s*:", ln))
        out.append(pat.sub(lambda m: '"{ki_path}' + m.group("rest") + '"', ln) if inside else ln)
    text = "\n".join(out)
    try:
        after = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        after = None
    want = dict(before)
    r2 = dict(runner)
    sub = dict(changed)
    if "command" in r2:
        r2["command"] = [sub.get(t, t) if isinstance(t, str) else t for t in r2["command"]]
    if isinstance(r2.get("cwd"), str):
        r2["cwd"] = sub.get(r2["cwd"], r2["cwd"])
    want["runner"] = r2
    if after != want:
        raise WorkflowNotSupported(f"{f} names the KI by its full path in the runner section and could not be re-pointed "
                        f"safely; edit the project's contract by hand")
    tmp = f.with_name(f.name + f".kdt_tmp_{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, f)
    finally:
        if os.path.lexists(tmp):
            os.unlink(tmp)
    return [f"{a} -> {b}" for a, b in changed]


def open_view(ki_path, project_dir) -> dict:
    """Set up the project's folder for one model's calibration workflow and return what the engine needs:
    {"view": the folder to pass as `ki_path`, "workdir": the folder to pass as `workdir`, "env": variables to set
    for every run, "report": ...}.

    The workflow's runner section must have the ONE shape every real workflow has; anything else is refused (not
    guessed at), so that by reading the contract it is certain the engine starts the PROJECT's runner and touches
    no file of the KI:
      injection.mode  runner          (in `applicator` mode the engine itself writes values into KI files)
      runner.kind     subprocess
      runner.command  [<python: a bare name, or a full path outside the KI and the project>,
                      "tools/calib_run.py" or "{ki_path}/tools/calib_run.py", args...]; all strings; no `..`
      runner.cwd      absent, or "{ki_path}"
      runner.metrics_file   absent, or "{metrics_json}"
    A contract that names the KI by its full path in command or cwd is first re-pointed to `{ki_path}`."""
    import re
    import yaml
    ki = Path(ki_path).resolve()
    proj = Path(project_dir).resolve()
    view, run_dir = proj / "ki_view", proj / "run"
    if any(c in str(x) for x in (ki, proj) for c in "{}"):
        raise ViewError(f"a folder name with `{{` or `}}` ({ki} / {proj}) would be mistaken for one of the engine's "
                        f"fill-ins")
    rr = os.path.realpath(run_dir)
    if run_dir.is_symlink() or rr != str(run_dir) or _inside(run_dir, ki) or _inside(ki, run_dir):
        raise ViewError(f"the project's run folder {run_dir} is a link, or lies in or around the KI")
    report = build_ki_view(ki, view)
    try:
        contract = yaml.safe_load((view / "calibration.yaml").read_text()) or {}
    except yaml.YAMLError as e:
        raise ViewError(f"the project's contract {view / 'calibration.yaml'} cannot be read: {str(e)[:200]}")
    if not isinstance(contract, dict):
        raise ViewError(f"the project's contract {view / 'calibration.yaml'} is not a mapping")
    mode = (contract.get("injection", {}) or {}).get("mode", "applicator")
    if mode != "runner":
        raise WorkflowNotSupported(f"the workflow's injection mode is '{mode}': the engine itself would write parameter values "
                        f"into the KI's files. Only `injection.mode: runner` can run from a project folder")
    report["re_pointed"] = _point_runner_at_view(view, ki)
    runner = (yaml.safe_load((view / "calibration.yaml").read_text()) or {}).get("runner") or {}
    cmd = runner.get("command")
    why = None
    if runner.get("kind") != "subprocess":
        why = f"runner kind is '{runner.get('kind')}', not subprocess"
    elif not (isinstance(cmd, list) and len(cmd) >= 2 and all(isinstance(t, str) for t in cmd)):
        why = "the command is not a list of at least two strings"
    elif not re.fullmatch(r"python[0-9.]*", os.path.basename(cmd[0])):
        why = f"the program is '{cmd[0]}', not a python interpreter"
    elif "/" in cmd[0] and (not os.path.isabs(cmd[0])
                            or any(_inside(Path(os.path.realpath(cmd[0])), x)
                                   for x in (ki, proj, (ki / "tools").resolve(), (ki / "calib").resolve()))):
        # a bare name (python3) or a full path outside the KI and the project: never a program the KI ships
        why = f"the program '{cmd[0]}' is not a bare name or a full path outside the KI and the project"
    elif str(cmd[1]) not in ("tools/calib_run.py", "{ki_path}/tools/calib_run.py"):
        why = f"the script is '{cmd[1]}', not tools/calib_run.py or {{ki_path}}/tools/calib_run.py"
    elif any(".." in Path(str(t)).parts for t in cmd):
        why = "an argument uses `..`"
    elif runner.get("cwd") not in (None, "{ki_path}"):
        why = f"cwd is '{runner.get('cwd')}', not absent or {{ki_path}}"
    elif runner.get("metrics_file") not in (None, "{metrics_json}"):
        why = f"metrics_file is '{runner.get('metrics_file')}', not absent or {{metrics_json}}"
    if why:
        raise WorkflowNotSupported(f"the project's contract cannot run from a project folder: {why}. Edit the runner section "
                        f"of {view / 'calibration.yaml'}")
    # the engine reads the contract named by KDT_CALIB_CONTRACT when that is set: pin it to the project's contract,
    # so a value left in the environment can never swap in another workflow
    env = dict(view_env(view), KDT_CALIB_CONTRACT=str(view / "calibration.yaml"))
    return {"view": str(view), "workdir": str(run_dir), "env": env, "report": report}


# ── has the KI changed? (an agent works on the project's runner; the links let it reach the KI) ──────────────────
_FP_SKIP = ("__pycache__", ".git")                           # left out at any depth
# left out ONLY at the KI's top level. `outputs/` and `detached/` ARE watched: a runner may read prepared inputs
# there, and a write there through the view's link is a write into the KI
_FP_SKIP_TOP = (".kdt_runs", ".kdt_snapshots", ".kdt_candidates")   # the harness's own run records


def ki_fingerprint(ki_path) -> dict:
    """A cheap record of the KI's code and settings: every file, folder and link under the KI (run and snapshot
    folders at the KI's top level left out) as kind, size, change time and permissions — and a content hash for the three workflow files. A link that
    leads OUT of the KI (a shared tools folder, a shared script) is recorded and what it points to is recorded too,
    because an edit made through the link lands there. Compare two with `ki_changes`."""
    import hashlib
    ki = Path(ki_path).resolve()
    out, seen = {}, set()

    def one_file(p, rel):
        st = os.stat(p)
        out[rel] = ["F", st.st_size, st.st_mtime_ns, st.st_mode]

    def walk(root, prefix):
        real = os.path.realpath(root)
        if real in seen:
            return
        seen.add(real)
        def cannot_list(err):                                # a folder that cannot be read cannot be watched
            raise ViewError(f"cannot read {getattr(err, 'filename', err)} in the KI: its content cannot be watched")
        for dp, dns, fns in os.walk(root, followlinks=False, onerror=cannot_list):
            top = prefix == "" and dp == root                # the KI's own top level
            dns[:] = sorted(d for d in dns if d not in _FP_SKIP and not (top and d in _FP_SKIP_TOP))
            rel_dp = os.path.normpath(os.path.join(prefix, os.path.relpath(dp, root)))
            for n in sorted(dns + fns):
                p = os.path.join(dp, n)
                rel = os.path.normpath(os.path.join(rel_dp, n))
                if os.path.islink(p):
                    out[rel] = ["L", os.readlink(p)]
                    target = os.path.realpath(p)
                    unwatched = (not _inside(Path(target), ki)
                                 or Path(target).relative_to(ki).parts[:1] in [(d,) for d in _FP_SKIP_TOP])
                    if unwatched:        # leads out of the KI, or into a top-level folder the walk leaves out
                        if os.path.isdir(target):
                            out[rel + "/->"] = ["D", os.stat(target).st_mode]
                            walk(target, rel + "/->")
                        elif os.path.isfile(target):
                            one_file(target, rel + "/->")
                elif n in fns:
                    one_file(p, rel)
                else:
                    out[rel] = ["D", os.stat(p).st_mode]
    walk(str(ki), "")
    for rel in OWN + OWN_OPTIONAL:
        f = ki / rel
        if f.is_file():
            out[rel] = ["F", f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mode]
    return out


def ki_changes(before: dict, after: dict) -> list:
    """What differs between two fingerprints: ["tools/x.py (changed)", "new.py (added)", "old.py (removed)"]."""
    out = []
    for rel in sorted(set(before) | set(after)):
        if rel not in before:
            out.append(f"{rel} (added)")
        elif rel not in after:
            out.append(f"{rel} (removed)")
        elif before[rel] != after[rel]:
            out.append(f"{rel} (changed)")
    return out
