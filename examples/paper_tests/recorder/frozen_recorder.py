"""Record every call of a search run by the PAPER-FROZEN calibration kit, without changing the search.

Why: the paper's reruns must use the frozen kit and the paper's own runners (they count only if they reproduce the
reported result exactly), but the new convergence check needs, for every call: the parameters, all scores, whether it
was a cache hit, which phase, which seed, and the optimizer's own step (SCE-UA loop, pymoo generation). The frozen kit
keeps none of that on disk. This module records it from the outside.

Two kinds of line, one file (<out_dir>/calls.jsonl), in the order they happen, numbered by `i`:
  - "model_run": every real model run made through a runner from `runner.make_run_model` — including the runs the
    frozen kit makes outside Evaluator.evaluate (runner certification, the objective probe, a baseline): the params
    file handed to the runner, the split, the evaluation id and the runner's raw reply (`payload`), exactly as
    returned;
  - "evaluator" (calibrate() runs) / "problem" (drivers that call a backend directly): every call the optimizer or the
    kit makes to score a parameter set: x, named parameters, losses, split, whether a model run happened (and which
    `model_run_i`), a cache hit, or a rejection before any run; the metrics of THIS call only (the raw reply of its own
    model run, or the cached payload on a hit) — never an earlier call's.
Each line also has phase (certification / objective_probe / commissioning / sensitivity / search / front_select /
holdout / outside, or a phase a driver sets), search number, seed, algorithm, SCE-UA loop (0 = the start-up sample;
read from SPOTPY's own printed "ComplexEvo loop #N" lines, passed through unchanged) and pymoo batch number (`gen`:
one batch per generation for NSGA-II/III; one per offspring for MOEA/D). <out_dir>/searches.jsonl: one line per
backend search.

A driver must call install() BEFORE it imports calibrate (or anything else) from the frozen kit by name, so it gets
the wrapped versions. Nothing the optimizer sees is changed: every wrapper calls the original with the same arguments and returns its result
unchanged. Recording is best-effort: a failure to write is logged to recorder_errors.jsonl and never reaches the
search (set KDT_RECORDER_STRICT=1 to raise instead, for tests). One search at a time per process.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time

FROZEN = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code"
_LOOP_RE = re.compile(r"ComplexEvo loop #(\d+)")

_state = {"installed": False, "out": None, "fh": None, "fs": None, "fe": None, "n": 0, "phases": ["outside"],
          "search": None, "seed": None, "algorithm": None, "loop": None, "gen": None, "active": False,
          "runs": 0, "lock": threading.Lock()}


def _strict() -> bool:
    return os.environ.get("KDT_RECORDER_STRICT") == "1"


def _plain(o):
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")     # JSON has no nan / inf
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    try:
        return _plain(float(o))                  # numpy scalars too (np.float32 nan / inf included)
    except Exception:
        return str(o)


def _err(where: str, e: BaseException):
    try:
        _state["fe"].write(json.dumps({"t": round(time.time(), 3), "where": where, "error": f"{type(e).__name__}: {e}"}) + "\n")
        _state["fe"].flush()
    except Exception:
        pass
    if _strict():
        raise e


def _write(rec: dict) -> int | None:
    """Append one line; returns its number. Never raises into the search (unless strict)."""
    try:
        with _state["lock"]:
            i = _state["n"]
            rec = dict(rec, i=i, t=round(time.time(), 3), phase=_state["phases"][-1], search=_state["search"],
                       seed=_state["seed"], algorithm=_state["algorithm"], loop=_state["loop"], gen=_state["gen"])
            _state["fh"].write(json.dumps(_plain(rec), allow_nan=False) + "\n")
            _state["fh"].flush()
            _state["n"] += 1
            return i
    except Exception as e:
        _err("write", e)
        return None


class _Tee(io.TextIOBase):
    """stdout passed through unchanged; SPOTPY's 'ComplexEvo loop #N' lines set the current loop."""
    def __init__(self, inner):
        self.inner = inner
        self._buf = ""

    def write(self, s):
        try:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                m = _LOOP_RE.search(line)
                if m:
                    _state["loop"] = int(m.group(1))
        except Exception as e:
            _err("tee", e)
        return self.inner.write(s)

    def flush(self):
        return self.inner.flush()


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _state["phases"].append(self.name)

    def __exit__(self, *a):
        _state["phases"].pop()


def phase(name: str):
    """A driver may mark its own phase: `with frozen_recorder.phase("holdout"): ...`."""
    return _Phase(name)


def set_phase(name: str):
    """Set the base phase (outside any kit phase)."""
    _state["phases"][0] = name


def _wrap_phase(mod, fname, name, cls=None):
    owner = cls if cls is not None else mod
    orig = getattr(owner, fname)

    def w(*a, **k):
        with _Phase(name):
            return orig(*a, **k)
    w.__wrapped__ = orig
    setattr(owner, fname, w)


def _recorded_run_model(inner):
    if getattr(inner, "_kdt_recorded", False):
        return inner

    def run_model(*a, **k):
        snap = {}
        try:                                 # what is handed to the model, read BEFORE it runs
            pf = os.environ.get("KDT_CALIB_PARAMS")
            params = None
            if pf and os.path.isfile(pf):
                with open(pf) as fh:
                    params = json.load(fh)
            snap = {"params_handed": params, "split_env": os.environ.get("KDT_CALIB_SPLIT"),
                    "eval_id_env": os.environ.get("KDT_CALIB_EVAL_ID")}
        except Exception as e:
            _err("model_run_pre", e)
        try:
            out = inner(*a, **k)
        except BaseException as exc:            # the run was attempted and failed: recorded, then passed on
            try:
                i = _write(dict(snap, kind="model_run", payload=None, raised=f"{type(exc).__name__}: {exc}"[:500]))
                _state["runs"] += 1
                _state["last_run_i"] = i
            except Exception as e:
                _err("model_run_raised", e)
            raise
        try:
            i = _write(dict(snap, kind="model_run", payload=out))
            _state["runs"] += 1
            _state["last_run_i"] = i
        except Exception as e:
            _err("model_run", e)
        return out
    run_model._kdt_recorded = True
    return run_model


def install(out_dir: str):
    """Wrap the frozen kit. Call after FROZEN is first on sys.path and before any search runs."""
    if _state["installed"]:
        raise RuntimeError("recorder already installed")
    import calibration_kit
    if not calibration_kit.__file__.startswith(FROZEN):
        raise RuntimeError(f"not the paper-frozen kit: {calibration_kit.__file__}")
    from calibration_kit.backends import spotpy_backend as SB, pymoo_backend as PB
    from calibration_kit import evaluator as EV, runner as RN, calib as CA, holdout as HO, sensitivity as SE
    os.makedirs(out_dir, exist_ok=True)
    _state.update(out=out_dir, fh=open(os.path.join(out_dir, "calls.jsonl"), "a"),
                  fs=open(os.path.join(out_dir, "searches.jsonl"), "a"),
                  fe=open(os.path.join(out_dir, "recorder_errors.jsonl"), "a"), installed=True, search=0)

    # ── every real model run made through a runner (incl. certification, probe, baseline) ─────────────
    _orig_mrm = RN.make_run_model

    def make_run_model(runner_spec, ki_path, workdir):
        return _recorded_run_model(_orig_mrm(runner_spec, ki_path, workdir))
    RN.make_run_model = make_run_model

    # a model callable handed to calibrate() / calibrate_staged() directly is wrapped the same way
    for _fname in ("calibrate", "calibrate_staged"):
        _orig_c = getattr(CA, _fname)

        def _c(*a, _orig=_orig_c, **k):
            if k.get("run_model") is not None:
                k["run_model"] = _recorded_run_model(k["run_model"])
            elif len(a) >= 4 and a[3] is not None:
                a = list(a); a[3] = _recorded_run_model(a[3]); a = tuple(a)
            return _orig(*a, **k)
        _c.__wrapped__ = _orig_c
        setattr(CA, _fname, _c)

    # ── the frozen Evaluator: one line per scoring call, with ITS OWN metrics ─────────────────────────
    _orig_eval = EV.Evaluator.evaluate

    def evaluate(self, x):
        eid0, runs0 = self._eval_id, _state["runs"]
        split = EV.resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
        named, hit_payload = None, None
        try:
            named = self._named(x)
            hit_payload = self._metrics_cache.get(self._cache_key(named, split))
        except Exception as e:
            _err("evaluator_pre", e)
        own = {}
        inner_rm = self.run_model

        def rm(*a, **k):                         # the raw reply of THIS call's own model run
            own["attempted"] = True
            own["payload"] = inner_rm(*a, **k)
            return own["payload"]
        self.run_model = rm
        try:
            losses = _orig_eval(self, x)
        finally:
            self.run_model = inner_rm
        try:
            ran = bool(own.get("attempted"))         # a run that raised still counts as a run (with no payload)
            _write({"kind": "evaluator", "x": list(x), "params": named, "losses": list(losses), "split": split,
                    "eval_id": self._eval_id if self._eval_id != eid0 else None, "ran_model": ran,
                    "model_run_i": _state.get("last_run_i") if _state["runs"] != runs0 else None,
                    "cache_hit": (not ran) and hit_payload is not None,
                    "rejected_before_run": (not ran) and hit_payload is None,
                    "run_raised": ran and "payload" not in own,
                    "metrics": own.get("payload") if ran else hit_payload})
        except Exception as e:
            _err("evaluator", e)
        return losses
    EV.Evaluator.evaluate = evaluate

    # ── phases of calibrate() ──────────────────────────────────────────────────────────────────────
    _wrap_phase(CA, "certify_runner", "certification")
    _wrap_phase(CA, "_probe_runner_metrics", "objective_probe")
    _wrap_phase(EV, "responsiveness_check", "commissioning", cls=EV.Evaluator)
    _wrap_phase(SE, "morris_screen", "sensitivity")
    _wrap_phase(HO, "validate_holdout", "holdout")
    _wrap_phase(CA, "_select_front_member", "front_select")

    # ── backends: the search phase, its seed and the optimizer's own step ────────────────────────────
    def _wrap_problem(problem):
        if isinstance(problem, EV.Evaluator) or getattr(problem, "_kdt_recorded", False):
            return problem
        inner = problem.evaluate

        def ev(x):
            losses = inner(x)
            try:
                rec = {"kind": "problem", "x": [float(v) for v in x],
                       "losses": list(losses) if losses is not None else None}
                extra = getattr(problem, "last_record", None)
                if extra is not None:
                    rec["extra"] = extra
                _write(rec)
            except Exception as e:
                _err("problem", e)
            return losses
        problem.evaluate = ev
        problem._kdt_recorded = True
        return problem

    def _search(algo_name, fn, self, problem, budget, seed=0, **kw):
        with _state["lock"]:
            if _state["active"]:
                raise RuntimeError("recorder: a second search started while one is running (one search at a time)")
            _state.update(active=True, seed=seed, algorithm=algo_name, loop=(0 if algo_name == "sceua" else None),
                          gen=None)
        _state["phases"].append("search")
        n0 = _state["n"]
        old = sys.stdout
        if algo_name == "sceua":
            sys.stdout = _Tee(old)
        try:
            return fn(self, _wrap_problem(problem), budget, seed=seed, **kw)
        finally:
            sys.stdout = old
            _state["phases"].pop()
            try:
                _state["fs"].write(json.dumps(_plain({"search": _state["search"], "algorithm": algo_name,
                                                      "budget": budget, "seed": seed, "lines": _state["n"] - n0,
                                                      "first_line": n0})) + "\n")
                _state["fs"].flush()
            except Exception as e:
                _err("searches", e)
            with _state["lock"]:
                _state.update(active=False, search=_state["search"] + 1, loop=None, gen=None, seed=None, algorithm=None)

    _orig_sp = SB.SpotpyBackend.optimize

    def sp_optimize(self, problem, budget, seed=0, **kw):
        return _search(getattr(self, "algorithm", "spotpy"), _orig_sp, self, problem, budget, seed=seed, **kw)
    SB.SpotpyBackend.optimize = sp_optimize

    _orig_pm = PB.PymooBackend.optimize

    def pm_optimize(self, problem, budget, seed=0, **kw):
        import pymoo.core.problem as PP
        orig_pe = PP.Problem.evaluate

        def pe(pself, X, *a, **k):              # one batch pymoo hands to the problem: one generation for
            _state["gen"] = 1 if _state["gen"] is None else _state["gen"] + 1   # NSGA-II/III; one offspring for MOEA/D
            return orig_pe(pself, X, *a, **k)
        PP.Problem.evaluate = pe
        try:
            return _search(getattr(self, "algorithm", "pymoo"), _orig_pm, self, problem, budget, seed=seed, **kw)
        finally:
            PP.Problem.evaluate = orig_pe
    PB.PymooBackend.optimize = pm_optimize
    return _state
