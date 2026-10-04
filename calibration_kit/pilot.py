"""PILOT — measure the model before deciding how long to search it (handoff §5.3).

A budget written into a contract by hand is a guess about run time. The pilot replaces the guess
with a measurement: ten live evaluations (the default vector plus a Latin-hypercube spread over
the search box) timed on THIS machine, with their CPU use and peak memory. From that,
budget.plan_budget turns the user's wall-clock allowance into an evaluation cap.

Three other things fall out of the same ten runs, which is why it is cheap:
  * stability — more than 30% failures means the ranges or the runner are wrong, and there is no
    point searching at all (`pilot_unstable`, handed back to the agent);
  * how many copies of the model can run side by side (threads per run, peak RSS → compute.py);
  * the paired series for the metric tolerances (panel.bootstrap_tolerances), when the runner can
    emit them.
Every pilot evaluation goes through ev.evaluate, so it lands in the cache and the search can
re-use the points instead of paying for them twice.
"""
from __future__ import annotations
import math
import json
import os
import resource
import time
from pathlib import Path


def lhs_points(lower, upper, m: int, seed: int = 0) -> list:
    """m Latin-hypercube points in the search box: one per stratum in every dimension, so a small
    sample still covers each parameter's range instead of clustering the way uniform draws can."""
    import numpy as np
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    d = len(lo)
    if m <= 0:
        return []
    u = (np.argsort(rng.random((d, m)), axis=1).T + rng.random((m, d))) / m
    return [list(lo + ui * (hi - lo)) for ui in u]


def _rusage_now():
    return (resource.getrusage(resource.RUSAGE_CHILDREN), resource.getrusage(resource.RUSAGE_SELF))


#: a pilot this small (the 3-run confirmation) is not judged by a failure RATE
SMALL_PILOT_MAX = 5


def run_pilot(ev, lower, upper, x_default, n: int = 10, seed: int = 0,
              emit_series: bool = True, fail_rate_max: float = 0.30) -> dict:
    """n live evaluations, timed. Returns the records and a summary (see module docstring).

    The default vector runs FIRST and, when `emit_series` is on, with KDT_CALIB_EMIT_SERIES=1 so a
    runner that supports it writes the paired series the tolerances are measured from. Timing is
    wall clock; CPU time over the same interval (this process + its children) gives threads per
    run, which is what decides how many copies of the model fit on the machine.
    """
    import numpy as np
    # Points that break the contract's constraints never run the model (C5), so they are not failed
    # runs and have no timing (Opus round 3). They are left out and replaced by fresh draws, so the
    # pilot still times n runs where the feasible region allows it; how many were rejected is reported.
    _cok, _named = getattr(ev, "constraints_ok", None), getattr(ev, "_named", None)

    def _feasible(x):
        if not (_cok and _named):
            return True
        try:
            return bool(_cok(_named(x)))
        except Exception:
            return True                    # a broken constraint is the evaluator's to report
    _want = max(0, int(n) - 1)
    _pts: list = []
    n_rejected = 0
    for _r in range(20):                   # at most 20 fresh draws of the same size
        if len(_pts) >= _want:
            break
        for x in lhs_points(lower, upper, _want, seed=seed + _r * 7919):
            if len(_pts) >= _want:
                break
            if _feasible(x):
                _pts.append(x)
            else:
                n_rejected += 1
    X = [list(x_default)] + _pts
    recs: list = []
    series_paths: dict = {}
    series_error = None                  # the runner's own reason when it could not save its series
    default_block: dict = {}
    default_metrics: dict = {}
    _prev_live = getattr(ev, "_force_live", False)
    with ev.phase_as("pilot"):
        ev._force_live = True            # a cached replay would measure nothing — time the model
        try:
            for k, x in enumerate(X):
                _emit = emit_series and k == 0
                _prev_emit = os.environ.get("KDT_CALIB_EMIT_SERIES")
                if _emit:
                    os.environ["KDT_CALIB_EMIT_SERIES"] = "1"
                # the series files that exist BEFORE this evaluation are not this run's evidence
                _before = _series_stamps(ev.workdir)
                # the call's start on the FILE SYSTEM's clock (a marker file), so the stale-file check below
                # compares two timestamps of the same clock (Opus 8b/9 r7 nit)
                _t_call0 = None
                try:
                    _mk = Path(ev.workdir) / ".kdt_series_call_start"
                    _mk.touch()
                    _t_call0 = _mk.stat().st_mtime
                except Exception:
                    _t_call0 = None
                r0, s0 = _rusage_now()
                t0 = time.perf_counter()
                try:
                    losses = ev.evaluate(x)
                except Exception:
                    losses = None
                wall = time.perf_counter() - t0
                r1, s1 = _rusage_now()
                if _emit:
                    if _prev_emit is None:                 # restore EXACTLY what was there
                        os.environ.pop("KDT_CALIB_EMIT_SERIES", None)
                    else:
                        os.environ["KDT_CALIB_EMIT_SERIES"] = _prev_emit
                    series_paths, series_error = _series_from_run(ev, before=_before, with_error=True,
                                                                  started=_t_call0)
                    # KEEP the default run's own series now: the runner may overwrite the same file on every
                    # later call, and the tolerances and the 2n check must read THIS run's (Opus 8b/9 r1 #3)
                    series_paths = _snapshot_series(ev.workdir, series_paths)
                if k == 0 and losses and any(math.isfinite(l) for l in losses):
                    # the default run's kit block (only if THAT run worked: the evaluator keeps the
                    # last good run's metrics, which could be an earlier one), for "mean near zero" of variables that write no
                    # series (design §1.4: decided once, at the pilot's default run)
                    _m = {}
                    for _sp in ("calibration", None):
                        _m = (getattr(ev, "_last_metrics", {}) or {}).get(_sp) or {}
                        if isinstance(_m, dict) and _m:
                            break
                    _b = ((_m.get("__kdt__") or {}).get("panel")) if isinstance(_m, dict) else None
                    try:
                        if isinstance(_b, dict):
                            default_block = json.loads(json.dumps(_b, default=float))
                    except Exception:
                        default_block = {}
                    try:
                        # the protection anchor: numbers only (plain keys, var-scoped dicts of numbers)
                        # plus the kit block — anything else in a payload is not a metric
                        def _nums(d):
                            out = {}
                            for kk, vv in (d or {}).items():
                                if isinstance(vv, bool):
                                    continue
                                if isinstance(vv, (int, float)):
                                    out[kk] = float(vv)
                                elif isinstance(vv, dict) and kk != "__kdt__":
                                    sub = _nums(vv)
                                    if sub:
                                        out[kk] = sub
                            return out
                        default_metrics = _nums(_m) if isinstance(_m, dict) else {}
                        if default_block:
                            default_metrics["__kdt__"] = {"panel": default_block}
                    except Exception:
                        default_metrics = {}
                cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
                       + (s1.ru_utime - s0.ru_utime) + (s1.ru_stime - s0.ru_stime))
                recs.append({"i": k, "x": list(x), "wall_s": wall, "cpu_s": cpu,
                             # Linux ru_maxrss is KB and a HIGH-WATER MARK, so it is the largest
                             # footprint seen so far, not this run's alone — deliberately
                             # conservative for the memory-based lane limit.
                             "peak_rss_mb": max(r1.ru_maxrss, s1.ru_maxrss) / 1024.0,
                             "ok": bool(losses) and all(math.isfinite(l) for l in losses),
                             # finite objective losses of this run (triage no_baseline reads the default's)
                             "n_finite": sum(1 for l in (losses or []) if math.isfinite(l)),
                             "default": k == 0})
        finally:
            ev._force_live = _prev_live
    # a default that breaks the constraints never ran either: not a timing, not a failed run
    _dflt_rejected = bool(recs) and not _feasible(X[0])
    if _dflt_rejected:
        recs[0]["rejected_by_constraints"] = True
    _ran = [r for r in recs if not r.get("rejected_by_constraints")]
    walls = np.array([r["wall_s"] for r in _ran], float) if _ran else np.array([0.0])
    oks = [bool(r["ok"]) for r in _ran] or [True]      # nothing ran: nothing failed (Opus round 5)
    fail_rate = float(1.0 - np.mean(oks))
    out = {"records": recs, "n": len(recs),
           "median_s": float(np.median(walls)), "p90_s": float(np.percentile(walls, 90)),
           "min_s": float(walls.min()), "max_s": float(walls.max()),
           "fail_rate": fail_rate,
           "threads_per_run": float(np.median([r["cpu_s"] / max(r["wall_s"], 1e-9) for r in _ran]))
                              if _ran else 1.0,
           "peak_rss_mb": float(max((r["peak_rss_mb"] for r in recs), default=0.0)),
           "series": series_paths,
           **({"series_error": series_error} if series_error else {}),
           "default_panel_block": default_block,
           "default_metrics": default_metrics,
           "default_ok": bool(recs and recs[0]["ok"]),
           "default_n_finite": (int(recs[0].get("n_finite", 0)) if recs else 0),
           "n_rejected_by_constraints": n_rejected,
           "status": "ok"}
    if len(_pts) < _want:
        out["warning"] = (f"only {len(_pts)} of {_want} non-default pilot points satisfy the contract's "
                          f"constraints ({n_rejected} drawn points were rejected without a run)")
    # more than 30 % of a full (10-run) pilot failing is not something a longer search fixes — it is a
    # wrong range or a broken runner. A SMALL pilot (the 3-run confirmation when the user gave a run
    # time) is too small for a rate: one failed point is 33 %. There, only "every non-default run failed"
    # stops the run; other failures are a warning. A failed DEFAULT run is the triage's to judge
    # (no_baseline when it gave no finite objective metric; build step 8).
    if _dflt_rejected:
        # a contract error the kit sees without a model run — the same reason at every pilot size
        out["status"] = "pilot_unstable"
        out["reason"] = ("the default parameters break the contract's constraints — fix the defaults or the "
                         "constraints before searching")
    elif len(recs) >= SMALL_PILOT_MAX + 1:
        if fail_rate > fail_rate_max:
            out["status"] = "pilot_unstable"
            out["reason"] = (f"{fail_rate:.0%} of {len(recs)} pilot evaluations failed (> "
                             f"{fail_rate_max:.0%}) — check parameter ranges and the runner before searching")
    # (a failed DEFAULT run is no longer a pilot stop: the triage decides — no finite objective metric is
    # `no_baseline`, a partly finite one is still a baseline; build step 8, Opus 8a r1)
    elif len(recs) >= 3 and not any(r["ok"] for r in recs[1:]):
        out["status"] = "pilot_unstable"
        out["reason"] = (f"every non-default pilot run failed ({len(recs) - 1} of {len(recs) - 1}) — only "
                         f"the defaults run; check the parameter ranges before searching")
    elif fail_rate > 0:
        _fw = (f"{sum(1 for r in recs if not r['ok'])} of {len(recs)} pilot runs failed (a small "
               f"pilot is not judged by its failure rate"
               + ("; the default run worked)" if recs and recs[0]["ok"] else
                  "; the DEFAULT run failed — the triage decides whether there is a baseline)"))
        out["warning"] = f"{out['warning']}; {_fw}" if out.get("warning") else _fw
    return out


def _series_stamps(workdir) -> dict:
    """{path: mtime} of the series files present before an evaluation."""
    out = {}
    try:
        for p in Path(workdir).glob("kdt_series_*.npz"):
            out[str(p)] = p.stat().st_mtime
    except Exception:
        pass
    return out


def _snapshot_series(workdir, paths: dict) -> dict:
    """Copy each series file of the default run to <workdir>/kdt_pilot_default_series/<var>.npz and return
    the copies' paths (a file that cannot be copied is left out: its variable has no series)."""
    import shutil
    out: dict = {}
    d = Path(workdir) / "kdt_pilot_default_series"
    for var, p in (paths or {}).items():
        try:
            d.mkdir(exist_ok=True)
            dst = d / f"{''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(var))}.npz"
            shutil.copy2(str(p), dst)
            out[var] = str(dst)
        except Exception:
            continue
    return out


def _series_from_run(ev, before: dict | None = None, with_error: bool = False, started: float | None = None):
    """Paths of the paired series THIS evaluation wrote: whatever the runner declared in
    __kdt__.series, plus — for targets it did not declare — the kdt_series_<var>.npz files that appeared or changed
    during it (Opus 8b/9 r5 #4: a two-target runner that declared only one). When the runner says it could not save
    its series (`__kdt__.series_error`), no file is searched for (a half-written one could be picked up) and the
    reason is returned with `with_error=True` as (paths, reason).

    A file that was already there and untouched is NOT evidence from this run (codex review
    2026-09-27): it could be another candidate's or another split's series, and measuring
    tolerances from it would report them as if they came from this model run.
    """
    out: dict = {}
    err = None
    try:
        for split in ("calibration", None):
            m = (getattr(ev, "_last_metrics", {}) or {}).get(split) or {}
            kdt = (m.get("__kdt__") or {}) if isinstance(m, dict) else {}
            if kdt.get("series_error"):
                err = str(kdt.get("series_error"))[:200]
            decl = (kdt.get("series") or {})
            if isinstance(decl, dict):
                for var, p in decl.items():
                    if p and Path(str(p)).exists():
                        out[str(var)] = str(p)
            if out or err:
                break
    except Exception:
        pass
    if not err:
        try:
            for p in sorted(Path(ev.workdir).glob("kdt_series_*.npz")):
                v = p.stem.replace("kdt_series_", "")
                if v in out:
                    continue                   # declared: the declared file wins
                if before is not None:
                    was = before.get(str(p))
                    if was is not None and p.stat().st_mtime <= was:
                        continue               # pre-existing and untouched -> not this run's
                if started is not None and p.stat().st_mtime < started - 1.0:
                    continue                   # copied in with an older timestamp (copy2, rsync -a) -> not this run's
                out[v] = str(p)
        except Exception:
            pass
    return (out, err) if with_error else out


def pilot_series(pilot: dict, variables, why: dict | None = None) -> dict:
    """{var: (sim, obs, dates)} read from the series the pilot's DEFAULT run wrote; variables
    without a readable series are left out (they get the labelled fixed fallback). When `why` is
    a dict, the reason per left-out variable is written into it."""
    import numpy as np
    from .panel import load_series, _finite_pairs
    out: dict = {}
    for var in [str(v) for v in (variables or [])]:
        p = ((pilot or {}).get("series") or {}).get(var)
        if not p:
            if why is not None:
                _se = (pilot or {}).get("series_error")
                why[var] = f"no series (the runner could not save it: {_se})" if _se else "no series"
            continue
        try:
            sim, obs, dates = load_series(p)
            if sim.ndim != 1 or obs.ndim != 1 or (dates is not None and dates.ndim != 1):
                raise ValueError(f"series must be 1-D: sim {sim.shape}, obs {obs.shape}")
            if len(sim) != len(obs) or (dates is not None and len(dates) != len(sim)):
                raise ValueError(f"lengths differ: sim {len(sim)}, obs {len(obs)}, "
                                 f"date {None if dates is None else len(dates)}")
            if dates is not None and dates.dtype.kind in ("U", "S"):
                # ISO dates only (the order the bootstrap needs); a "05/01/1981" date would sort by day of month
                # and shrink the tolerances silently (Opus 8b/9 r5 #3)
                try:
                    dates = np.asarray(dates, dtype="datetime64[m]")
                except (ValueError, TypeError):
                    raise ValueError("dates are not ISO (use YYYY-MM-DD or YYYY-MM-DDTHH:MM)")
            _finite_pairs(sim, obs, dates)          # the same checks the panel will make
            out[var] = (sim, obs, dates)
        except Exception as e:
            if why is not None:
                why[var] = f"series unreadable ({type(e).__name__}: {str(e)[:160]})"
            continue
    return out


def pilot_tolerances(pilot: dict, variables, kinds: dict | None = None, seed: int = 0,
                     series: dict | None = None, why: dict | None = None) -> tuple[dict, str]:
    """Tolerance records per variable (design §2.11): bootstrap where the pilot's default run wrote
    a series, labelled fixed fallback otherwise. The bootstrap generator seed is FIXED (0), not the
    search seed; `seed` is kept for older callers and ignored. `series` (already read) and `why`
    (reasons for variables without one) may be passed in.
    Returns ({var: record}, overall source label)."""
    from .panel import tolerances_for
    why = {} if why is None else why
    ser = series if series is not None else pilot_series(pilot, variables, why)
    recs = tolerances_for([str(v) for v in (variables or [])], kinds or {}, ser, no_series_reason=why)
    srcs = {s for r in recs.values() for s in (r.get("source") or {}).values()}
    label = ("bootstrap" if srcs == {"bootstrap"} else "fixed_fallback" if srcs == {"fixed_fallback"}
             else "bootstrap+fixed_fallback" if srcs else "none")
    return recs, label
