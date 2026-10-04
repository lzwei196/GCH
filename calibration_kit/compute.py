"""Compute probe — what this machine can actually run at once (handoff §5.4).

The budget is a wall-clock allowance divided by a measured run time, so it has to know how many
copies of the model fit side by side. Two limits decide that, and both are measured, not assumed:

  * threads per run (from the pilot's CPU/wall ratio) against the cores that are actually free —
    a model that already uses 8 threads does not get 8 lanes on 16 free cores;
  * peak memory per run against MemAvailable, with a 20% margin.

Then an optional concurrency test checks the answer: run k copies at once and measure the slowdown.
Efficiency e(k) = t(1)/median(t(k)); lanes = the k that maximizes k·e(k) while e(k) stays at or
above 0.7. Below that, extra lanes buy wall-clock that the shared disk or memory bandwidth takes
straight back, and on this machine an over-parallel read off /mnt/disk3 wedges the filesystem.

Parallel copies need separate PROCESSES and separate workdirs: the kit passes candidates through
the process-global env (KDT_CALIB_PARAMS, KDT_CALIB_SPLIT), so two evaluations in one process
would read each other's parameters.
"""
from __future__ import annotations
import math
import os
import shutil
import time
from pathlib import Path

#: below this efficiency an extra lane is wasted work
MIN_EFFICIENCY = 0.70
#: leave a fifth of available memory alone — a run's peak is a high-water mark, not a promise
MEM_MARGIN = 0.80


def _meminfo_bytes(key: str = "MemAvailable") -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return float(line.split()[1]) * 1024.0
    except Exception:
        pass
    return 0.0


def probe_machine() -> dict:
    """Cores this process may use (affinity, not the machine's total), free cores after the
    current load, and available memory. Never raises."""
    try:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    except Exception:
        cores = os.cpu_count() or 1
    try:
        load1 = os.getloadavg()[0]
    except Exception:
        load1 = 0.0
    mem = _meminfo_bytes("MemAvailable")
    return {"cores": int(cores), "load1": float(load1),
            "free_cores": int(max(1, int(cores - load1))),
            "mem_available_gb": round(mem / 1e9, 2)}


def lane_ceiling(machine: dict, pilot: dict) -> tuple[int, dict]:
    """The largest number of concurrent runs the machine can hold: cores/threads-per-run and
    memory/peak-RSS, whichever binds. Returns (k_max, why)."""
    threads = max(1.0, float(pilot.get("threads_per_run") or 1.0))
    rss_mb = max(1.0, float(pilot.get("peak_rss_mb") or 1.0))
    by_cpu = int(max(1, int(machine.get("free_cores", 1)) // max(1, int(round(threads)))))
    mem_mb = float(machine.get("mem_available_gb", 0.0)) * 1000.0
    by_mem = int(max(1, math.floor(MEM_MARGIN * mem_mb / rss_mb))) if mem_mb > 0 else by_cpu
    k = int(max(1, min(by_cpu, by_mem)))
    return k, {"threads_per_run": round(threads, 2), "peak_rss_mb": round(rss_mb, 1),
               "by_cpu": by_cpu, "by_mem": by_mem,
               "binding": ("cpu" if by_cpu <= by_mem else "memory")}


def clone_workdir(src, dst, symlink_large_mb: float | None = None) -> str:
    """A private workdir for one lane. The caches are NOT copied: two lanes sharing an eval cache
    would replay each other's metrics, and the history log must stay one file per run."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True,
                    ignore=shutil.ignore_patterns("eval_metrics_cache_*", "eval_history.jsonl",
                                                  "calib_detached", "kdt_consumption_proof",
                                                  "kdt_params.json"))
    if symlink_large_mb:                      # replace big read-only inputs with symlinks
        limit = float(symlink_large_mb) * 1e6
        for p in dst.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink() and p.stat().st_size > limit:
                    rel = p.relative_to(dst)
                    p.unlink()
                    p.symlink_to((src / rel).resolve())
            except Exception:
                continue
    return str(dst)


def concurrency_test(clone_and_eval, ks=(1, 2, 4), timeout_s: float | None = None) -> dict:
    """Measure e(k) = t(1)/median(t(k)) by running k copies of ONE default evaluation at once.

    `clone_and_eval(i)` must run a single evaluation in its own cloned workdir, in its own process,
    and return its wall time (or raise). Uses threads only to WAIT on those processes. Stops
    climbing as soon as efficiency drops below the floor, so a bad k is paid for once.
    """
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    out: dict = {"efficiency": {}, "median_s": {}, "floor": MIN_EFFICIENCY, "probe_wall_s": 0.0}
    t1 = None
    for k in sorted({int(x) for x in ks if int(x) >= 1}):
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=k) as pool:
                ts = [f.result(timeout=timeout_s) for f in [pool.submit(clone_and_eval, i) for i in range(k)]]
        except Exception as e:
            out.setdefault("errors", {})[k] = str(e)[:200]
            out["probe_wall_s"] = round(out["probe_wall_s"] + (time.perf_counter() - t0), 3)
            break
        ts = [float(t) for t in ts if t and math.isfinite(float(t))]
        if not ts:
            out.setdefault("errors", {})[k] = "no timing returned"
            out["probe_wall_s"] = round(out["probe_wall_s"] + (time.perf_counter() - t0), 3)
            break
        med = float(np.median(ts))
        out["median_s"][k] = med
        out["wall_s_total"] = round(time.perf_counter() - t0, 3)
        out["probe_wall_s"] = round(out["probe_wall_s"] + out["wall_s_total"], 3)   # all of the probe
        if k == 1:
            t1 = med
            out["efficiency"][1] = 1.0
            continue
        e = (t1 / med) if (t1 and med > 0) else 0.0
        out["efficiency"][k] = round(float(e), 3)
        if e < MIN_EFFICIENCY:
            break                              # more lanes only make each one slower
    return out


def choose_lanes(k_max: int, measured: dict | None = None) -> tuple[int, float, list]:
    """(lanes, efficiency, notes). With measurements, take the k ≤ k_max that maximizes k·e(k)
    among those at or above the efficiency floor. Without them, stay at 1 lane and say so —
    an unmeasured parallel speed-up is not evidence."""
    notes: list = []
    eff = (measured or {}).get("efficiency") or {}
    usable = {int(k): float(v) for k, v in eff.items() if int(k) <= int(k_max) and v >= MIN_EFFICIENCY}
    if not usable:
        if eff:
            notes.append(f"no tested concurrency held efficiency >= {MIN_EFFICIENCY} "
                         f"(measured {({int(k): round(float(v), 2) for k, v in eff.items()})}) — 1 lane")
        else:
            notes.append("no concurrency measurement — assuming 1 lane")
        return 1, 1.0, notes
    best = max(usable, key=lambda k: k * usable[k])
    if best < k_max:
        notes.append(f"lanes {best} < machine ceiling {k_max}: efficiency decided, not capacity")
    return int(best), float(usable[best]), notes


def probe_lanes(pilot: dict, *, parallel_safe: bool = True, runner_mode: str = "",
                clone_and_eval=None, ks=(1, 2, 4)) -> dict:
    """The whole §5.4 step: machine → ceiling → optional measurement → lanes.
    A detached runner (it queues its own jobs) or a contract that declares the runner not
    parallel-safe stays at one lane; nothing is measured behind its back."""
    m = probe_machine()
    k_max, why = lane_ceiling(m, pilot)
    out = {"machine": m, "ceiling": k_max, "ceiling_why": why, "lanes": 1, "efficiency": 1.0,
           "measured": None, "notes": []}
    if not parallel_safe or str(runner_mode).strip().lower() == "detached":
        out["notes"].append("strategy.budget.parallel_safe is not true (or the runner is detached) — "
                            "1 lane, no concurrency test")
        return out
    if k_max <= 1:
        out["notes"].append(f"machine ceiling is 1 lane ({why['binding']}-bound)")
        return out
    if clone_and_eval is None:
        out["notes"].append("no cloned-workdir evaluator supplied — 1 lane (ceiling recorded only)")
        return out
    measured = concurrency_test(clone_and_eval, ks=[k for k in ks if k <= k_max])
    lanes, eff, notes = choose_lanes(k_max, measured)
    out.update({"measured": measured, "lanes": lanes, "efficiency": eff})
    out["notes"].extend(notes)
    return out


def make_clone_and_eval(runner_spec: dict, ki_path: str, workdir: str, named_default: dict,
                        base_dir: str | None = None):
    """A `clone_and_eval(i)` for the machine probe (design §1.3, gap 2m): runs ONE default-parameter
    evaluation of a `subprocess` runner in its own cloned workdir and its own process, with its own
    KDT_CALIB_PARAMS file, and returns the wall time. Only for runner-mode contracts that declared
    `strategy.budget.parallel_safe: true` — the caller checks that. Returns None when the runner kind
    cannot be cloned this way."""
    import json as _json
    import subprocess
    import tempfile
    from .runner import _subst
    if (runner_spec or {}).get("kind") != "subprocess":
        return None
    root = Path(base_dir or tempfile.mkdtemp(prefix="kdt_lane_probe_", dir=str(Path(workdir).parent)))

    def clone_and_eval(i):
        lane = root / f"lane_{i}"
        clone_workdir(workdir, lane)
        mapping = {"workdir": str(lane), "ki_path": str(ki_path),
                   "metrics_json": os.path.join(str(lane), "calib_metrics.json")}
        cmd = [_subst(c, mapping) for c in runner_spec["command"]]
        cwd = _subst(runner_spec.get("cwd", ki_path), mapping)
        pfile = lane / "kdt_params.json"
        pfile.write_text(_json.dumps(named_default))
        env = dict(os.environ, KDT_CALIB_PARAMS=str(pfile), KDT_CALIB_SPLIT="calibration")
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"lane {i}: runner exit {r.returncode}")
        return wall

    clone_and_eval.cleanup = lambda: shutil.rmtree(root, ignore_errors=True)
    return clone_and_eval
