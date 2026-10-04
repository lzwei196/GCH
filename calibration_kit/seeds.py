"""SEEDS — run the same calibration from several starting points and check they agree (design §1.7, §2.7).

One search from one seed tells you what that search found. It cannot tell you whether the answer
is a property of the model and the data or of that particular random start — and for a
population/perturbation optimizer the difference is real (Kavetski, Qin & Kuczera 2018,
doi:10.1029/2017WR022051). So the default is three seeds (slots), and the report carries:

- each slot's result and seed verdict; a slot whose search CRASHED is replaced once by a new seed
  number, and a replacement that also crashes leaves the slot missing;
- AGREEMENT on each variable's required panel, compared on the seeds' final CALIBRATION incumbents
  (never a validation-selected member): max - min across the seeds <= the tolerance from the one pilot.
  A categorical variable (no panel) is "not compared" — kept apart from missing comparisons; if no
  variable has a required panel metric, agreement is unknown ("seeds not compared");
- the RETURNED seed: the completed seed whose incumbent has the lowest calibration loss, whatever the
  agreement verdict (trade-off searches: admissible under protection first, then the smallest worst
  normalized objective on ONE frozen scale, see pick_returned_seed);
- the run verdict and each variable's verdict across seeds (§2.7 aggregation); parameter spread across
  seeds, reported and never a gate. A single-seed run reports "seeds: not checked".

A disagreement is not a failure to hide. It says the result is seed-dependent, which is either
equifinality or too small a budget, and both belong in the record.
"""
from __future__ import annotations
import math
from pathlib import Path

from .compute import clone_workdir
# ONE spelling of the verdicts: verdict.py writes every seed and variable verdict (Opus step 6 r1)
from .verdict import CONVERGED, NOT_CONVERGED, UNKNOWN, aggregate as _agg


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def seed_agreement(panels: list, required: dict, tol: dict, not_compared_why: dict | None = None) -> dict:
    """Do the seeds' final calibration incumbents agree (§2.7)?

    panels:   one entry per SLOT, in slot order: the incumbent's panel {var: {metric: value}}, or None
              for a slot with no completed seed (crashed twice, or no finite incumbent).
    required: {var: [required panel metrics]} — an empty list means the variable has no panel
              (categorical): it is "not compared", which is not the same as missing.
    tol:      {var: {metric: tolerance}} from the one pilot (seed 0's tolerances, used for all seeds).

    Returns {"agree": True | False | None, "reason", "per_variable": {var: {...} | "not compared"},
    "not_compared": [vars], "exceeds": [[var, metric, spread, tol]], "missing": [[var, metric, why]],
    "n_slots", "n_compared"}. False as soon as any compared metric exceeds its tolerance (even with
    something missing: §2.7 aggregation); otherwise None if anything could not be compared."""
    slots = list(panels or [])
    present = [p for p in slots if isinstance(p, dict)]
    out: dict = {"agree": None, "reason": None, "per_variable": {}, "not_compared": [], "exceeds": [],
                 "missing": [], "n_slots": len(slots), "n_compared": len(present)}
    if len(slots) < 2:
        out["reason"] = "seeds: not checked (a single seed)"
        return out
    compared = [v for v, mm in (required or {}).items() if mm]
    out["not_compared"] = [v for v, mm in (required or {}).items() if not mm]
    for v in out["not_compared"]:
        out["per_variable"][v] = "not compared (" + ((not_compared_why or {}).get(v) or "no panel metric") + ")"
    if not compared:
        out["reason"] = "seeds not compared (no variable has a required panel metric)"
        return out
    n_missing_slots = len(slots) - len(present)
    for v in compared:
        pv = {"agree": None, "spread": {}, "exceeds": [], "missing": []}
        for m in required[v]:
            t = _f((tol.get(v) or {}).get(m))
            vals = [_f(((p or {}).get(v) or {}).get(m)) for p in present]
            if t is None:
                pv["missing"].append(m)
                out["missing"].append([v, m, "no tolerance"])
                continue
            if len(present) < 2 or any(x is None for x in vals):
                pv["missing"].append(m)
                out["missing"].append([v, m, "not recorded at every seed's incumbent"
                                       if len(present) >= 2 else "fewer than two completed seeds"])
                ok = [x for x in vals if x is not None]
                if len(ok) >= 2 and (max(ok) - min(ok)) > t:     # what WAS compared still counts
                    pv["spread"][m] = round(max(ok) - min(ok), 9)
                    pv["exceeds"].append(m)
                    out["exceeds"].append([v, m, pv["spread"][m], t])
                continue
            sp = max(vals) - min(vals)
            pv["spread"][m] = round(sp, 9)
            if sp > t:
                pv["exceeds"].append(m)
                out["exceeds"].append([v, m, pv["spread"][m], t])
        pv["agree"] = (False if pv["exceeds"] else
                       None if (pv["missing"] or n_missing_slots) else True)
        out["per_variable"][v] = pv
    if out["exceeds"]:
        out["agree"] = False
        out["reason"] = "; ".join(f"{v}:{m} spread {sp:g} > tol {t:g}" for v, m, sp, t in out["exceeds"])
    elif n_missing_slots:
        out["reason"] = f"{n_missing_slots} slot(s) have no completed seed — agreement is unknown"
    elif out["missing"]:
        out["reason"] = "not compared at every seed: " + ", ".join(f"{v}:{m} ({w})" for v, m, w in out["missing"])
    else:
        out["agree"] = True
        out["reason"] = "every required panel metric agrees within its tolerance"
    return out


def pick_returned_seed(cands: list, trade_off: bool) -> tuple:
    """(index into cands, how) of the RETURNED seed (§1.7), or (None, why).

    cands: one dict per completed slot with a finite incumbent, in slot order:
      {"slot", "scalar" (single-objective loss), "losses" (loss vector), "admissible" (bool),
       "t0", "z", "s" (the slot's frozen trade-off scale, None before t0)}.
    Single objective: the lowest calibration loss (ties: the lowest slot). Trade-off: admissible
    under protection first; then the smallest worst normalized objective, then the smallest sum,
    using the frozen z, s of the lowest-numbered slot that reached t0 for EVERY seed (one scale);
    if no slot reached t0, the smallest sum of raw losses."""
    cs = [c for c in (cands or []) if c]
    if not cs:
        return None, "no completed seed has a finite incumbent"
    if not trade_off:
        ok = [(i, c) for i, c in enumerate(cs) if _f(c.get("scalar")) is not None]
        if not ok:
            return None, "no completed seed has a finite calibration loss"
        i, c = min(ok, key=lambda ic: (_f(ic[1]["scalar"]), ic[1]["slot"]))
        return i, f"lowest calibration loss ({_f(c['scalar']):.6g}, slot {c['slot']})"
    scaled = sorted((c for c in cs if c.get("t0") is not None and c.get("z") and c.get("s")),
                    key=lambda c: c["slot"])
    z, s = (scaled[0]["z"], scaled[0]["s"]) if scaled else (None, None)

    def key(ic):
        i, c = ic
        f = [_f(x) for x in (c.get("losses") or [])]
        if not f or any(x is None for x in f):
            return (2, math.inf, math.inf, c["slot"])
        adm = 0 if c.get("admissible", True) else 1
        if z is not None:
            nf = [(fk - zk) / sk for fk, zk, sk in zip(f, z, s)]
            return (adm, max(nf), sum(nf), c["slot"])
        return (adm, sum(f), sum(f), c["slot"])
    i, c = min(enumerate(cs), key=key)
    if key((i, c))[0] == 2:
        return None, "no completed seed has a fully finite incumbent loss vector"
    _fin_c = [x for x in cs if x.get("losses") and all(_f(v) is not None for v in x["losses"])]
    _adm = [bool(x.get("admissible", True)) for x in _fin_c]
    # when NO seed passes protection the result is never called protected (design §2.4, §5.1b)
    how = ("no seed is admissible under protection (protection infeasible); " if not any(_adm) else
           "admissible under protection, then " if not all(_adm) else "")
    how += (f"smallest worst normalized objective on the frozen scale of slot {scaled[0]['slot']}"
            if z is not None else "smallest sum of raw losses (no slot reached t0)")
    return i, how + f" (slot {c['slot']})"


def param_spread(named_list: list) -> dict:
    """{param: max - min} across the seeds' incumbents (numbers only). Reported, never a gate."""
    out = {}
    ds = [d for d in (named_list or []) if isinstance(d, dict)]
    if len(ds) < 2:
        return out
    for k in ds[0]:
        vals = [_f(d.get(k)) for d in ds]
        if all(v is not None for v in vals):
            out[k] = round(max(vals) - min(vals), 9)
    return out


def across_seeds(slot_verdicts: list, slot_var_verdicts: list, agreement: dict) -> dict:
    """The run verdict and each variable's verdict across seeds (§2.7).

    slot_verdicts: the seed verdict of each slot (None for a missing slot).
    slot_var_verdicts: {var: verdict} per slot (None for a missing slot).
    A single slot: its own verdict, "seeds: not checked". Otherwise not converged if any seed is not
    converged or any compared metric exceeds its tolerance; unknown if anything is unknown, missing or
    not compared; else converged. A variable: converged only if converged in every seed and its seeds
    agree on its own panel; a variable with no panel is at most unknown across seeds."""
    n = len(slot_verdicts)
    if n == 1:
        v0 = slot_verdicts[0] if slot_verdicts[0] is not None else UNKNOWN
        return {"run_verdict": v0, "how": "seeds: not checked (a single seed)",
                "variables": {v: vv for v, vv in (slot_var_verdicts[0] or {}).items()}}
    ag = agreement or {}
    parts = [v if v is not None else "missing" for v in slot_verdicts]
    run = _agg(parts, exceeded=(ag.get("agree") is False))
    if run == CONVERGED and ag.get("agree") is not True:
        run = UNKNOWN
    how = (f"seed verdicts {parts}; agreement: "
           + ("agree" if ag.get("agree") is True else
              "disagree" if ag.get("agree") is False else "unknown")
           + (f" ({ag.get('reason')})" if ag.get("reason") else ""))
    variables = {}
    names = []
    for d in slot_var_verdicts:
        for v in (d or {}):
            if v not in names:
                names.append(v)
    for v in names:
        vs = [((d or {}).get(v) if d is not None else "missing") for d in slot_var_verdicts]
        vs = [x if x is not None else "missing" for x in vs]
        pv = (ag.get("per_variable") or {}).get(v)
        if isinstance(pv, dict):
            vv = _agg(vs, exceeded=(pv.get("agree") is False))
            if vv == CONVERGED and pv.get("agree") is not True:
                vv = UNKNOWN
        else:                        # no panel: not compared -> at most unknown
            vv = _agg(vs)
            if vv == CONVERGED:
                vv = UNKNOWN
        variables[v] = vv
    return {"run_verdict": run, "how": how, "variables": variables}


def seed_lanes(n_seeds: int, lanes: int) -> tuple[int, int]:
    """(seeds at a time, waves) — the same arithmetic the budget plan used to size the cap."""
    n_seeds = max(1, int(n_seeds or 1))
    par = max(1, min(n_seeds, max(1, int(lanes or 1))))
    return par, int(math.ceil(n_seeds / par))


def seed_workdirs(workdir, seeds, parent=None) -> dict:
    """{seed: cloned workdir}. Clones share the inputs and NOT the caches (compute.clone_workdir),
    so two seeds can never replay each other's evaluations."""
    src = Path(workdir)
    base = Path(parent) if parent else src.parent
    out = {}
    for s in seeds:
        out[int(s)] = clone_workdir(src, base / f"{src.name}_seed{int(s)}")
    return out


def run_seeds(run_one, seeds, *, lanes: int = 1, timeout_s: float | None = None) -> dict:
    """Run `run_one(seed, workdir)` for each seed, `seeds_parallel` at a time, in separate
    PROCESSES. `run_one` must be a module-level function (it is pickled) and return the seed's
    report dict. Returns {"reports": {seed: report}, "errors": {seed: message}, ...}.

    Workdirs must already be cloned (seed_workdirs) — this function does not guess where a seed's
    inputs should live.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    seeds = [int(s) for s in seeds]
    par, waves = seed_lanes(len(seeds), lanes)
    out: dict = {"reports": {}, "errors": {}, "seeds": seeds, "seeds_parallel": par, "waves": waves}
    with ProcessPoolExecutor(max_workers=par) as pool:
        futs = {pool.submit(run_one, s): s for s in seeds}
        for f in as_completed(futs, timeout=timeout_s):
            s = futs[f]
            try:
                out["reports"][s] = f.result()
            except Exception as e:
                out["errors"][s] = f"{type(e).__name__}: {e}"
    return out
