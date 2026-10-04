"""VERDICTS (design HANDOFF_CONVERGENCE_2026-09-27_v2.md §2.7, §2.9, §2.10; build step 3).

Reads a finished ConvergenceRule (one seed's search) and says:

  * the SEED verdict (§2.7): converged / not converged / unknown, with how it was reached —
    "rule triggered" (our rule ended the search at its stop point), "confirmed" (it ran past the stop
    point and nothing moved afterwards, §2.10 safe), premature, undecidable, or (no stop point)
    whether missing evidence was the only obstacle. Missing evidence ALONE never makes a result
    "not converged". DREAM: R-hat decides; everything else is a diagnostic beside it.
  * the PER-VARIABLE verdict (§2.7 rules 0-5), which explains the seed verdict but is not combined
    into it — except through the UNKNOWN CAP: a required metric the kit set missing (mean near zero,
    runner/kit mismatch) turns a "converged" into "unknown", for the variable and (not DREAM) the seed.
  * §2.10 SAFETY of the stop point, each metric's largest move after it, and the final-value settle
    point per metric (informational).
  * §2.9 STANDARDS per variable: the KI validation convention's headline metrics against its
    pass_band, read from the runner's own metrics as the KI defines them.
  * aggregation of verdicts (seeds into a run; a variable's per-seed verdicts).
"""
from __future__ import annotations
import math

from .panel import KIND_TABLE
from .rule import LOSS_FLOOR, eps_indicator

CONVERGED, NOT_CONVERGED, UNKNOWN = "converged", "not_converged", "unknown"


def _fin(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def aggregate(verdicts, exceeded: bool = False) -> str:
    """§2.7 verdict aggregation: not converged if any part is (or a compared metric exceeds its
    tolerance); otherwise unknown if anything is unknown or missing; otherwise converged."""
    vs = list(verdicts)
    if exceeded or NOT_CONVERGED in vs:
        return NOT_CONVERGED
    if not vs or any(v not in (CONVERGED, NOT_CONVERGED) for v in vs):
        return UNKNOWN
    return CONVERGED


# ── movement after a reference call (§2.10 bounds) ───────────────────────────────────────────────
class _Suffix:
    """Per tracked series (each objective's v_k, each judged variable's required metrics at the
    incumbent), the max / min of the finite values AFTER each call and whether any value after it is
    missing — computed once, so a bound check from any reference call is O(1) per series (long
    searches have thousands of candidate reference calls)."""

    def __init__(self, rule):
        self.rule = rule
        n = len(rule.F)
        self.loss = {k: self._build([rule.V[j][k] for j in range(n)]) for k in range(rule.K)}
        self.metric = {}
        for v in rule.variables:
            for m in rule.panel.required(v):
                self.metric[(v, m)] = self._build([_fin((rule.X[j].get(v) or {}).get(m)) for j in range(n)])

    @staticmethod
    def _build(vals):
        n = len(vals)
        smax, smin, smiss = [None] * n, [None] * n, [False] * n
        hi = lo = None
        miss = False
        for j in range(n - 1, -1, -1):
            smax[j], smin[j], smiss[j] = hi, lo, miss            # over calls j+1 .. n-1
            x = vals[j]
            if x is None:
                miss = True
            else:
                hi = x if hi is None else max(hi, x)
                lo = x if lo is None else min(lo, x)
        return vals, smax, smin, smiss


def moved_after(rule, ref: int, variables=None, objectives=None, front: bool = False, _sx=None) -> dict:
    """The §2.10 bounds measured from call `ref` to the end, on the given variables' REQUIRED metrics
    and the given objectives' tracked values (and, if `front`, the archive's eps-indicator).

    Returns {"moved": bool, "missing": bool, "missing_what": [...], "calls_after": int,
    "largest": {...}, "moved_what": [...]}. A series with no value AT `ref` cannot be measured from
    `ref` and is reported missing; a value missing at any later call is missing evidence."""
    sx = _sx or _Suffix(rule)
    n = len(rule.F)
    variables = rule.variables if variables is None else list(variables)
    objectives = list(range(rule.K)) if objectives is None else list(objectives)
    largest, moved_what, missing_what = {}, [], []

    def span(series, ref):
        vals, smax, smin, smiss = series
        a = vals[ref]
        if a is None:
            return None, True
        if smax[ref] is None:
            return 0.0, smiss[ref]
        return max(abs(smax[ref] - a), abs(a - smin[ref])), smiss[ref]

    for k in objectives:
        name = f"loss:{rule.objectives[k][0]}"
        d, miss = span(sx.loss[k], ref)
        if miss:
            missing_what.append(name)
        if d is None:
            continue
        a = sx.loss[k][0][ref]
        rel = d / max(abs(a), LOSS_FLOOR)
        largest[name] = rel
        if rel >= rule.rel_gain:
            moved_what.append(name)
    for v in variables:
        for m in rule.panel.required(v):
            name = f"{v}:{m}"
            t = _fin((rule.tol.get(v) or {}).get(m))
            d, miss = span(sx.metric[(v, m)], ref)
            if t is None:
                missing_what.append(f"{name} (no tolerance)")
                continue
            if miss:
                missing_what.append(name)
            if d is None:
                continue
            largest[name] = d
            if d > t:
                moved_what.append(name)
    if front and rule.trade_off and rule.s is not None:
        old = [rule.F[k] for k in rule._archive_at(ref)]
        new = [rule.F[k] for k in rule._arch]
        if old and new:
            e = eps_indicator(old, new, rule.s)
            largest["front:eps"] = e
            if e > rule.eps_front:
                moved_what.append("front")
        else:
            missing_what.append("front (no archive)")
    return {"moved": bool(moved_what), "missing": bool(missing_what), "missing_what": missing_what,
            "calls_after": n - 1 - ref, "largest": largest, "moved_what": moved_what}


def safety(rule, _sx=None) -> dict | None:
    """§2.10, for a search that ran past its stop point s: premature (anything moved beyond a bound
    — decides first), else undecidable (fewer than W calls after s, or a required metric missing
    after s), else safe. None when there is no stop point."""
    s = rule.stop_point
    if s is None:
        return None
    mv = moved_after(rule, s, front=True, _sx=_sx)
    if mv["moved"]:
        verdict = "premature"
    elif mv["calls_after"] < rule.W or mv["missing"]:
        verdict = "undecidable"
    else:
        verdict = "safe"
    return {"verdict": verdict, "stop_point": s, "calls_after": mv["calls_after"],
            "largest_move_after": mv["largest"], "moved": mv["moved_what"],
            "missing_after": mv["missing_what"]}


# ── sole-blocker calls (§2.7 seed rule 3 and per-variable rule 2) ─────────────────────────────────
def _only_missing_blocks(rec, variables, objectives, need_front: bool) -> bool:
    """At this call: every listed objective's loss test held (and the front test, when needed), no
    required metric MOVED, and at least one was missing ('unknown'). 'early' (before W) is not
    missing evidence."""
    if not all(rec["cond1"][k] for k in objectives):
        return False
    if need_front and rec["cond4"] is not True:
        return False
    states = [s for v in variables for s in rec["cond2"].get(v, {}).values()]
    if any(s in ("moved", "early") for s in states):
        return False
    return any(s == "unknown" for s in states)


def sole_blocker_calls(rule, variable: str | None = None) -> list[int]:
    """Calls where a missing required metric was the ONLY thing that kept the rule (seed level) or
    this variable (variable level) from settling."""
    if variable is None:
        vs, ks, front = rule.variables, list(range(rule.K)), rule.trade_off
    else:
        vs, ks, front = [variable], rule.obj_of.get(variable, []), False
    return [rec["i"] for rec in rule.cond if _only_missing_blocks(rec, vs, ks, front)]


def _missing_at(rule, j, variables):
    """The (variable, metric) pairs whose panel test was 'unknown' at call j — the missing evidence
    that blocked it — with "(no tolerance)" when the tolerance, not the value, was missing."""
    out = []
    for v in variables:
        for m, st in (rule.cond[j]["cond2"].get(v) or {}).items():
            if st != "unknown":
                continue
            if _fin((rule.tol.get(v) or {}).get(m)) is None:
                out.append(f"{v}:{m} (no tolerance)")
            else:
                out.append(f"{v}:{m}")
    return out


def _first_quiet_sole_blocker(rule, J, variables, objectives, front, sx):
    """The latest sole-blocker call after which nothing RECORDED moved (metrics missing at j cannot
    be measured from j and are the missing evidence). Latest first: the first hit decides."""
    for j in reversed(J):
        if not moved_after(rule, j, variables=variables, objectives=objectives, front=front, _sx=sx)["moved"]:
            return j
    return None


# ── per-variable verdict (§2.7 rules 0-5 + cap) ───────────────────────────────────────────────────
def _rules_3_to_5(rule, v, p, sx) -> tuple[str, str]:
    mv = moved_after(rule, p, variables=[v], objectives=rule.obj_of.get(v, []), _sx=sx)
    if mv["moved"]:                                                              # rule 3
        return NOT_CONVERGED, f"moved after call {p}: {', '.join(mv['moved_what'])}"
    if mv["missing"]:                                                            # rule 4
        return UNKNOWN, f"missing after call {p}: {', '.join(mv['missing_what'])}"
    if mv["calls_after"] < rule.W:
        return UNKNOWN, f"only {mv['calls_after']} calls after call {p} (W = {rule.W})"
    return CONVERGED, f"nothing moved after call {p}"                           # rule 5


def variable_verdict(rule, v: str, ended_at_stop: bool = False, _sx=None, _J=None) -> dict:
    sx = _sx or _Suffix(rule)
    panel = rule.panel
    out = {"variable": v, "first_settle": rule.first_settle.get(v)}
    if v in rule.rule0:                                                         # rule 0
        out.update(verdict=UNKNOWN, reason="no objective and no panel metric")
        return out
    s = rule.stop_point
    if ended_at_stop and s is not None:                                         # rule 1
        verdict, reason = CONVERGED, "search ended at its stop point (settled there by definition)"
    elif s is not None:                                                         # rule 2: p = s
        verdict, reason = _rules_3_to_5(rule, v, s, sx)
    else:
        fs = rule.first_settle.get(v)
        J = sole_blocker_calls(rule) if _J is None else _J
        # a sole-blocker call with fewer than W calls after it has no vote, whatever it would show
        # (Leo, 2026-09-29; design §2.7 rule 2, rev 28)
        J_vote = [j for j in J if len(rule.F) - 1 - j >= rule.W]
        if fs is not None and J:
            answers, reasons = set(), []
            for p in [fs] + J_vote:
                a, why = _rules_3_to_5(rule, v, p, sx)
                answers.add(a)
                reasons.append(why)
                if len(answers) > 1 or a == UNKNOWN:
                    break
            # every metric that blocked at ANY sole-blocker call (different calls can lack different ones)
            blockers = ", ".join(dict.fromkeys(b for j in J for b in _missing_at(rule, j, rule.variables))) or "a metric"
            if len(answers) == 1 and answers <= {CONVERGED, NOT_CONVERGED}:
                verdict = answers.pop()
                reason = (f"judged from its first settle point {fs} and from every call where {blockers} "
                          f"was the only blocker (and at least W calls followed); all agree"
                          if J_vote else
                          f"judged from its first settle point {fs} alone (every call where {blockers} was the "
                          f"only blocker had fewer than W calls after it)")
            elif not J_vote:
                verdict = UNKNOWN
                reason = (f"judged from its first settle point {fs} alone (every call where {blockers} was "
                          f"the only blocker had fewer than W calls after it): {reasons[-1]}")
            else:
                verdict = UNKNOWN
                reason = (f"the answer depends on where the search would have stopped had {blockers} been "
                          f"recorded ({'; '.join(reasons[-2:])})")
        elif fs is not None:
            verdict, reason = _rules_3_to_5(rule, v, fs, sx)
        else:
            Jv = sole_blocker_calls(rule, v)
            j = _first_quiet_sole_blocker(rule, Jv, [v], rule.obj_of.get(v, []), False, sx)
            if j is not None:
                verdict = UNKNOWN
                reason = f"never settled only because {', '.join(_missing_at(rule, j, [v])) or 'a metric'} was missing"
            else:
                verdict, reason = NOT_CONVERGED, "never settled although its evidence was there"
    # the unknown cap (after rules 0-5)
    if verdict == CONVERGED and panel.capped(v):
        why = "; ".join(f"{v}:{m} missing ({r})" for m, r in panel.kit_missing[v].items()
                        if m in KIND_TABLE[panel.kinds[v]]["required"])
        verdict, reason = UNKNOWN, f"{reason}; capped: {why}"
        out["capped"] = True
        out["cap_reason"] = why
    out.update(verdict=verdict, reason=reason)
    return out


# ── seed verdict (§2.7) ───────────────────────────────────────────────────────────────────────────
TRACKED_BACKENDS = ("dds", "sceua", "dream", "nsga2", "nsga3", "moead")


def not_tracked(reason: str) -> dict:
    """§1.8: backends that do not report each call (surrogate, PEST++, MADR) — no verdict is made."""
    return {"verdict": UNKNOWN, "how": f"convergence not tracked ({reason})", "stop_point": None,
            "safety": None, "variables": {}}


def seed_verdict(rule, ended_at_stop: bool = False, dream: dict | None = None,
                 algorithm: str | None = None) -> dict:
    """ended_at_stop: our rule ended the search (§2.8 reason "our rule"). dream: for a DREAM seed,
    {"rhat_recorded": bool, "rhat_reached": bool} — R-hat then decides (§5.1b). algorithm: a backend
    outside the six tracked ones gets "convergence not tracked" (§1.8)."""
    if algorithm is not None and algorithm not in TRACKED_BACKENDS:
        return not_tracked(f"{algorithm} does not report each call")
    if not rule.F or not any(x is not None for f in rule.F for x in f):
        return not_tracked("the rule saw no call with a finite loss")
    sx = _Suffix(rule)
    J = sole_blocker_calls(rule)
    variables = {v: variable_verdict(rule, v, ended_at_stop, _sx=sx, _J=J)
                 for v in list(rule.variables) + list(rule.rule0)}
    s = rule.stop_point
    saf = safety(rule, _sx=sx) if (s is not None and not ended_at_stop) else None
    if dream is not None:
        if not dream.get("rhat_recorded"):
            verdict, how = UNKNOWN, "DREAM: R-hat not recorded"
        elif dream.get("rhat_reached"):
            verdict, how = CONVERGED, "DREAM: R-hat < 1.2 reached"
        else:
            verdict, how = NOT_CONVERGED, "DREAM: the cap came before R-hat < 1.2"
        return {"verdict": verdict, "how": how, "stop_point": s, "safety": saf, "variables": variables,
                "diagnostic_only": "our rule (DREAM: R-hat decides)"}
    if s is not None and ended_at_stop:                                         # rule 1
        verdict, how = CONVERGED, "rule triggered"
    elif s is not None:                                                         # rule 2
        verdict = {"premature": NOT_CONVERGED, "undecidable": UNKNOWN, "safe": CONVERGED}[saf["verdict"]]
        how = {"premature": "premature",
               "undecidable": ("undecidable (" + (f"missing after the stop point: {', '.join(saf['missing_after'])}; "
                                                  if saf["missing_after"] else "")
                               + f"{saf['calls_after']} calls after it, W = {rule.W})"),
               "safe": "confirmed — nothing moved afterwards"}[saf["verdict"]]
    else:                                                                       # rule 3
        j = _first_quiet_sole_blocker(rule, J, rule.variables, list(range(rule.K)), rule.trade_off, sx)
        if j is not None:
            miss = _missing_at(rule, j, rule.variables)
            verdict, how = UNKNOWN, (f"no stop point only because {', '.join(miss) or 'a metric'} was not "
                                     f"recorded or had no tolerance")
        else:
            verdict, how = NOT_CONVERGED, "no stop point"
    capped = [v for v, r in variables.items() if r.get("capped")]
    if verdict == CONVERGED and capped:                                         # the unknown cap
        verdict = UNKNOWN
        how = f"{how}; capped: " + "; ".join(variables[v]["cap_reason"] for v in capped)
    return {"verdict": verdict, "how": how, "stop_point": s, "safety": saf, "variables": variables}


def final_value_settle_points(rule) -> dict:
    """§2.10, informational: per required metric, one past the last call whose incumbent value
    differs from the final value by more than tol."""
    out = {}
    n = len(rule.X)
    if not n:
        return out
    for v in rule.variables:
        for m in rule.panel.required(v):
            fin = _fin((rule.X[-1].get(v) or {}).get(m))
            t = _fin((rule.tol.get(v) or {}).get(m))
            if fin is None or t is None:
                out[f"{v}:{m}"] = None
                continue
            last = -1
            for j in range(n):
                x = _fin((rule.X[j].get(v) or {}).get(m))
                if x is None or abs(x - fin) > t:
                    last = j
            out[f"{v}:{m}"] = last + 1
    return out


# ── standards (§2.9) ──────────────────────────────────────────────────────────────────────────────
def _entry(convention, var, obs_shape):
    for e in ((convention or {}).get("validation") or []):
        if isinstance(e, dict) and e.get("dag_variable") == var and e.get("obs_shape") == obs_shape:
            return e
    return None


def _metric_value(metrics, var, name, single):
    src = metrics.get(var) if isinstance((metrics or {}).get(var), dict) else (metrics if single else {})
    low = {str(k).lower(): v for k, v in (src or {}).items()}
    return _fin(low.get(str(name).lower()))


def standards(convention, var, obs_shape, metrics: dict | None, single_variable: bool = True,
              band_override: dict | None = None) -> dict:
    """§2.9 for one variable on one call's runner metrics (as the KI defines them).
    fails if any evaluable headline metric misses its pass_band; else unknown if any headline metric
    is unavailable or its band unusable; else meets; 'no standard declared' when there is no entry
    or it lists no headline metric."""
    if obs_shape is None:
        return {"outcome": "no standard declared", "metrics": {},
                "why": "the variable's obs_shape is not known, so no entry can be matched"}
    e = _entry(convention, var, obs_shape)
    heads = [h for h in ((e or {}).get("headline_metrics") or []) if isinstance(h, dict)]
    if not heads:
        return {"outcome": "no standard declared", "metrics": {}}
    per, fails, unknown = {}, False, False
    for h in heads:
        name = str(h.get("metric", ""))
        band_name = ((band_override or {}).get(name.lower()) or h.get("pass_band") or "satisfactory")
        band = _fin((h.get("bands") or {}).get(band_name))
        direction = str(h.get("direction", "")).lower()
        val = _metric_value(metrics or {}, var, name, single_variable)
        rec = {"value": val, "band": band_name, "threshold": band, "direction": direction}
        if val is None:
            rec["result"] = "unavailable"; unknown = True
        elif band is None or direction not in ("maximize", "minimize", "lower_is_better", "zero_centered"):
            rec["result"] = "band unusable"; unknown = True
        else:
            ok = (val >= band if direction == "maximize" else
                  abs(val) <= band if direction == "zero_centered" else val <= band)
            rec["result"] = "meets" if ok else "fails"
            fails = fails or not ok
        per[name] = rec
    return {"outcome": "fails" if fails else ("unknown" if unknown else "meets"), "metrics": per}
