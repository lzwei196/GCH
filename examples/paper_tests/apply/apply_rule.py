"""Part D tool: apply the FROZEN, validated step-end rule (SCE-UA, NSGA-II) and the descriptive settle point (every
search; the only result for DDS / random / LLM arms) to one recorded search. Not to be run on a paper search before
the validation (part B) has passed for the optimizer and tolerance version used.

A search is given to it as a standard list of calls, in order: {"losses": [...] | None, "scalar": float | None,
"panel": {var: {metric: value}}, "step": int | None}. Adapters per record format build that list (adapters.py).
"""
from __future__ import annotations
import sys
import hashlib, importlib.util, os
_VAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation", "frozen_rule")
sys.path.insert(0, _VAL)                                     # the SAME snapshot the validation ran on
from kdt_rule_frozen import panel as P                       # noqa: E402
from kdt_rule_frozen.rule import _fin                        # noqa: E402
from kdt_rule_frozen.rule_steps import StepRule              # noqa: E402
_SETTLE = "/home/server/kdt_convergence_dev/calibration_kit/settle.py"
_SETTLE_SHA = "75e6b21a7a7791e2fb4847809c93a4d4d7c06e027ff5fd89ae5b7706e39355a4"                         # settle.py as reviewed (codex A4)
if hashlib.sha256(open(_SETTLE, "rb").read()).hexdigest() != _SETTLE_SHA:      # not assert: -O must not skip it
    raise RuntimeError("settle.py changed since its review")
_src = open(_SETTLE).read().replace("from .rule import", "from kdt_rule_frozen.rule import")
_m = type(sys)("kdt_settle_frozen"); exec(compile(_src, _SETTLE, "exec"), _m.__dict__)
SettlePoint = _m.SettlePoint

STEP_RULE_OPTIMIZERS = ("sceua", "nsga2")


def apply(calls, optimizer: str, objectives, kinds: dict, tol: dict, trade_off: bool = False,
          last_step_complete: bool = False, weights=None) -> dict:
    pn = P.Panel(list(kinds), kinds=dict(kinds))
    out = {"optimizer": optimizer, "calls": len(calls), "tolerance": tol}
    if optimizer in STEP_RULE_OPTIMIZERS:
        rule = StepRule(objectives, pn, tol, trade_off=trade_off, weights=weights)
        for c in calls:
            if c.get("step") is None:
                raise ValueError(f"{optimizer}: a call has no step tag")
            rule.add(c.get("losses"), c.get("scalar"), c.get("panel"), step=c["step"])   # raw: StepRule checks it (codex A4 r4)
        s = rule.finish(last_step_complete=last_step_complete)
        ca = s["converged_at"]
        out["rule"] = s
        out["wording"] = (f"converged at {'loop' if optimizer == 'sceua' else 'generation'} {ca['to_step']} "
                          f"(run {ca['calls']} of {len(calls)}) by the pre-specified, validated rule" if ca else
                          ("too short to judge" if s["verdict"] == "too short to judge" else
                           "not converged within the budget"))
    else:
        out["rule"] = None
        out["wording"] = None
    if not trade_off:
        sp = SettlePoint(pn, tol)
        for c in calls:
            sc = c.get("scalar")
            ls = c.get("losses")
            if ls is not None and len(ls) != len(objectives):
                raise ValueError(f"call: {len(ls)} losses for {len(objectives)} objectives")   # as StepRule (codex A4 r3)
            fs = [_fin(x) for x in ls] if ls is not None else None     # the kit's own check (refuses bool, nan, inf)
            if fs is not None and not all(x is not None for x in fs):
                sc = None                    # a partial loss vector is never a best, as in StepRule (codex A4 r2)
            elif sc is None and fs is not None:
                # the same weighted mean StepRule builds (codex A4 r1): every component is finite here
                w = [float(x) for x in weights] if weights is not None else [1.0] * len(fs)
                sc = sum(wk * x for wk, x in zip(w, fs)) / (sum(w) or 1.0)
            sp.add(sc, c.get("panel"))
        out["settle"] = sp.summary()
        if out["wording"] is None:
            out["wording"] = out["settle"]["wording"]
    return out
