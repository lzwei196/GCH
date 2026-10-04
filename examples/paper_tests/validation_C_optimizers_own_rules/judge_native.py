"""Judge for the built-in-rule validation (PREREGISTRATION_C.md). Mechanical; no choices made here.
usage: python judge_native.py <variant>     variants: spotpy_loss, spotpy_scores, pymoo_p50, pymoo_p5
-> results/<variant>.json (+ prints the summary)"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "validation", "frozen_rule"))
from kdt_rule_frozen import panel as P                                       # noqa: E402
from kdt_rule_frozen.rule import eps_indicator, LOSS_FLOOR                    # noqa: E402
from kdt_rule_frozen.rule_steps import StepRule                              # noqa: E402

MIN_AFTER, NEED, FIRST_SEED, LAST_SEED = 10, 59, 3000, 3199
KSTOP, PCENTO, PEPS = 10, 0.1, 0.001            # classic SCE-UA settings (Duan et al.)
TOL = {"Q": {m: 0.01 for m in ("r", "alpha", "beta", "lnnse")}}
VARIANTS = {"spotpy_loss": "sce", "spotpy_scores": "sce", "pymoo_p50": "nsga", "pymoo_p5": "nsga"}


def cp_upper(x, n, conf=0.95):
    """One-sided exact (Clopper-Pearson) upper bound, bisection (as validation/judge.py)."""
    if n == 0 or x >= n:
        return 1.0
    a, lo, hi = 1.0 - conf, 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = sum(math.comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(x + 1))
        lo, hi = (mid, hi) if cdf > a else (lo, mid)
    return 0.5 * (lo + hi)


def _pct_change(series, nloop):
    """SPOTPY sceua.py's objective test, verbatim: |c[n-1] - c[n-kstop]| * 100 / mean(|c[n-kstop:n]|)."""
    c = series
    absolute_change = abs(c[nloop - 1] - c[nloop - KSTOP]) * 100
    denominator = float(np.mean(np.abs(c[(nloop - KSTOP):nloop])))
    return 0.0 if denominator == 0.0 else absolute_change / denominator


def states(path, two):
    rows = [json.loads(l) for l in open(os.path.join(path, "calls.jsonl"))]
    rows = [r for r in rows if r.get("kind") == "problem" and r.get("phase") == "search"]
    pn = P.Panel(["Q"], kinds={"Q": "flow"})
    objs = [("Q:nse", "Q"), ("Q:lnnse", "Q")] if two else [("Q:nse", "Q")]
    rule = StepRule(objs, pn, TOL, trade_off=two)
    for r in rows:
        rule.add(r.get("losses"), None, (r.get("extra") or {}).get("panel"), step=int(r.get("gen") if two else r.get("loop")))
    rule.finish(last_step_complete=False)            # the last step is never counted
    return rule, objs, pn, len(rows)


def native_stop(path, variant, rule):
    """(stop step, index into rule.ends) where the built-in rule first fires, or (None, None)."""
    if VARIANTS[variant] == "sce":
        loops = json.load(open(os.path.join(path, "loops.json")))
        if [d["loop"] for d in loops] != list(range(1, len(loops) + 1)):
            raise RuntimeError(f"{path}: loop numbers are not 1..n as SPOTPY's criter indexing assumes")
        crit = [d["bestf"] for d in loops]
        end_of = {e["step"]: j for j, e in enumerate(rule.ends)}
        series = {m: [] for m in ("r", "alpha", "beta", "lnnse")}
        for d in loops:
            j = end_of.get(d["loop"])
            X = (rule.ends[j]["X"].get("Q") or {}) if j is not None else {}
            for m in series:
                series[m].append(X.get(m) if X.get(m) is not None else float("nan"))
        for i, d in enumerate(loops):
            n = d["loop"]
            if n not in end_of:                         # the last (uncounted) loop, or past the record
                continue
            if d["gnrng"] < PEPS:
                return n, end_of[n]
            if n >= KSTOP:
                ok = _pct_change(crit[:i + 1], n) <= PCENTO
                if variant == "spotpy_scores":
                    ok = ok and all(math.isfinite(_pct_change(series[m][:i + 1], n)) and
                                    _pct_change(series[m][:i + 1], n) <= PCENTO for m in series)
                if ok:
                    return n, end_of[n]
        return None, None
    gens = json.load(open(os.path.join(path, "gens.json")))
    key = "p50" if variant == "pymoo_p50" else "p5"
    by_calls = {e["calls"]: j for j, e in enumerate(rule.ends)}
    for g in gens:
        # convergence = pymoo's own design-space (x) or objective-space (f) test passing (perc >= 1); its max-gen /
        # max-eval caps inside DefaultTermination are budget limits, never convergence (tracer seed 998: "p50" hit
        # perc 1 only at n_gen 1000 = the max-gen cap)
        if max(g[key]["x"], g[key]["f"]) >= 1.0:
            if g["n_eval"] not in by_calls:
                if g["n_eval"] > rule.ends[-1]["calls"]:
                    return None, None                   # fired only in the uncounted last step
                raise RuntimeError(f"{path}: generation {g['n_gen']} ends at call {g['n_eval']}, not a step end")
            return g["n_gen"], by_calls[g["n_eval"]]
    return None, None


def judge_one(path, variant):
    two = VARIANTS[variant] == "nsga"
    rule, objs, pn, ncalls = states(path, two)
    out = {"run": os.path.basename(path), "calls": ncalls, "steps_after_start": len(rule.transitions)}
    stop, k = native_stop(path, variant, rule)
    if stop is None:
        out["judge"] = "no stop"
        return out
    s, later = rule.ends[k], rule.ends[k + 1:]
    out.update(stop_step=stop, stop_calls=s["calls"], ends_after=len(later))
    moved, missing, largest = [], [], {}
    for e in later:                                     # the same premature test as validation/judge.py
        for j in range(len(objs)):
            x, y = s["V"][j], e["V"][j]
            if x is None or y is None:
                missing.append(["loss", objs[j][0], e["step"]]); continue
            g = (x - y) / max(abs(x), LOSS_FLOOR)
            largest[f"loss:{objs[j][0]}"] = max(largest.get(f"loss:{objs[j][0]}", -1e9), g)
            if g >= rule.rel_gain:
                moved.append(["loss", objs[j][0], e["step"], g])
        for m_ in pn.required("Q"):
            x = (s["X"].get("Q") or {}).get(m_); y = (e["X"].get("Q") or {}).get(m_)
            if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
                missing.append(["score", m_, e["step"]]); continue
            d = abs(y - x)
            largest[f"Q:{m_}"] = max(largest.get(f"Q:{m_}", 0.0), d)
            if d > TOL["Q"][m_]:
                moved.append(["score", m_, e["step"], x, y])
        if two:
            ef = eps_indicator([rule.F[i] for i in s["archive"]], [rule.F[i] for i in e["archive"]], rule.scale[1])
            largest["front:eps"] = max(largest.get("front:eps", -1e9), ef)
            if ef > rule.eps_front:
                moved.append(["front", e["step"], ef])
    out.update(largest_after=largest, moved=moved[:20], n_moved=len(moved), missing=missing[:20])
    out["judge"] = ("premature" if moved else
                    "undecidable" if (len(later) < MIN_AFTER or missing) else "safe")
    out["budget_share_saved"] = 1.0 - s["calls"] / ncalls
    return out


def main(variant):
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {list(VARIANTS)}")
    sys.path.insert(0, HERE)
    import deps_native as DM
    frozen = DM.frozen_digest()
    if DM.deps_digest() != frozen:
        raise RuntimeError("run-time dependencies changed since DEPS_MANIFEST.json")
    kind = VARIANTS[variant]
    res = []
    for seed in range(FIRST_SEED, LAST_SEED + 1):
        p = os.path.join(HERE, "runs", f"{kind}_s{seed}")
        if not os.path.exists(p):
            break                                       # judged strictly in seed order
        err = os.path.join(p, "recorder_errors.jsonl")
        if os.path.exists(err) and os.path.getsize(err) > 0:
            res.append({"seed": seed, "judge": "invalid", "why": "recorder errors"}); continue
        dj = os.path.join(p, "done.json")
        if not os.path.exists(dj) or json.load(open(dj)).get("deps_digest") != frozen:
            res.append({"seed": seed, "judge": "invalid", "why": "did not finish or other dependencies"}); continue
        res.append(dict(judge_one(p, variant), seed=seed))
        if sum(r["judge"] in ("safe", "premature") for r in res) >= NEED:
            break
    cnt = {v: sum(r["judge"] == v for r in res) for v in ("premature", "safe", "undecidable", "no stop", "invalid")}
    n_dec = cnt["premature"] + cnt["safe"]
    stopped = [r for r in res if r["judge"] in ("safe", "premature")]
    summ = {"variant": variant, "seeds_judged": len(res), "counts": cnt, "decidable": n_dec,
            "cp95_upper": cp_upper(cnt["premature"], n_dec),
            "result": ("PASS" if (n_dec >= NEED and cnt["premature"] == 0) else
                       "FAIL" if cnt["premature"] > 0 else "INCONCLUSIVE (so far)"),
            "median_stop_step": float(np.median([r["stop_step"] for r in stopped])) if stopped else None,
            "median_budget_share_saved": float(np.median([r["budget_share_saved"] for r in stopped])) if stopped else None,
            "premature_cases": [r for r in res if r["judge"] == "premature"],
            "invalid_runs": [r for r in res if r["judge"] == "invalid"], "deps_digest": frozen}
    json.dump({"summary": summ, "searches": res}, open(os.path.join(HERE, "results", f"{variant}.json"), "w"),
              indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("premature_cases", "invalid_runs")}, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
