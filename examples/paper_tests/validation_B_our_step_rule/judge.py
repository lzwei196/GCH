"""The validation judge (PREREGISTRATION_B.md sections 2, 4, 5, 6). Mechanical; no choices are made here.
usage: python judge.py <sce|nsga> <boot|fixed>   -> results/<kind>_<tol>.json (+ prints the summary)"""
import glob, importlib.util, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frozen_rule"))       # the snapshot only, never the live kit (codex B2 r4 #1)
from kdt_rule_frozen import panel as P                                       # noqa: E402
from kdt_rule_frozen.rule import eps_indicator, LOSS_FLOOR                    # noqa: E402
from kdt_rule_frozen.rule_steps import StepRule                              # noqa: E402

RULE_SHA = "e3b1f5ef9ef4904f9cc3d2a9d2f849fccff1e91885e4a5f4b77a392350f3d7a6"
FROZEN_SHA = {"rule_steps.py": RULE_SHA, "rule.py": "5d5c844e924a1fe2b575f5a081b583f78b79f112f3f23bbf0506f71cf98e8079", "panel.py": "032b1c78f3ff414cb17fbf1c86847e0fa312b51519b3160d80a4bb50c4e598fb"}
MIN_AFTER = 10            # counted step ends after the stop needed to decide
NEED = 59                 # stopped-and-decidable searches per optimizer x tolerance
LAST_SEED = 1199
MIDPOINT = [(1.0 + 500.0) / 2, (0.1 + 2.0) / 2, (0.1 + 0.99) / 2, (0.001 + 0.10) / 2, (0.1 + 0.99) / 2]


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tolerances(version):
    pn = P.Panel(["Q"], kinds={"Q": "flow"})
    req = pn.required("Q")
    if version == "fixed":
        return {"Q": {m: 0.01 for m in req}}, {m: "fixed" for m in req}
    sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/models/spotpy/source/repo/src")
    from spotpy.examples.spot_setup_hymod_python import spot_setup
    s = spot_setup()
    sim = np.array(s.simulation(MIDPOINT), float); obs = np.array(s.evaluation(), float)
    m = np.isfinite(sim) & np.isfinite(obs)
    b = P.bootstrap_tolerances(sim[m], obs[m], kind="flow")
    return {"Q": {k: b["tol"][k] for k in req}}, {k: b["source"][k] for k in req}


def cp_upper(x, n, conf=0.95):
    """One-sided exact (Clopper-Pearson) upper bound: the p with P(X <= x | p) = 1 - conf. Bisection, no scipy
    (codex B2 r4 #2)."""
    if n == 0:
        return 1.0
    if x >= n:
        return 1.0
    a, lo, hi = 1.0 - conf, 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = sum(math.comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(x + 1))
        if cdf > a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def judge_one(path, kind, tol):
    rows = [json.loads(l) for l in open(os.path.join(path, "calls.jsonl"))]
    rows = [r for r in rows if r.get("kind") == "problem" and r.get("phase") == "search"]
    two = kind == "nsga"
    pn = P.Panel(["Q"], kinds={"Q": "flow"})
    objs = [("Q:nse", "Q"), ("Q:lnnse", "Q")] if two else [("Q:nse", "Q")]
    rule = StepRule(objs, pn, tol, trade_off=two)
    for r in rows:
        step = r.get("gen") if two else r.get("loop")
        rule.add(r.get("losses"), None, ((r.get("extra") or {}).get("panel")), step=int(step))
    s = rule.finish(last_step_complete=False)
    out = {"run": os.path.basename(path), "calls": len(rows), "steps_after_start": s["steps_after_start"],
           "verdict_rule": s["verdict"]}
    c = s["converged_at"]
    if c is None:
        out["judge"] = "no stop"
        return out
    k = next(j for j, e in enumerate(rule.ends) if e["step"] == c["to_step"])
    stop, later = rule.ends[k], rule.ends[k + 1:]
    out.update(stop_step=c["to_step"], stop_calls=c["calls"], stop_m=c["m"], ends_after=len(later))
    moved, missing, largest = [], [], {}
    for e in later:
        for j in range(len(objs)):
            x, y = stop["V"][j], e["V"][j]
            if x is None or y is None:
                missing.append(["loss", objs[j][0], e["step"]]); continue
            g = (x - y) / max(abs(x), LOSS_FLOOR)
            largest[f"loss:{objs[j][0]}"] = max(largest.get(f"loss:{objs[j][0]}", -1e9), g)
            if g >= rule.rel_gain:
                moved.append(["loss", objs[j][0], e["step"], g])
        for m_ in pn.required("Q"):
            x = (stop["X"].get("Q") or {}).get(m_); y = (e["X"].get("Q") or {}).get(m_)
            t = tol["Q"][m_]
            if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
                missing.append(["score", m_, e["step"]]); continue
            d = abs(y - x)
            largest[f"Q:{m_}"] = max(largest.get(f"Q:{m_}", 0.0), d)
            if d > t:
                moved.append(["score", m_, e["step"], x, y, t])
        if two:
            ef = eps_indicator([rule.F[i] for i in stop["archive"]], [rule.F[i] for i in e["archive"]], rule.scale[1])
            largest["front:eps"] = max(largest.get("front:eps", -1e9), ef)
            if ef > rule.eps_front:
                moved.append(["front", e["step"], ef])
    out.update(largest_after=largest, moved=moved[:20], n_moved=len(moved), missing=missing[:20])
    if moved:
        out["judge"] = "premature"
    elif len(later) < MIN_AFTER or missing:
        out["judge"] = "undecidable"
    else:
        out["judge"] = "safe"
    out["budget_share_saved"] = 1.0 - c["calls"] / len(rows)
    return out


def main(kind, version):
    for f, h in FROZEN_SHA.items():
        if _sha(os.path.join(HERE, "frozen_rule", "kdt_rule_frozen", f)) != h:
            raise RuntimeError(f + " changed")
    sys.path.insert(0, HERE)
    import deps_manifest as DM
    frozen = DM.frozen_digest()
    if DM.deps_digest() != frozen:
        raise RuntimeError("run-time dependencies changed since DEPS_MANIFEST.json")
    tol, src = tolerances(version)
    _bad = DM.check_loaded(sys.modules)
    if _bad:
        raise RuntimeError(f"modules loaded from outside the hashed roots: {_bad[:5]}")
    res = []
    for seed in range(1000, LAST_SEED + 1):
        p = os.path.join(HERE, "runs", f"{kind}_s{seed}")
        if not os.path.exists(p):
            break                                         # seeds are judged strictly in order
        err = os.path.join(p, "recorder_errors.jsonl")
        if os.path.exists(err) and os.path.getsize(err) > 0:
            res.append({"seed": seed, "run": os.path.basename(p), "judge": "invalid", "why": "recorder errors"}); continue
        if not os.path.exists(os.path.join(p, "done.json")):
            res.append({"seed": seed, "run": os.path.basename(p), "judge": "invalid", "why": "did not finish"}); continue
        if json.load(open(os.path.join(p, "done.json"))).get("deps_digest") != frozen:
            res.append({"seed": seed, "run": os.path.basename(p), "judge": "invalid", "why": "other dependencies"}); continue
        res.append(dict(judge_one(p, kind, tol), seed=seed))
        if sum(r["judge"] in ("safe", "premature") for r in res) >= NEED:
            break
    cnt = {v: sum(r["judge"] == v for r in res) for v in ("premature", "safe", "undecidable", "no stop", "invalid")}
    n_dec = cnt["premature"] + cnt["safe"]
    stopped = [r for r in res if r["judge"] in ("safe", "premature")]
    summ = {"kind": kind, "tolerance": version, "tol": tol, "tol_source": src, "seeds_judged": len(res),
            "counts": cnt, "decidable": n_dec, "cp95_upper": cp_upper(cnt["premature"], n_dec),
            "result": ("PASS" if (n_dec >= NEED and cnt["premature"] == 0) else
                       "FAIL" if cnt["premature"] > 0 and n_dec >= 1 else "INCONCLUSIVE (so far)"),
            "median_stop_step": float(np.median([r["stop_step"] for r in stopped])) if stopped else None,
            "median_budget_share_saved": float(np.median([r["budget_share_saved"] for r in stopped])) if stopped else None,
            "premature_cases": [r for r in res if r["judge"] == "premature"],
            "invalid_runs": [r for r in res if r["judge"] == "invalid"], "frozen_sha256": FROZEN_SHA}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"summary": summ, "searches": res}, open(os.path.join(HERE, "results", f"{kind}_{version}.json"), "w"),
              indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "premature_cases"}, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
