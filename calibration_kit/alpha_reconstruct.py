"""Exact reconstruction of alpha (variability ratio) from recorded NSE, KGE, r and PBIAS (design §2.13).

Why: many runners (and every old history) record NSE, KGE, r and PBIAS but not alpha. KGE alone gives
|alpha - 1| but not its sign. Within ONE search the observations are fixed, so c = mean(obs)/sd(obs)
is one constant, and with population moments

    NSE = 2 r alpha - alpha^2 - (beta - 1)^2 c^2            (exact)
    (alpha - 1)^2 = D^2 = (1 - KGE)^2 - (r - 1)^2 - (beta - 1)^2   (KGE, Gupta et al. 2009)

For each call the two roots alpha = 1 +- D give two candidate c^2 = (2 r alpha - alpha^2 - NSE)/(beta-1)^2.
The true c^2 is common to all calls: the value on which >= 80 % of usable calls agree, uniquely. Once
c^2 is fixed, each call's root is the one whose PREDICTED NSE matches the recorded NSE (no division);
the two roots' predictions differ by 4 D |1 - r|. Anything ambiguous -> alpha is missing, never guessed.

Preconditions (the caller states them; unknown -> every alpha missing): NSE and KGE on the same
observation pairs, the same observation mask for every call, KGE in the 2009 form (alpha = sd ratio,
not the 2012 CV form), c with the population sd.

Precision: `decimals` gives the number of recorded decimals per metric (None = full precision). Rounded
records get bounds by propagating half a unit of the last decimal through the formulas (linear,
worst case, numerical partial derivatives).
"""
from __future__ import annotations
import math

AGREE_FRACTION = 0.8          # >= 80 % of usable calls must agree on c^2
FULL_PRECISION_REL = 1e-6     # agreement tolerance on c^2 for full-precision records (relative)
BETA_MIN = 0.01               # |beta - 1| must exceed this for a call to help estimate c^2
BOUND_MAX_REL = 0.01          # a rounded call helps only if its c^2 bound is below 1 % of c^2
NSE_FLOOR = 1e-9              # floor of the NSE-match bound
METRICS = ("nse", "kge", "r", "pbias")


def _f(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _half_units(decimals: dict | None) -> dict:
    """Half a unit of the last recorded decimal per metric; 0 for full precision."""
    h = {}
    for m in METRICS:
        d = (decimals or {}).get(m)
        h[m] = 0.0 if d is None else 0.5 * 10.0 ** (-int(d))
    return h


def _d2(nse, kge, r, pbias):
    beta = 1.0 + pbias / 100.0
    return (1.0 - kge) ** 2 - (r - 1.0) ** 2 - (beta - 1.0) ** 2


def _roots(nse, kge, r, pbias) -> list[float]:
    d2 = _d2(nse, kge, r, pbias)
    if d2 < 0:
        return []
    d = math.sqrt(d2)
    roots = sorted({1.0 + d, 1.0 - d})          # de-duplicated (D = 0 gives one root)
    return [a for a in roots if a >= 0.0]       # negative sd ratio is impossible


def _c2(nse, kge, r, pbias, sign: int):
    """c^2 from the root alpha = 1 + sign*D; None where undefined."""
    beta = 1.0 + pbias / 100.0
    d2 = _d2(nse, kge, r, pbias)
    if d2 < 0 or beta == 1.0:
        return None
    a = 1.0 + sign * math.sqrt(d2)
    if a < 0:
        return None
    return (2.0 * r * a - a * a - nse) / (beta - 1.0) ** 2


def _propagate(fn, x: dict, h: dict) -> float:
    """Linear worst-case bound sum_k |d fn / d x_k| * h_k by central differences. inf when fn is
    undefined near x (e.g. D close to 0, where sqrt's slope blows up)."""
    tot = 0.0
    for k in METRICS:
        if h[k] == 0.0:
            continue
        step = max(1e-7, 1e-7 * abs(x[k]))
        xp = dict(x); xm = dict(x)
        xp[k] += step; xm[k] -= step
        fp, fm = fn(**xp), fn(**xm)
        if fp is None or fm is None:
            return math.inf
        tot += abs((fp - fm) / (2.0 * step)) * h[k]
    return tot


def reconstruct_alpha(records, decimals: dict | None = None, preconditions_ok: bool | None = None,
                      c2_known: float | None = None) -> dict:
    """alpha for every call of ONE variable within ONE search (design §2.13).

    `records`: list of dicts with nse, kge, r, pbias (PBIAS in percent) — one per call, in order.
    `decimals`: recorded decimals per metric (None/absent = full precision).
    `preconditions_ok`: must be True (see module doc); False or None -> every alpha missing.
    `c2_known`: c^2 computed directly from the observations, when the caller has them.

    Returns {"alpha": [float|None per call], "why": [reason|None per call], "c2": float|None,
    "support": fraction|None, "n_usable": int, "delta_c2": float|None, "status": str}.
    Labelled use: alpha values from here are `alpha_reconstructed`."""
    n = len(records or [])
    out = {"alpha": [None] * n, "why": [None] * n, "c2": None, "support": None,
           "n_usable": 0, "delta_c2": None, "status": ""}
    if preconditions_ok is not True:
        out["why"] = ["preconditions not shown (same pairs / same mask / KGE 2009 / population sd)"] * n
        out["status"] = "preconditions_unknown"
        return out
    h = _half_units(decimals)
    full = all(v == 0.0 for v in h.values())
    xs = []
    for rec in records:
        x = {m: _f((rec or {}).get(m)) for m in METRICS}
        xs.append(x if None not in x.values() else None)

    # ── 1. c^2: known, or agreed by >= 80 % of usable calls, uniquely ─────────────────────────
    if c2_known is not None:
        c2 = float(c2_known)
        delta = FULL_PRECISION_REL * abs(c2)     # same tolerance as an agreed full-precision c^2
        out.update(c2=c2, delta_c2=delta, status="c2_known")
    else:
        cands = []                 # (call, c2, bound)
        usable = set()
        for i, x in enumerate(xs):
            if x is None:
                continue
            beta = 1.0 + x["pbias"] / 100.0
            if abs(beta - 1.0) <= BETA_MIN:
                continue
            for sign in (+1, -1):
                v = _c2(**x, sign=sign)
                if v is None or v < 0:            # c^2 = (mean/sd)^2 cannot be negative
                    continue
                if full:
                    b = None                      # relative 1e-6, applied at the centre
                else:
                    b = _propagate(lambda **kw: _c2(**kw, sign=sign), x, h)
                    if not (b < BOUND_MAX_REL * abs(v)):
                        continue
                cands.append((i, v, b))
                usable.add(i)
            # the D = 0 case gives the same c^2 twice; that is one call, counted once below
        k = len(usable)
        out["n_usable"] = k
        if k == 0:
            out["why"] = ["no usable call to estimate c^2 (|beta-1| <= 0.01 or imprecise)"] * n
            out["status"] = "c2_unestimable"
            return out

        def agrees(v_centre, cand):
            _, v, b = cand
            tol = FULL_PRECISION_REL * abs(v_centre) if b is None else b
            return abs(v - v_centre) <= tol

        centres = []                                  # (support, centre value, centre's own tol, agreeing)
        for c in cands:
            agreeing = [d for d in cands if agrees(c[1], d)]
            own = FULL_PRECISION_REL * abs(c[1]) if c[2] is None else c[2]
            centres.append((len({d[0] for d in agreeing}), c[1], own, agreeing))
        sup, c2, c2_tol, agreeing = max(centres, key=lambda t: t[0])
        out["support"] = sup / k
        if sup < AGREE_FRACTION * k:
            out["why"] = [f"no c^2 value agreed by >= {AGREE_FRACTION:.0%} of usable calls "
                          f"(best {sup}/{k})"] * n
            out["status"] = "c2_no_agreement"
            return out
        # unique: no second, separate value that also has >= 80 % support
        for s2, v2, tol2, _ in centres:
            if s2 >= AGREE_FRACTION * k and abs(v2 - c2) > c2_tol + tol2:
                out["why"] = ["two different c^2 values have the same support"] * n
                out["status"] = "c2_not_unique"
                return out
        if full:
            delta = FULL_PRECISION_REL * abs(c2)
        else:
            delta = 2.0 * min(d[2] for d in agreeing)
        out.update(c2=c2, delta_c2=delta, status="c2_agreed")

    # ── 2. per call: the root whose predicted NSE matches the recorded NSE ────────────────────
    c2 = out["c2"]; delta = out["delta_c2"]
    for i, x in enumerate(xs):
        if x is None:
            out["why"][i] = "nse, kge, r or pbias not recorded"
            continue
        roots = _roots(**x)
        if not roots:
            out["why"][i] = "D^2 < 0 (metrics inconsistent)"
            continue
        beta = 1.0 + x["pbias"] / 100.0

        def resid(nse, kge, r, pbias, _sign):
            d2 = _d2(nse, kge, r, pbias)
            if d2 < 0:
                return None
            a = 1.0 + _sign * math.sqrt(d2)
            b_ = 1.0 + pbias / 100.0
            return 2.0 * r * a - a * a - (b_ - 1.0) ** 2 * c2 - nse

        matches = []
        for a in roots:
            sign = +1 if a >= 1.0 else -1
            p = 2.0 * x["r"] * a - a * a - (beta - 1.0) ** 2 * c2
            prec = 0.0 if full else _propagate(lambda **kw: resid(**kw, _sign=sign), x, h)
            bound = prec + (beta - 1.0) ** 2 * delta + NSE_FLOOR
            if abs(p - x["nse"]) <= bound:
                matches.append(a)
        if len(matches) == 1:
            out["alpha"][i] = matches[0]
        elif len(matches) > 1:
            out["why"][i] = "both roots match (r close to 1 or D close to 0)"
        else:
            out["why"][i] = "no root matches the recorded NSE"
    return out


def verify_against_measured(records, measured_alpha, **kw) -> dict:
    """The §2.13 check before use on old histories: on runs where alpha IS measured, the
    reconstruction must match to 1e-6. Returns {"max_abs_err", "n_compared", "n_missing", "ok"}."""
    if len(records or []) != len(measured_alpha or []):
        raise ValueError(f"{len(records or [])} records but {len(measured_alpha or [])} measured alphas")
    res = reconstruct_alpha(records, **kw)
    errs, miss, why_missing = [], 0, {}
    for i, (a, m) in enumerate(zip(res["alpha"], measured_alpha)):
        if m is None:
            continue
        if a is None:
            miss += 1
            why_missing[i] = res["why"][i]
            continue
        errs.append(abs(a - float(m)))
    mx = max(errs) if errs else None
    return {"max_abs_err": mx, "n_compared": len(errs), "n_missing": miss, "why_missing": why_missing,
            # every measured call must be reconstructed AND match: a method that leaves most calls
            # missing has not been verified
            "ok": bool(errs) and mx <= 1e-6 and miss == 0, "reconstruction": res}
