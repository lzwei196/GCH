"""BUDGET — turn a wall-clock allowance into an evaluation cap (handoff §5.5).

A contract used to declare `max_evaluations` as a number someone chose. That number carries no
information about the model: the same 500 evaluations is generous for a lumped rainfall-runoff
model and nothing at all for a distributed one. Here the cap is DERIVED:

    cap per seed = floor( (allowance - shared overhead - final overhead) / waves x e / t_slow )
                                                                     (design §1.3, one ledger)
  t_slow      1.2 x p90 of a 10-run pilot, or 1.2 x max(the user's run time, the slowest of a 3-run
              pilot) — conservative on purpose, because slow runs cluster late in a search
  waves       seeds / seeds that fit in the measured lanes: three seeds in one lane is three waves
  e           the measured parallel efficiency of those lanes (compute.py), never assumed
  shared      time spent once before the seeds: the certification replay, the objective probe, the
              pilot (reused by every seed), the machine probe, commissioning and the consumption
              proof — measured
  final       time spent once after the seeds: the validation assessment of the one reported
              result (4 runs x t_slow)

κ = cap/(d+1) reports the cap in units of the problem's dimension, the standard way to compare
budgets across models of different size (Moré & Wild 2009, doi:10.1137/080724083). A small κ is
not an error but it IS a warning on the record: the allowance was too short for this model, and a
poor score from such a run says nothing about the model.

`fixed` mode keeps the legacy behaviour — the contract's `max_evaluations` is the cap — but still
reports κ and the expected finish time from the pilot, so a hand-set budget stops being opaque.
"""
from __future__ import annotations
import math
import re

#: κ below this is flagged: the allowance is short for the number of parameters (provisional, §5.15)
KAPPA_WARN = 10.0


def parse_allowance(v) -> float:
    """Seconds from `72h`, `90m`, `3d`, `5400`, `1.5 h`. Returns 0.0 when there is nothing to parse."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().lower().replace(" ", "")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([smhd]?)", s)
    if not m:
        return 0.0
    n = float(m.group(1))
    return n * {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[m.group(2)]


def estimate_overhead_evals(n_params: int, *, runner_mode: str = "applicator", front_size: int = 0,
                            consumption_proof: bool = True, pilot_n: int = 10) -> dict:
    """Evaluations a run spends OUTSIDE the search, so the cap does not quietly include them.
    Commissioning perturbs each parameter once (1 + d); the consumption proof does the same
    against the raw series; the holdout gate costs
    about 4 runs and front selection 2 per gated member."""
    d = max(0, int(n_params))
    # the pilot's evaluations are charged to the overhead even though the search can re-use them
    # from the cache: it under-promises the cap by at most `pilot_n`, which is the safe direction
    # for a wall-clock allowance. Deliberate — do not "optimize" it away.
    parts = {"pilot": int(max(0, pilot_n))}
    if str(runner_mode).strip().lower() == "runner":
        parts["commission"] = 1 + d
        if consumption_proof:
            parts["consumption_proof"] = 1 + d
    parts["holdout"] = FINAL_OVERHEAD_RUNS       # probe (2) + best on both splits (2) + default on holdout (1)
    if front_size:
        parts["front_select"] = 2 * int(front_size)
    parts["total"] = int(sum(v for k, v in parts.items() if k != "total"))
    return parts


T_SLOW_FACTOR = 1.2
FINAL_OVERHEAD_RUNS = 5          # split probe on both splits + the reported result on both splits + the DEFAULT
                                 # parameters on the holdout split (the "beats the default" check; HBV tracer 2026-09-30)
MIN_CAP = 20


def t_slow_from(pilot: dict | None, run_time_s: float | None = None) -> float:
    """The slow-end time per call (design §1.3): 1.2 x p90 of a 10-run pilot, or, when the user gave
    a typical run time (3-run pilot), 1.2 x max(the user's time, the slowest of the 3 runs)."""
    p = pilot or {}
    if run_time_s:
        return T_SLOW_FACTOR * max(float(run_time_s), float(p.get("max_s") or 0.0))
    return T_SLOW_FACTOR * float(p.get("p90_s") or 0.0)


def plan_budget(allowance_s, pilot: dict, lanes: int, efficiency: float, n_params: int,
                n_seeds: int = 3, overhead_evals: int = 0, hard_max: int | None = None,
                kappa_warn: float = KAPPA_WARN, mode: str = "measured",
                max_evaluations: int | None = None, min_cap: int = MIN_CAP,
                shared_overhead_s: float | None = None, final_overhead_s: float | None = None,
                run_time_s: float | None = None) -> dict:
    """The cap per seed, and every number it came from (design §1.3). Never raises.

    ONE allowance ledger: shared overhead (certification replay, objective probe, pilot — reused by every
    seed —, machine probe, commissioning, consumption proof) and final overhead (the one validation assessment, 4 runs) are
    charged ONCE, as time, before the seeds share what is left:
        cap per seed = floor( (allowance - shared - final) / waves x e / t_slow )
    `overhead_evals` is the older per-run form; when the times are not given it is converted with
    t_slow and charged as shared overhead. In measured mode a cap below `min_cap` (20) is marked
    `refuse` with the reason — the caller refuses the run."""
    d = max(0, int(n_params))
    t90 = float((pilot or {}).get("p90_s") or 0.0)
    tmed = float((pilot or {}).get("median_s") or 0.0)
    t_slow = t_slow_from(pilot, run_time_s)
    lanes = max(1, int(lanes or 1))
    eff = float(efficiency or 1.0)
    if not (0.0 < eff <= 1.0):
        eff = min(1.0, max(1e-3, eff))
    n_seeds = max(1, int(n_seeds or 1))
    seeds_parallel = max(1, min(n_seeds, lanes))
    waves = int(math.ceil(n_seeds / seeds_parallel))
    warn: list = []
    allowance_s = parse_allowance(allowance_s)
    if not math.isfinite(allowance_s):
        allowance_s = 0.0                 # a non-finite allowance is no allowance ("never raises")
    mode = (mode or "measured").strip().lower()
    final_s = float(final_overhead_s) if final_overhead_s is not None else FINAL_OVERHEAD_RUNS * t_slow
    shared_s = (float(shared_overhead_s) if shared_overhead_s is not None
                else float(overhead_evals or 0) * t_slow)
    refuse, refuse_why = False, None
    search_s = None

    if mode == "fixed" or not allowance_s or t_slow <= 0:
        if mode != "fixed":
            warn.append("no usable allowance or pilot timing — falling back to the contract's "
                        "max_evaluations (fixed mode)" if max_evaluations else
                        "no usable allowance or pilot timing, and no max_evaluations to fall back to")
            mode = "fixed" if max_evaluations else "unplanned"
        cap = int(max_evaluations or 0)
    else:
        search_s = allowance_s - shared_s - final_s
        cap = int(math.floor(max(0.0, search_s) / waves * eff / t_slow))
        if max_evaluations:                      # in measured mode this is an optional CEILING
            if cap > int(max_evaluations):
                warn.append(f"cap {cap} trimmed to the contract's max_evaluations {int(max_evaluations)}")
            cap = min(cap, int(max_evaluations))
    # what the allowance alone buys, and every ceiling under the minimum — a refusal names ALL the
    # limits the user has to raise, so one fix does not just lead to the next refusal
    by_allowance = (int(math.floor(max(0.0, search_s) / waves * eff / t_slow)) if search_s is not None else None)
    low_ceilings = []
    if mode == "measured" and max_evaluations and int(max_evaluations) < min_cap:
        low_ceilings.append(f"max_evaluations ({int(max_evaluations)})")
    if hard_max:
        if cap > int(hard_max):
            warn.append(f"cap trimmed to hard_max {int(hard_max)}")
        if mode == "measured" and int(hard_max) < min_cap:
            low_ceilings.append(f"hard_max ({int(hard_max)})")
        cap = min(cap, int(hard_max))
    if mode == "measured" and cap < min_cap and by_allowance is not None and by_allowance >= min_cap:
        refuse = True
        refuse_why = (f"the cap is {cap} search calls per seed (under {min_cap}): the allowance buys "
                      f"{by_allowance}, but {' and '.join(low_ceilings)} cut it — raise "
                      f"{' and '.join(c.split(' (')[0] for c in low_ceilings)} to at least {min_cap}")
    elif mode == "measured" and cap < min_cap:
        refuse = True
        refuse_why = (f"the allowance buys {by_allowance if by_allowance is not None else cap} search calls "
                      f"per seed (under {min_cap}): "
                      f"{allowance_s:.0f} s minus {shared_s:.0f} s shared and {final_s:.0f} s final overhead, "
                      f"over {waves} wave(s) at {t_slow:.1f} s per call (1.2 x the slow end, "
                      f"efficiency {eff:.2f}) — raise the allowance or use a cheaper case"
                      + (f"; {' and '.join(low_ceilings)} must also be raised to at least {min_cap}"
                         if low_ceilings else ""))
    cap = max(cap, 0)
    kappa = cap / (d + 1)
    if cap > 0 and kappa < kappa_warn:
        warn.append(f"kappa={kappa:.1f} < {kappa_warn}: the cap is short for {d} parameters — a "
                    f"weak result from this run is a budget statement, not a model statement")
    fail_rate = float((pilot or {}).get("fail_rate") or 0.0)
    if fail_rate > 0:
        # failed evaluations still cost their run time, so the cap has to expect them
        warn.append(f"pilot fail rate {fail_rate:.0%} — expect that share of the cap to be spent "
                    f"on failed evaluations")
    expected = (shared_s + final_s + waves * cap * t_slow / eff) if t_slow > 0 else None
    return {"mode": mode, "allowance_s": allowance_s, "t_p90_s": t90, "t_median_s": tmed,
            "t_slow_s": t_slow, "run_time_given_s": run_time_s,
            "lanes": lanes, "efficiency": eff, "seeds": n_seeds,
            "seeds_parallel": seeds_parallel, "waves": waves,
            "overhead_evals": int(overhead_evals or 0),
            # the allowance ledger (design §1.3): charged once, not per seed
            # unrounded, so anyone can reproduce the cap from these numbers
            "ledger": {"allowance_s": allowance_s, "shared_overhead_s": shared_s,
                       "final_overhead_s": final_s, "search_s": search_s,
                       "per_seed_search_s": (max(0.0, search_s) / waves if search_s is not None else None)},
            "cap_per_seed": int(cap), "refuse": refuse, "refuse_reason": refuse_why,
            "kappa": round(kappa, 2), "kappa_warn": kappa_warn,
            "expected_finish_s": (round(expected, 1) if expected is not None else None),
            "expected_finish_h": (round(expected / 3600.0, 2) if expected is not None else None),
            "pilot": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in (pilot or {}).items() if k not in ("records", "series")},
            "warnings": warn}
