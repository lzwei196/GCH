"""REPLAY — apply the convergence rule to a search that already ran (design §4 step 3, gap 2u; build step 9).

A replay feeds a recorded search, call by call, through the SAME rule (rule.ConvergenceRule) and verdicts
(verdict.seed_verdict) the live kit uses. It starts with an EVIDENCE MANIFEST — everything the rule needs and
where it came from:

  - each variable's kind and the obs_shape it was scored on;
  - the optimizer's losses (one per objective, per call) and their weights;
  - protected metrics and the default run's panel (the protection anchor);
  - the tolerances (or their source);
  - the phase and the split of every record;
  - whether failures, cache hits and the call order are complete.

A missing prerequisite makes the affected result UNDECIDABLE — never guessed. Two levels, never mixed:

  Level A — decided: every required metric recorded (directly, or alpha by §2.13), the history complete (every
            call, in order, with phases) and the manifest complete. The §2.7 stop point and §2.10 safety are the
            rule's answer.
  Level B — diagnostic on available evidence: anything less. The same computation runs on the metrics and calls
            that exist and is always labelled "on available evidence: <metrics>, <what is missing>"; it is never
            reported as converged, safe or premature.

What a replayed stop point means: for SCE-UA and NSGA-II stopping only removes the calls after the stop point, so the
replay gives exactly where the rule would have stopped THIS recorded trajectory; for DDS it is a diagnostic only
(DDS cannot stop, and a shorter DDS budget would have searched differently). A replay applies the rule to a recorded
trajectory; it is not a run of the revised live kit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .panel import Panel
from .rule import ConvergenceRule
from .verdict import seed_verdict

PASS_THROUGH_WHY = "not recorded"


def _fin(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_history(workdir) -> list:
    """Every record of <workdir>/eval_history.jsonl (torn / non-dict lines skipped)."""
    out = []
    p = Path(workdir) / "eval_history.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def sessions(records) -> list:
    """Split a log into calibrate() calls: the counter `i` restarts at 0 for every call."""
    out, cur, prev = [], [], None
    for r in records:
        i = r.get("i")
        if prev is not None and isinstance(i, int) and i <= prev:
            out.append(cur)
            cur = []
        cur.append(r)
        prev = i if isinstance(i, int) else prev
    if cur:
        out.append(cur)
    return [s for s in out if s]


def evidence_manifest(report: dict | None) -> dict:
    """What the rule needs, from the run's own report (kit >= 0.4.0-dev). Missing pieces are named."""
    rep = report or {}
    cv = rep.get("convergence") or {}
    ru = cv.get("rule") or {}
    stds = ((cv.get("verdict") or {}).get("standards") or {}) if isinstance((cv.get("verdict") or {}).get("standards"), dict) else {}
    m = {
        "algorithm": rep.get("algorithm"),
        "objectives": list(ru.get("objectives") or []),
        "objective_vars": list(ru.get("objective_vars") or []),
        "weights": ru.get("weights"),
        "trade_off": ru.get("trade_off"),
        "window": ru.get("window"), "rel_gain": ru.get("rel_gain"), "eps_front": ru.get("eps_front"),
        "kinds": dict(cv.get("kinds") or {}),
        "obs_shapes": {v: (s or {}).get("obs_shape") for v, s in stds.items() if isinstance(s, dict)},
        "pbias_percent": dict(cv.get("pbias_percent") or {}),
        "kit_missing": dict(cv.get("kit_missing") or {}),
        "tolerances": {v: dict((r or {}).get("tol") or {}) for v, r in (cv.get("tolerance_records") or {}).items()},
        "tolerance_source": cv.get("tolerance_source"),
        "protect": (ru.get("protect_active") if isinstance(ru.get("protect_active"), dict) else {}),
        "default_panel": ru.get("default_panel") or {},
        "mode": cv.get("mode"),
    }
    missing = []
    if not m["objectives"] or len(m["objective_vars"]) != len(m["objectives"]):
        missing.append("objectives and their variables")
    if not m["trade_off"] and len(m["objectives"]) > 1 and not m["weights"]:
        missing.append("objective weights")
    if not m["kinds"]:
        missing.append("each variable's kind")
    if not m["window"]:
        missing.append("the window W")
    if not m["tolerances"]:
        missing.append("tolerances")
    if m["protect"] and not m["default_panel"]:
        missing.append("the protection anchor (default run's panel)")
    m["missing"] = missing
    return m


def _split_gaps(rows) -> tuple:
    """(missing pieces, notes) for the split of each search record (design §4 step 3): every search call must
    have been asked for the CALIBRATION split and, where the runner echoed one, echoed that split."""
    miss, notes = [], []
    # a call that never ran the model (rejected by a constraint, or its runner raised) scored nothing: its
    # missing split (logs written before the evaluator recorded it on every call) is not missing evidence
    no_split = sum(1 for r in rows if not r.get("split") and r.get("reason") not in ("infeasible", "exception"))
    other = sum(1 for r in rows if r.get("split") and str(r.get("split")).lower() != "calibration")
    wrong = sum(1 for r in rows if r.get("split_echo") not in (None, "calibration"))
    silent = sum(1 for r in rows if "split_echo" in r and r.get("split_echo") is None)
    if no_split:
        miss.append(f"the split of {no_split} search calls")
    if other:
        miss.append(f"{other} search calls were asked for a split other than calibration")
    if wrong:
        miss.append(f"{wrong} search calls echoed a split other than calibration")
    if silent:
        notes.append(f"{silent} search calls did not echo the split they scored")
    return miss, notes


def _seed_groups(search) -> dict:
    groups: dict = {}
    for r in search:
        groups.setdefault(r.get("seed"), []).append(r)
    return groups


def replay_seed(records, manifest: dict, ended_at_stop: bool = False, dream: dict | None = None) -> dict:
    """One seed's recorded search through the rule. Returns the rule's answer and the evidence level.
    `dream` = the seed's recorded R-hat outcome {"rhat_recorded", "rhat_reached"}: R-hat decides a DREAM
    seed (design §2.7), exactly as in the live kit."""
    variables = list(dict.fromkeys(list(manifest["kinds"]) + list(manifest["objective_vars"])))
    panel = Panel(variables, kinds=manifest["kinds"], pbias_percent=manifest.get("pbias_percent"))
    for v, mm in (manifest.get("kit_missing") or {}).items():
        for mt, why in (mm or {}).items():
            if v in panel.kinds:
                panel.set_kit_missing(v, mt, why)
    rule = ConvergenceRule(list(zip(manifest["objectives"], manifest["objective_vars"])), panel,
                           manifest["tolerances"], int(manifest["window"]), trade_off=bool(manifest["trade_off"]),
                           weights=manifest.get("weights"),
                           # 0.0 is a real value (codex A3d r3): only a missing value takes the default
                           rel_gain=float(0.005 if manifest.get("rel_gain") is None else manifest["rel_gain"]),
                           eps_front=float(0.01 if manifest.get("eps_front") is None else manifest["eps_front"]),
                           protect=manifest.get("protect") or {}, default_panel=manifest.get("default_panel") or {})
    gaps: dict = {}                                      # (var, metric) -> calls where it was not recorded
    for k, r in enumerate(records):
        losses = [_fin(x) for x in (r.get("losses") or [])]
        if not losses or len(losses) != rule.K:
            losses = None
        any_fin = bool(losses) and any(x is not None for x in losses)
        pn = (r.get("panel") or {}) if any_fin else {}
        if any_fin:
            for v in rule.variables:
                for mt in panel.required(v):
                    if _fin((pn.get(v) or {}).get(mt)) is None:
                        gaps.setdefault((v, mt), []).append(k)
        rule.add(losses, None, pn)
    vd = seed_verdict(rule, ended_at_stop=ended_at_stop, algorithm=manifest.get("algorithm"),
                      dream=(dream if str(manifest.get("algorithm")) == "dream" else None))
    return {"rule": rule, "verdict": vd, "gaps": gaps, "calls": len(records)}


def replay_workdir(workdir, report: dict | None = None, session: str = "last") -> dict:
    """Replay the search of one calibrate() call (the last one by default) from its eval_history.jsonl,
    per seed, with the evidence manifest and the level (A decided / B diagnostic)."""
    recs = read_history(workdir)
    ss = sessions(recs)
    notes: list = []
    if not ss:
        return {"level": "B", "status": "nothing to replay", "notes": ["no eval_history.jsonl records"], "seeds": {}}
    if len(ss) > 1:
        notes.append(f"{len(ss)} calibrate() calls in this log; replaying the {session} one")
    chosen = ss[-1] if session == "last" else ss[int(session)]
    search = [r for r in chosen if r.get("phase") == "search"]
    manifest = evidence_manifest(report)
    if not (manifest["objectives"] and manifest["window"] and manifest["kinds"]
            and len(manifest["objective_vars"]) == len(manifest["objectives"])):
        return {"level": "B", "status": "undecidable: the manifest lacks " + "; ".join(manifest["missing"]),
                "manifest": manifest, "notes": notes, "seeds": {}}
    if not search:
        return {"level": "B", "status": "nothing to replay", "manifest": manifest,
                "notes": notes + ["no search-phase records"], "seeds": {}}
    # completeness of the history: phase tags on every record, the call counts the run reported per seed
    hist_missing = []
    if any("phase" not in r for r in chosen):
        hist_missing.append("phase tags")
    # every call, in order (codex step 9 r1 #1): the kit numbers each call once, 0, 1, 2 …, within a calibrate()
    # call; a gap, a repeat or a reorder means the log is not the complete, ordered history (a count can still match)
    _ii = [r.get("i") for r in chosen]
    if not (all(isinstance(x, int) for x in _ii) and _ii == list(range(len(_ii)))):   # 0, 1, 2 … (codex r2)
        hist_missing.append("an unbroken, ordered call log (the call numbers have gaps, repeats or reorders)")
    slot_list = [s for s in (((report or {}).get("convergence") or {}).get("seeds") or {}).get("slots", [])
                 if isinstance(s, dict)]
    slots = {s.get("seed"): s for s in slot_list}
    # a crashed attempt is not a complete search: never replayed as one (no verdict), listed apart (Opus 8b/9 r1 #2)
    crashed_first = {(s.get("replaced") or {}).get("seed"): s for s in slot_list if s.get("replaced")}
    out_seeds, attempts = {}, {}
    groups = _seed_groups(search)
    for sd, rows in groups.items():
        if sd in crashed_first or (slots.get(sd) or {}).get("crashed"):
            attempts[sd] = {"calls": len(rows), "level": "B", "verdict": None,
                            "label": "crashed — not a complete search (no verdict)",
                            "replaced_by": (crashed_first.get(sd) or {}).get("seed")}
            continue
        m = dict(manifest)
        _algo = str(m.get("algorithm"))
        dream = (slots.get(sd) or {}).get("dream") if _algo == "dream" else None
        rs = replay_seed(rows, m, ended_at_stop=bool(((slots.get(sd) or {}).get("ended") or {}).get("reason") == "our rule"),
                         dream=dream)
        seed_missing = list(hist_missing)
        if report is not None and slot_list and sd not in slots:
            seed_missing.append("this seed's slot in the report")
        if _algo == "dream" and not isinstance(dream, dict):
            seed_missing.append("DREAM's R-hat record (R-hat decides a DREAM seed)")
        _sm, _sn = _split_gaps(rows)
        seed_missing += _sm
        rep_calls = (slots.get(sd) or {}).get("calls")
        if rep_calls is not None and int(rep_calls) != rs["calls"]:
            seed_missing.append(f"calls: the report says {rep_calls}, the log holds {rs['calls']}")
        gaps = {f"{v}:{mt}": len(ks) for (v, mt), ks in rs["gaps"].items()}
        level = "A" if not (m["missing"] or seed_missing or gaps) else "B"
        rule, vd = rs["rule"], rs["verdict"]
        row = {"calls": rs["calls"], "stop_point": rule.stop_point,
               "first_settle": dict(rule.first_settle), "level": level,
               "algorithm": m.get("algorithm")}
        if level == "A":
            row.update(verdict=vd.get("verdict"), how=vd.get("how"), safety=vd.get("safety"),
                       variables={v: (x or {}).get("verdict") for v, x in (vd.get("variables") or {}).items()})
        else:
            recorded = sorted({f"{v}:{mt}" for v in rule.variables for mt in rule.panel.required(v)} - set(gaps))
            what = m["missing"] + seed_missing + [f"{k} not recorded on {n} calls" for k, n in gaps.items()]
            # Level B is never called converged, safe or premature (design): only the stop point is shown
            row.update(verdict="diagnostic only", how=None, safety=None,
                       label=f"on available evidence: {', '.join(recorded) or 'losses only'}; missing: {'; '.join(what)}",
                       diagnostic={"stop_point": rule.stop_point})
        if _sn:
            row["notes"] = _sn
        if str(m.get("algorithm")) == "dds":
            row["note"] = ("DDS: the replayed stop point is a diagnostic only — DDS cannot stop, and a shorter DDS "
                           "budget would have searched differently")
        # self-check against the run's own rule (same kit): the same stop point AND the same seed verdict
        if slots.get(sd) and "ended" in slots[sd]:
            own = (slots[sd].get("ended") or {}).get("stop_point")
            # the per-call step rule's own verdict (since 2026-10-04 kept as step_rule_verdict; older reports: verdict)
            _own_vd = slots[sd].get("step_rule_verdict", slots[sd].get("verdict"))
            row["matches_the_runs_own_rule"] = (own == rule.stop_point
                                                and (level != "A" or _own_vd == vd.get("verdict")))
        out_seeds[sd] = row
    # a slot the report lists (and that was not crashed) with no search record in this log
    absent = [s.get("seed") for s in slot_list if not s.get("crashed") and s.get("seed") not in groups]
    if absent:
        notes.append(f"seeds {absent} are in the report but have no search records in this log")
    return {"level": "A" if out_seeds and not absent and all(r["level"] == "A" for r in out_seeds.values()) else "B",
            "manifest": {k: v for k, v in manifest.items() if k not in ("default_panel",)},
            "notes": notes, "seeds": out_seeds, "crashed_attempts": attempts}


def fill_alpha(records, variables, preconditions_ok: bool = False) -> dict:
    """alpha for calls that recorded nse, kge, r and pbias but not alpha (design §2.13), per variable, in
    place, labelled `alpha_reconstructed`. Only when the caller vouches for the §2.13 preconditions (the KGE
    2009 definition, PBIAS in percent, the same series); otherwise nothing is filled. Returns {var: status}."""
    from .alpha_reconstruct import reconstruct_alpha
    out = {}
    for v in variables:
        rows = [((r.get("panel") or {}).get(v) or {}) for r in records]
        if not any(_fin(x.get("alpha")) is None and all(_fin(x.get(m)) is not None for m in ("nse", "kge", "r", "pbias"))
                   for x in rows):
            continue
        res = reconstruct_alpha([{m: x.get(m) for m in ("nse", "kge", "r", "pbias")} for x in rows],
                                preconditions_ok=bool(preconditions_ok))
        n = 0
        for r, a in zip(records, res.get("alpha") or []):
            pv = (r.setdefault("panel", {})).setdefault(v, {})
            if _fin(pv.get("alpha")) is None and a is not None:
                pv["alpha"] = a
                pv.setdefault("_reconstructed", []).append("alpha")
                n += 1
        out[v] = {"status": res.get("status"), "filled": n}
    return out


def replay_records(records, report: dict | None, complete: bool = False, notes: list | None = None,
                   alpha_preconditions_ok: bool = False, dream: dict | None = None) -> dict:
    """Replay records that did NOT come from this kit's eval_history.jsonl (an older kit's log or its cache):
    `records` = [{"losses": [...], "panel": {var: {metric: value}}}] in call order, one search. The history is
    taken as INCOMPLETE unless `complete=True` is stated by the caller (older logs miss failures, cache hits or
    pre-run rejections), so the result is Level B unless every piece is there."""
    manifest = evidence_manifest(report)
    variables = list(dict.fromkeys(list(manifest["kinds"]) + list(manifest["objective_vars"])))
    alpha = fill_alpha(records, variables, alpha_preconditions_ok) if variables else {}
    if not (manifest["objectives"] and manifest["window"] and manifest["kinds"]
            and len(manifest["objective_vars"]) == len(manifest["objectives"])):
        return {"level": "B", "status": "undecidable: the manifest lacks " + "; ".join(manifest["missing"]),
                "manifest": manifest, "notes": list(notes or []), "seeds": {}}
    rs = replay_seed(records, manifest, dream=dream)
    gaps = {f"{v}:{mt}": len(ks) for (v, mt), ks in rs["gaps"].items()}
    what = list(manifest["missing"]) + ([] if complete else ["a complete call history (failures, cache hits, "
                                                             "rejections and their order)"])
    if str(manifest.get("algorithm")) == "dream" and not isinstance(dream, dict):
        what.append("DREAM's R-hat record (R-hat decides a DREAM seed)")
    what += [f"{k} not recorded on {n} calls" for k, n in gaps.items()]
    rule, vd = rs["rule"], rs["verdict"]
    row = {"calls": rs["calls"], "stop_point": rule.stop_point, "first_settle": dict(rule.first_settle),
           "level": "A" if not what else "B", "algorithm": manifest.get("algorithm"), "alpha": alpha}
    if row["level"] == "A":
        row.update(verdict=vd.get("verdict"), how=vd.get("how"), safety=vd.get("safety"))
    else:
        recorded = sorted({f"{v}:{mt}" for v in rule.variables for mt in rule.panel.required(v)} - set(gaps))
        row.update(verdict="diagnostic only", how=None, safety=None,
                   label=f"on available evidence: {', '.join(recorded) or 'losses only'}; missing: {'; '.join(what)}",
                   diagnostic={"stop_point": rule.stop_point})
    if str(manifest.get("algorithm")) == "dds":
        row["note"] = ("DDS: the replayed stop point is a diagnostic only — DDS cannot stop, and a shorter DDS "
                       "budget would have searched differently")
    return {"level": row["level"], "manifest": manifest, "notes": list(notes or []), "seeds": {None: row}}


def replay_native(records, convergence: dict | None, algorithm: str, objectives, kinds: dict, tol: dict,
                  weights=None) -> dict:
    """The kit's convergence rule (2026-10-04) replayed on one recorded seed: SCE-UA = SPOTPY's own loop-end test on
    the objective and watched scores (native_rules.SpotpyTest), NSGA-II / NSGA-III / MOEA-D = pymoo's own rule (only
    from the live run's per-generation record: the rule needs pymoo's population, which call records do not hold),
    every single-objective search = the settle point (settle.SettlePoint).

    `records`: the seed's search calls in order, [{"losses": [...], "panel": {...}}]; `convergence`: that seed's
    report block (optimizer_termination with loop_starts / loop_record / native_verdict). Without SPOTPY's per-loop
    record (loop_record: gnrng and the best objective) the SCE-UA replay uses the best point's loss and leaves out the
    population-spread part — that can only make it fire later or not at all, and it is said in the result."""
    from .native_rules import SpotpyTest
    from .rule_steps import StepRule
    from .settle import SettlePoint
    term = ((convergence or {}).get("optimizer_termination") or {})
    variables = list(dict.fromkeys(list(kinds) + [v for _, v in objectives]))
    cv = convergence or {}
    # the live panel's decisions (codex A3d #2): kit-set missing metrics leave the required set, PBIAS unit
    panel = Panel(variables, kinds={v: kinds.get(v, "series") for v in variables},
                  pbias_percent=cv.get("pbias_percent"))
    for v, mm in (cv.get("kit_missing") or {}).items():
        for mt, why in (mm or {}).items():
            if v in panel.kinds:
                panel.set_kit_missing(v, mt, why)
    trade_off = algorithm in ("nsga2", "nsga3", "moead")
    out = {"algorithm": algorithm, "calls": len(records)}
    if algorithm == "sceua":
        starts = sorted((int(c), int(l)) for c, l in (term.get("loop_starts") or []))
        if not starts:
            out["native"] = {"verdict": "undecidable", "why": "no loop tags (loop_starts) in the record"}
        else:
            rule = StepRule(list(objectives), panel, tol, trade_off=False, weights=weights)
            j = 0
            for i, r in enumerate(records):
                while j + 1 < len(starts) and starts[j + 1][0] <= i:
                    j += 1
                rule.add(r.get("losses"), None, r.get("panel"), step=starts[j][1])
            # the last loop is complete when SPOTPY or the kit ended the search at a loop end, not when the cap cut it
            complete = term.get("status") in ("stopped_by_kit_rule", "population_converged", "improvement_below_pcento")
            rule.finish(last_step_complete=bool(complete))
            lr = {int(d["loop"]): d for d in (term.get("loop_record") or [])}
            t = SpotpyTest(watch_scores=True, variables=list(rule.variables))
            for e in rule.ends:
                if e["step"] < 1:
                    continue                                   # loop 0 = the start-up sample (SPOTPY's criter starts at 1)
                d = lr.get(e["step"])
                bestf = d["bestf"] if d else (e["V"][0] if len(e["V"]) == 1 else None)
                t.add_loop(e["step"], bestf, d["gnrng"] if d else None, e["X"])
            out["native"] = dict(t.summary(), gnrng_recorded=bool(lr),
                                 note=(None if lr else "SPOTPY's per-loop record is missing: the population-spread "
                                       "part is left out (it can only make the rule fire later, never earlier)"))
    elif trade_off:
        nv = term.get("native_verdict") or {}
        if nv.get("per_generation") is not None:
            out["native"] = {k: nv.get(k) for k in ("rule", "settings", "generations_seen", "fired_at", "verdict")}
        else:
            out["native"] = {"verdict": "not replayable",
                             "why": "pymoo's own rule needs the population, which call records do not hold"}
    else:
        out["native"] = {"rule": None, "verdict": f"{algorithm} has no convergence rule of its own; see the settle point"}
    if not trade_off:
        # the run's own rel_gain (codex A3d r2): the live settle point uses the contract's value
        _rg = (cv.get("rule") or {}).get("rel_gain")
        sp = SettlePoint(panel, tol, rel_gain=float(0.005 if _rg is None else _rg))   # 0.0 is a real value
        w = [float(x) for x in weights] if weights is not None else None
        for r in records:
            ls = [_fin(x) for x in (r.get("losses") or [])]
            sc = None
            if ls and len(ls) == len(objectives) and all(x is not None for x in ls):
                ww = w or [1.0] * len(ls)
                sc = sum(a * b for a, b in zip(ww, ls)) / (sum(ww) or 1.0)
            sp.add(sc, r.get("panel"))
        out["settle"] = sp.summary()
    return out


def main(argv=None):
    """python -m calibration_kit.replay <workdir> [--report report.json] [--json out.json]"""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workdir")
    ap.add_argument("--report", help="the run's kit report (JSON); default: <workdir>/../report.json if present")
    ap.add_argument("--json", help="write the full replay record here")
    a = ap.parse_args(argv)
    rp = Path(a.report) if a.report else None
    if rp is None:
        for cand in (Path(a.workdir) / "calibration_report.json", Path(a.workdir).parent / "report.json"):
            if cand.exists():
                rp = cand
                break
    report = json.loads(rp.read_text()) if rp and rp.exists() else None
    out = replay_workdir(a.workdir, report)
    print(f"replay: level {out['level']}" + (f" — {out.get('status')}" if out.get("status") else ""))
    for n in out.get("notes") or []:
        print(f"  note: {n}")
    for sd, att in (out.get("crashed_attempts") or {}).items():
        print(f"seed {sd}: {att['label']} ({att['calls']} calls, replaced by seed {att.get('replaced_by')})")
    for sd, row in out["seeds"].items():
        print(f"seed {sd}: level {row['level']} calls {row['calls']} stop point {row['stop_point']} "
              f"verdict {row['verdict']}" + (f" — {row['label']}" if row.get("label") else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
