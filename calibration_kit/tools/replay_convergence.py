#!/usr/bin/env python
"""SUPERSEDED (build step 9, 2026-09-30): this tool replays the OLD convergence marker (convergence.py). The
replay of the rebuilt rule — evidence manifest, Level A (decided) / Level B (diagnostic) — is
`calibration_kit/replay.py` (`python -m calibration_kit.replay <workdir>`). Kept for the paper's earlier numbers.

Replay the convergence marker over runs that are already finished (handoff §5.13).

Why: the marker was built after the six-model campaign ran, so those runs cannot be asked
directly where they converged. But they left their evaluation records behind, and the marker is a
function of those records — so it can be applied offline. That answers, for runs already reported,
"was the search finished, or was it cut off at the budget?" without rerunning a single model.

Two input formats:

  new     <workdir>/eval_history.jsonl — ordered, phase-tagged, with the decoded parameter vector
          and the panel per evaluation. Everything the marker wants.

  legacy  <dir>/eval_metrics_cache_*.jsonl — the resumability cache. It is a hash-keyed store, so
          read it honestly:
            * it holds only SUCCESSFUL evaluations — failures are invisible, so a replayed marker
              sees a smoother search than really happened;
            * it has no phase tag, so holdout/front-selection re-runs sit in the same file. Rows
              whose `scored_split` is not the calibration split are dropped, and a missing
              `scored_split` is warned about instead of assumed;
            * file order is write order, which is evaluation order — but a resumed run appends
              after a restart, so an interrupted run's order is only within-session;
            * it DOES carry `__kdt__.applied_params`, so parameter width at the marker is
              recoverable here (the handoff expected otherwise).
          Objective losses are rebuilt from the objective names in the run's own report
          (`rerun_conv_report.json` / `kit_report.json`) through the same family→metric mapping the
          kit uses, so the replayed losses are the ones the run optimized.

Output: one row per run — marker evaluation and fraction, status, budget that would have been
saved, which panel metric moved last, and the parameter width at the marker. `--json` writes the
full record. Nothing is written into the run's own directory.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from calibration_kit.convergence import (ConvergenceMarker, Incumbent, auto_window,   # noqa: E402
                                         marker_evidence)
from calibration_kit.objectives import _FAMILY_LOSS                                   # noqa: E402
from calibration_kit.panel import (ALWAYS_RECORDED, FIXED_FALLBACK_TOL, PANEL_DEFAULT,  # noqa: E402
                                   derive_panel)

REPORT_NAMES = ("rerun_conv_report.json", "kit_report.json", "calib_report.json", "report.json")


# ── reading the two formats ──────────────────────────────────────────────────────────
def read_new(workdir: Path, session: str = "last") -> tuple[list, list]:
    """Search-phase records from eval_history.jsonl, ONE optimizer history at a time.

    The log is append-only, so a restarted calibration writes a second run after the first and the
    per-process counter `i` starts again at 0. Concatenating them would replay a history no
    optimizer ever saw — including the cached prefix the resume re-proposed. Sessions are split
    where `i` stops increasing; `session="last"` (the default) replays the most recent one.
    """
    recs, warn = [], []
    for line in (workdir / "eval_history.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict):
            recs.append(r)
    sessions, cur, prev_i = [], [], None
    for r in recs:
        i = r.get("i")
        if prev_i is not None and isinstance(i, int) and i <= prev_i:
            sessions.append(cur); cur = []
        cur.append(r)
        prev_i = i if isinstance(i, int) else prev_i
    if cur:
        sessions.append(cur)
    sessions = [ss for ss in sessions if ss]
    if not sessions:
        return [], warn + ["eval_history.jsonl holds no usable records — nothing to replay"]
    if len(sessions) > 1:
        warn.append(f"{len(sessions)} sessions in this log (the run was restarted); replaying "
                    f"'{session}' — a concatenated history is not one optimizer's search")
    chosen = sessions[-1] if session == "last" else [r for ss in sessions for r in ss]
    search = [r for r in chosen if r.get("phase") == "search"]
    if not search:
        # NO search records is not a search that failed to converge: it is nothing to judge.
        warn.append("no `search`-phase records in this log — nothing to replay")
        return [], warn
    out = []
    for r in search:
        out.append({"losses": [_num(v) for v in (r.get("losses") or [])],
                    "panel": r.get("panel") or {},
                    "x": list((r.get("x") or {}).values()),
                    "names": list((r.get("x") or {}).keys()),
                    "ok": bool(r.get("ok"))})
    return out, warn


def read_legacy(path: Path, objectives: list, split: str = "calibration",
                overrides: dict | None = None) -> tuple[list, list]:
    """Records from an eval_metrics_cache_*.jsonl, with the caveats in the module docstring."""
    warn, out = [], []
    n_all = n_nosplit = n_dropped = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        n_all += 1
        m = row.get("metrics") or {}
        sp = m.get("scored_split")
        if sp is None:
            n_nosplit += 1
        elif str(sp) != split:
            n_dropped += 1
            continue
        x = ((m.get("__kdt__") or {}).get("applied_params") or {})
        panel = {}
        for var in _vars_of(objectives) or [k for k, v in m.items() if isinstance(v, dict) and "nse" in v]:
            got, _flags = derive_panel(m.get(var) if isinstance(m.get(var), dict) else m)
            if got:
                panel[var] = got
        out.append({"losses": _losses_from_metrics(m, objectives, overrides),
                    "panel": panel, "x": [_num(v) for v in x.values()],
                    "names": list(x.keys()), "ok": True})
    if n_nosplit:
        warn.append(f"{n_nosplit}/{n_all} rows carry no `scored_split` — they could be holdout "
                    f"re-runs; they were KEPT, so the replayed tail may be contaminated")
    if n_dropped:
        warn.append(f"dropped {n_dropped}/{n_all} rows scored on another split")
    warn.append("legacy cache: successful evaluations only (failures are not recorded) and no "
                "phase tags — a replayed marker sees a smoother search than the real one")
    warn.append("losses are RECONSTRUCTED from the reported objective names through the kit's "
                "family->metric map (approximate: per-objective weights and any undeclared "
                "determining metric are not in the cache)")
    return out, warn


def _vars_of(objectives) -> list:
    return list(dict.fromkeys(str(o).split(":")[0] for o in (objectives or []) if ":" in str(o)))


def _losses_from_metrics(metrics: dict, objectives: list, overrides: dict | None = None) -> list:
    """Rebuild the loss vector the run optimized, from its objective names.

    APPROXIMATE, and the report says so: the family→metric map is the kit's own, but a run may have
    been scored on a DETERMINING metric other than the family default (e.g. r instead of nse for
    soil moisture) — pass it in `overrides` when the report records it. A non-finite metric is
    +inf, never a good score: `max(0, 1 - nan)` would otherwise read as a perfect fit.
    """
    ov = {str(k).lower(): str(v).lower() for k, v in (overrides or {}).items() if v}
    out = []
    for name in (objectives or []):
        var, _, fam = str(name).partition(":")
        if fam not in _FAMILY_LOSS:
            continue
        key, fn = _FAMILY_LOSS[fam]
        key = ov.get(fam, key)
        mv = metrics.get(var) if isinstance(metrics.get(var), dict) else metrics
        v = (mv or {}).get(key)
        try:
            fv = float(v)
            loss = float(fn(fv)) if math.isfinite(fv) else float("inf")
        except (TypeError, ValueError):
            loss = float("inf")
        out.append(loss if math.isfinite(loss) else float("inf"))
    return out or [float("inf")]


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def find_report(d: Path) -> dict:
    for n in REPORT_NAMES:
        p = d / n
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return {}


# ── the replay itself ────────────────────────────────────────────────────────────────
def _scalarize(records: list) -> None:
    """A SINGLE-objective run minimizes ONE number: the weighted mean over a variable's families.
    The cache does not hold the weights, so the replay uses EQUAL weights and says so — ranking by
    the first family's loss would hold a candidate the search had already beaten (codex round 2).
    """
    for r in records:
        ls = [v for v in (r.get("losses") or []) if v is not None]
        if len(ls) > 1:
            try:
                fin = [float(v) for v in ls]
                r["losses"] = [sum(fin) / len(fin)] if all(math.isfinite(v) for v in fin) \
                    else [float("inf")]
            except (TypeError, ValueError):
                r["losses"] = [float("inf")]


def replay(records: list, *, n_params: int, window=None, rel_gain: float = 0.005,
           tolerances: dict | None = None, panel_metrics=None, multi: bool = False) -> dict:
    metrics = tuple(panel_metrics or (PANEL_DEFAULT + ALWAYS_RECORDED))
    src = "given"
    if not tolerances:
        # No paired series offline, so the fixed fallbacks — and ONLY for metrics this run actually
        # reported. Asking for a metric the runner never emitted (lnnse, alpha on an old runner)
        # would leave the panel permanently short of evidence and no marker could ever fire.
        seen: dict = {}
        for r in records:
            for v, mm in (r.get("panel") or {}).items():
                for m, val in (mm or {}).items():
                    try:
                        if val is None or not math.isfinite(float(val)):
                            continue
                    except (TypeError, ValueError):
                        continue
                    seen.setdefault(v, {}).setdefault(m, 0)
                    seen[v][m] += 1
        n = max(1, len(records))
        tolerances = {v: {m: FIXED_FALLBACK_TOL[m] for m, c in mm.items()
                          if m in FIXED_FALLBACK_TOL and m in metrics and c >= 0.5 * n}
                      for v, mm in seen.items()}
        tolerances = {v: mm for v, mm in tolerances.items() if mm}
        src = "fixed_fallback"
    W = auto_window(n_params) if window in (None, "auto") else int(window)
    scalarized = False
    if not multi and any(len(r.get("losses") or []) > 1 for r in records):
        _scalarize(records)
        scalarized = True
    mk = ConvergenceMarker(tolerances, window=W, rel_gain=rel_gain, mode="observe")
    inc = Incumbent(multi=multi)
    for r in records:
        mk.update(*inc.add(r.get("losses") or [float("inf")], r.get("panel")))
    out = mk.summary(at_cap=None)
    out["tolerance_source"] = src
    out["tolerances"] = tolerances
    if scalarized:
        out["loss_reconstruction"] = "equal_weight_scalarization (the cache holds no weights)"
    out["n_records"] = len(records)
    out["n_params"] = n_params
    if mk.marker is not None and records and records[0].get("x"):
        out["param_width_at_marker"] = _offline_width(records, mk.marker)
    return out


def _offline_width(records: list, marker_i: int, topk_frac: float = 0.10) -> dict:
    """Parameter width of the top set at the marker, normalized by the RANGE SEEN in the records.
    The contract's search box is not in the cache and the seen range is at most that box, so this
    OVERSTATES the normalized width — it is an upper bound, not a lower one."""
    try:
        import numpy as np
        rows = [r for r in records[:marker_i + 1] if r.get("x")]
        if len(rows) < 5:
            return {}
        X = np.asarray([r["x"] for r in rows], float)
        L = np.asarray([(r["losses"] or [np.inf])[0] for r in rows], float)
        fin = np.isfinite(L)
        if fin.sum() < 5:
            return {}
        order = np.where(fin)[0][np.argsort(L[fin])]
        k = max(5, int(round(topk_frac * len(order))))
        top = X[order[:k]]
        seen = X.max(0) - X.min(0)
        seen[seen == 0] = 1.0
        w = (top.max(0) - top.min(0)) / seen
        return {"k": int(k), "max_normalized_range": round(float(w.max()), 4),
                "per_param": [round(float(v), 3) for v in w],
                "names": rows[0].get("names"),
                "note": "normalized by the range SEEN in the cache, which is at most the "
                        "contract's search box — so this is an UPPER bound on the true normalized "
                        "width, not a lower one"}
    except Exception:
        return {}


def replay_dir(d: Path, *, window=None, rel_gain=0.005, split="calibration",
               session="last") -> list:
    """Every run under `d`: a new-format workdir, or one row per legacy cache file."""
    rows = []
    rep = find_report(d)
    objectives = rep.get("objectives") or []
    multi = bool(rep.get("multi_objective"))
    n_params = len(rep.get("best_params") or rep.get("best_x") or []) or 1
    overrides = (rep.get("metric_overrides") or rep.get("determining_metric_by_family") or {})
    if (d / "eval_history.jsonl").exists():
        recs, warn = read_new(d, session=session)
        if not recs:
            rows.append({"run": str(d), "format": "new", "warnings": warn,
                         "status": "not_observed", "n_records": 0})
        else:
            r = replay(recs, n_params=n_params, window=window, rel_gain=rel_gain, multi=multi)
            rows.append({"run": str(d), "format": "new", "warnings": warn,
                         "reported_n_evaluations": rep.get("n_evaluations"), **r})
    for f in sorted(d.glob("eval_metrics_cache_*.jsonl")):
        recs, warn = read_legacy(f, objectives, split=split, overrides=overrides)
        if not recs:
            rows.append({"run": str(f), "format": "legacy", "warnings": warn + ["no usable rows"]})
            continue
        np_ = len(recs[0]["x"]) or n_params
        r = replay(recs, n_params=np_, window=window, rel_gain=rel_gain, multi=multi)
        rows.append({"run": str(f), "format": "legacy", "warnings": warn,
                     "objectives": objectives, "algorithm": rep.get("algorithm"),
                     "reported_n_evaluations": rep.get("n_evaluations"),
                     "reported_stop": rep.get("stop"), **r})
    return rows


def _fmt(v, n=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run directories (workdir, or a dir with legacy caches)")
    ap.add_argument("--window", default="auto", help="marker window in evaluations (default auto)")
    ap.add_argument("--rel-gain", type=float, default=0.005)
    ap.add_argument("--split", default="calibration", help="legacy: which scored_split to keep")
    ap.add_argument("--session", default="last", choices=("last", "all"),
                    help="new format: which appended run to replay (default the most recent)")
    ap.add_argument("--json", dest="json_out", default=None, help="write the full records here")
    a = ap.parse_args(argv)

    rows = []
    for p in a.paths:
        d = Path(p)
        if not d.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        rows.extend(replay_dir(d, window=(None if a.window == "auto" else int(a.window)),
                               rel_gain=a.rel_gain, split=a.split, session=a.session))

    hdr = f"{'run':<34} {'fmt':<7} {'evals':>6} {'W':>5} {'marker':>7} {'frac':>6} {'saved':>6} {'status':<22} last-moving"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        last = "-"
        lc = r.get("last_change") or {}
        pairs = [(v, m, i) for v, mm in lc.items() for m, i in (mm or {}).items() if i is not None]
        if pairs:
            v, m, i = max(pairs, key=lambda t: t[2])
            last = f"{v}:{m}@{i}"
        print(f"{Path(r['run']).parent.name + '/' + Path(r['run']).name:<34.34} "
              f"{r.get('format','-'):<7} {_fmt(r.get('n_records'),0):>6} {_fmt(r.get('window'),0):>5} "
              f"{_fmt(r.get('marker_eval'),0):>7} {_fmt(r.get('marker_fraction'),2):>6} "
              f"{_fmt(r.get('budget_saved_fraction'),2):>6} {str(r.get('status','-')):<22} {last}")
    for r in rows:
        for w in (r.get("warnings") or []):
            print(f"  note [{Path(r['run']).name}]: {w}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nfull records -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
