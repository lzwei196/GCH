"""Metric PANEL — judge a calibration on several metrics, never on NSE alone (design §1.4, §2.11).

Why a panel: NSE is one number that mixes correlation, variability and bias, and it hides bias
when variability is high (Gupta et al. 2009, doi:10.1016/j.jhydrol.2009.08.003). A search can sit
on a flat NSE while the bias ratio is still drifting — so "NSE stopped moving" is not "the fit
stopped moving". The panel splits the fit into parts that move independently:

    r      correlation (shape/timing)
    alpha  variability ratio  sd(sim)/sd(obs)            (population sd, ddof 0)
    beta   bias ratio         mean(sim)/mean(obs)
    pbias  100*(beta-1), the field's usual bias number (percent)
    nrmse  100*RMSE/|mean(obs)| (percent)
    lnnse  NSE of log(x + 0.01*mean(obs)) — low-flow weighted (Krause et al. 2005,
           doi:10.5194/adgeo-5-89-2005); only for variables of kind `flow`
    nse, kge  recorded, never required (they combine r, alpha, beta)

WHICH metrics a variable has depends on its KIND (design §1.4, one table for the panel and the
protect load check): flow / series / snapshot / categorical. The kind comes from the contract
(`strategy.convergence.variables.<var>.kind`) or, if absent, from the matched dag entry.

KIT metrics vs KI metrics: several KIs use the names `nrmse` / `pbias` for other numbers (NRMSE by
sd(obs), PBIAS in days or mm). The panel always has the kit's definitions and reads them from a
kit-owned block `__kdt__.panel.<var>` (written by the workflow with the kit's formulas, schema C14; keys read
case-insensitively). Beside a block, only the runner's plain `nse`, `kge` and `r` are read, and only where the
block lacks them (their names and formulas are the kit's; the 2n check compares them). Without a block the
runner's plain keys are read — never plain `nrmse`, and plain `pbias` only when no KI entry of that variable
gives pbias a unit other than percent.

TOLERANCES answer "how much movement is real?" (§2.11). A metric computed from a finite record has
sampling noise (Clark et al. 2021, doi:10.1029/2020WR029001); a change smaller than that noise is
not improvement. `bootstrap_tolerances` measures it with a moving-block bootstrap of the pilot's
default-run series. Without series (or per metric when the bootstrap cannot measure it), fixed
fallbacks are used and LABELLED `fixed_fallback`.
"""
from __future__ import annotations
import json
import math

# ── the one table (design §1.4) ───────────────────────────────────────────────────────────────
#: kind -> required metrics (settle test) and recorded metrics (panel record; what may be protected)
KIND_TABLE = {
    "flow":        {"required": ("r", "alpha", "beta", "lnnse"),
                    "recorded": ("r", "alpha", "beta", "pbias", "lnnse", "nse", "kge", "nrmse")},
    "series":      {"required": ("r", "alpha", "beta"),
                    "recorded": ("r", "alpha", "beta", "pbias", "nse", "kge", "nrmse")},
    "snapshot":    {"required": ("pbias", "nrmse"),
                    "recorded": ("pbias", "nrmse")},
    "categorical": {"required": (), "recorded": ()},
}
KINDS = tuple(KIND_TABLE)
#: the kit-owned keys of a __kdt__.panel.<var> block
BLOCK_KEYS = ("r", "alpha", "beta", "pbias", "nse", "kge", "lnnse", "nrmse")
#: plain runner keys read for a variable WITHOUT a block (never plain nrmse; pbias only if percent)
PLAIN_KEYS = ("r", "alpha", "beta", "nse", "kge", "lnnse")
#: metrics that divide by mean(obs) and go missing when the mean is near zero (Leo, 2026-09-29)
MEAN_DIVIDED = ("beta", "pbias", "nrmse")
#: |mean(obs)| < MEAN_NEAR_ZERO * sd(obs) -> "mean near zero" (provisional cut-off, design §1.4)
MEAN_NEAR_ZERO = 0.1
#: log offset for lnnse, as a fraction of mean(obs)
LN_EPS_FRAC = 0.01

# kept for older callers (the replay tool is migrated in step 9); the panel itself uses KIND_TABLE
PANEL_DEFAULT = ("r", "alpha", "beta", "pbias", "lnnse")
ALWAYS_RECORDED = ("nse", "kge")

#: fixed fallbacks (design §2.11): 0.01 for ratios and efficiencies, 1 percentage point for percents
FIXED_FALLBACK_TOL = {"r": 0.01, "alpha": 0.01, "beta": 0.01, "nse": 0.01, "kge": 0.01,
                      "lnnse": 0.01, "pbias": 1.0, "nrmse": 1.0}

_PERCENT_UNITS = {"percent", "%", "pct", "percentage", "per cent"}


def _num(v):
    """A finite float, else None (bools are not numbers here)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# ── kind ──────────────────────────────────────────────────────────────────────────────────────
def default_kind(obs_shape: str | None, comparison_mode: str | None = None) -> str | None:
    """Kind from the matched dag entry (design §1.4): categorical comparisons first, whatever
    their obs_shape; then *_time_series -> series, *_snapshot -> snapshot. None if unknown."""
    if (comparison_mode or "") == "categorical_event_comparison" or (obs_shape or "") == "categorical_event":
        return "categorical"
    s = obs_shape or ""
    if s.endswith("_time_series"):
        return "series"
    if s.endswith("_snapshot"):
        return "snapshot"
    return None


def _fits(stated: str, dflt: str | None) -> bool:
    if dflt is None:
        return True
    if stated == dflt:
        return True
    return stated == "flow" and dflt == "series"      # flow is a series the KI marks as discharge-type


def resolve_kind(var: str, dag: dict | None, obs_shape: str | None, stated: str | None = None):
    """(kind, warnings) for one target. `stated` = the contract's kind; else the dag default.

    The dag entry is the output's comparable_obs_shapes item whose obs_shape matches. A stated kind
    that does not fit that entry (e.g. `series` on a `*_snapshot`) is used but WARNED about.
    An unknown stated kind is an error (the contract is wrong, not the data)."""
    warns: list[str] = []
    entry = None
    for o in (dag or {}).get("outputs") or []:
        if isinstance(o, dict) and o.get("var") == var:
            for s in (o.get("observability") or {}).get("comparable_obs_shapes") or []:
                if isinstance(s, dict) and s.get("obs_shape") == obs_shape:
                    entry = s
                    break
            break
    dflt = default_kind(obs_shape, (entry or {}).get("comparison_mode"))
    if stated is not None:
        stated = str(stated).strip().lower()
        if stated not in KIND_TABLE:
            raise ValueError(f"strategy.convergence.variables.{var}.kind = {stated!r}; "
                             f"must be one of {list(KIND_TABLE)}")
        if not _fits(stated, dflt):
            warns.append(f"{var}: stated kind {stated!r} does not fit the dag entry "
                         f"(obs_shape {obs_shape!r} -> {dflt!r})")
        return stated, warns
    if dflt is None:
        warns.append(f"{var}: no kind stated and obs_shape {obs_shape!r} gives none; using 'series'")
        return "series", warns
    return dflt, warns


def pbias_is_percent(convention: dict | None, var: str) -> bool:
    """False when ANY KI validation-convention entry of `var` gives `pbias` a unit other than
    percent (e.g. days, mm, m, K). A missing unit is not "other than percent" (design §1.4)."""
    return pbias_other_unit(convention, var) is None


def pbias_other_unit(convention: dict | None, var: str) -> str | None:
    """The first non-percent unit a KI entry of `var` gives `pbias`, else None."""
    for e in ((convention or {}).get("validation") or []):
        if not isinstance(e, dict) or e.get("dag_variable") != var:
            continue
        for grp in ("headline_metrics", "secondary_metrics"):
            for m in e.get(grp) or []:
                if isinstance(m, dict) and str(m.get("metric", "")).lower() == "pbias":
                    u = m.get("unit")
                    if u is not None and str(u).strip() and str(u).strip().lower() not in _PERCENT_UNITS:
                        return str(u).strip()
    return None


# ── metrics from a paired series ──────────────────────────────────────────────────────────────
def _finite_pairs(sim, obs, dates=None):
    """Valid pairs = both finite, in date order when dates are given (design §2.11)."""
    import numpy as np
    s = np.asarray(sim, float).ravel(); o = np.asarray(obs, float).ravel()
    if len(s) != len(o):
        raise ValueError(f"sim has {len(s)} values but obs has {len(o)}")
    n = len(s)
    if dates is not None:
        d = np.asarray(dates).ravel()
        if len(d) != n:
            raise ValueError(f"series has {n} pairs but {len(d)} dates")
        order = np.argsort(d, kind="stable")
        s, o = s[order], o[order]
    m = np.isfinite(s) & np.isfinite(o)
    return s[m], o[m]


def mean_near_zero(obs) -> bool:
    """|mean(obs)| < MEAN_NEAR_ZERO * sd(obs) (population sd). A constant record is not 'near
    zero' by this test (sd 0); its metrics are undefined for other reasons."""
    import numpy as np
    o = np.asarray(obs, float)
    o = o[np.isfinite(o)]
    if len(o) < 2:
        return False
    sd = float(o.std())
    return sd > 0 and abs(float(o.mean())) < MEAN_NEAR_ZERO * sd


def _metrics_from_pairs(s, o, kind: str) -> dict:
    """Kit formulas on already-cleaned pairs; None where a metric is undefined."""
    import numpy as np
    out: dict = {}
    so, oo = float(s.std()), float(o.std())
    sm, om = float(s.mean()), float(o.mean())
    out["r"] = float(np.corrcoef(s, o)[0, 1]) if so > 0 and oo > 0 and len(s) >= 3 else None
    out["alpha"] = (so / oo) if oo > 0 else None
    out["beta"] = (sm / om) if om != 0 else None
    out["pbias"] = (100.0 * (out["beta"] - 1.0)) if out["beta"] is not None else None
    denom = float(np.sum((o - om) ** 2))
    out["nse"] = float(1.0 - np.sum((s - o) ** 2) / denom) if denom > 0 else None
    if None not in (out["r"], out["alpha"], out["beta"]):
        out["kge"] = float(1.0 - math.sqrt((out["r"] - 1.0) ** 2 + (out["alpha"] - 1.0) ** 2
                                           + (out["beta"] - 1.0) ** 2))
    else:
        out["kge"] = None
    rmse = float(math.sqrt(np.mean((s - o) ** 2)))
    out["nrmse"] = (100.0 * rmse / abs(om)) if om != 0 else None
    out["lnnse"] = None
    if kind == "flow" and om > 0:
        eps = LN_EPS_FRAC * om
        if float(s.min()) + eps > 0 and float(o.min()) + eps > 0:
            ls, lo = np.log(s + eps), np.log(o + eps)
            ldenom = float(np.sum((lo - lo.mean()) ** 2))
            out["lnnse"] = float(1.0 - np.sum((ls - lo) ** 2) / ldenom) if ldenom > 0 else None
    return out


def panel_from_series(sim, obs, kind: str = "series", dates=None, mean_rule: bool = True) -> dict:
    """The panel for ONE variable from its paired (sim, obs) series, with the kit's formulas,
    restricted to the kind's recorded metrics. Undefined metrics are left out.

    `mean_rule`: when |mean(obs)| < 0.1*sd(obs), beta / pbias / nrmse are left out and the reason
    is given under "_why" (design §1.4). The bootstrap passes False (the rule is decided once, at
    the pilot's default run, never per resample)."""
    if kind not in KIND_TABLE:
        raise ValueError(f"unknown kind {kind!r}")
    s, o = _finite_pairs(sim, obs, dates)
    out: dict = {"n": int(len(s))}
    # a snapshot (e.g. one season's yield) has PBIAS and NRMSE from a single pair; series metrics
    # (r, alpha, NSE ...) need at least 3
    if len(s) < (1 if kind == "snapshot" else 3) or not KIND_TABLE[kind]["recorded"]:
        return out
    got = _metrics_from_pairs(s, o, kind)
    why: dict = {}
    if mean_rule and mean_near_zero(o):
        for m in MEAN_DIVIDED:
            if m in KIND_TABLE[kind]["recorded"]:
                got[m] = None
                why[m] = "mean near zero"
    for m in KIND_TABLE[kind]["recorded"]:
        if got.get(m) is not None:
            out[m] = got[m]
    if why:
        out["_why"] = why
    return out


def panel_block(series_by_var: dict, kinds: dict) -> dict:
    """What a runner adds to its metrics payload (design §1.4, gap 2n): the kit-owned block
    {"__kdt__": {"panel": {var: {...}}}} with the kit formulas. The helper never writes plain
    keys, so a KI's own `nrmse` / `pbias` under the plain names is never overwritten.

    `series_by_var` = {var: (sim, obs) or (sim, obs, dates)}; `kinds` = {var: kind}."""
    block: dict = {}
    for var, ser in (series_by_var or {}).items():
        sim, obs = ser[0], ser[1]
        dates = ser[2] if len(ser) > 2 else None
        # all eight keys wherever defined, whatever the runner thinks the kind is: the kit picks
        # what the variable's kind needs (a runner cannot see the contract's kind)
        s, o = _finite_pairs(sim, obs, dates)
        rec: dict = {"n": int(len(s))}
        if len(s) >= 1:
            got = _metrics_from_pairs(s, o, "flow")
            if mean_near_zero(o):
                rec["_why"] = {}
                for m in MEAN_DIVIDED:
                    got[m] = None
                    rec["_why"][m] = "mean near zero"
            rec.update({k: v for k, v in got.items() if v is not None and k in BLOCK_KEYS})
        block[str(var)] = rec
    return {"__kdt__": {"panel": block}}


def merge_panel_block(metrics: dict, block: dict) -> dict:
    """Add a `panel_block` result to a runner's metrics WITHOUT replacing its other __kdt__ keys
    (applied_params, case_id, split, series). Returns `metrics`, changed in place."""
    kdt = metrics.setdefault("__kdt__", {})
    kdt.setdefault("panel", {}).update(((block or {}).get("__kdt__") or {}).get("panel") or {})
    return metrics


def load_series(path) -> tuple:
    """Read a runner-written paired series file (.npz with `sim`, `obs`, optional `date`).
    Returns (sim, obs, dates_or_None)."""
    import numpy as np
    with np.load(str(path), allow_pickle=False) as z:
        dates = np.asarray(z["date"]) if "date" in z.files else None
        return np.asarray(z["sim"], float), np.asarray(z["obs"], float), dates


# ── metrics from a runner payload ─────────────────────────────────────────────────────────────
def derive_panel(metrics_for_var: dict, pbias_percent: bool = True) -> tuple[dict, list[str]]:
    """Panel values from a runner's PLAIN keys for one variable (no block). Returns (panel, flags).

    Never reads plain `nrmse` (KIs define it several ways). Reads plain `pbias` only when
    `pbias_percent`. beta from pbias is exact (beta = 1 + pbias/100) and allowed; alpha from KGE
    alone has an unknown sign and is NOT derived (design §1.4; exact reconstruction across a
    history is alpha_reconstruct.py)."""
    m = {str(k).lower(): v for k, v in (metrics_for_var or {}).items()}
    flags: list[str] = []
    out: dict = {}
    for k in PLAIN_KEYS:
        v = _num(m.get(k))
        if v is not None:
            out[k] = v
    if pbias_percent:
        v = _num(m.get("pbias"))
        if v is not None:
            out["pbias"] = v
    elif _num(m.get("pbias")) is not None:
        flags.append("pbias_not_percent_ignored")
    if "beta" not in out and "pbias" in out:
        out["beta"] = 1.0 + out["pbias"] / 100.0
        flags.append("beta_from_pbias")
    if "pbias" not in out and "beta" in out:
        out["pbias"] = 100.0 * (out["beta"] - 1.0)
        flags.append("pbias_from_beta")
    return out, flags


class Panel:
    """Turns a runner's metrics payload into {var: {metric: value}} for the history log, and says
    which metrics each variable REQUIRES (design §1.4).

    Reading order per variable: the kit-owned block `__kdt__.panel.<var>` if present (plus the plain
    nse / kge / r it lacks, if the kind records them, flagged `headline_from_plain`); otherwise the runner's plain keys under the
    variable's own name (`payload[var]`); a FLAT payload counts
    only when exactly one variable is declared (flagged `flat_single_variable`) — unscoped metrics
    are never copied to several variables.

    Kit-set missing metrics (mean near zero, runner/kit mismatch — decided once, at the pilot)
    leave the variable's required set; `kit_missing[var][metric]` holds the reason.
    `metrics` (the old contract-wide panel list) is accepted and ignored: the panel comes from
    each variable's kind now.
    """

    def __init__(self, variables, metrics: tuple | list | None = None, kinds: dict | None = None,
                 pbias_percent: dict | None = None, pbias_unit: dict | None = None):
        self.variables = [str(v) for v in (variables or [])]
        self.kinds = {v: (kinds or {}).get(v, "series") for v in self.variables}
        for v, k in self.kinds.items():
            if k not in KIND_TABLE:
                raise ValueError(f"{v}: unknown kind {k!r}")
        self.pbias_percent = {v: bool((pbias_percent or {}).get(v, True)) for v in self.variables}
        self.pbias_unit = {v: (pbias_unit or {}).get(v) for v in self.variables}
        self.ignored_panel_list = list(metrics) if metrics else None
        self.kit_missing: dict = {v: {} for v in self.variables}
        self.flags: dict = {}
        # union of recorded metrics, for callers that want one list
        self.metrics = tuple(dict.fromkeys(m for v in self.variables
                                           for m in KIND_TABLE[self.kinds[v]]["recorded"]))

    def apply_pilot(self, decisions: dict) -> list[str]:
        """Apply `pilot_decisions` (kinds after the flow->series check, kit-set missing metrics).
        Returns the notes, for the report."""
        for v, k in ((decisions or {}).get("kinds") or {}).items():
            if v in self.kinds:
                self.kinds[v] = k
        for v, mm in ((decisions or {}).get("kit_missing") or {}).items():
            for m, why in mm.items():
                if v in self.kinds:
                    self.set_kit_missing(v, m, why)
        self.metrics = tuple(dict.fromkeys(m for v in self.variables
                                           for m in KIND_TABLE[self.kinds[v]]["recorded"]))
        return list((decisions or {}).get("notes") or [])

    # what each variable records / requires
    def recorded(self, var: str) -> tuple:
        return KIND_TABLE[self.kinds[var]]["recorded"]

    def required(self, var: str) -> tuple:
        km = self.kit_missing.get(var) or {}
        return tuple(m for m in KIND_TABLE[self.kinds[var]]["required"] if m not in km)

    def set_kit_missing(self, var: str, metric: str, reason: str) -> None:
        """Mark a metric missing for the whole search, for a kit reason. "mean near zero" wins
        over any other reason (design §1.4)."""
        cur = self.kit_missing.setdefault(var, {})
        if cur.get(metric) == "mean near zero":
            return
        cur[metric] = reason

    def capped(self, var: str) -> bool:
        """True when a REQUIRED metric of `var` is kit-set missing (the unknown cap, §2.7)."""
        km = self.kit_missing.get(var) or {}
        return any(m in km for m in KIND_TABLE[self.kinds[var]]["required"])

    def missing_required(self, var: str, record: dict | None) -> list[str]:
        rec = record or {}
        return [m for m in self.required(var) if _num(rec.get(m)) is None]

    def _flag(self, var, flags):
        if flags:
            # the SET of flags for this variable — extract() runs once per evaluation
            self.flags[var] = sorted(set(self.flags.get(var, [])) | set(flags))

    def extract(self, metrics: dict) -> dict:
        payload = metrics if isinstance(metrics, dict) else {}
        kdt = payload.get("__kdt__") if isinstance(payload.get("__kdt__"), dict) else {}
        blocks = kdt.get("panel") if isinstance(kdt.get("panel"), dict) else {}
        out = {}
        for var in self.variables:
            rec = self.recorded(var)
            if not rec:
                continue
            blk = blocks.get(var)
            if isinstance(blk, dict):
                # NSE / lnNSE as the scorer spells them; an exact lower-case key wins over another spelling
                blk = {**{str(k).lower(): v for k, v in blk.items() if str(k) != str(k).lower()},
                       **{k: v for k, v in blk.items() if str(k) == str(k).lower()}}
                got = {k: _num(blk.get(k)) for k in BLOCK_KEYS}
                got = {k: v for k, v in got.items() if v is not None}
                flags = ["from_block"]
                # a reserved section without NSE / KGE / r: the runner's own plain ones (their names and formulas
                # are the kit's; the 2n check compares them) — so a protected NSE is never lost (Opus r5 #1)
                _pl = payload.get(var) if isinstance(payload.get(var), dict) else (
                    payload if len(self.variables) == 1 else {})
                _pl = {str(a).lower(): b for a, b in (_pl or {}).items() if not str(a).startswith("__")}
                for _h in ("nse", "kge", "r"):
                    if _h not in got and _h in rec and _num(_pl.get(_h)) is not None:
                        got[_h] = _num(_pl.get(_h))
                        if "headline_from_plain" not in flags:
                            flags.append("headline_from_plain")
            else:
                src = payload.get(var)
                flags = []
                if not isinstance(src, dict):
                    if len(self.variables) == 1:
                        src = payload
                        flags.append("flat_single_variable")
                    else:
                        src = {}
                km = self.kit_missing.get(var) or {}
                # a kit-set missing value is never used to derive another (beta <-> pbias)
                src = {k: v for k, v in (src or {}).items() if str(k).lower() not in km}
                got, f2 = derive_panel(src, self.pbias_percent.get(var, True))
                if "pbias_not_percent_ignored" in f2 and "beta" not in got:
                    f2.append(f"runner's pbias is in {self.pbias_unit.get(var) or 'another unit'}; "
                              f"no beta given")
                flags += [f for f in f2 if not (f == "pbias_from_beta" and "pbias" in km)
                          and not (f == "beta_from_pbias" and "beta" in km)]
            for m in (self.kit_missing.get(var) or {}):
                got.pop(m, None)
            keep = {m: got[m] for m in rec if m in got}
            self._flag(var, flags)
            if keep:
                out[var] = keep
        return out


# ── pilot decisions (design §1.4: decided once, at the pilot's default run) ───────────────────
def pilot_decisions(series_by_var: dict, kinds: dict) -> dict:
    """From the pilot DEFAULT run's series: {"kinds": {var: kind}, "kit_missing": {var: {m: why}},
    "notes": [...]}.

    * a `flow` variable whose default-run sim or obs has a value <= -0.01*mean(obs) (log undefined)
      is judged as `series` (log-flow NSE dropped) — the run still goes ahead;
    * mean near zero -> beta, pbias, nrmse kit-set missing ("mean near zero").
    Variables with no series keep their kind and get no kit-set missing metrics here."""
    import numpy as np
    new_kinds = dict(kinds or {})
    km: dict = {}
    notes: list[str] = []
    for var, ser in (series_by_var or {}).items():
        k = new_kinds.get(var, "series")
        try:
            sim, obs = ser[0], ser[1]
            dates = ser[2] if len(ser) > 2 else None
            s, o = _finite_pairs(sim, obs, dates)
        except Exception as e:
            notes.append(f"{var}: pilot series not usable ({type(e).__name__}); no pilot decision")
            continue
        if len(o) < (1 if k == "snapshot" else 3):
            continue
        om = float(o.mean())
        if k == "flow":
            lim = -LN_EPS_FRAC * om
            if om <= 0 or float(s.min()) <= lim or float(o.min()) <= lim:
                new_kinds[var] = "series"
                k = "series"
                cause = ("mean(obs) <= 0" if om <= 0 else
                         "default-run sim has a value <= -0.01*mean(obs)" if float(s.min()) <= lim else
                         "obs has a value <= -0.01*mean(obs)")
                notes.append(f"{var}: judged as series (log-flow NSE dropped): {cause}, so the log "
                             f"is undefined")
        if mean_near_zero(o):
            for m in MEAN_DIVIDED:
                if m in KIND_TABLE[k]["recorded"]:
                    km.setdefault(var, {})[m] = "mean near zero"
    return {"kinds": new_kinds, "kit_missing": km, "notes": notes}


def pilot_decisions_from_payload(payload: dict | None, kinds: dict, variables) -> dict:
    """For variables WITHOUT a series: the kit-set missing metrics a workflow recorded in
    the default run's block (`__kdt__.panel.<var>._why`, e.g. "mean near zero"). Same shape as
    `pilot_decisions`; kinds are unchanged (the flow check needs the series)."""
    km: dict = {}
    kdt = (payload or {}).get("__kdt__") if isinstance((payload or {}).get("__kdt__"), dict) else {}
    blocks = kdt.get("panel") if isinstance(kdt.get("panel"), dict) else {}
    for var in variables or []:
        blk = blocks.get(var)
        why = blk.get("_why") if isinstance(blk, dict) else None
        if not isinstance(why, dict):
            continue
        k = (kinds or {}).get(var, "series")
        for m, reason in why.items():
            # only the kit's own reason, and only for the mean-divided metrics of the kind: a
            # hand-written block cannot free a required metric by naming some other reason
            if reason == "mean near zero" and m in MEAN_DIVIDED and m in KIND_TABLE[k]["recorded"]:
                km.setdefault(var, {})[m] = "mean near zero"
    return {"kinds": dict(kinds or {}), "kit_missing": km, "notes": []}


# ── the runner/kit cross-check (gap 2n: decided once, at the pilot's default run) ─────────────
#: KI headline metrics whose name and definition are the kit's; every other plain key is not checked
HEADLINE_SAME = ("nse", "kge", "r")
#: added to the recorded-precision bound
CROSS_CHECK_ABS = 1e-6


def recorded_precision(v) -> float:
    """Half a unit of the last decimal the value was written with (its shortest repr, which is what
    json wrote), e.g. 0.8123 -> 5e-05. A full-precision float gives a bound near 1e-17."""
    from decimal import Decimal
    try:
        exp = Decimal(repr(float(v))).as_tuple().exponent
    except Exception:
        return 0.0
    return 0.5 * 10.0 ** exp if isinstance(exp, int) else 0.0


def cross_check_2n(series_by_var: dict, default_metrics: dict | None, kinds: dict,
                   variables, pbias_percent: dict | None = None, precision_from: dict | None = None,
                   no_series_why: dict | None = None) -> dict:
    """At the pilot's DEFAULT run: recompute each variable's panel from the runner's series and
    compare it with what the runner reported (design gap 2n, Leo 2026-09-29).

    * EVERY value the panel reads is checked: the kit block `__kdt__.panel.<var>` when there is one,
      otherwise the runner's plain keys the panel uses (r, alpha, beta, nse, kge, lnnse, and pbias when
      it is in percent). A difference above the recorded precision + 1e-6 makes that metric kit-set
      MISSING for the whole search ("2n mismatch: runner …, kit …").
    * Beside a block, a plain NSE / KGE / r the block lacks is a PANEL value (the panel takes it) if the
      variable's kind records it, checked as above and marked "from the runner's plain key". A mismatch on a score
      the variable's kind never uses is reported only ("not used for this kind"). A plain NSE / KGE / r the block also has is the KI's
      headline value: compared, a disagreement reported only (the panel reads the block; the standards check
      reads the KI's value). Every other plain key is listed "not checked (the KI may define it differently)".
    The recorded precision is PER METRIC (Leo, 2026-09-30): half a unit of the finest decimal the runner
    wrote for THAT metric, over the default run and the pilot's other runs (`precision_from` =
    {var: {metric: [values]}}) — a runner that writes PBIAS to 1 decimal and the ratios to 4 is judged
    at 0.05 and 0.00005 respectively; a lone 1.0 among 4-decimal values of the same metric is judged
    at 4 decimals. When the runner's PBIAS fails and it wrote no beta of its own, beta (the panel would
    work it out from that PBIAS) is missing too, with that reason (Leo, 2026-09-30).
    Returns {"kinds": {}, "kit_missing": {var: {m: why}}, "notes": [...], "checks": {var: {...}}}
    in the shape `Panel.apply_pilot` takes. Variables without a series are "not checked (no series)"."""
    payload = default_metrics if isinstance(default_metrics, dict) else {}
    kdt = payload.get("__kdt__") if isinstance(payload.get("__kdt__"), dict) else {}
    blocks = kdt.get("panel") if isinstance(kdt.get("panel"), dict) else {}
    variables = [str(v) for v in (variables or [])]
    km: dict = {}
    notes: list[str] = []
    checks: dict = {}

    def _prec(var, m, rv):
        vals = [rv] + [v for v in (((precision_from or {}).get(var) or {}).get(m) or []) if _num(v) is not None]
        ps = [recorded_precision(v) for v in vals if v is not None]
        return min(ps) if ps else 0.0

    def _cmp(rv, kv, prec):
        if kv is None or (isinstance(kv, float) and not math.isfinite(kv)):
            # the kit's formula is UNDEFINED on this series (e.g. r with constant obs): a value the runner reports
            # for it cannot be right — it is missing like a mismatch (codex step 9 r1 #2)
            return ({"runner": rv, "kit": None, "bound": None, "status": "mismatch (undefined on the series)"},
                    f"2n mismatch: runner {rv!r}, kit undefined on the series")
        bound = prec + CROSS_CHECK_ABS
        ok = abs(rv - float(kv)) <= bound
        return ({"runner": rv, "kit": float(kv), "bound": bound, "status": "agrees" if ok else "mismatch"},
                None if ok else f"2n mismatch: runner {rv!r}, kit {float(kv)!r}")

    for var in variables:
        ser = (series_by_var or {}).get(var)
        if ser is None:
            checks[var] = {"status": f"not checked ({(no_series_why or {}).get(var) or 'no series'})"}
            continue
        k = (kinds or {}).get(var, "series")
        try:
            dts = ser[2] if len(ser) > 2 else None
            kit = panel_from_series(ser[0], ser[1], k if k in KIND_TABLE else "series", dts)
            if k in KIND_TABLE and KIND_TABLE[k]["recorded"]:
                # the block has all eight keys whatever the kind; compare those the kit can compute
                kit = {**_metrics_from_pairs(*_finite_pairs(ser[0], ser[1], dts), "flow"), **kit}
        except Exception as e:
            checks[var] = {"status": f"not checked (series unusable: {type(e).__name__})"}
            continue
        plain = payload.get(var)
        if not isinstance(plain, dict):
            plain = payload if len(variables) == 1 else {}
        plain = {str(a).lower(): b for a, b in plain.items() if not str(a).startswith("__")}
        blk = blocks.get(var)
        row: dict = {"metrics": {}}
        if isinstance(blk, dict):
            row["source"] = "block"
            # exact lower-case keys win over other spellings of the same score (Opus 8b/9 r6 nit)
            panel_src = {str(k).lower(): v for k, v in blk.items() if str(k) != str(k).lower()}
            panel_src.update({k: v for k, v in blk.items() if str(k) == str(k).lower()})
            # the plain NSE / KGE / r the panel takes when the section lacks them are panel values too
            _filled = set()
            _rec_k = KIND_TABLE.get(k if k in KIND_TABLE else "series", {}).get("recorded", ())
            for _h in HEADLINE_SAME:                   # only what this kind records is ever taken (Opus r8 #2)
                if _h in _rec_k and _num(panel_src.get(_h)) is None and _num(plain.get(_h)) is not None:
                    panel_src[_h] = plain[_h]
                    _filled.add(_h)
            panel_names = BLOCK_KEYS
        else:
            row["source"] = "plain"
            _filled = set()
            panel_src = plain
            panel_names = PLAIN_KEYS + (("pbias",) if (pbias_percent or {}).get(var, True) else ())
        for m in panel_names:                          # the values the panel reads: a mismatch is missing
            rv = _num(panel_src.get(m))
            if rv is None:
                continue
            rec, why = _cmp(rv, kit.get(m), _prec(var, m, rv))
            if m in _filled:
                rec["from"] = "the runner's plain key (the block lacks it)"
            if why and k in KIND_TABLE and m not in KIND_TABLE[k]["recorded"]:
                # a score this kind never uses (e.g. NSE for a snapshot target): reported, never "missing for the
                # search" (Opus 8b/9 r9 nit)
                rec["status"] = "mismatch (not used for this kind)"
                notes.append(f"{var}:{m} {why} ({'bound %.3g' % rec['bound'] if rec.get('bound') is not None else 'undefined on the series'}); not used for kind {k}")
                why = None
            row["metrics"][m] = rec
            if why:
                km.setdefault(var, {})[m] = why
                notes.append(f"{var}:{m} {why} ({'bound %.3g' % rec['bound'] if rec.get('bound') is not None else 'undefined on the series'}); missing for the search")
        if (row["source"] == "plain" and "pbias" in (km.get(var) or {})
                and _num(panel_src.get("beta")) is None
                and "beta" in KIND_TABLE.get(k if k in KIND_TABLE else "series", {}).get("recorded", ())):
            # the panel would work beta out from this PBIAS (beta = 1 + PBIAS/100): it is missing too
            km[var]["beta"] = f"worked out from PBIAS, which failed the 2n check ({km[var]['pbias']})"
            notes.append(f"{var}:beta missing for the search: the runner gives no beta and its PBIAS failed "
                         f"the 2n check")
        if row["source"] == "block":                   # the KI's own headline values beside the block
            head = {}
            for m in HEADLINE_SAME:
                rv = _num(plain.get(m))
                if rv is not None and m not in _filled:        # a filled-in one was compared as a panel value
                    rec, why = _cmp(rv, kit.get(m), recorded_precision(rv))
                    if why:
                        rec["status"] = "mismatch (the KI's headline value; the panel reads the kit block)"
                        _kv = kit.get(m)
                        notes.append(f"{var}:{m} the KI's own value {rv!r} differs from the kit's "
                                     f"{(repr(float(_kv)) if _kv is not None else 'undefined (on the series)')} "
                                     f"(reported only)")
                    head[m] = rec
            if head:
                row["ki_headline"] = head
        # the plain keys already compared: the panel's (plain source) or the KI headline ones (block source)
        listed = ((set(row.get("ki_headline") or {}) | _filled) if row["source"] == "block"
                  else set(row["metrics"]))
        for m in sorted(set(plain) - listed):
            if _num(plain.get(m)) is not None:
                row.setdefault("not_checked", {})[m] = {
                    "runner": _num(plain.get(m)), "status": "not checked (the KI may define it differently)"}
        checks[var] = row
    return {"kinds": {}, "kit_missing": km, "notes": notes, "checks": checks}



# ── tolerances (design §2.11) ─────────────────────────────────────────────────────────────────
N_BOOT = 500
BOOT_SEED = 0
BLOCK_LEN = 30
MIN_RESAMPLES = 100
#: an SD this small relative to the metric's size is floating-point noise, i.e. SD 0 (§2.11)
SD_ZERO_REL = 1e-12


def bootstrap_tolerances(sim, obs, kind: str = "series", dates=None, n_boot: int = N_BOOT,
                         block: int = BLOCK_LEN, seed: int = BOOT_SEED, z: float = 1.0,
                         min_ok: int = MIN_RESAMPLES, metrics=None) -> dict:
    """Per-metric tolerance = z * SD (ddof 0) of the metric across moving-block resamples of the
    pilot default run's valid pairs (design §2.11).

    Block length 30, reduced to round(sqrt(n)) when n < 90; blocks are consecutive valid pairs
    (they may span missing dates — an approximation). Generator seed fixed (0), not the search
    seed. A resample where a metric is undefined is skipped; fewer than `min_ok` usable resamples,
    or SD 0, -> that metric uses the fixed fallback. n < 10 -> every metric falls back.

    Returns {"tol": {m: v}, "source": {m: "bootstrap"|"fixed_fallback"}, "why": {m: reason},
    "n": n, "block": b, "z": z}."""
    import numpy as np
    want = tuple(metrics) if metrics else KIND_TABLE[kind]["recorded"]
    s, o = _finite_pairs(sim, obs, dates)
    n = int(len(s))
    tol, src, why = {}, {}, {}

    def fallback(m, reason):
        if m in FIXED_FALLBACK_TOL:
            tol[m] = z * FIXED_FALLBACK_TOL[m]
            src[m] = "fixed_fallback"
            why[m] = reason

    if n < 10:
        for m in want:
            fallback(m, f"record too short (n={n} < 10)")
        return {"tol": tol, "source": src, "why": why, "n": n, "block": None, "z": z}
    b = int(block) if n >= 90 else max(1, int(round(math.sqrt(n))))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / b))
    vals: dict = {m: [] for m in want}
    for _ in range(int(n_boot)):
        starts = rng.integers(0, n - b + 1, size=n_blocks)
        idx = np.concatenate([np.arange(st, st + b) for st in starts])[:n]
        got = _metrics_from_pairs(s[idx], o[idx], kind)
        for m in want:
            v = got.get(m)
            if v is not None and math.isfinite(v):
                vals[m].append(v)
    for m in want:
        xs = vals[m]
        if len(xs) < min_ok:
            fallback(m, f"only {len(xs)} of {n_boot} resamples defined (< {min_ok})")
            continue
        arr = np.asarray(xs, float)
        sd = float(np.std(arr))
        if sd <= SD_ZERO_REL * max(1.0, float(np.max(np.abs(arr)))):
            # exactly flat up to floating-point noise: the record cannot measure this metric
            fallback(m, "bootstrap SD is 0 (up to rounding)")
            continue
        tol[m] = z * sd
        src[m] = "bootstrap"
    return {"tol": tol, "source": src, "why": why, "n": n, "block": b, "z": z}


def fallback_tolerances(kind: str, z: float = 1.0, reason: str = "no series") -> dict:
    """The fixed fallback for every recorded metric of a kind, labelled (design §2.11)."""
    rec = KIND_TABLE[kind]["recorded"]
    return {"tol": {m: z * FIXED_FALLBACK_TOL[m] for m in rec},
            "source": {m: "fixed_fallback" for m in rec},
            "why": {m: reason for m in rec}, "n": None, "block": None, "z": z}


def tolerances_for(variables, kinds: dict, series_by_var: dict | None, z: float = 1.0,
                   no_series_reason: dict | None = None) -> dict:
    """{var: tolerance record} for every variable: bootstrap where the pilot wrote a series,
    fixed fallback (labelled) otherwise. Categorical variables get an empty record."""
    out = {}
    for var in variables:
        k = (kinds or {}).get(var, "series")
        ser = (series_by_var or {}).get(var)
        if ser is None:
            out[var] = fallback_tolerances(k, z, (no_series_reason or {}).get(var, "no series"))
        else:
            try:
                out[var] = bootstrap_tolerances(ser[0], ser[1], kind=k,
                                                dates=ser[2] if len(ser) > 2 else None, z=z)
            except Exception as e:
                out[var] = fallback_tolerances(k, z, f"series unreadable ({type(e).__name__})")
    return out


def setup_kinds(variables, dag: dict | None, obs_shape_by_var: dict | None, contract: dict | None,
                convention: dict | None = None) -> dict:
    """Kinds and the plain-pbias rule for the calibrated variables (design §1.4).
    Returns {"kinds": {var: kind}, "pbias_percent": {var: bool}, "pbias_unit": {var: unit|None},
    "warnings": [...], "shapes": {var: the obs_shape used, after the one-obs_shape rule}}."""
    stated = ((((contract or {}).get("strategy") or {}).get("convergence") or {}).get("variables") or {})
    kinds, pp, pu, warns, used_shape = {}, {}, {}, [], {}
    if not isinstance(stated, dict):
        raise ValueError("strategy.convergence.variables must be a mapping {var: {kind: ...}}; got "
                         f"{type(stated).__name__}")
    declared = {str(x) for x in (variables or [])}
    for name, spec in stated.items():
        if str(name) not in declared:
            warns.append(f"strategy.convergence.variables.{name}: not a declared variable; its settings "
                         f"are ignored")
        elif not isinstance(spec, dict):
            warns.append(f"strategy.convergence.variables.{name}: must be a mapping such as "
                         f"{{kind: flow}}; got {spec!r}, ignored")
    shapes = obs_shape_by_var or {}
    named = {str(t.get("var")) for t in ((contract or {}).get("targets") or []) if isinstance(t, dict)}
    seen = set(shapes.values())
    for v in [str(x) for x in (variables or [])]:
        spec = stated.get(v) if isinstance(stated.get(v), dict) else {}
        shape = shapes.get(v)
        if shape is None and v in named and len(seen) == 1:
            shape = next(iter(seen))      # the same one-obs_shape rule objectives_from_dag uses
        k, w = resolve_kind(v, dag, shape, spec.get("kind"))
        used_shape[v] = shape
        kinds[v] = k
        warns += w
        pp[v] = pbias_is_percent(convention, v)
        pu[v] = pbias_other_unit(convention, v)
    return {"kinds": kinds, "pbias_percent": pp, "pbias_unit": pu, "warnings": warns, "shapes": used_shape}


def write_tolerances(path, tol_by_var: dict) -> None:
    """Persist the frozen tolerances (design §2.11: frozen after the pilot, saved in the plan file)."""
    with open(path, "w") as f:
        json.dump(tol_by_var, f, indent=1, sort_keys=True)


def read_tolerances(path) -> dict:
    with open(path) as f:
        return json.load(f)


def tolerances_from_metrics(metrics: dict, variables, panel_metrics=None, kinds: dict | None = None
                            ) -> tuple[dict, str]:
    """Series-free tolerances: the fixed fallbacks for every recorded metric of each variable's
    kind (not only the metrics the payload happens to carry — a variable is never exempt from the
    flatness test because one payload lacked a key). Returns (tolerances_by_var, "fixed_fallback")."""
    tol = {}
    for v in [str(x) for x in (variables or [])]:
        k = (kinds or {}).get(v, "series")
        rec = KIND_TABLE[k]["recorded"]
        if rec:
            tol[v] = {m: FIXED_FALLBACK_TOL[m] for m in rec}
    return tol, "fixed_fallback"
