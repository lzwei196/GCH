"""The convergence RULE (design HANDOFF_CONVERGENCE_2026-09-27_v2.md §2.1–§2.6; build step 2).

Call by call, within ONE search, the rule tracks:

  * the INCUMBENT (§2.4) — single objective: the lowest loss so far (ties -> earliest); several
    objectives (trade-off search): the calibration-only COMPROMISE of the non-dominated archive on a
    scale FROZEN once at t0, with the contract's protected metrics (`strategy.protect`);
  * condition 1 (§2.2), the LOSS test per objective: the tracked value v_k moved by less than
    rel_gain (relative, floor 1e-3) over the window. Trade-off: v_k = B_k, the best loss of k so far.
    Single objective: v_k = objective k's own loss AT THE INCUMBENT (never the weighted scalar);
  * conditions 2-3 (§2.5), the PANEL test: every required metric of every variable, read at the
    incumbent, is finite at the call AND at every call of the window, and stays within its tolerance
    of the current value; anything missing makes that (variable, metric) unknown;
  * condition 4 (§2.6, trade-off only), the FRONT test: the additive epsilon-indicator of the archive
    W calls ago over the current archive, on the frozen scale, is at most eps_front.

The STOP POINT is the first call where conditions 1-3 (and 4) hold for everything. Each variable's
first SETTLE point (its own objectives pass condition 1 and its own required metrics condition 2) is
recorded too. Nothing here stops a search or gives a verdict: termination is build step 4, verdicts
step 3. Everything needed for those is kept per call.

W (§2.3) = min(max(20, c(d+1)), floor(cap/2)), provisional c = 5.
"""
from __future__ import annotations
import math

REL_GAIN = 0.005
LOSS_FLOOR = 1e-3
EPS_FRONT = 0.01
C_WINDOW = 5
W_MIN = 20


def design_window(n_params: int, cap: int, c: float = C_WINDOW) -> int:
    """W = min(max(20, c(d+1)), floor(cap/2)) (design §2.3). At least 1."""
    return int(max(1, min(max(W_MIN, int(math.ceil(c * (int(n_params) + 1)))), int(cap) // 2)))


def _fin(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── protection (§2.4, §1.4 load check and pilot drop) ────────────────────────────────────────────
def check_protect(spec, panel, declared=None) -> dict:
    """Load check of `strategy.protect` (design §1.4). Returns {var: [metrics]}; raises ValueError for
    a spec that is not a mapping of lists, a variable that is not a declared target, or a metric the
    variable's kind does not record (e.g. any metric of a `categorical` variable)."""
    if spec is None or spec is False or spec == {} or spec == []:
        return {}
    if not isinstance(spec, dict):
        raise ValueError("strategy.protect must be a mapping {var: [metrics]}")
    targets = [str(v) for v in (declared if declared is not None else panel.variables)]
    out = {}
    for var, mets in spec.items():
        var = str(var)
        if var not in targets or var not in panel.variables:
            raise ValueError(f"strategy.protect names {var!r}, which is not a declared target")
        if isinstance(mets, str) or not isinstance(mets, (list, tuple)):
            raise ValueError(f"strategy.protect.{var} must be a list of metrics")
        rec = panel.recorded(var)
        for m in mets:
            m = str(m).lower()
            if m not in rec:
                raise ValueError(f"strategy.protect.{var}: {m!r} is not a metric a "
                                 f"{panel.kinds[var]!r} variable records ({list(rec)})")
        out[var] = [str(m).lower() for m in mets]
    return out


def resolve_protect(protect: dict, panel, default_panel: dict | None,
                    no_default_reason: str | None = None) -> tuple[dict, list]:
    """After the pilot's default run (design §1.4): a protected metric that is missing there for a
    known reason — the runner did not emit it, the kit set it missing (mean near zero, mismatch), or
    the flow->series switch removed it — is DROPPED with that reason, so protection is never failed
    by a metric nobody could check. Returns (active {var: [m]}, dropped [{var, metric, reason}])."""
    active, dropped = {}, []
    for var, mets in (protect or {}).items():
        keep = []
        for m in mets:
            km = (panel.kit_missing.get(var) or {}).get(m)
            if m not in panel.recorded(var):
                dropped.append({"var": var, "metric": m,
                                "reason": f"no longer recorded (kind is now {panel.kinds[var]!r})"})
            elif km:
                dropped.append({"var": var, "metric": m, "reason": km})
            elif default_panel is None:
                dropped.append({"var": var, "metric": m,
                                "reason": no_default_reason or "no pilot default run to anchor it"})
            elif _fin(((default_panel or {}).get(var) or {}).get(m)) is None:
                dropped.append({"var": var, "metric": m,
                                "reason": "not given at the default run (runner did not emit it)"})
            else:
                keep.append(m)
        if keep:
            active[var] = keep
    return active, dropped


def admissible(cand_panel: dict, active: dict, default_panel: dict, tol: dict) -> bool:
    """§2.4: each protected metric no worse than at the default parameters by more than its tolerance.
    r, NSE, KGE, log-flow NSE: x >= default - tol; alpha, beta: |x-1| <= |x_def-1| + tol;
    PBIAS: |x| <= |x_def| + tol; NRMSE: x <= x_def + tol. A missing protected metric -> not admissible."""
    for var, mets in (active or {}).items():
        for m in mets:
            x = _fin(((cand_panel or {}).get(var) or {}).get(m))
            d = _fin(((default_panel or {}).get(var) or {}).get(m))
            t = _fin(((tol or {}).get(var) or {}).get(m))
            if x is None or d is None or t is None:
                return False
            if m in ("r", "nse", "kge", "lnnse"):
                ok = x >= d - t
            elif m in ("alpha", "beta"):
                ok = abs(x - 1.0) <= abs(d - 1.0) + t
            elif m == "pbias":
                ok = abs(x) <= abs(d) + t
            elif m == "nrmse":
                ok = x <= d + t
            else:
                return False
            if not ok:
                return False
    return True


def eps_indicator(A_old, A_new, scale) -> float:
    """Additive epsilon-indicator on a fixed scale (§2.6):
    I(A_old, A_new) = max_{a in A_new} min_{b in A_old} max_k (b_k - a_k) / s_k."""
    import numpy as np
    B = np.asarray(A_old, float)                      # (nb, K)
    A = np.asarray(A_new, float)                      # (na, K)
    sc = np.asarray(scale, float)
    e = ((B[None, :, :] - A[:, None, :]) / sc).max(axis=2)     # (na, nb): max over objectives
    return float(e.min(axis=1).max())


def _dominates(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


class ConvergenceRule:
    """Feed it every call of ONE search, in order (design §2.1: the call index is the order of add()).

    objectives: [(name, var)] in the order of the loss vector given to add().
    trade_off:  True when each objective is an optimizer axis (NSGA-II/III, MOEA/D); False for a
                single-objective search (one variable or a weighted scalar over several).
    panel:      the step-1 Panel (variables, kinds, required metrics).
    tol:        {var: {metric: tolerance}} (the step-1 tolerance records' `tol`).
    protect:    active protections {var: [metrics]} (after resolve_protect); used by the compromise.
    default_panel: the pilot's default-run panel {var: {metric: value}} (protection anchor).
    """

    def __init__(self, objectives, panel, tol: dict, window: int, trade_off: bool = False,
                 rel_gain: float = REL_GAIN, eps_front: float = EPS_FRONT,
                 protect: dict | None = None, default_panel: dict | None = None,
                 weights=None):
        self.objectives = [(str(n), str(v)) for n, v in objectives]
        # the weights of the single-objective scalar (the weighted mean the optimizer minimizes)
        self.weights = [float(w) for w in weights] if weights is not None else None
        self.K = len(self.objectives)
        self.panel = panel
        self.tol = {str(v): dict(m) for v, m in (tol or {}).items()}
        self.W = int(window)
        self.trade_off = bool(trade_off)
        self.rel_gain = float(rel_gain)
        self.eps_front = float(eps_front)
        self.protect = {v: list(m) for v, m in (protect or {}).items()} if self.trade_off else {}
        self.default_panel = default_panel or {}
        # variables judged: declared ones with an objective or a required metric (rule-0 variables —
        # no objective and no panel metric — are left out and never block, §2.7)
        self.obj_of = {}
        for k, (_, v) in enumerate(self.objectives):
            self.obj_of.setdefault(v, []).append(k)
        # (a variable whose metrics are all kit-set missing is NOT rule 0: `recorded`, not `required`)
        self.variables = [v for v in panel.variables if self.obj_of.get(v) or panel.recorded(v)]
        self.rule0 = [v for v in panel.variables if v not in self.variables]
        # per call
        self.F: list = []            # loss vector (list of float|None) per call
        self.S: list = []            # scalar loss per call (single objective)
        self.P: list = []            # panel per call
        self.inc: list = []          # incumbent call index (or None) per call
        self.V: list = []            # tracked value v_k per call (list, None where not finite)
        self.X: list = []            # incumbent panel per call
        self.infeasible: list = []   # protection infeasible at this call (trade-off)
        self._arch_hist: list = []   # (call, archive) only when the archive changed — memory
        self.cond: list = []         # per-call condition record (see _conditions)
        self._arch: list = []
        self._best: list = [None] * self.K
        self._best_single = None
        self.t0 = None
        self.z = None
        self.s = None
        self.stop_point = None
        self.first_settle = {v: None for v in self.variables}

    # ── incumbent ──────────────────────────────────────────────────────────────────────────────
    def _norm(self, f):
        return [(fk - zk) / sk for fk, zk, sk in zip(f, self.z, self.s)]

    def _compromise(self, i):
        cands = list(self._arch)
        if not cands:
            return None, False
        adm = [c for c in cands if admissible(self.P[c], self.protect, self.default_panel, self.tol)] \
            if self.protect else cands
        infeasible = bool(self.protect) and not adm
        pool = adm if adm else cands
        if self.t0 is None:
            key = lambda c: (sum(self.F[c]), c)                                    # noqa: E731
        else:
            key = lambda c: (max(self._norm(self.F[c])), sum(self._norm(self.F[c])), c)  # noqa: E731
        return min(pool, key=key), infeasible

    def _update_incumbent(self, i, f, scalar):
        if not self.trade_off:
            sc = self.S[i]
            if sc is not None and (self._best_single is None or sc < self.S[self._best_single]):
                self._best_single = i                      # ties -> earliest (strict <)
            return self._best_single, False
        # B_k: the lowest finite loss of EACH objective so far, from every finite component (§2.2)
        for k in range(self.K):
            x = f[k] if f is not None else None
            if x is not None and (self._best[k] is None or x < self._best[k]):
                self._best[k] = x
        if f is not None and all(x is not None for x in f):
            # the archive holds fully finite, distinct, non-dominated points (a copy of a member —
            # a cache hit, equal rounded metrics — adds nothing)
            if not any(_dominates(self.F[k], f) or self.F[k] == f for k in self._arch):
                self._arch = [k for k in self._arch if not _dominates(f, self.F[k])] + [i]
        if self.t0 is None and i >= self.W and len({tuple(self.F[k]) for k in self._arch}) >= 2:
            A = [self.F[k] for k in self._arch]
            self.t0 = i
            self.z = [min(a[k] for a in A) for k in range(self.K)]
            self.s = []
            for k in range(self.K):
                col = [a[k] for a in A]
                self.s.append(max(max(col) - min(col), 1e-6 * max(abs(x) for x in col), 1e-12))
        return self._compromise(i)

    # ── one call ───────────────────────────────────────────────────────────────────────────────
    def add(self, losses, scalar=None, panel=None) -> dict:
        """One call (in order). `losses`: loss per objective (None for a rejected or failed call);
        `scalar`: the loss the single-objective optimizer minimizes (weighted mean of `losses`);
        `panel`: {var: {metric: value}} of this call. Returns this call's condition record.
        All-or-nothing: if anything fails, the call is not recorded at all (the lists stay in step)."""
        snap = self._snapshot()
        try:
            return self._add(losses, scalar, panel)
        except Exception:
            self._restore(snap)
            raise

    _PER_CALL = ("F", "S", "P", "inc", "V", "X", "infeasible", "cond")

    def _snapshot(self):
        return ({k: len(getattr(self, k)) for k in self._PER_CALL}, list(self._best), self._best_single,
                list(self._arch), len(self._arch_hist), self.t0, self.z, self.s, self.stop_point,
                dict(self.first_settle))

    def _restore(self, snap):
        lens, best, bs, arch, nah, t0, z, s, sp, fs = snap
        for k, n in lens.items():
            del getattr(self, k)[n:]
        self._best, self._best_single, self._arch = best, bs, arch
        del self._arch_hist[nah:]
        self.t0, self.z, self.s, self.stop_point, self.first_settle = t0, z, s, sp, fs

    def _add(self, losses, scalar=None, panel=None) -> dict:
        i = len(self.F)
        f = None
        if losses is not None:
            f = [_fin(x) for x in losses]
            if len(f) != self.K:
                raise ValueError(f"call {i}: {len(f)} losses for {self.K} objectives")
        self.F.append(f if f is not None else [None] * self.K)
        full = f is not None and all(x is not None for x in f)
        if not self.trade_off and scalar is None and full:
            w = self.weights or [1.0] * self.K
            scalar = sum(wk * x for wk, x in zip(w, f)) / (sum(w) or 1.0)
        # a single-objective incumbent needs a finite full vector AND a finite scalar
        self.S.append(_fin(scalar) if full else None)
        self.P.append(panel or {})
        inc, infeas = self._update_incumbent(i, f, scalar)
        self.inc.append(inc)
        self.infeasible.append(infeas)
        if self.trade_off and (not self._arch_hist or self._arch_hist[-1][1] != self._arch):
            self._arch_hist.append((i, list(self._arch)))
        # tracked values (§2.2)
        if self.trade_off:
            self.V.append(list(self._best))
        else:
            self.V.append(list(self.F[inc]) if inc is not None else [None] * self.K)
        self.X.append(self.P[inc] if inc is not None else {})
        rec = self._conditions(i)
        self.cond.append(rec)
        for v in self.variables:
            if self.first_settle[v] is None and rec["settled"].get(v):
                self.first_settle[v] = i
        if self.stop_point is None and rec["stop"]:
            self.stop_point = i
        return rec

    # ── conditions ─────────────────────────────────────────────────────────────────────────────
    def _cond1(self, i, k) -> bool:
        if i < self.W:
            return False
        a, b = self.V[i - self.W][k], self.V[i][k]
        if a is None or b is None:
            return False
        return abs(a - b) / max(abs(a), LOSS_FLOOR) < self.rel_gain

    def _cond2(self, i, var, m) -> str:
        """'ok' | 'moved' | 'unknown' | 'early' for (var, metric) at call i (§2.5). 'early' = fewer
        than W calls so far: not testable yet, which is NOT missing evidence (the verdict's
        sole-blocker rules must not read it as a missing metric)."""
        if i < self.W:
            return "early"
        x = _fin((self.X[i].get(var) or {}).get(m))
        t = _fin((self.tol.get(var) or {}).get(m))
        if x is None or t is None:
            return "unknown"
        vals = []
        for k in range(i - self.W, i):
            u = _fin((self.X[k].get(var) or {}).get(m))
            if u is None:
                return "unknown"
            vals.append(u)
        return "ok" if max(abs(u - x) for u in vals) <= t else "moved"

    def _archive_at(self, i):
        """The archive as it was after call i."""
        last = []
        for c, a in self._arch_hist:
            if c > i:
                break
            last = a
        return last

    def _cond4(self, i):
        """True / False / None (not testable: before t0 or no archive W calls ago)."""
        if not self.trade_off:
            return None
        if self.t0 is None or i < self.t0 or i < self.W:
            return None
        old = [self.F[k] for k in self._archive_at(i - self.W)]
        new = [self.F[k] for k in self._arch]
        if not old or not new:
            return None
        return eps_indicator(old, new, self.s) <= self.eps_front

    def _conditions(self, i) -> dict:
        c1 = [self._cond1(i, k) for k in range(self.K)]
        c2 = {v: {m: self._cond2(i, v, m) for m in self.panel.required(v)} for v in self.variables}
        c4 = self._cond4(i)
        settled = {}
        for v in self.variables:
            ok1 = all(c1[k] for k in self.obj_of.get(v, []))
            ok2 = all(s == "ok" for s in c2[v].values())
            settled[v] = bool(ok1 and ok2)
        unknown = [[v, m] for v in self.variables for m, s in c2[v].items() if s in ("unknown", "early")]
        stop = (all(c1) and all(settled.values()) and not unknown
                and (c4 is True if self.trade_off else True))
        return {"i": i, "inc": self.inc[i], "cond1": c1, "cond2": c2, "cond3": not unknown,
                "cond4": c4, "settled": settled, "stop": bool(stop),
                "protection_infeasible": self.infeasible[i]}

    # ── report ─────────────────────────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        n = len(self.F)
        return {"window": self.W, "rel_gain": self.rel_gain, "eps_front": self.eps_front,
                "trade_off": self.trade_off, "calls": n, "stop_point": self.stop_point,
                "first_settle": dict(self.first_settle), "variables_judged": list(self.variables),
                "variables_rule0": list(self.rule0),
                "t0": self.t0, "frozen_offset": self.z, "frozen_scale": self.s,
                "incumbent_final": (self.inc[-1] if n else None),
                "best_losses": (list(self._best) if self.trade_off else None),
                "protect_active": ({v: list(m) for v, m in self.protect.items()} if self.trade_off
                                   else "not used (single-objective search)"),
                "protection_infeasible_final": bool(self.infeasible[-1]) if n else None,
                # when protection is infeasible the candidates are reported (§2.4)
                "candidates_final": ([{"call": k, "losses": self.F[k]} for k in self._arch]
                                     if (n and self.infeasible[-1]) else None),
                "objectives": [n_ for n_, _ in self.objectives],
                # what a replay needs to rebuild this rule (build step 9, gap 2u)
                "objective_vars": [v_ for _, v_ in self.objectives], "weights": self.weights,
                "default_panel": self.default_panel}

    def state(self) -> dict:
        """What must be saved to reuse the rule's frozen quantities after the fact (§2.6)."""
        return {"t0": self.t0, "z": self.z, "s": self.s, "W": self.W}


# ── modes and the old strategy.stop block (design §1.2, §1.6, §5.1b; build step 4) ──────────────
MODES = ("keep_going", "stop")
#: optimizers our stop point may end in stop mode (DDS is budget-scheduled; DREAM: R-hat decides)
STOPPABLE = ("sceua", "nsga2", "nsga3", "moead")
_LEGACY_MODES = {"observe": "keep_going", "enforce": "stop"}


def resolve_mode(contract: dict | None) -> tuple[str, list, dict | None]:
    """(mode, warnings, translated_strategy_stop) from the contract, at LOAD.

    strategy.convergence.mode: "keep_going" (default when nobody answers) or "stop". The old names
    "observe"/"enforce" are translated with a warning. An old `strategy.stop` block is translated
    (design §5.1b): it means mode "stop" (unless strategy.convergence.mode says otherwise) and its
    floor band becomes the band the standards check uses for its main_stat; anything else in it is
    ignored with a warning, never an error. Raises ValueError for an unknown mode."""
    strat = (contract or {}).get("strategy", {}) or {}
    conv = strat.get("convergence") if isinstance(strat.get("convergence"), dict) else {}
    warns: list[str] = []
    raw = conv.get("mode")
    translated = None
    old = strat.get("stop")
    if old is not None and not isinstance(old, dict):
        warns.append(f"strategy.stop is removed and was not a mapping ({type(old).__name__}); ignored")
    elif isinstance(old, dict) and old:
        floor = old.get("floor") if isinstance(old.get("floor"), dict) else {}
        translated = {"mode": "stop", "main_stat": str(old.get("main_stat", "nse")).lower(),
                      "band": floor.get("band"), "dag_variable": floor.get("dag_variable")}
        ignored = (sorted(set(old) - {"main_stat", "floor"})
                   + sorted(set(floor) - {"band", "source", "dag_variable"}))
        translated["_ignored"] = ignored
    if raw is None:
        mode = "stop" if translated else "keep_going"
        explicit = False
    else:
        explicit = True
        m = str(raw).strip().lower()
        if m in _LEGACY_MODES:
            warns.append(f"strategy.convergence.mode {raw!r} is an old name; read as {_LEGACY_MODES[m]!r}")
            m = _LEGACY_MODES[m]
        if m not in MODES:
            raise ValueError(f"strategy.convergence.mode must be one of {list(MODES)}; got {raw!r}")
        mode = m
    if translated:
        translated["mode"] = mode
        ign = translated.pop("_ignored")
        warns.append("strategy.stop is removed; translated: mode "
                     + (f"{mode} (the explicit strategy.convergence.mode wins)" if explicit else "stop")
                     + (f"; the standards band {translated['band']!r} for {translated['main_stat']}"
                        + (f" of {translated['dag_variable']}" if translated.get("dag_variable") else "")
                        if translated["band"] else "")
                     + (f"; ignored: {ign}" if ign else ""))
    return mode, warns, translated


def sceua_settings(mode: str, cap: int, n_params: int, rel_gain: float = REL_GAIN, ngs: int = 20) -> dict:
    """SCE-UA settings per mode (design §1.6). The cap is enforced by the kit's call hook, so SPOTPY
    is given repetitions = 2*cap + ngs(2d+1) (its counter counts every trial plus one per step)."""
    pop = int(ngs) * (2 * int(n_params) + 1)
    loops_available = max(0.0, (int(cap) - pop) / (2.0 * pop))
    # SPOTPY's own INTERNAL settings stay at its 1.6.7 defaults in every mode (2026-10-04): the kit's convergence
    # rule is applied at each loop end through the loop hook (native_rules.SpotpyTest, classic settings kstop 10,
    # pcento 0.1, peps 0.001). The old stop-mode values (kstop = loops_available / 3) depended on the budget.
    kstop, pcento = 100, 1e-7
    return {"ngs": int(ngs), "kstop": kstop, "pcento": pcento, "peps": 1e-7,
            "repetitions": 2 * int(cap) + pop, "loops_available": round(loops_available, 2)}
