"""The STEP-END convergence rule (rev 30, decided 2026-10-03; plan
/mnt/datasets/HANDOFF_CONVERGENCE_REV30_FINAL_PLAN_2026-10-03.md, part A2).

A STEP is one SCE-UA loop or one NSGA-II generation. Each optimizer keeps its own logic and is judged
only at the END of a step, after its random start (SCE-UA loop 0; NSGA-II generation 1), on the loss
AND our watched scores. Feed every call in order with its step tag; a step ends when a call of a
LATER step arrives (a partial last step never counts as a step end unless the caller says the last
step was complete).

At each step end after the start the rule compares this step end with the previous one (a
TRANSITION). The transition is QUIET when:
  * the loss: each tracked value moved by less than rel_gain (relative, floor 1e-3). Single
    objective: the objective losses AT THE BEST POINT (lowest scalar so far). Trade-off: the best
    loss of each objective so far;
  * every required watched score of every judged variable, read at the best point (trade-off: at the
    compromise), is finite at both step ends and stayed within its tolerance; a missing score makes
    the transition NOT quiet (fail closed) and is listed;
  * trade-off only: the archive front moved by no more than eps_front (the kit's additive
    eps-indicator, previous step-end archive over this one, on the scale frozen at the end of the
    start-up step).

Look-back m (quiet transitions in a row needed), q = transitions so far (= step ends after the start):
  q < Q_MIN (5): no test; Q_MIN <= q < H (20): m = M_MIN (5); q >= H: from the last H transitions
  up to the last non-quiet one (the quiet run being tested is left out), p_L = one-sided 80 % exact
  (Clopper-Pearson) lower bound on the share of non-quiet transitions (a "pre-quiet activity rate": the
  window always ends on a move, so it leans toward a higher rate and a shorter m than an unconditional
  window would; fixed in advance and validated exactly as used),
  m = ceil(log ALPHA / log(1 - p_L)) clamped to M_MIN..M_MAX (M_MAX when p_L = 0).
CONVERGED at the first transition where the last m transitions were all quiet. (SCE-UA's own
population-spread test, gnrng < peps, is NOT part of this rule: frozen-kit records have no per-loop
gnrng, and the rule validated must be the rule applied — codex plan review 2026-10-03, point 1.
SPOTPY still applies its own built-in stop inside the search, as always.)

These values are set once by the kit's developers, the same for every model; nobody tunes them per
search. Nothing here stops a search; it only records.
"""
from __future__ import annotations
import math

from .rule import EPS_FRONT, LOSS_FLOOR, REL_GAIN, _dominates, _fin, admissible, eps_indicator

Q_MIN = 5          # no test before 5 transitions
M_MIN = 5          # the smallest look-back
M_MAX = 50         # the largest look-back
H = 20             # the transitions the adaptive look-back learns from
CONF = 0.80        # one-sided confidence of the lower bound p_L
ALPHA = 0.05       # chance of a quiet run of length m when still improving at rate p_L


def _binom_sf(y: int, n: int, p: float) -> float:
    """P(X >= y), X ~ Binomial(n, p), exact sum."""
    if y <= 0:
        return 1.0
    if y > n:
        return 0.0
    return sum(math.comb(n, k) * p ** k * (1.0 - p) ** (n - k) for k in range(y, n + 1))


def cp_lower(y: int, n: int, conf: float = CONF) -> float:
    """One-sided exact (Clopper-Pearson) lower confidence bound on a binomial share: the p with
    P(X >= y | p) = 1 - conf (0 when y = 0). Bisection, no scipy needed."""
    if n <= 0 or y < 0 or y > n:
        raise ValueError(f"bad counts y={y} n={n}")
    if y == 0:
        return 0.0
    a, lo, hi = 1.0 - conf, 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _binom_sf(y, n, mid) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lookback(quiet: list) -> int | None:
    """The look-back m after the transitions `quiet` (True = quiet), or None (q < Q_MIN: no test).

    For q >= H the share of non-quiet transitions is learned from the H transitions that END AT THE
    LAST NON-QUIET ONE: the quiet run now being tested is left out. (Counting it would make every
    quiet step lower p_L and so RAISE m — the run would count against itself; found by the tracer
    test test_adaptive_lookback_after_twenty, 2026-10-03.) Fewer than H transitions before that
    point -> all of them."""
    q = len(quiet)
    if q < Q_MIN:
        return None
    if q < H:
        return M_MIN
    k = max((j + 1 for j, x in enumerate(quiet) if not x), default=0)   # transitions up to the last move
    if k == 0:
        return M_MAX
    hist = quiet[max(0, k - H):k]
    y = sum(1 for x in hist if not x)
    p = cp_lower(y, len(hist))
    if p <= 0.0:
        return M_MAX
    if p >= 1.0:
        return M_MIN
    m = math.ceil(math.log(ALPHA) / math.log(1.0 - p))
    return max(M_MIN, min(M_MAX, m))


class StepRule:
    """Feed it every call of ONE search, in order, with its step tag (SCE-UA loop, NSGA-II
    generation). The first step tag seen is the random start.

    objectives / panel / tol / trade_off / protect / default_panel / weights: as in rule.ConvergenceRule.
    """

    def __init__(self, objectives, panel, tol: dict, trade_off: bool = False,
                 rel_gain: float = REL_GAIN, eps_front: float = EPS_FRONT,
                 protect: dict | None = None, default_panel: dict | None = None, weights=None):
        self.objectives = [(str(n), str(v)) for n, v in objectives]
        self.K = len(self.objectives)
        self.weights = [float(w) for w in weights] if weights is not None else None
        self.panel = panel
        self.tol = {str(v): dict(m) for v, m in (tol or {}).items()}
        self.trade_off = bool(trade_off)
        self.rel_gain = float(rel_gain)
        self.eps_front = float(eps_front)
        self.protect = {v: list(m) for v, m in (protect or {}).items()} if self.trade_off else {}
        self.default_panel = default_panel or {}
        self.obj_of = {}
        for k, (_, v) in enumerate(self.objectives):
            self.obj_of.setdefault(v, []).append(k)
        self.variables = [v for v in panel.variables if self.obj_of.get(v) or panel.recorded(v)]
        # per call
        self.F: list = []
        self.S: list = []
        self.P: list = []
        self.step_of: list = []
        # running state
        self._best_single = None
        self._best = [None] * self.K
        self._arch: list = []
        self._cur_step = None
        self.scale = None                # frozen at the end of the start-up step (trade-off)
        self.ends: list = []             # one record per step end (the first = the start-up end)
        self.transitions: list = []      # one record per transition (after the start)
        self.converged_at = None         # the transition record that fired first, or None
        self.closed = False

    # ── calls ──────────────────────────────────────────────────────────────────────────────────
    def add(self, losses, scalar=None, panel=None, step=None) -> None:
        if self.closed:
            raise ValueError("add() after finish()")
        if step is None or isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"call {len(self.F)}: step tag must be an int, got {step!r}")
        if self._cur_step is not None and step < self._cur_step:
            raise ValueError(f"call {len(self.F)}: step tag went back from {self._cur_step} to {step}")
        i = len(self.F)
        f = None
        if losses is not None:                   # checked BEFORE the previous step is closed: a refused call
            f = [_fin(x) for x in losses]        # leaves the rule exactly as it was (codex A2 r1 #1)
            if len(f) != self.K:
                raise ValueError(f"call {i}: {len(f)} losses for {self.K} objectives")
        if self._cur_step is not None and step > self._cur_step:
            self._end_step(len(self.F) - 1)
        self._cur_step = step
        self.F.append(f if f is not None else [None] * self.K)
        full = f is not None and all(x is not None for x in f)
        if not self.trade_off and scalar is None and full:
            w = self.weights or [1.0] * self.K
            scalar = sum(wk * x for wk, x in zip(w, f)) / (sum(w) or 1.0)
        self.S.append(_fin(scalar) if full else None)
        self.P.append(panel or {})
        self.step_of.append(step)
        if not self.trade_off:
            sc = self.S[i]
            if sc is not None and (self._best_single is None or sc < self.S[self._best_single]):
                self._best_single = i
        else:
            for k in range(self.K):
                x = f[k] if f is not None else None
                if x is not None and (self._best[k] is None or x < self._best[k]):
                    self._best[k] = x
            if full and not any(_dominates(self.F[k], f) or self.F[k] == f for k in self._arch):
                self._arch = [k for k in self._arch if not _dominates(f, self.F[k])] + [i]

    def finish(self, last_step_complete: bool = False) -> dict:
        """End of the record. The last step counts as a step end only when the caller knows it was
        complete (live: the optimizer finished the step; never for a budget cut mid-step)."""
        if not self.closed:
            if last_step_complete and self.F:
                self._end_step(len(self.F) - 1)
            self.closed = True
        return self.summary()

    # ── step ends ──────────────────────────────────────────────────────────────────────────────
    def _compromise(self):
        cands = list(self._arch)
        if not cands:
            return None
        if self.protect:
            adm = [c for c in cands if admissible(self.P[c], self.protect, self.default_panel, self.tol)]
            cands = adm or cands
        if self.scale is None:
            return min(cands, key=lambda c: (sum(self.F[c]), c))
        z, s = self.scale

        def nrm(c):
            return [(fk - zk) / sk for fk, zk, sk in zip(self.F[c], z, s)]
        return min(cands, key=lambda c: (max(nrm(c)), sum(nrm(c)), c))

    def _end_step(self, last_call: int) -> None:
        step = self.step_of[last_call]
        if self.trade_off and self.scale is None and len(self.ends) == 0:
            A = [self.F[k] for k in self._arch]
            if A:
                z = [min(a[k] for a in A) for k in range(self.K)]
                s = []
                for k in range(self.K):
                    col = [a[k] for a in A]
                    s.append(max(max(col) - min(col), 1e-6 * max(abs(x) for x in col), 1e-12))
                self.scale = (z, s)
        inc = self._compromise() if self.trade_off else self._best_single
        if self.trade_off:
            V = list(self._best)
        else:
            V = list(self.F[inc]) if inc is not None else [None] * self.K
        X = self.P[inc] if inc is not None else {}
        end = {"step": step, "last_call": last_call, "calls": last_call + 1, "incumbent": inc,
               "V": V, "X": {v: dict((X.get(v) or {})) for v in self.variables},
               "archive": list(self._arch) if self.trade_off else None}
        self.ends.append(end)
        if len(self.ends) >= 2:
            self._transition(self.ends[-2], end)

    def _transition(self, a: dict, b: dict) -> None:
        reasons = []
        loss_ok = []
        for k in range(self.K):
            x, y = a["V"][k], b["V"][k]
            ok = x is not None and y is not None and abs(x - y) / max(abs(x), LOSS_FLOOR) < self.rel_gain
            loss_ok.append(ok)
            if not ok:
                reasons.append(["loss", self.objectives[k][0], x, y])
        missing = []
        for v in self.variables:
            for m in self.panel.required(v):
                x = _fin((a["X"].get(v) or {}).get(m))
                y = _fin((b["X"].get(v) or {}).get(m))
                t = _fin((self.tol.get(v) or {}).get(m))
                if x is None or y is None or t is None:
                    missing.append([v, m])
                elif abs(y - x) > t:
                    reasons.append(["score", v, m, x, y, t])
        front = None
        if self.trade_off:
            if self.scale is None or not a["archive"] or not b["archive"]:
                front = None
                reasons.append(["front", "not measurable"])
            else:
                front = eps_indicator([self.F[k] for k in a["archive"]], [self.F[k] for k in b["archive"]],
                                      self.scale[1])
                if front > self.eps_front:
                    reasons.append(["front", front])
        for vm in missing:
            reasons.append(["score_missing"] + vm)
        quiet = not reasons
        self.transitions.append({"q": len(self.transitions) + 1, "from_step": a["step"], "to_step": b["step"],
                                 "calls": b["calls"], "quiet": quiet, "reasons": reasons, "front_eps": front})
        flags = [t["quiet"] for t in self.transitions]
        m = lookback(flags)
        rec = self.transitions[-1]
        rec["m"] = m
        rec["fired"] = bool(m is not None and len(flags) >= m and all(flags[-m:]))
        if rec["fired"] and self.converged_at is None:
            self.converged_at = dict(rec)

    # ── report ─────────────────────────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        q = len(self.transitions)
        last_change = None
        for t in reversed(self.transitions):
            if not t["quiet"]:
                last_change = {"step": t["to_step"], "calls": t["calls"], "reasons": t["reasons"]}
                break
        if self.converged_at is not None:
            verdict = "converged"
        elif q < Q_MIN:
            verdict = "too short to judge"
        else:
            verdict = "not converged within the budget"
        return {"verdict": verdict, "steps_after_start": q, "calls": len(self.F),
                "start_step": self.ends[0]["step"] if self.ends else None,
                "start_calls": self.ends[0]["calls"] if self.ends else None,
                "converged_at": self.converged_at, "last_change": last_change,
                "values": {"Q_MIN": Q_MIN, "M_MIN": M_MIN, "M_MAX": M_MAX, "H": H, "CONF": CONF,
                           "ALPHA": ALPHA, "rel_gain": self.rel_gain, "eps_front": self.eps_front},
                "trade_off": self.trade_off, "objectives": [n for n, _ in self.objectives],
                "variables_judged": list(self.variables), "frozen_scale": self.scale}
