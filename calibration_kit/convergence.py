"""Convergence MARKER — record where the search stopped improving, without stopping it (§5.7).

The problem this solves: a budget alone cannot say whether a calibration searched enough. A run
that used 500 evaluations may have been done at 120, or may still have been improving at 500 —
and the report looked identical either way. The marker makes that visible.

How it works. After every SEARCH evaluation the marker is handed the current INCUMBENT (the
best-so-far point) and its metric PANEL. It fires at the first evaluation where, over a window
of W evaluations, (1) the incumbent's loss gained less than `rel_gain` in relative terms, and
(2) EVERY panel metric moved less than its tolerance — the noise floor of the metric on this
record (panel.py). Both conditions matter: a flat loss with a drifting bias ratio is not
convergence, it is compensation.

Two modes, and `observe` is the default:
  observe   the marker is recorded and the search runs on to the cap. The report then says where
            it would have stopped, how much budget that would have saved, and — the honest part —
            how much the panel STILL changed after the marker. A marker whose post-marker change
            exceeds tolerance is reported as `premature_marker`, not hidden.
  enforce   the search stops at the marker. Only for optimizers whose sampling does not depend on
            the budget: SCE-UA and NSGA-II. DDS is FORCED to observe, because its perturbation
            probability is scheduled as P = 1 - ln(i)/ln(m) against the declared budget m
            (Tolson & Shoemaker 2007) — stopping it early truncates a schedule that was already
            planned, so an early-stopped DDS run is a DIFFERENT algorithm, not the same one cut
            short.

Provisional numbers: W = max(50, 10*(d+1)) evaluations (budgets in units of d+1 follow
Moré & Wild 2009) and rel_gain = 0.005. §5.15 of the handoff calibrates both on cheap models
before any claim is made from them.
"""
from __future__ import annotations
import math

#: optimizers whose sampling is budget-scheduled — they may NEVER be stopped early
BUDGET_SCHEDULED = ("dds",)


def _fin(v):
    """A finite float, or None. NaN and inf are MISSING evidence, never a flat value."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def auto_window(n_params: int) -> int:
    """W = max(50, 10*(d+1)): a window in units of the problem's dimension + 1, floored so a
    1-parameter problem still needs a real stretch of no-improvement."""
    return int(max(50, 10 * (int(n_params) + 1)))


def resolve_mode(requested: str | None, algorithm: str | None) -> tuple[str, list[str]]:
    """(mode, warnings). `enforce` on a budget-scheduled optimizer is downgraded to observe and
    said out loud — silently honouring it would change the algorithm."""
    mode = (requested or "observe").strip().lower()
    warns: list[str] = []
    if mode not in ("observe", "enforce"):
        warns.append(f"convergence.mode '{requested}' is not observe|enforce — using observe")
        mode = "observe"
    algo = (algorithm or "").strip().lower()
    if mode == "enforce" and algo in BUDGET_SCHEDULED:
        warns.append(f"convergence.mode=enforce ignored for {algo}: its sampling is scheduled "
                     f"against the declared budget (Tolson & Shoemaker 2007), so stopping early "
                     f"truncates that schedule — running in observe mode instead")
        mode = "observe"
    return mode, warns


class ConvergenceMarker:
    """Records the first evaluation at which loss AND every panel metric went flat."""

    def __init__(self, tolerances: dict, window: int, rel_gain: float = 0.005,
                 min_evals: int | None = None, mode: str = "observe",
                 min_coverage: float = 1.0, fallback_table: dict | None = None):
        self.tol = {str(v): {str(m): float(t) for m, t in (mets or {}).items()}
                    for v, mets in (tolerances or {}).items()}
        self.W = int(window)
        self.eps = float(rel_gain)
        self.mode = mode
        self.min_evals = int(min_evals if min_evals is not None else window)
        self.L: list = []            # incumbent loss per search evaluation
        self.P: list = []            # incumbent panel per search evaluation
        self.marker: int | None = None
        #: a marker needs panel EVIDENCE: this fraction of the declared (var, metric) pairs must
        #: have been reported as finite numbers inside the window, or it does not fire at all.
        self.min_coverage = float(min_coverage)
        #: when the caller could not measure tolerances up front (an applicator-mode contract has
        #: no pre-search evaluation to read metrics from), the marker learns them from the FIRST
        #: scored evaluation's panel using this table. Without it a run would either fire on the
        #: loss alone — the NSE-only claim this class exists to refuse — or never fire at all.
        self.fallback_table = dict(fallback_table or {})
        self.tolerances_learned = False
        self._coverage: dict = {}
        self._coverage_short: dict = {}

    # ---- the rule -----------------------------------------------------------------
    def update(self, inc_loss, inc_panel) -> bool:
        """Call once per SEARCH evaluation with the current incumbent. Returns True only in
        enforce mode, at the evaluation where the marker fires (the backend's stop signal)."""
        try:
            l = float(inc_loss)
        except (TypeError, ValueError):
            l = float("inf")
        self.L.append(l)
        self.P.append(inc_panel or {})
        if not self.tol and self.fallback_table and inc_panel:
            learned = {}
            for var, mets in (inc_panel or {}).items():
                got = {m: self.fallback_table[m] for m, v in (mets or {}).items()
                       if m in self.fallback_table and _fin(v) is not None}
                if got:
                    learned[var] = got
            if learned:
                self.tol = learned
                self.tolerances_learned = True
        i = len(self.L) - 1
        if self.marker is not None or i < self.min_evals:
            return False
        j = i - self.W
        if j < 0:
            return False
        a, b = self.L[j], self.L[i]
        if not (math.isfinite(a) and math.isfinite(b)):
            return False                      # nothing finite to compare — the search is not done
        gain = (a - b) / max(abs(a), 1e-9)
        if gain >= self.eps:
            return False
        # PANEL FLATNESS. Two rules, both needed (codex review 2026-09-27):
        #  * compare the incumbent metric against EVERY point in the window, not just the window's
        #    far end — an oscillation that happens to return to its starting value would otherwise
        #    read as flat;
        #  * a metric can only be called flat if it was actually REPORTED, as a finite number, at
        #    both ends. Missing or NaN panels used to skip the check, so a run with no panel at all
        #    fired the marker on no evidence. Now the marker needs coverage.
        if not self.tol:
            # NO tolerances at all means no panel evidence exists. The loss going flat on its own is
            # not convergence — that is the NSE-alone claim this class was written to refuse. An
            # empty tolerance set used to skip the coverage check entirely and fire (codex round 2).
            self._coverage_short = {"covered": 0, "wanted": 0, "reason": "no_tolerances"}
            return False
        covered = 0
        wanted = 0
        per_var: dict = {}
        for var, mets in self.tol.items():
            per_var.setdefault(var, 0)
            for m, tol in mets.items():
                wanted += 1
                v = _fin((self.P[i].get(var) or {}).get(m))
                if v is None:
                    continue                  # not reported now — cannot claim this metric is flat
                seen = False
                for k in range(j, i):         # an EARLIER point: a value compared with itself says
                    u = _fin((self.P[k].get(var) or {}).get(m))   # nothing about movement
                    if u is None:
                        continue
                    seen = True
                    if abs(v - u) > tol:
                        return False          # a panel metric is STILL moving -> not converged
                if seen:
                    covered += 1
                    per_var[var] += 1
        short = [v for v, c in per_var.items() if c == 0]
        if short or covered < max(1, int(math.ceil(self.min_coverage * wanted))):
            self._coverage_short = {"covered": covered, "wanted": wanted,
                                    "required_fraction": self.min_coverage,
                                    "variables_without_evidence": short}
            return False                       # not enough panel evidence to call anything flat
        self._coverage = {"covered": covered, "wanted": wanted, "per_variable": per_var}
        self.marker = i
        return self.mode == "enforce"

    # ---- what the report says -----------------------------------------------------
    def last_change_points(self) -> dict:
        """Per metric, the LAST evaluation at which it moved by more than its tolerance. Answers
        'which part of the fit was still moving latest' — the panel metric that decides the marker."""
        out: dict = {}
        for var, mets in self.tol.items():
            for m, tol in mets.items():
                last = None
                for i in range(1, len(self.P)):
                    u = (self.P[i - 1].get(var) or {}).get(m)
                    v = (self.P[i].get(var) or {}).get(m)
                    if u is None or v is None:
                        continue
                    if abs(float(v) - float(u)) > tol:
                        last = i
                out.setdefault(var, {})[m] = last
        return out

    def summary(self, at_cap: bool | None = None, cap: int | None = None) -> dict:
        """`at_cap`: did the run actually reach its cap? `cap`: the declared cap, so a run stopped
        AT the marker still reports the budget it would have saved against the budget it was given
        (codex review 2026-09-27 — measuring savings against an already-shortened run reports 0%)."""
        n = len(self.L)
        if self.marker is None:
            # no records at all means the backend never called the hook (a backend that cannot
            # report per-evaluation). That is "not observed", which is not the same claim as
            # "searched to the cap without converging" — never conflate the two.
            return {"marker_eval": None, "marker_fraction": None,
                    "status": ("not_observed" if n == 0 else
                               ("not_converged_at_cap" if at_cap is not False
                                else "not_converged_stopped_early")),
                    "evals": n, "window": self.W, "rel_gain": self.eps, "mode": self.mode,
                    "panel_coverage": (self._coverage_short or self._coverage or None),
                    "last_change": self.last_change_points()}
        mk = (self.P[self.marker] or {})
        post: dict = {}         # the LARGEST move after the marker, not just the end-to-end one:
        post_end: dict = {}     # an excursion that comes back would otherwise be invisible
        for v, mets in self.tol.items():
            for m in mets:
                a = _fin((mk.get(v) or {}).get(m))
                if a is None:
                    continue
                worst, last = None, None
                for k in range(self.marker, n):
                    b = _fin((self.P[k].get(v) or {}).get(m))
                    if b is None:
                        continue
                    last = b - a
                    if worst is None or abs(b - a) > abs(worst):
                        worst = b - a
                if worst is None:
                    continue
                post.setdefault(v, {})[m] = worst
                post_end.setdefault(v, {})[m] = last
        exceeded = [[v, m] for v in post for m, dv in post[v].items()
                    if abs(dv) > self.tol[v][m]]
        # the LOSS must also have stayed put. A marker the search then improved past by more than
        # rel_gain was premature even if every panel metric happened to stay inside tolerance.
        lm, le = _fin(self.L[self.marker]), _fin(self.L[-1])
        post_gain = None
        if lm is not None and le is not None:
            post_gain = (lm - le) / max(abs(lm), 1e-9)
        loss_moved = bool(post_gain is not None and post_gain > self.eps)
        denom = int(cap) if (cap and int(cap) > 0) else n
        return {"marker_eval": self.marker, "marker_fraction": self.marker / max(denom - 1, 1),
                "post_marker_change": post, "post_marker_change_at_end": post_end,
                "post_marker_exceeds_tol": exceeded,
                "post_marker_loss_gain": (None if post_gain is None else round(post_gain, 6)),
                "post_marker_loss_moved": loss_moved,
                "budget_saved_fraction": 1 - (self.marker + 1) / max(denom, 1),
                "budget_saved_against": ("cap" if denom != n else "observed_evals"),
                # the honest verdict: a marker the run then walked away from was PREMATURE, and
                # says the window/tolerances were too loose — it is never quietly called converged.
                "status": ("converged" if not (exceeded or loss_moved) else "premature_marker"),
                "evals": n, "window": self.W, "rel_gain": self.eps, "mode": self.mode,
                "panel_coverage": (self._coverage or None),
                "tolerances_learned_from_first_eval": self.tolerances_learned,
                "at_cap": at_cap,
                "last_change": self.last_change_points(),
                "loss_at_marker": self.L[self.marker], "loss_at_end": self.L[-1]}


class Incumbent:
    """Tracks the current incumbent across search evaluations, single- or multi-objective.

    Single objective: the best-so-far point by loss.
    Multi objective: the non-dominated archive's member chosen by the engine's OWN rule — smallest
    worst normalized objective (minimax), the same criterion `_select_front_member` commits on,
    minus the holdout gate. Progress for the plateau test is then the archive's IDEAL point (the
    per-objective best-so-far, summed): it can only improve, so a 'gain' is never an artefact of
    the front moving sideways.
    """

    def __init__(self, multi: bool = False):
        self.multi = bool(multi)
        self.best_i: int | None = None
        self.best_loss = float("inf")
        self._F: list = []
        self._panels: list = []
        self._ideal: list | None = None
        self._arch: list = []          # indices of non-dominated evaluations

    @staticmethod
    def _dominates(a, b) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def add(self, losses, panel) -> tuple:
        """Returns (incumbent_loss, incumbent_panel) after folding in this evaluation."""
        f = [float(v) for v in (losses or [float("inf")])]
        i = len(self._F)
        self._F.append(f)
        self._panels.append(panel or {})
        finite = all(math.isfinite(v) for v in f)
        if not self.multi:
            if finite and f[0] < self.best_loss:
                self.best_loss, self.best_i = f[0], i
            return self.best_loss, (self._panels[self.best_i] if self.best_i is not None else {})
        if finite:
            self._ideal = f[:] if self._ideal is None else [min(a, b) for a, b in zip(self._ideal, f)]
            if not any(self._dominates(self._F[k], f) for k in self._arch):
                self._arch = [k for k in self._arch if not self._dominates(f, self._F[k])] + [i]
        if not self._arch:
            return float("inf"), {}
        self.best_i = self._minimax_member()
        self.best_loss = sum(self._ideal) if self._ideal else float("inf")
        return self.best_loss, self._panels[self.best_i]

    def _minimax_member(self) -> int:
        import numpy as np
        F = np.asarray([self._F[k] for k in self._arch], float)
        span = np.ptp(F, axis=0)
        scale = np.maximum(np.abs(F).max(0), 1e-12)
        disc = (span > 1e-9) & (span > 1e-6 * scale)       # same rule as _select_front_member
        if disc.any():
            Fd = F[:, disc]
            Fn = (Fd - Fd.min(0)) / (np.ptp(Fd, axis=0) + 1e-12)
            crit = Fn.max(1)
        else:
            crit = F.sum(1)
        return self._arch[int(np.argmin(crit))]


def make_hook(marker: ConvergenceMarker, incumbent: Incumbent, panel_of, losses_of=None,
              legacy_on_eval=None):
    """The backend's on_eval hook: update the marker from this evaluation, then let the legacy
    `strategy.stop` rule have its say. Order matters — the marker must SEE every evaluation even
    when the legacy rule is the one that stops the run, or the two would disagree about where the
    search converged. A fault in the marker can never stop or fail a run.

    `panel_of(i)` / `losses_of(i)` read evaluation i. A backend's history carries only the
    SCALARIZED loss, so a multi-objective marker needs `losses_of` — calib.py records the
    per-objective vector where it is still intact, in its own evaluate wrapper.
    """
    def _hook(history) -> bool:
        stop_marker = False
        try:
            i = len(history) - 1
            h = history[i] if i >= 0 else {}
            losses = None
            if losses_of is not None:
                losses = losses_of(i)
            if not losses:
                losses = [h.get("loss", float("inf"))]
            stop_marker = marker.update(*incumbent.add(losses, panel_of(i)))
        except Exception:
            stop_marker = False
        stop_legacy = False
        if legacy_on_eval is not None:
            stop_legacy = bool(legacy_on_eval(history))
        return bool(stop_marker or stop_legacy)
    return _hook


def marker_evidence(history, marker_i: int, *, lower=None, upper=None, names=None,
                    topk_frac: float = 0.10, param_width_max: float = 0.30) -> dict:
    """Equifinality AT THE MARKER: how wide are the top set's parameters at the point where the
    search went flat? Same measure as the removed StopRule.assess (normalized range over the search box),
    computed on the history TRUNCATED at the marker — so it describes what was known then, not
    what the extra evaluations to the cap added. Reported, never gating."""
    try:
        import numpy as np
        hs = [h for h in history[:max(marker_i + 1, 1)] if h.get("x") is not None]
        losses = np.array([float(h.get("loss", np.inf)) for h in hs], float)
        fin = np.isfinite(losses) & (losses < 1e29)
        if fin.sum() < 5 or lower is None or upper is None:
            return {}
        idx = np.where(fin)[0]
        order = idx[np.argsort(losses[idx])]
        k = max(5, int(round(topk_frac * len(order))))
        top = order[:k]
        X = np.array([hs[j]["x"] for j in top], float)
        rng = np.array(upper, float) - np.array(lower, float)
        rng[rng == 0] = 1.0
        widths = (X.max(axis=0) - X.min(axis=0)) / rng
        wide = [i for i, w in enumerate(widths) if w > param_width_max]
        return {"k": int(k), "max_normalized_range": round(float(np.max(widths)), 4),
                "limit": param_width_max,
                "per_param": [round(float(w), 3) for w in widths],
                "wide_params": [(names[i] if names and i < len(names) else i) for i in wide],
                "identifiable": bool(float(np.max(widths)) <= param_width_max)}
    except Exception:
        return {}
