"""The optimizers' OWN stopping rules, on our scores (validated 2026-10-04, part C: paper_replay/validation_native/,
PREREGISTRATION_C.md, codex SOUND; result: spotpy_loss and spotpy_scores PASS 0/59).

SCE-UA — SpotpyTest: SPOTPY sceua.py's own loop-end test, verbatim: stop when the population has shrunk
(gnrng < peps), or, once nloop >= kstop, when the best objective changed by at most pcento percent over the last
kstop loops: |c[n-1] - c[n-kstop]| * 100 / mean(|c[n-kstop:n]|) <= pcento. "On our scores" (watch_scores=True): the
same percent test must ALSO pass for every watched score of the best point at each loop end (fail closed: a missing
score never passes). Settings: the classic SCE-UA values kstop 10, pcento 0.1, peps 0.001 (SPOTPY's shipped defaults
kstop 100, pcento 1e-7, peps 1e-7 never fired in any validation run). Wording: "the SCE-UA objective test with
classic settings" — never "SPOTPY defaults".

NSGA-II / NSGA-III / MOEA-D — PymooTest: pymoo's own DefaultMultiObjectiveTermination (xtol 0.0005, ftol 0.005,
n_skip 5, period 50 as shipped), updated after every generation exactly as pymoo updates its own termination;
convergence = its design-space (x) or objective-space (f) part reaching 1 (its max-generation / max-evaluation caps
are budget limits, never convergence).

DDS / random / LLM arms have no rule of their own: settle.SettlePoint ("settled by run N of M").

Nothing here stops a search by itself; the backends ask it at each loop / generation end.
"""
from __future__ import annotations
import math

KSTOP, PCENTO, PEPS = 10, 0.1, 0.001
WATCHED_DEFAULT = ("r", "alpha", "beta", "lnnse")


def _fin(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def pct_change(series, nloop, kstop=KSTOP):
    """SPOTPY's formula: |c[n-1] - c[n-kstop]| * 100 / mean(|c[n-kstop:n]|); 0 when that mean is 0; None if any
    value in the window is missing."""
    win = series[nloop - kstop:nloop]
    if len(win) != kstop or any(_fin(x) is None for x in win):
        return None
    den = sum(abs(float(x)) for x in win) / kstop
    absolute_change = abs(float(series[nloop - 1]) - float(series[nloop - kstop])) * 100
    return 0.0 if den == 0.0 else absolute_change / den


class SpotpyTest:
    """Feed it every SCE-UA loop end in order (loop 1, 2, ...): add_loop(loop, bestf, gnrng, scores)."""

    def __init__(self, kstop=KSTOP, pcento=PCENTO, peps=PEPS, watch_scores=True, watched=WATCHED_DEFAULT,
                 variables=None):
        self.kstop, self.pcento, self.peps = int(kstop), float(pcento), float(peps)
        self.watch_scores = bool(watch_scores)
        self.watched = tuple(watched)
        self.variables = list(variables) if variables is not None else None
        self.crit: list = []
        self.scores: dict = {}         # (var, metric) -> list per loop
        self.loops: list = []
        self.fired_at = None           # (loop, why) of the first firing
        self.records: list = []

    def add_loop(self, loop, bestf, gnrng=None, scores=None) -> bool:
        n = len(self.loops) + 1
        if loop != n:
            raise ValueError(f"loop {loop} given, expected {n} (loops must be 1, 2, ... in order)")
        self.loops.append(loop)
        self.crit.append(_fin(bestf))
        sc = scores or {}
        vars_ = self.variables if self.variables is not None else sorted(sc)
        for k in set((v, m) for v in vars_ for m in self.watched) | set(self.scores):
            self.scores.setdefault(k, [None] * (n - 1)).append(_fin((sc.get(k[0]) or {}).get(k[1])))
        # every (variable, score) ever watched is checked, so one that disappears fails closed (codex A3a r1 #1)
        keys = sorted(self.scores)
        why = None
        g = _fin(gnrng)
        if g is not None and g < self.peps:
            why = "population spread gnrng < peps"
        elif n >= self.kstop:
            pc = pct_change(self.crit, n, self.kstop)
            ok = pc is not None and pc <= self.pcento
            parts = {"objective": pc}
            if ok and self.watch_scores:
                for k in keys:
                    p = pct_change(self.scores[k], n, self.kstop)
                    parts[f"{k[0]}:{k[1]}"] = p
                    if p is None or p > self.pcento:
                        ok = False
                if not keys:
                    ok = False                         # watching scores but none given: fail closed
            if ok:
                why = f"best objective{' and watched scores' if self.watch_scores else ''} changed by <= " \
                      f"{self.pcento} % over the last {self.kstop} loops"
            self.records.append({"loop": loop, "pct": parts, "fired": ok})
        fired = why is not None
        if fired and self.fired_at is None:
            self.fired_at = {"loop": loop, "why": why}
        return fired

    def summary(self) -> dict:
        return {"rule": "SCE-UA objective test" + (" on the objective and watched scores" if self.watch_scores else ""),
                "settings": {"kstop": self.kstop, "pcento": self.pcento, "peps": self.peps,
                             "note": "classic SCE-UA settings, not SPOTPY's shipped defaults"},
                "loops_seen": len(self.loops), "fired_at": self.fired_at,
                "verdict": ("converged" if self.fired_at else
                            ("too short to judge" if len(self.loops) < self.kstop else "not converged within the budget"))}


class PymooTest:
    """pymoo's own DefaultMultiObjectiveTermination as an observer: call update(algorithm) once per generation,
    after pymoo has updated its own termination (from a Callback). period / settings as shipped unless given."""

    def __init__(self, period=50, xtol=0.0005, ftol=0.005, n_skip=5):
        from pymoo.termination.default import DefaultMultiObjectiveTermination
        self.t = DefaultMultiObjectiveTermination(xtol=xtol, ftol=ftol, n_skip=n_skip, period=period)
        self.settings = {"period": period, "xtol": xtol, "ftol": ftol, "n_skip": n_skip}
        self.gens = 0
        self.fired_at = None
        self.records: list = []

    def update(self, algorithm) -> bool:
        self.t.update(algorithm)
        self.gens += 1
        x, f = float(self.t.x.perc), float(self.t.f.perc)
        fired = max(x, f) >= 1.0
        self.records.append({"n_gen": int(getattr(algorithm, "n_gen", self.gens)), "x": x, "f": f, "fired": fired})
        if fired and self.fired_at is None:
            self.fired_at = {"n_gen": int(getattr(algorithm, "n_gen", self.gens)),
                             "n_eval": int(getattr(getattr(algorithm, "evaluator", None), "n_eval", -1)),
                             "why": "design-space" if x >= 1.0 else "objective-space"}
        return fired

    def summary(self) -> dict:
        return {"rule": "pymoo DefaultMultiObjectiveTermination (replayed on the same generation states)",
                "settings": self.settings, "generations_seen": self.gens, "fired_at": self.fired_at,
                "verdict": ("converged" if self.fired_at else
                            # earliest possible firing: 1 update without a previous one + n_skip + a full period
                            ("too short to judge" if self.gens < self.settings["period"] + self.settings["n_skip"] + 1
                             else "not converged within the budget"))}
