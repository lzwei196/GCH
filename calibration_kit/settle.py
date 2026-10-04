"""The DESCRIPTIVE settle point, for optimizers with no convergence rule of their own (DDS, random search, the LLM
in-loop arm) — plan rev 30, Method: "settled by run N of M".

Feed every call in order (single objective). The best point is the lowest scalar so far. A CHANGE is a call where the
best point moved and, compared with the state at the LAST change (not the previous call — so slow drift adds up and
is caught), the best loss improved by >= rel_gain (relative, floor 1e-3) or a required watched score of a judged
variable moved by more than its tolerance. N = the last change (1-based run number); a missing watched score at a new
best point counts as a change (fail closed: we cannot show it did not move). Nothing here judges convergence; it only
says when the search last changed in a way that matters.
"""
from __future__ import annotations

from .rule import LOSS_FLOOR, REL_GAIN, _fin


class SettlePoint:
    def __init__(self, panel, tol: dict, rel_gain: float = REL_GAIN, variables=None):
        self.panel = panel
        self.tol = {str(v): dict(m) for v, m in (tol or {}).items()}
        self.rel_gain = float(rel_gain)
        self.variables = list(variables) if variables is not None else list(panel.variables)
        self.n = 0
        self._best = None             # (call index, scalar, panel)
        self._ref = None              # state at the last change: (scalar, panel)
        self.changes: list = []       # [{"run": N (1-based), "why": [...]}]

    def add(self, scalar, panel=None) -> None:
        i = self.n
        self.n += 1
        sc = _fin(scalar)
        if sc is None or (self._best is not None and not sc < self._best[1]):
            return
        self._best = (i, sc, panel or {})
        if self._ref is None:
            self._ref = (sc, panel or {})
            self.changes.append({"run": i + 1, "why": [["first finite run"]]})
            return
        why = []
        x = self._ref[0]
        g = (x - sc) / max(abs(x), LOSS_FLOOR)
        if g >= self.rel_gain:
            why.append(["loss", x, sc])
        for v in self.variables:
            for m in self.panel.required(v):
                a = _fin((self._ref[1].get(v) or {}).get(m))
                b = _fin(((panel or {}).get(v) or {}).get(m))
                t = _fin((self.tol.get(v) or {}).get(m))
                if a is None or b is None or t is None:
                    why.append(["score_missing", v, m])
                elif abs(b - a) > t:
                    why.append(["score", v, m, a, b, t])
        if why:
            self._ref = (sc, panel or {})
            self.changes.append({"run": i + 1, "why": why})

    def summary(self) -> dict:
        last = self.changes[-1] if self.changes else None
        return {"runs": self.n, "settled_by_run": last["run"] if last else None,
                "share_after": (1.0 - last["run"] / self.n) if (last and self.n) else None,
                "last_change": last, "n_changes": len(self.changes), "rel_gain": self.rel_gain,
                "wording": (f"settled by run {last['run']} of {self.n}" if last else "no finite run")}
