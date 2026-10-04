"""pymoo backend — true MULTI-OBJECTIVE (NSGA-II / NSGA-III / MOEA/D).

Use when the dag yields >=2 genuinely-conflicting objectives (e.g. crop yield
magnitude vs phenology timing). Returns the Pareto front; best_x is a knee/min-sum
representative for callers that want a single point.

Non-finite losses (failed/infeasible evals) are clamped to a large finite penalty —
pymoo's selection mishandles inf/NaN, so we map them to a big number that the search
strongly avoids without breaking the dominance sort.
"""
from __future__ import annotations
from .base import Backend, Problem, CalibResult, EarlyStop

_PENALTY = 1e12        # finite stand-in for inf/NaN losses (pymoo-safe)
#: pymoo's own running-metric defaults (Blank & Deb 2020, doi:10.1109/CEC48606.2020.9185546).
#: Verified on this server's pymoo 0.6.1.6: as an OBSERVER they never stop the run, and at
#: ftol=0.005/period=10 the metric did not fire on ZDT1 (NSGA-II, pop 40) within 4,000
#: evaluations, while ftol=0.05 fired at generation 84. So a non-firing observer at our budgets
#: means "not confirmed at this tolerance", never "did not converge" — §5.15 sets the values.
FRONT_FTOL, FRONT_XTOL, FRONT_PERIOD, FRONT_NSKIP = 0.005, 0.0005, 50, 5   # pymoo's shipped defaults (2026-10-04)
PYMOO_PINNED = "0.6.1.6"          # the version these settings were verified against (design §1.6)


def _nondominated(points):
    """(X, F) of the non-dominated, distinct points among [(x, f), ...] — a front rebuilt after the
    kit ended the search must not contain dominated calls (it feeds the reported pick and the
    holdout gate)."""
    import numpy as np
    keep = []
    for i, (xi, fi) in enumerate(points):
        dominated = False
        for j, (xj, fj) in enumerate(points):
            if j != i and all(a <= b for a, b in zip(fj, fi)) and any(a < b for a, b in zip(fj, fi)):
                dominated = True
                break
        if not dominated and not any(points[k][1] == fi for k in keep):
            keep.append(i)
    return (np.asarray([points[k][0] for k in keep], float),
            np.asarray([points[k][1] for k in keep], float))


def _pymoo_version():
    try:
        import pymoo
        return getattr(pymoo, "__version__", None)
    except Exception:
        return None


def make_native_observer(period: int = FRONT_PERIOD, ftol: float = FRONT_FTOL, xtol: float = FRONT_XTOL,
                         n_skip: int = FRONT_NSKIP, on_generation_end=None):
    """pymoo's own DefaultMultiObjectiveTermination (shipped settings unless given), run as an observer through
    native_rules.PymooTest after every generation — exactly as pymoo updates its own termination (2026-10-04
    decision: the kit's convergence rule for NSGA-II / NSGA-III / MOEA-D is pymoo's own). It records the first
    generation where its design-space or objective-space part passes (its max-gen / max-eval caps never count).
    `on_generation_end(gen, n_eval, fired_now, converged)` -> True ends the search at this generation's end (the
    kit's stop mode); the backend then rebuilds the front from every finite call."""
    from pymoo.core.callback import Callback
    from ..native_rules import PymooTest

    class NativeObserver(Callback):
        def __init__(self):
            super().__init__()
            self.test = PymooTest(period=period, xtol=xtol, ftol=ftol, n_skip=n_skip)
            self.generations_done = 0
            self.fired_gen = self.fired_eval = None
            self.stop_requested_gen = None
            self.error = None

        def notify(self, algorithm):
            try:
                self.generations_done = int(algorithm.n_gen)
                fired = self.test.update(algorithm)
                if fired and self.fired_gen is None:
                    self.fired_gen = int(algorithm.n_gen)
                    self.fired_eval = int(algorithm.evaluator.n_eval)
            except Exception as e:                  # the observer may never break a run
                self.error = f"{type(e).__name__}: {e}"
                fired = False
            if on_generation_end is not None and on_generation_end(
                    int(algorithm.n_gen), int(algorithm.evaluator.n_eval), bool(fired), self.fired_gen is not None):
                self.stop_requested_gen = int(algorithm.n_gen)
                raise EarlyStop()

    return NativeObserver()


class PymooBackend(Backend):
    name = "pymoo"

    def __init__(self, algorithm: str = "nsga2", pop_size: int = 40):
        self.algorithm = algorithm          # nsga2 | nsga3 | moead
        self.pop_size = pop_size

    @staticmethod
    def available() -> bool:
        import importlib.util as u
        return u.find_spec("pymoo") is not None

    def optimize(self, problem: Problem, budget: int, seed: int = 0, **kw) -> CalibResult:
        import numpy as np
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.optimize import minimize

        n_var = len(problem.names)
        n_obj = len(problem.objective_names)
        xl = np.asarray(problem.lower, float)
        xu = np.asarray(problem.upper, float)
        # per-call history ({"x", "loss", "gen"}); "loss" is the sum of the finite loss vector, inf for
        # a failed or penalized row. The kit's rule reads the per-objective vector from calib.py, not this.
        history: list[dict] = []
        obs_ref: dict = {}
        capped = {"n": 0}                 # population members skipped once the cap was spent
        on_eval = kw.get("on_eval")       # the kit's per-call hook: True (stop mode) ends the search; at the cap pymoo ends itself
        stopped_early = False
        # remember which evaluated x were FULLY FINITE (codex pymoo:53) — penalized
        # points must NOT be selectable as a "best"/Pareto solution.
        finite_pts = []   # list of (x_tuple, losses) with all-finite losses

        def _clamp(v):
            v = float(v)
            return v if np.isfinite(v) else _PENALTY

        class _P(PymooProblem):
            def __init__(s):
                super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

            def _evaluate(s, X, out, *a, **k):
                F = []
                for row in np.atleast_2d(X):
                    xr = [float(z) for z in row]
                    # THE CAP IS A CAP ON MODEL RUNS. pymoo checks termination between generations
                    # and evaluates a whole population in between, so a cap that is not a multiple
                    # of pop_size used to overshoot (41 -> 80 evaluations; codex round 2). Once the
                    # cap is spent, the remaining members of the generation are penalized WITHOUT
                    # running the model, and the overrun is recorded.
                    if len(history) >= int(budget):
                        F.append([_PENALTY] * n_obj)
                        capped["n"] += 1
                        continue
                    losses = problem.evaluate(xr)
                    # Finiteness is judged on the RAW evaluator result (codex/kimi 2026-08-30): an
                    # empty / wrong-length / non-finite result is a FAILED eval — it must NEVER become a
                    # selectable solution or a finite history loss. (A `[_PENALTY]*n_obj` fill would read
                    # as finite because _PENALTY is finite, re-introducing the pollution this guards.)
                    all_finite = (bool(losses) and len(losses) == n_obj
                                  and all(np.isfinite(z) for z in losses))
                    gen = int(obs_ref["obs"].generations_done) + 1 if obs_ref.get("obs") else None
                    if all_finite:
                        fl = [float(z) for z in losses]
                        finite_pts.append((xr, fl))
                        history.append({"x": xr, "loss": float(sum(fl)), "gen": gen})
                    else:
                        history.append({"x": xr, "loss": float("inf"), "gen": gen})
                    # the kit's per-call hook (stop mode; at the cap it lets pymoo end itself): True ends it. EarlyStop unwinds
                    # pymoo's minimize(); finite_pts already holds every finite call so far.
                    if on_eval is not None and on_eval(history):
                        raise EarlyStop()
                    # pymoo needs a FINITE objective row for its dominance sort -> clamp separately.
                    row_losses = losses if (losses and len(losses) == n_obj) else [_PENALTY] * n_obj
                    F.append([_clamp(z) for z in row_losses])
                out["F"] = np.asarray(F, float)

        algo = self._make_algorithm(n_obj)
        # pymoo's own multi-objective termination is RECORDED as the optimizer's native verdict
        # (period 10 generations, n_skip 1) and never ends the search (design §1.6, gap 2a: the old
        # front_mode="enforce" path is removed). The cap (this backend's own check and pymoo's n_eval limit) and the kit's stop mode (via on_eval) end it.
        period = int(kw.get("front_period", FRONT_PERIOD))
        ftol = float(kw.get("front_ftol", FRONT_FTOL))
        n_skip = int(kw.get("front_n_skip", FRONT_NSKIP))
        obs = make_native_observer(period=period, ftol=ftol, xtol=float(kw.get("front_xtol", FRONT_XTOL)),
                                   n_skip=n_skip, on_generation_end=kw.get("on_generation_end"))
        obs_ref["obs"] = obs
        # calls per generation: MOEA/D runs one call per reference direction, not pop_size
        _per_gen = int(getattr(algo, "pop_size", None) or 0) or (len(getattr(algo, "ref_dirs", []) or [])
                                                                  or int(self.pop_size))
        gens_available = int(budget) / max(1, _per_gen)
        warnings: list[str] = []
        _pv = _pymoo_version()
        if _pv and _pv != PYMOO_PINNED:
            warnings.append(f"pymoo {_pv} is not the pinned {PYMOO_PINNED}: the native verdict's "
                            f"settings were verified on {PYMOO_PINNED}")
        if gens_available < period + n_skip + 1:
            # pymoo's own test cannot pass before period + n_skip + 1 generations
            warnings.append(f"budget_too_small_to_confirm_front_convergence: {gens_available:.1f} "
                            f"generations available ({budget} evals / {_per_gen} per generation) < "
                            f"period + n_skip + 1 ({period + n_skip + 1})")
        term = ("n_eval", int(budget))
        try:
            res = minimize(_P(), algo, term, seed=seed, verbose=False, callback=obs)
        except EarlyStop:
            stopped_early = True          # the stop rule decided; finite_pts holds every finite eval so far
            res = None
        n_done = len(history)
        # the kit's hook ends the search AT the cap too: that is the cap, not an early stop
        _gen_stop = obs.stop_requested_gen is not None
        termination = {"status": ("stopped_by_kit_rule" if _gen_stop else
                                  "stopped_by_kit" if (stopped_early and n_done < int(budget)) else "max_evals"),
                       "stopped_by_kit_at_generation": obs.stop_requested_gen,
                       "pop_size": self.pop_size, "calls_per_generation": _per_gen,
                       "generations_available": round(gens_available, 2),
                       "generations_done": obs.generations_done,
                       # pymoo's native verdict, recorded only (never ends the search)
                       "native_verdict": dict(obs.test.summary(), fired_gen=obs.fired_gen, fired_eval=obs.fired_eval,
                                              per_generation=obs.test.records,
                                              pymoo_version=_pymoo_version(), error=obs.error),
                       "cap_reached_mid_generation": capped["n"],
                       "warnings": warnings}

        if not finite_pts:
            # every evaluation was infeasible/failed -> NOT a calibration (so
            # calib.py routes to the failure/restore path, not a bogus "completed").
            return CalibResult(best_x=[], best_loss=[float("inf")] * n_obj,
                               backend=f"pymoo:{self.algorithm}", n_evaluations=len(history),
                               history=history, stopped_early=stopped_early,
                               termination=termination,
                               notes="no fully-finite evaluation (all penalized)")
        _rebuilt = False
        if res is None or res.X is None:
            # ended by the kit (res is None) or pymoo returned no front -> rebuild it from every finite
            # call, filtered to the non-dominated, distinct points.
            _rebuilt = True
            X, F = _nondominated(finite_pts)
        else:
            # keep only returned-front points that were actually finite (drop penalized)
            Xr = np.atleast_2d(res.X); Fr = np.atleast_2d(res.F)
            keep = [i for i in range(len(Fr)) if np.all(np.isfinite(Fr[i]))
                    and Fr[i].max() < _PENALTY / 2]
            if keep:
                X = Xr[keep]; F = Fr[keep]
            else:
                _rebuilt = True     # returned front was all-penalized -> rebuild from finite evals
                X, F = _nondominated(finite_pts)
        # single representative = min normalized-sum across the front (a simple knee proxy)
        Fn = (F - F.min(0)) / (np.ptp(F, axis=0) + 1e-12)
        best_i = int(Fn.sum(1).argmin())
        return CalibResult(
            best_x=X[best_i].tolist(), best_loss=F[best_i].tolist(),
            pareto_x=X.tolist(), pareto_f=F.tolist(), n_evaluations=len(history),
            history=history, stopped_early=stopped_early, termination=termination,
            backend=f"pymoo:{self.algorithm}",
            notes=(f"{'non-dominated set of every finite call' if _rebuilt else 'Pareto front'} of {len(X)} "
                   f"solution(s); best = min-normalized-sum knee"
                   + ("; stopped early by the stop rule" if stopped_early else "")))

    def _make_algorithm(self, n_obj: int):
        if self.algorithm == "nsga2":
            from pymoo.algorithms.moo.nsga2 import NSGA2
            return NSGA2(pop_size=self.pop_size)
        from pymoo.util.ref_dirs import get_reference_directions
        # das-dennis partitions tuned so the reference set ~ pop_size
        n_part = max(1, 12 if n_obj <= 2 else 6)
        ref = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
        if self.algorithm == "nsga3":
            from pymoo.algorithms.moo.nsga3 import NSGA3
            return NSGA3(ref_dirs=ref, pop_size=self.pop_size)
        if self.algorithm == "moead":
            from pymoo.algorithms.moo.moead import MOEAD
            return MOEAD(ref_dirs=ref, n_neighbors=min(15, len(ref)))
        raise ValueError(f"unknown pymoo algorithm {self.algorithm!r}")
