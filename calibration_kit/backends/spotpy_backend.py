"""SPOTPY backend — the model-agnostic DEFAULT (DDS, SCE-UA, DREAM).

SPOTPY is purpose-built for environmental-model calibration/uncertainty and ships
DDS, SCE-UA, DREAM, PA-DDS, NSGA-II, and NSE/KGE/PBIAS. We use it as the
SINGLE-OBJECTIVE default (DDS first, then SCE-UA) and for DREAM posterior sampling.
For TRUE multi-objective use pymoo_backend (cleaner MOEA support).

Wiring notes (spotpy 1.6.x):
  - DDS HARDCODES optimization_direction="maximize", so our loss-minimization is
    expressed by returning -loss from objectivefunction.
  - We don't trust ParameterSet extraction for the best vector; instead we record
    every (x, loss) the evaluator sees and pick the min-loss vector ourselves.
"""
from __future__ import annotations
import io
import re
import sys
from .base import Backend, Problem, CalibResult, EarlyStop


class _Tee(io.TextIOBase):
    """Write to the real stdout AND keep a copy. SCE-UA says why it stopped only by printing it,
    so the run's log has to be read back — but a long calibration must still print live."""

    def __init__(self, real, on_line=None):
        self.real = real
        self.buf = io.StringIO()
        self.on_line = on_line            # called with each written chunk (loop index, R-hat crossing)

    def write(self, s):
        try:
            self.real.write(s)
        except Exception:
            pass
        if self.on_line is not None:
            try:
                self.on_line(s)
            except Exception:
                pass
        return self.buf.write(s)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def getvalue(self):
        return self.buf.getvalue()


def classify_sceua_output(text: str, kstop=None, pcento=None, peps=None, ngs=None) -> dict:
    """WHY did SCE-UA stop? SPOTPY only prints it, so this reads the printed verdict.

    The three endings mean different things and were indistinguishable in our records:
      max_trials                the evaluation cap ran out — the search was CUT OFF, and a
                                result from it says nothing about convergence;
      population_converged      the complexes collapsed into a small parameter space (gnrng < peps);
      improvement_below_pcento  the best point improved by less than pcento over the last kstop
                                loops — the algorithm's own "done".
    Decided on the LAST evolution loop's messages, because the per-loop lines repeat. Also returns
    the final geometric range and percent improvement, which is the evidence for the verdict.
    """
    t = text or ""
    # only the final loop's block decides; the earlier loops print the same lines
    marks = [m.start() for m in re.finditer(r"ComplexEvo loop #\d+ in progress", t)]
    tail = t[marks[-1]:] if marks else t
    status = "unknown"
    if "MAXIMUM NUMBER OF TRIALS" in tail:
        status = "max_trials"
    elif "POPULATION HAS CONVERGED" in tail:
        status = "population_converged"
    elif "CONVERGENCY HAS ACHIEVED" in tail or "IMPROVED IN LAST" in tail and "LESS THAN THE USER-SPECIFIED" in tail:
        status = "improvement_below_pcento"
    elif "MAXIMAL NUMBER OF LOOPS" in tail:
        status = "max_loops"
    elif "MAXIMUM NUMBER OF TRIALS" in t:
        status = "max_trials"                      # burn-in already exhausted the cap
    out = {"status": status, "settings": {k: v for k, v in
                                          (("kstop", kstop), ("pcento", pcento),
                                           ("peps", peps), ("ngs", ngs)) if v is not None}}
    m = re.search(r"SEARCH WAS STOPPED AT TRIAL NUMBER:\s*(\d+)", t)
    if m:
        out["stopped_at_trial"] = int(m.group(1))
    m = re.search(r"NORMALIZED GEOMETRIC RANGE\s*=\s*([0-9.eE+-]+)", t)
    if m:
        out["geometric_range"] = float(m.group(1))
    m = re.findall(r"IMPROVED IN LAST (\d+) LOOPS BY ([0-9.eE+-]+) PERCENT", t)
    if m:
        out["last_improvement_pct"] = float(m[-1][1])
        out.setdefault("settings", {}).setdefault("kstop", int(m[-1][0]))
    m = re.findall(r"Updated convergence criteria:\s*([0-9.eE+-]+)", t)
    if m:
        out["convergence_criteria"] = float(m[-1])
    m = re.findall(r"ComplexEvo loop #(\d+)", t)
    if m:
        out["loops"] = int(m[-1])
    m = re.search(r"NUMBER OF DISCARDED TRIALS:\s*(\d+)", t)
    if m:
        out["discarded_trials"] = int(m.group(1))
    return out


class SpotpyBackend(Backend):
    name = "spotpy"
    # Single-objective optimizers/samplers verified through this backend.
    # (PA-DDS is multi-objective — route it via the multi-objective path, not here.)
    _ALGOS = ("dds", "sceua", "dream")

    def __init__(self, algorithm: str = "dds"):
        if algorithm not in self._ALGOS:
            raise ValueError(f"unknown spotpy algorithm {algorithm!r}; choose {self._ALGOS}")
        self.algorithm = algorithm

    @staticmethod
    def available() -> bool:
        import importlib.util as u
        return u.find_spec("spotpy") is not None

    def optimize(self, problem: Problem, budget: int, seed: int = 0, **kw) -> CalibResult:
        import spotpy
        import numpy as np

        if problem.is_multi_objective:
            raise ValueError("SpotpyBackend is single-objective; use PymooBackend for "
                             "multi-objective problems.")
        try:
            np.random.seed(seed)
        except Exception:
            pass

        history: list[dict] = []          # every (x, loss) the optimizer evaluated
        names, lo, hi = problem.names, problem.lower, problem.upper
        on_eval = kw.get("on_eval")       # the kit's per-call hook (cap, stop mode); True ends the search
        stopped_early = False
        # what SPOTPY only PRINTS, read live (design §2.8, gap 2h): SCE-UA's evolution-loop index for
        # every call, and the call at which DREAM declared R-hat < 1.2
        live = {"loop": 0, "dream_converged_call": None}

        def _on_line(chunk):
            for m in re.finditer(r"ComplexEvo loop #(\d+)", chunk):
                live["loop"] = int(m.group(1))
            if "Convergence has been achieved" in chunk and live["dream_converged_call"] is None:
                live["dream_converged_call"] = len(history) - 1

        class _Setup:
            def parameters(self):
                return spotpy.parameter.generate([
                    spotpy.parameter.Uniform(n, float(a), float(b))
                    for n, a, b in zip(names, lo, hi)])

            def simulation(self, vector):
                x = [float(v) for v in vector]
                losses = problem.evaluate(x)
                loss = float(losses[0]) if losses else float("inf")
                # spotpy can't store non-finite "simulations"; clamp for storage but
                # keep the real loss for our own best-tracking.
                rec = {"x": x, "loss": loss}
                if self_algo == "sceua":
                    rec["loop"] = live["loop"]           # 0 = the burn-in population
                history.append(rec)
                if on_eval is not None and on_eval(history):
                    raise EarlyStop()          # caught around sampler.sample below
                return [loss if np.isfinite(loss) else 1e30]

            def evaluation(self):
                return [0.0]              # target loss is 0

            def objectivefunction(self, simulation, evaluation):
                # DDS HARDCODES maximize -> return -loss so maximize(-loss) minimizes
                # loss. Other algos accept optimization_direction="minimize" below and
                # take +loss directly. (best-vector tracking is via `history`, so it's
                # direction-independent — only the sampler's SEARCH direction matters.)
                s = simulation[0] if simulation else 1e30
                return -s if _maximize else s

        # dds HARDCODES maximize; DREAM is a Bayesian sampler that draws proportional to the
        # objective read AS A LIKELIHOOD (higher == better fit), so it too must be handed -loss and
        # maximize — feeding +loss/minimize made it sample toward the WORST fits (latent bug: no dream
        # run existed before). sceua minimizes +loss directly.
        _maximize = self.algorithm in ("dds", "dream")
        self_algo = self.algorithm
        sampler_cls = getattr(spotpy.algorithms, self.algorithm)
        on_loop_end = kw.get("on_loop_end")  # SCE-UA loop-end hook (loop, gnrng, bestf, calls) -> True to stop
        if self.algorithm == "sceua":
            # the kit's copy of SPOTPY 1.6.7 sceua.py: identical search, plus a loop-end hook and a per-loop record
            # of what SPOTPY's own test sees (backends/sceua_hooked.py; 2026-10-04)
            from .sceua_hooked import sceua_hooked as sampler_cls
        _dir_kw = {} if _maximize else {"optimization_direction": "minimize"}
        sampler = sampler_cls(_Setup(), dbname="calib", dbformat="ram",
                              save_sim=False, **_dir_kw)
        if self.algorithm == "sceua" and on_loop_end is not None:
            sampler.loop_hook = lambda n, g, f: bool(on_loop_end(n, g, f, len(history)))
        # DDS: hand it the DEFAULT vector as its start point when the caller supplies one. DDS
        # perturbs around the current best, so starting at the model's own defaults (a point the
        # model is known to run at) beats a random draw — and makes a seed's run reproducible from
        # a stated start. Ignored by the other algorithms.
        _x0 = kw.get("x_initial")
        _sample_kw = {}
        if self.algorithm == "dds" and _x0 is not None and len(list(_x0)) == len(names):
            _sample_kw["x_initial"] = np.asarray([float(v) for v in _x0], float)
        # SCE-UA settings are recorded WITH the verdict: "converged" means nothing without the
        # kstop/pcento/peps it converged against. These are spotpy's own defaults for 1.6.x unless
        # the caller overrides them.
        _sce, _sce_warn = {}, []
        if self.algorithm == "sceua":
            for k, dflt in (("ngs", 20), ("kstop", 100), ("pcento", 1e-7), ("peps", 1e-7)):
                v = kw.get(k, dflt)
                if v is not None:
                    _sce[k] = v
            _sample_kw.update(_sce)
            # REACHABILITY OF SCE-UA's OWN CONVERGENCE TEST (found 2026-09-27 in the GR4J C-test
            # logs): the objective-improvement test only runs once nloop >= kstop, and one loop
            # costs about ngs * npg evaluations with npg = 2*d+1. With spotpy's default kstop=100,
            # a 3,000-evaluation budget on a 4-parameter model buys ~7 loops — so the test was
            # never evaluated, every run ended on "MAXIMUM NUMBER OF TRIALS", and reading that as
            # "SCE-UA converged" was wrong. Say it up front instead of leaving it in the log.
            _npg = 2 * len(names) + 1
            # design §1.6: the burn-in costs one population and each step 1-3 model calls
            _pop = max(1, int(_sce.get("ngs", 20)) * _npg)
            _loops = max(0.0, (int(budget) - _pop) / (2.0 * _pop))
            # SPOTPY's own internal test (kstop 100 by default) cannot run at most budgets; the kit's convergence
            # rule (native_rules.SpotpyTest, kstop 10, applied through the loop hook) needs at least 10 loops
            if _loops < 10:
                _sce_warn.append(f"about {_loops:.1f} evolution loops fit in {int(budget)} evaluations "
                                 f"(ngs={_sce.get('ngs')}, npg={_npg}): fewer than the 10 loops the kit's convergence "
                                 f"rule needs, so it will report 'too short to judge'")
                print(f"  [calib] WARNING sceua: {_sce_warn[-1]}", flush=True)
        termination = None
        # how many repetitions SPOTPY is told: the CAP is enforced by the kit's call hook, never by
        # SPOTPY's counter (SCE-UA's counts every trial plus one per step, so it is given
        # 2*cap + ngs(2d+1); DDS's schedule is built on the budget, so it gets exactly the cap)
        _reps = int(kw.get("repetitions") or budget)
        _tee = _Tee(sys.stdout, _on_line) if self.algorithm in ("sceua", "dream") else None
        _prev_stdout = sys.stdout
        if _tee is not None:
            sys.stdout = _tee                 # SCE-UA prints its verdict; keep a copy, keep it live
        try:
            sampler.sample(_reps, **_sample_kw)
        except EarlyStop:
            stopped_early = True             # the kit's hook ended it (cap or stop mode); history holds every call
        finally:
            if _tee is not None:
                sys.stdout = _prev_stdout
            if self.algorithm == "sceua":
                termination = classify_sceua_output(_tee.getvalue(), **_sce)
                termination["repetitions_given"] = _reps
                if stopped_early:
                    # the kit's hook ended it (cap or stop mode) before SPOTPY printed a verdict
                    termination["status"] = "ended_by_kit"
                termination["early_stopped_by_rule"] = stopped_early
                _pop = max(1, int(_sce.get("ngs", 20)) * (2 * len(names) + 1))
                termination["loops_available"] = round(max(0.0, (int(budget) - _pop) / (2.0 * _pop)), 2)
                termination["warnings"] = _sce_warn
                # a cap-ended search is NOT a converged one — the report must not let the two read
                # the same, which is exactly how "stopped under the cap" was mis-read before.
                termination["converged_by_own_rule"] = (
                    termination.get("status") in ("population_converged", "improvement_below_pcento"))
                termination["loop_record"] = list(getattr(sampler, "loop_record", []) or [])
                _kl = getattr(sampler, "stopped_by_kit_at_loop", None)
                termination["stopped_by_kit_at_loop"] = _kl
                if _kl is not None:
                    termination["status"] = "stopped_by_kit_rule"       # ended at a loop end by the kit's rule
                    termination["converged_by_own_rule"] = False         # SPOTPY's own test did not end it
                    stopped_early = True
                    termination["early_stopped_by_rule"] = True
        if self.algorithm == "dds":
            # DDS has no termination test at all: its perturbation probability is scheduled
            # against the declared budget (P = 1 - ln(i)/ln(m), Tolson & Shoemaker 2007), so it
            # always ends exactly at the cap. Saying so is the point — a DDS run that used its
            # whole budget is not evidence of convergence.
            # the kit's hook also ends DDS at the cap (the last call of its schedule): that is the
            # schedule completing, not an early stop
            _complete = len(history) >= _reps
            termination = {"status": ("budget_schedule_complete" if (_complete or not stopped_early)
                                      else "stopped_by_kit"),
                           "note": "DDS is budget-scheduled; it has no convergence test",
                           "x_initial": ("default_vector" if "x_initial" in _sample_kw else None),
                           "early_stopped_by_rule": stopped_early}

        if self.algorithm == "dream":
            try:
                _rh = [list(map(float, np.asarray(r, float).ravel())) for r in (getattr(sampler, "r_hats", None) or [])]
            except Exception:
                _rh = []
            termination = {"status": ("rhat_converged" if live["dream_converged_call"] is not None
                                      else "no_rhat_convergence"),
                           "rhat_recorded": bool(_rh),
                           "rhat_reached": live["dream_converged_call"] is not None,
                           "rhat_converged_at_call": live["dream_converged_call"],
                           "rhat_final": (_rh[-1] if _rh else None),
                           "convergence_limit": 1.2, "runs_after_convergence": 100,
                           "ended_at_call": len(history) - 1,
                           "early_stopped_by_rule": stopped_early}

        finite = [h for h in history if h["loss"] < 1e30]
        if not finite:
            return CalibResult(best_x=[], best_loss=[float("inf")], backend=self.name,
                               n_evaluations=len(history), termination=termination,
                               notes="no finite evaluation — all runs infeasible/failed")
        best = min(finite, key=lambda h: h["loss"])

        # DREAM is a POSTERIOR sampler — its deliverable is the parameter distribution, not just the
        # best point. The posterior is the ACCEPTED post-burn-in chain (sampler.bestpar, shape
        # (nchains, niter, nparam) — the state AFTER Metropolis accept/reject; getdata() stores the
        # PROPOSED runs before accept/reject and best-X% filtering is GLUE, which understates
        # uncertainty — codex review). Discard the first half as burn-in, flatten chains, drop
        # non-finite, summarize per-parameter percentiles + 95% credible interval. dds/sceua leave
        # posterior=None. Best-effort: any failure degrades to None, never breaks the run.
        posterior = None
        if self.algorithm == "dream":
            try:
                bp = np.asarray(getattr(sampler, "bestpar"), float)   # (nchains, niter, nparam)
                if bp.ndim == 3 and bp.shape[2] == len(names):
                    # DREAM stops EARLY at convergence, leaving bestpar's tail pre-allocated
                    # (NaN/zero). Keep only the FILLED iterations, then discard the first half of
                    # those as burn-in — never split on the allocated length.
                    valid = (np.isfinite(bp).all(axis=(0, 2))
                             & (np.abs(bp).sum(axis=(0, 2)) > 0))       # (niter,) filled iterations
                    idx = np.where(valid)[0]
                    if idx.size < 8:
                        raise ValueError(f"only {idx.size} filled DREAM iterations")
                    lo, hi = int(idx[0]), int(idx[-1]) + 1
                    burn = lo + (hi - lo) // 2                          # first half of filled = burn-in
                    samples = bp[:, burn:hi, :].reshape(-1, bp.shape[2])
                    samples = samples[np.isfinite(samples).all(axis=1)]
                    posterior = {"method": "dream_postburnin_chain",
                                 "n_chains": int(bp.shape[0]), "burn_in_frac": 0.5,
                                 "filled_iters": hi - lo, "n_samples": int(samples.shape[0])}
                    for j, nm in enumerate(names):
                        col = samples[:, j]
                        col = col[np.isfinite(col)]
                        if col.size < 2:
                            continue
                        p2, p50, p97 = (float(np.percentile(col, q)) for q in (2.5, 50, 97.5))
                        posterior[nm] = {"p2_5": p2, "p50": p50, "p97_5": p97,
                                         "mean": float(np.mean(col)), "std": float(np.std(col)),
                                         "ci95_width": p97 - p2}
                else:
                    posterior = {"_error": f"unexpected bestpar shape {getattr(bp,'shape',None)}"}
            except Exception as e:
                posterior = {"_error": f"{type(e).__name__}: {e}"}

        return CalibResult(
            best_x=best["x"], best_loss=[best["loss"]],
            n_evaluations=len(history),
            history=history, backend=f"spotpy:{self.algorithm}", posterior=posterior,
            stopped_early=stopped_early, termination=termination,
            notes=f"{len(finite)}/{len(history)} finite evals; best loss {best['loss']:.5g}"
                  + ("; stopped early by the stop rule" if stopped_early else ""))
