"""Built-in-rule validation run (PREREGISTRATION_C.md). Paper-frozen kit + recorder, HYMOD, as validation/run_val.py (SCE-UA 40,000 SPOTPY trials; NSGA-II budget 40,000 = 1,000 generations; no
per-call series for NSGA-II),
plus RECORDING ONLY of what each optimizer's OWN stopping rule sees:
  sce : SPOTPY SCE-UA through a recording copy of its sceua.py (sceua_recording.py: two added lines) -> per loop end
        the population spread gnrng, the best objective bestf and the trial count -> loops.json;
  nsga: pymoo NSGA-II with a read-only callback after every generation -> per generation n_gen, n_eval, the
        algorithm's own opt (F, X), and the state of two observers of pymoo's own DefaultMultiObjectiveTermination
        (period 50 = pymoo default; period 5) updated exactly as pymoo updates its termination -> gens.json.
Neither changes the search (checked: same calls as validation/run_val.py for the same seed).
usage: python run_native.py <sce|nsga> <seed>     -> runs/<kind>_s<seed>/"""
import importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code"
sys.path.insert(0, FROZEN); sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/models/spotpy/source/repo/src")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recorder"))
sys.path.insert(0, HERE)
import deps_native as DM
DEPS = DM.deps_digest()
if DEPS != DM.frozen_digest():
    sys.exit("REFUSED: run-time dependencies differ from DEPS_MANIFEST.json")
if len(sys.argv) != 3 or sys.argv[1] not in ("sce", "nsga"):
    sys.exit("usage: run_native.py <sce|nsga> <seed>")
kind, seed = sys.argv[1], int(sys.argv[2])
OUT = os.path.join(HERE, "runs", f"{kind}_s{seed}")
os.makedirs(os.path.join(OUT, "series"), exist_ok=False)
import frozen_recorder as R
R.install(OUT)
import numpy as np
from spotpy.examples.spot_setup_hymod_python import spot_setup
_sp = importlib.util.spec_from_file_location(
    "kdt_new_panel", os.path.join(os.path.dirname(HERE), "validation", "frozen_rule", "kdt_rule_frozen", "panel.py"))
NP = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(NP)
NAMES = ['cmax', 'bexp', 'alpha', 'Ks', 'Kq']; LOWER = [1.0, 0.1, 0.1, 0.001, 0.1]; UPPER = [500.0, 2.0, 0.99, 0.10, 0.99]


class Hymod:   # verbatim from validation/run_val.py
    names, lower, upper = NAMES, LOWER, UPPER

    def __init__(self, two):
        self.two = two
        self.is_multi_objective = two
        self.objective_names = ["Q:nse", "Q:lnnse"] if two else ["Q:nse"]
        self.s = spot_setup(); self.obs = np.array(self.s.evaluation(), float)
        self.last_record = None; self.i = 0

    def evaluate(self, x):
        self.last_record = None
        i = self.i; self.i += 1
        bad = [1e30, 1e30] if self.two else [1e30]
        try:
            sim = np.array(self.s.simulation(list(x)), float)
        except Exception:
            return bad
        o = self.obs
        if not self.two:      # NSGA-II: no per-call series (40,000 calls; only the dropped bootstrap tolerance used them)
            np.save(os.path.join(OUT, "series", f"sim_{i:05d}.npy"), sim.astype(np.float32))
        if self.two:
            m = np.isfinite(o) & np.isfinite(sim) & (sim > 0) & (o > 0)
            oo, ss = o[m], sim[m]
            nse = 1 - np.sum((ss - oo) ** 2) / np.sum((oo - oo.mean()) ** 2)
            lo, ls = np.log(oo), np.log(ss)
            lnnse = 1 - np.sum((ls - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)
            loss = [float(1 - nse), float(1 - lnnse)]
        else:
            m = np.isfinite(o) & np.isfinite(sim)
            if m.sum() < 2:
                return bad
            oo, ss = o[m], sim[m]; den = np.sum((oo - oo.mean()) ** 2)
            if not den > 0:
                return bad
            loss = [float(1.0 - (1 - np.sum((ss - oo) ** 2) / den))]
        mm = np.isfinite(o) & np.isfinite(sim)
        p = NP.panel_from_series(sim[mm], o[mm], "flow", mean_rule=False)
        self.last_record = {"panel": {"Q": {k: v for k, v in p.items() if not k.startswith("_")}},
                            "series": (None if self.two else f"series/sim_{i:05d}.npy")}
        return loss


if kind == "sce":
    import spotpy.algorithms
    import sceua_recording as SR
    spotpy.algorithms.sceua = SR.sceua                     # the kit looks it up by name at run time
    from calibration_kit.backends.spotpy_backend import SpotpyBackend
else:
    import pymoo.optimize as PO
    from pymoo.core.callback import Callback
    from pymoo.termination.default import DefaultMultiObjectiveTermination
    from calibration_kit.backends.pymoo_backend import PymooBackend
    GENS = []

    class Observe(Callback):
        """Read-only: after each generation (pymoo calls it right after updating its own termination), update two
        observers of pymoo's own default multi-objective termination the same way, and record."""
        def __init__(self):
            super().__init__()
            self.obs = {"p50": DefaultMultiObjectiveTermination(), "p5": DefaultMultiObjectiveTermination(period=5)}

        def notify(self, algorithm):
            rec = {"n_gen": int(algorithm.n_gen), "n_eval": int(algorithm.evaluator.n_eval),
                   "opt_F": algorithm.opt.get("F").tolist(), "opt_X": algorithm.opt.get("X").tolist()}
            for k, t in self.obs.items():
                t.update(algorithm)
                rec[k] = {"perc": float(t.perc), "terminated": bool(t.has_terminated()),
                          "x": float(t.x.perc), "cv": float(t.cv.perc), "f": float(t.f.perc)}
            GENS.append(rec)

    _orig_min = PO.minimize

    def minimize(*a, **k):
        k["callback"] = Observe()
        return _orig_min(*a, **k)
    PO.minimize = minimize                                  # the kit imports it inside optimize() at call time
_bad = DM.check_loaded(sys.modules)
if _bad:
    sys.exit(f"REFUSED: modules loaded from outside the hashed roots: {_bad[:5]}")
if kind == "sce":
    SpotpyBackend("sceua").optimize(Hymod(False), budget=40000, seed=seed)   # ~65 loops: room to judge after a stop
    json.dump(SR.LOOP_RECORD, open(os.path.join(OUT, "loops.json"), "w"))
else:
    PymooBackend("nsga2").optimize(Hymod(True), budget=40000, seed=seed)   # 1,000 generations = pymoo's own max-gen default
    json.dump(GENS, open(os.path.join(OUT, "gens.json"), "w"))
if not os.path.exists(os.path.join(HERE, "runs", "obs.npy")):
    np.save(os.path.join(HERE, "runs", "obs.npy"), np.array(spot_setup().evaluation(), float))
R._state["fe"].flush()
if os.path.getsize(os.path.join(OUT, "recorder_errors.jsonl")) > 0:
    sys.exit(f"INVALID {kind} {seed}: recorder errors")
if DM.deps_digest() != DEPS or DM.check_loaded(sys.modules):
    sys.exit(f"INVALID {kind} {seed}: dependencies changed or loaded from outside")
json.dump({"kind": kind, "seed": seed, "done": True, "deps_digest": DEPS}, open(os.path.join(OUT, "done.json"), "w"))
print("DONE", kind, seed)
