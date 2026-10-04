"""Validation run (part B1; PREREGISTRATION_B.md section 3). Paper-frozen kit + recorder; every call with its score
panel (flow kind, new kit's panel_from_series), loop / generation, and its simulated flow (float32).
usage: python run_val.py <sce|nsga> <seed>      -> runs/<sce|nsga>_s<seed>/"""
import importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code"
sys.path.insert(0, FROZEN); sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/models/spotpy/source/repo/src")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recorder"))
sys.path.insert(0, HERE)
import deps_manifest as DM                       # codex B2 r2: run-time dependencies must be the frozen ones
DEPS = DM.deps_digest()
if DEPS != DM.frozen_digest():
    sys.exit("REFUSED: run-time dependencies differ from DEPS_MANIFEST.json")
kind, seed = sys.argv[1], int(sys.argv[2])
if kind not in ("sce", "nsga"):
    sys.exit("usage: run_val.py <sce|nsga> <seed>")
OUT = os.path.join(HERE, "runs", f"{kind}_s{seed}")
os.makedirs(os.path.join(OUT, "series"), exist_ok=False)
import frozen_recorder as R
R.install(OUT)
import numpy as np
from spotpy.examples.spot_setup_hymod_python import spot_setup
_sp = importlib.util.spec_from_file_location("kdt_new_panel", os.path.join(HERE, "frozen_rule", "kdt_rule_frozen", "panel.py"))
NP = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(NP)
NAMES = ['cmax', 'bexp', 'alpha', 'Ks', 'Kq']; LOWER = [1.0, 0.1, 0.1, 0.001, 0.1]; UPPER = [500.0, 2.0, 0.99, 0.10, 0.99]


class Hymod:
    names, lower, upper = NAMES, LOWER, UPPER

    def __init__(self, two):
        self.two = two
        self.is_multi_objective = two
        self.objective_names = ["Q:nse", "Q:lnnse"] if two else ["Q:nse"]
        self.s = spot_setup(); self.obs = np.array(self.s.evaluation(), float)
        self.last_record = None; self.i = 0

    def evaluate(self, x):
        """Single objective: the loss of reruns/hymod_tagged (1 - NSE over finite pairs). Two objectives: the loss of
        demo_nsga2_hymod (1 - NSE, 1 - lnNSE over positive pairs). last_record is recording only."""
        self.last_record = None
        i = self.i; self.i += 1
        bad = [1e30, 1e30] if self.two else [1e30]
        try:
            sim = np.array(self.s.simulation(list(x)), float)
        except Exception:
            return bad
        o = self.obs
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
                            "series": f"series/sim_{i:05d}.npy"}
        return loss


if kind == "sce":
    from calibration_kit.backends.spotpy_backend import SpotpyBackend
else:
    from calibration_kit.backends.pymoo_backend import PymooBackend
    import pymoo.algorithms.moo.nsga2  # noqa: F401  (load what the backend uses before the check)
_bad = DM.check_loaded(sys.modules)
if _bad:
    sys.exit(f"REFUSED: modules loaded from outside the hashed roots: {_bad[:5]}")
if kind == "sce":
    from calibration_kit.backends.spotpy_backend import SpotpyBackend
    res = SpotpyBackend("sceua").optimize(Hymod(False), budget=20000, seed=seed)
else:
    from calibration_kit.backends.pymoo_backend import PymooBackend
    res = PymooBackend("nsga2").optimize(Hymod(True), budget=4000, seed=seed)
if not os.path.exists(os.path.join(OUT, "..", "obs.npy")):
    np.save(os.path.join(HERE, "runs", "obs.npy"), np.array(spot_setup().evaluation(), float))
R._state["fe"].flush()
if os.path.getsize(os.path.join(OUT, "recorder_errors.jsonl")) > 0:
    sys.exit(f"INVALID {kind} {seed}: recorder errors (no done.json; the judge lists it as invalid)")
if DM.deps_digest() != DEPS:
    sys.exit(f"INVALID {kind} {seed}: run-time dependencies changed during the run")
_bad = DM.check_loaded(sys.modules)
if _bad:
    sys.exit(f"INVALID {kind} {seed}: modules loaded from outside the hashed roots: {_bad[:5]}")
json.dump({"kind": kind, "seed": seed, "done": True, "deps_digest": DEPS}, open(os.path.join(OUT, "done.json"), "w"))
print("DONE", kind, seed)
