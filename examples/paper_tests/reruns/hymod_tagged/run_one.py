"""HYMOD sweep (SI Table S8a) rerun with the recorder: frozen paper kit, the original problem and loss, every call
recorded with the full score panel (problem.last_record) and SCE-UA loop numbers. Counts only if repetitions and
objective equal the original log.  python run_one.py <budget> <dds|sceua> <seed>"""
import contextlib, importlib.util, io, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code"
sys.path.insert(0, FROZEN); sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/models/spotpy/source/repo/src")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "recorder"))
budget, algo, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
tag = f"b{budget}_{algo}_s{seed}"
import frozen_recorder as R
R.install(os.path.join(HERE, "runs", tag))
import numpy as np
from spotpy.examples.spot_setup_hymod_python import spot_setup
from calibration_kit.backends.spotpy_backend import SpotpyBackend
_sp = importlib.util.spec_from_file_location("kdt_new_panel", "/home/server/kdt_convergence_dev/calibration_kit/panel.py")
NP = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(NP)
LOG = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/03_supporting_evidence/original_axisAB_study/logs/l1_sweep.log"


class HymodProblem:   # loss verbatim from l1_budget_sweep.py; last_record is recording only
    names = ['cmax', 'bexp', 'alpha', 'Ks', 'Kq']; lower = [1.0, 0.1, 0.1, 0.001, 0.1]; upper = [500.0, 2.0, 0.99, 0.10, 0.99]
    is_multi_objective = False; objective_names = ["Q:nse"]

    def __init__(self):
        self.s = spot_setup(); self.obs = np.array(self.s.evaluation(), float); self.last_record = None

    def evaluate(self, x):
        self.last_record = None
        try: sim = np.array(self.s.simulation(list(x)), float)
        except Exception: return [1e30]
        o = self.obs; m = np.isfinite(o) & np.isfinite(sim)
        if m.sum() < 2: return [1e30]
        oo, ss = o[m], sim[m]; den = np.sum((oo - oo.mean()) ** 2)
        loss = [float(1.0 - (1 - np.sum((ss - oo) ** 2) / den))] if den > 0 else [1e30]
        if den > 0:
            p = NP.panel_from_series(ss, oo, "flow", mean_rule=False)
            self.last_record = {"panel": {"Q": {k: v for k, v in p.items() if not k.startswith("_")}}}
        return loss


runs = re.findall(r"Total Repetitions: (\d+)\n(Maximal|Minimal) objective value: ([-\d.e+]+)", open(LOG).read())
k = [(b, a, s) for b in [200, 500, 2000, 5000, 10000, 20000] for a in ("dds", "sceua") for s in (0, 1, 2)].index((budget, algo, seed))
o_reps, o_obj = runs[k][0], runs[k][2]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):          # the recorder's tee sits inside: it still sees SPOTPY's loop lines
    SpotpyBackend(algo).optimize(HymodProblem(), budget=budget, seed=seed)
m = re.search(r"Total Repetitions: (\d+)\n(?:Maximal|Minimal) objective value: ([-\d.e+]+)", buf.getvalue())
same = (m.group(1), m.group(2)) == (o_reps, o_obj)
json.dump({"tag": tag, "original": [o_reps, o_obj], "rerun": [m.group(1), m.group(2)], "identical": same},
          open(os.path.join(HERE, "runs", tag, "check.json"), "w"))
print(tag, "SAME" if same else "DIFFERENT", m.group(1), m.group(2))
sys.exit(0 if same else 1)
