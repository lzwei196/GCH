"""Tracer-only aid for the pre-registration review (seed 999 is NOT a validation seed): bootstrap tolerances from
(a) the run at the midpoint of the parameter ranges and (b) the best point at the end of the start-up step of a search."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/server/kdt_convergence_dev")
sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/models/spotpy/source/repo/src")
from calibration_kit import panel as P
from spotpy.examples.spot_setup_hymod_python import spot_setup
s = spot_setup(); obs = np.array(s.evaluation(), float)
MID = [(1.0 + 500.0) / 2, (0.1 + 2.0) / 2, (0.1 + 0.99) / 2, (0.001 + 0.10) / 2, (0.1 + 0.99) / 2]


def boot(sim):
    m = np.isfinite(sim) & np.isfinite(obs)
    b = P.bootstrap_tolerances(sim[m], obs[m], kind="flow")
    p = P.panel_from_series(sim[m], obs[m], "flow", mean_rule=False)
    return {k: round(b["tol"][k], 4) for k in ("r", "alpha", "beta", "lnnse")}, {k: round(p[k], 3) for k in ("nse", "r", "alpha", "beta", "lnnse")}


print("midpoint", *boot(np.array(s.simulation(MID), float)))
for run, key in (("sce_s999", "loop"), ("nsga_s999", "gen")):
    rows = [json.loads(l) for l in open(f"runs/{run}/calls.jsonl")]
    rows = [r for r in rows if r.get("kind") == "problem" and r.get("phase") == "search"]
    first = rows[0][key]
    start = [r for r in rows if r[key] == first and r["losses"] and all(x is not None for x in r["losses"])]
    best = min(start, key=lambda r: sum(r["losses"]))
    sim = np.load(f"runs/{run}/" + best["extra"]["series"]).astype(float)
    print(run, "start-up best", *boot(sim))
