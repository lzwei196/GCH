#!/usr/bin/env python3
"""Duan comparison driver: SCE-UA vs DDS vs random on the REAL Sac-SMA @ ibex.

Objective (minimize) = 1 - NSE(CAL 2006-2010). Every model eval's NSE is logged.
Writes results incrementally so partial completion is still usable.
Run from model root with the python_env interpreter.
"""
import os, sys, time, csv, json
sys.path.insert(0, "work_ibex")
sys.path.insert(0, "tools")
import numpy as np
import spotpy
from spotpy.parameter import Uniform
import duan_harness as H

OUTDIR = "work_ibex/duan_results"
os.makedirs(OUTDIR, exist_ok=True)
SUMMARY = os.path.join(OUTDIR, "summary.csv")
EVALLOG = os.path.join(OUTDIR, "evals.csv")

_evalfile = open(EVALLOG, "a", newline="")
_evalw = csv.writer(_evalfile)
if os.path.getsize(EVALLOG) == 0:
    _evalw.writerow(["tag", "eval_i", "FULL", "CAL", "VAL"] + H.CAL_PARAMS)

def log_summary(row):
    exists = os.path.exists(SUMMARY) and os.path.getsize(SUMMARY) > 0
    with open(SUMMARY, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "budget", "seed", "n_evals",
                        "best_NSE_CAL", "best_NSE_FULL", "best_NSE_VAL", "wall_s"])
        w.writerow(row)

class Setup:
    def __init__(self, tag, direction):
        self.tag = tag; self.direction = direction; self.i = 0
        self.best = {"CAL": -1e9}
        self.params = [Uniform(p, low=H.LO[k], high=H.HI[k])
                       for k, p in enumerate(H.CAL_PARAMS)]
    def parameters(self):
        return spotpy.parameter.generate(self.params)
    def simulation(self, vector):
        vec = list(vector)
        try:
            r = H.simulate(vec)
            cal = float(r["CAL"])
            _evalw.writerow([self.tag, self.i, round(r["FULL"],5), round(r["CAL"],5),
                             round(r["VAL"],5)] + [round(float(x),5) for x in vec])
            if self.i % 25 == 0:
                _evalfile.flush()
            if cal > self.best["CAL"]:
                self.best = {"CAL": cal, "FULL": float(r["FULL"]), "VAL": float(r["VAL"])}
            self.i += 1
            return [cal]
        except Exception as e:
            _evalw.writerow([self.tag, self.i, "ERR", "ERR", "ERR"] + [round(float(x),5) for x in vec])
            self.i += 1
            return [-10.0]
    def evaluation(self):
        return [1.0]
    def objectivefunction(self, simulation, evaluation):
        nse = simulation[0]
        return (1.0 - nse) if self.direction == "min" else nse

def run_spotpy(algo, budget, seed):
    tag = "%s_b%d_s%d" % (algo, budget, seed)
    np.random.seed(seed)
    import random; random.seed(seed)
    direction = "min" if algo == "sceua" else "max"
    setup = Setup(tag, direction)
    t = time.time()
    dbname = os.path.join(OUTDIR, "db_" + tag)
    if algo == "sceua":
        sampler = spotpy.algorithms.sceua(setup, dbname=dbname, dbformat="csv",
                                          save_sim=False, random_state=seed)
        sampler.sample(budget, ngs=6)
    elif algo == "dds":
        sampler = spotpy.algorithms.dds(setup, dbname=dbname, dbformat="csv",
                                        save_sim=False, random_state=seed)
        sampler.sample(budget, trials=1)
    _evalfile.flush()
    wall = time.time() - t
    b = setup.best
    log_summary([algo, budget, seed, setup.i, round(b["CAL"],5),
                 round(b.get("FULL",float('nan')),5), round(b.get("VAL",float('nan')),5),
                 round(wall,1)])
    print("DONE", tag, "n=%d best_CAL=%.4f wall=%.0fs" % (setup.i, b["CAL"], wall), flush=True)
    return b

def run_random(budget, seed):
    tag = "random_b%d_s%d" % (budget, seed)
    rng = np.random.default_rng(seed)
    t = time.time()
    best = {"CAL": -1e9}
    for i in range(budget):
        vec = rng.uniform(H.LO, H.HI)
        try:
            r = H.simulate(list(vec))
            _evalw.writerow([tag, i, round(r["FULL"],5), round(r["CAL"],5), round(r["VAL"],5)]
                            + [round(float(x),5) for x in vec])
            if r["CAL"] > best["CAL"]:
                best = {"CAL": float(r["CAL"]), "FULL": float(r["FULL"]), "VAL": float(r["VAL"])}
        except Exception:
            _evalw.writerow([tag, i, "ERR","ERR","ERR"] + [round(float(x),5) for x in vec])
        if i % 25 == 0:
            _evalfile.flush()
    _evalfile.flush()
    wall = time.time() - t
    log_summary(["random", budget, seed, budget, round(best["CAL"],5),
                 round(best.get("FULL",float('nan')),5), round(best.get("VAL",float('nan')),5),
                 round(wall,1)])
    print("DONE", tag, "best_CAL=%.4f wall=%.0fs" % (best["CAL"], wall), flush=True)
    return best

if __name__ == "__main__":
    plan = json.loads(sys.argv[1])  # list of [method, budget, seed]
    for method, budget, seed in plan:
        try:
            if method == "random":
                run_random(budget, seed)
            else:
                run_spotpy(method, budget, seed)
        except Exception as e:
            print("CONFIGFAIL", method, budget, seed, str(e)[:300], flush=True)
    print("ALL DONE", flush=True)
