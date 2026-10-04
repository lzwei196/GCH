"""SAC-SMA optimizer comparison (Experiment 6, "Ibex") — ONE cell rerun with recording. The paper's duan_driver.py
(copied unchanged) and duan_harness.py (copied; only ROOT points at the record addendum's copy of the model folder,
as its README says) run in a cell folder of their own, so the driver's evals.csv / summary.csv / SPOTPY database are
written here. Recording only: (1) SPOTPY's printed "ComplexEvo loop #N" lines give each call's SCE-UA loop (0 = the
start-up sample), as in the kit recorder; (2) the harness's parse_output() is wrapped to keep each call's simulated
flow (tci, the model's total channel inflow, converted as the harness does) -> series/sim_<i>.npy float32, dates once.
usage: python rec_cell.py <method> <budget> <seed>
"""
import io, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S = ("/mnt/disk1/Hydrocraft_server/agent_calibration_study/RECORD_ADDENDUM_2026-09-16/2_EXP6_SACSMA_Duan_work_ibex/"
     "files/mnt/disk1/Hydrocraft_server/models/Sacramento_SMA")
method, budget, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tag = "%s_b%d_s%d" % (method, budget, seed)
C = os.path.join(HERE, "cells", tag)
os.makedirs(os.path.join(C, "work_ibex"), exist_ok=False)
for name in ("tools", "bin", "example"):                       # read-only links into the record copy
    os.symlink(os.path.join(S, name), os.path.join(C, name))
os.symlink(os.path.join(S, "work_ibex", "forcing"), os.path.join(C, "work_ibex", "forcing"))
for f in ("duan_driver.py", "duan_harness.py"):
    os.symlink(os.path.join(HERE, f), os.path.join(C, "work_ibex", f))
os.chdir(C)
os.makedirs("series")

state = {"loop": 0 if method == "sceua" else None, "i": 0, "last": None}
_LOOP = re.compile(r"ComplexEvo loop #(\d+)")


class Tee(io.TextIOBase):
    def __init__(self, inner):
        self.inner = inner
    def write(self, s):
        for m in _LOOP.finditer(s):
            state["loop"] = int(m.group(1))
        return self.inner.write(s)
    def flush(self):
        self.inner.flush()


sys.stdout = Tee(sys.stdout)
sys.argv = ["duan_driver.py", json.dumps([[method, budget, seed]])]
sys.path.insert(0, "work_ibex"); sys.path.insert(0, "tools")
import duan_harness as H                                        # noqa: E402
_po = H.parse_output
calls = open("calls.jsonl", "w")
AREA = H.AREA


def parse_output(*a, **k):
    p = _po(*a, **k)
    try:
        dates = [(d.date() if hasattr(d, "date") else d).isoformat() for d in p["dates"]]
        sim = np.asarray(p["tci"], float) * AREA / (1000 * 86400)
        if not os.path.exists("series/dates.json"):
            json.dump(dates, open("series/dates.json", "w"))
        elif json.load(open("series/dates.json")) != dates:
            raise RuntimeError("dates differ from the first call")
        np.save("series/sim_%05d.npy" % state["i"], sim.astype(np.float32))
        state["last"] = "series/sim_%05d.npy" % state["i"]
    except Exception as e:                                       # recording must never change the search
        state["last"] = "record_error: %s" % str(e)[:200]
    return p


H.parse_output = parse_output
_sim = H.simulate


def simulate(vec):
    state["last"] = None
    loop = state["loop"]
    try:
        r = _sim(vec)
        calls.write(json.dumps({"i": state["i"], "tag": tag, "loop": loop, "x": [float(v) for v in vec],
                                "FULL": float(r["FULL"]), "CAL": float(r["CAL"]), "VAL": float(r["VAL"]),
                                "n": r["n"], "series": state["last"]}) + "\n")
        return r
    except Exception as e:
        calls.write(json.dumps({"i": state["i"], "tag": tag, "loop": loop, "x": [float(v) for v in vec],
                                "error": str(e)[:200]}) + "\n")
        raise
    finally:
        state["i"] += 1
        calls.flush()


H.simulate = simulate
import runpy                                                     # noqa: E402
runpy.run_path("work_ibex/duan_driver.py", run_name="__main__")
calls.close()

# ── does it reproduce the paper's evals.csv rows for this cell? ──────────────────────────────────
import csv                                                       # noqa: E402
ORIG = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/literature/SAC_SMA_duan_results"
orig = [r[1:] for r in csv.reader(open(os.path.join(ORIG, "evals.csv"))) if r and r[0] == tag]
new = [r[1:] for r in csv.reader(open("work_ibex/duan_results/evals.csv")) if r and r[0] == tag]
so = [r for r in csv.reader(open(os.path.join(ORIG, "summary.csv"))) if r[:3] == [method, str(budget), str(seed)]]
sn = [r for r in csv.reader(open("work_ibex/duan_results/summary.csv")) if r[:3] == [method, str(budget), str(seed)]]
chk = {"tag": tag, "rows": [len(orig), len(new)], "rows_identical": orig == new,
       "summary_same_except_wall": [x[:-1] for x in so] == [x[:-1] for x in sn], "orig_summary": so, "rerun_summary": sn}
chk["identical"] = chk["rows_identical"] and chk["summary_same_except_wall"]
json.dump(chk, open("check.json", "w"), indent=1)
sys.stdout.write("CHECK %s %s rows %s\n" % (tag, "identical" if chk["identical"] else "DIFFERS", chk["rows"]))
