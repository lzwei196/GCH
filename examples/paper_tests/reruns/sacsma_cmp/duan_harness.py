#!/usr/bin/env python3
"""Duan et al. reproduction harness on the REAL Sac-SMA @ work_ibex (b2 config).

Reuses the pre-staged b2 forcing (HRUs b20..b24, degree-day snow + Hamon PET
already applied) and varies ONLY the 13 classic Sac-SMA parameters. Each eval
runs the real bin/sac ELF once over 2005-2015 and returns loss = 1 - NSE(CAL).
"""
import os, sys, time, tempfile, sqlite3, calendar, datetime
sys.path.insert(0, "tools")
import numpy as np
from build_parameters import build_parameters, BOUNDS, DEFAULTS
from run_sac import run as sac_run
from parse_output import parse_output

ROOT = "/mnt/disk1/Hydrocraft_server/agent_calibration_study/RECORD_ADDENDUM_2026-09-16/2_EXP6_SACSMA_Duan_work_ibex/files/mnt/disk1/Hydrocraft_server/models/Sacramento_SMA"   # RERUN: the record addendum copy (its README: the old folder was archived)
WI = os.path.join(ROOT, "work_ibex")
HRU_IDS = ["b20", "b21", "b22", "b23", "b24"]
HRU_AREAS = [129.6] * 5
FORCING = {h: os.path.join(WI, "forcing", f"forcing.sacbmi.rates.{h}.csv") for h in HRU_IDS}
START, END = "2005010112", "2015123112"
AREA = 648e6
STN = "09AC007"
DB = "/mnt/datasets/observed_data/dischargeandwatershed/National Water Data Archive HYDAT/Hydat.sqlite3"

# 13 classic Duan-calibrated params (fixed: riva, side, rserv, adimp? -> we calibrate adimp too => 13)
CAL_PARAMS = ["uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm", "adimp",
              "uzk", "lzpk", "lzsk", "zperc", "rexp", "pctim", "pfree"]
# fixed at b2 / defaults
FIXED = {"riva": 0.01, "side": 0.0, "rserv": 0.30}
LO = np.array([BOUNDS[p][0] for p in CAL_PARAMS], float)
HI = np.array([BOUNDS[p][1] for p in CAL_PARAMS], float)

_OBS = None
def get_obs():
    global _OBS
    if _OBS is not None:
        return _OBS
    c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True); cur = c.cursor()
    cur.execute("SELECT YEAR,MONTH," + ",".join("FLOW%d" % i for i in range(1, 32)) +
                " FROM DLY_FLOWS WHERE STATION_NUMBER=? AND YEAR BETWEEN 2006 AND 2015", (STN,))
    o = {}
    for r in cur.fetchall():
        yr, mo = r[0], r[1]; fl = r[2:]
        for d in range(1, calendar.monthrange(yr, mo)[1] + 1):
            f = fl[d - 1]
            if f is not None:
                o[datetime.date(yr, mo, d)] = float(f)
    c.close(); _OBS = o
    return o

def _nse(o, s):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

def simulate(param_vec):
    """Run real sac once; return dict of NSE for FULL/CAL/VAL. Raises on failure."""
    overrides = dict(FIXED)
    for p, v in zip(CAL_PARAMS, param_vec):
        overrides[p] = float(v)
    tmp = tempfile.mkdtemp(prefix="duan_")
    try:
        pf = os.path.join(tmp, "params.txt")
        build_parameters(HRU_IDS, HRU_AREAS, pf, overrides)  # enforces BOUNDS
        outd = os.path.join(tmp, "out")
        res = sac_run("cal", HRU_IDS, FORCING, pf, START, END, outd)
        p = parse_output(res["main_output"])
        sim = {}
        for d, v in zip(p["dates"], p["tci"]):
            dd = d.date() if hasattr(d, "date") else d
            sim[dd] = v * AREA / (1000 * 86400)
        obs = get_obs(); common = sorted(set(sim) & set(obs))
        od = np.array([obs[d] for d in common]); sd = np.array([sim[d] for d in common])
        def split(y0, y1):
            idx = [i for i, d in enumerate(common) if y0 <= d.year <= y1]
            return _nse(od[idx], sd[idx])
        return {"FULL": _nse(od, sd), "CAL": split(2006, 2010), "VAL": split(2011, 2015),
                "n": len(common)}
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)

def loss(param_vec):
    """1 - NSE(CAL); large penalty on failure."""
    try:
        r = simulate(param_vec)
        return 1.0 - r["CAL"], r
    except Exception as e:
        return 10.0, {"error": str(e)[:200]}

if __name__ == "__main__":
    # timing + reproduce b2 calibrated point
    b2 = {"uztwm": 45, "uzfwm": 25, "lztwm": 170, "lzfpm": 160, "lzfsm": 45,
          "adimp": 0.0, "uzk": 0.30, "lzpk": 0.009, "lzsk": 0.055, "zperc": 100,
          "rexp": 2.5, "pctim": 0.01, "pfree": 0.40}
    vec = [b2[p] for p in CAL_PARAMS]
    t = time.time()
    r = simulate(vec)
    dt = time.time() - t
    print("b2-reproduce:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
          "  eval_seconds=%.2f" % dt)
    print("LO", LO.tolist()); print("HI", HI.tolist())
