#!/usr/bin/env python3
"""E2 SAC-SMA / CAMELS reproduction on Blanco TX (08171300).

Faithful blind reproduction driven by the frozen agent_authored.yaml decisions.
- Model: real NOAA-OWP Sac-SMA Fortran binary (no surrogate).
- Optimizer: calibration_kit spotpy SCE-UA backend (its Problem/Backend contract).
- Objective: NSE on daily tci vs USGS gauge discharge.
- Forcing bridge: CAMELS Maurer -> Sac-SMA CSV; PET via Hamon.
"""
from __future__ import annotations
import os, sys, subprocess, math, json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code")
from calibration_kit.backends.base import Problem
from calibration_kit.backends.spotpy_backend import SpotpyBackend

# ---------------- fixed inputs ----------------
BINARY = "/home/server/knowledge-dissection-toolkit/auto_dissect_multi_agent/_work/Sacramento-SMA/source/cmake_build/sac"
FORCING = "/mnt/disk1/Hydrocraft_server/models/CAMELS/data/raw/basin_dataset_public_v1p2/basin_mean_forcing/maurer/12/08171300_lump_maurer_forcing_leap.txt"
STREAMFLOW = "/mnt/disk1/Hydrocraft_server/models/CAMELS/data/raw/basin_dataset_public_v1p2/usgs_streamflow/12/08171300_streamflow_qc.txt"
AREA_KM2 = 1067.47
BASIN = "BLANCO"
WORK = Path("/mnt/disk1/Hydrocraft_server/agent_calibration_study/RERUN_FULL_HISTORY_2026-09-27/paper_replay/reruns/e2/blanco_work")

# frozen authored params
CAL_PARAMS = ["uztwm","uzfwm","lztwm","lzfsm","lzfpm","uzk","lzpk","lzsk","zperc","rexp","pfree"]
RANGES = {"uztwm":(10,300),"uzfwm":(5,150),"lztwm":(10,500),"lzfsm":(5,400),"lzfpm":(10,1000),
          "uzk":(0.1,0.95),"lzpk":(0.001,0.05),"lzsk":(0.01,0.4),"zperc":(1,350),"rexp":(1,5),"pfree":(0.0,0.6)}
FIXED = {"adimp":0.0,"pctim":0.0,"riva":0.0,"side":0.0,"rserv":0.3}
ROW_ORDER = ["uztwm","uzfwm","lztwm","lzfsm","lzfpm","adimp","uzk","lzpk","lzsk",
             "zperc","rexp","pctim","pfree","riva","side","rserv"]

# ---------------- forcing bridge + PET (Hamon) ----------------
def load_forcing():
    df = pd.read_csv(FORCING, sep=r"\s+", skiprows=4,
                     names=["Year","Mnth","Day","Hr","Dayl","PRCP","SRAD","SWE","Tmax","Tmin","Vp"])
    df["date"] = pd.to_datetime(dict(year=df.Year, month=df.Mnth, day=df.Day))
    df["Tmean"] = (df.Tmax + df.Tmin) / 2.0
    # Hamon PET (mm/day)
    Ld = (df.Dayl/3600.0)/12.0
    es = 6.108*np.exp(17.27*df.Tmean/(df.Tmean+237.3))          # hPa
    df["PET"] = 0.1651*Ld*216.7*es/(df.Tmean+273.3)
    df["PET"] = df["PET"].clip(lower=0.0)
    return df

def load_obs():
    o = pd.read_csv(STREAMFLOW, sep=r"\s+", names=["id","Year","Mnth","Day","Qcfs","flag"])
    o["date"] = pd.to_datetime(dict(year=o.Year, month=o.Mnth, day=o.Day))
    q_m3s = o.Qcfs*0.0283168
    o["Q_mmday"] = q_m3s*86400.0/(AREA_KM2*1000.0)
    o.loc[o.Qcfs <= -998, "Q_mmday"] = np.nan
    return o[["date","Q_mmday"]]

def write_forcing_csv(df, path):
    with open(path,"w",newline="\n") as f:
        f.write("year,mo,dy,hr,prcp_rate,tavg_degC,pet_rate\n")
        for r in df.itertuples():
            f.write(f"{r.date.year},{r.date.month},{r.date.day},12,"
                    f"{r.PRCP/86400.0:.4e},{r.Tmean:.2f},{r.PET/86400.0:.4e}\n")

# ---------------- namelist / params / run ----------------
def setup_dirs():
    for sub in ["run","input/forcing","input/params","output","state"]:
        (WORK/sub).mkdir(parents=True, exist_ok=True)

def write_namelist(start_dh, end_dh):
    nl = WORK/"run"/f"namelist.bmi.{BASIN}"
    param_rel = os.path.relpath((WORK/"input/params"/f"sac_params.{BASIN}.txt"), WORK/"run")
    body = f"""&SAC_CONTROL
main_id               = "{BASIN}"
n_hrus                = 1
forcing_root          = "../input/forcing/forcing.sacbmi.rates."
output_root           = "../output/output.sacbmi."
sac_param_file        = "{param_rel}"
output_hrus           = 1
start_datehr          = {start_dh}
end_datehr            = {end_dh}
model_timestep        = 86400
warm_start_run        = 0
write_states          = 1
sac_state_in_root     = "../state/sac_states."
sac_state_out_root    = "../state/sac_states."
/
"""
    nl.write_text(body)
    return nl

def write_params(values):
    p = WORK/"input/params"/f"sac_params.{BASIN}.txt"
    allv = dict(FIXED); allv.update(values)
    with open(p,"w",newline="\n") as f:
        f.write(f"hru_id {BASIN}\n")
        f.write(f"hru_area {AREA_KM2}\n")
        for name in ROW_ORDER:
            f.write(f"{name} {allv[name]:.6g}\n")

def run_binary():
    r = subprocess.run([BINARY, f"namelist.bmi.{BASIN}"], cwd=str(WORK/"run"),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sac failed rc={r.returncode}: {r.stdout[-500:]} {r.stderr[-500:]}")
    return r

def read_sim():
    out = WORK/"output"/f"output.sacbmi.{BASIN}.txt"
    cols = ["year","mo","dy","hr","tair","precip","pet","qs","qg","tci","eta",
            "roimp","sdro","ssur","sif","bfs","bfp","bfncc"]
    s = pd.read_csv(out, sep=r"\s+", names=cols, skiprows=0)
    # first row may be header-less numeric; ensure numeric
    s = s[pd.to_numeric(s["year"], errors="coerce").notna()].copy()
    for c in cols: s[c] = pd.to_numeric(s[c])
    s["date"] = pd.to_datetime(dict(year=s.year.astype(int), month=s.mo.astype(int), day=s.dy.astype(int)))
    return s[["date","tci"]]

def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    if len(obs) < 30: return -1e9, 0
    denom = np.sum((obs-np.mean(obs))**2)
    if denom == 0: return -1e9, len(obs)
    return 1.0 - np.sum((sim-obs)**2)/denom, len(obs)

def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    if len(obs) < 30: return -1e9
    r = np.corrcoef(sim,obs)[0,1]
    alpha = np.std(sim)/np.std(obs)
    beta = np.mean(sim)/np.mean(obs)
    return 1.0 - math.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)

def pbias(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    return 100.0*np.sum(sim-obs)/np.sum(obs)

# ---------------- scored run over a window ----------------
OBS = None
LAST_SERIES = None
PROBLEM_REF = {}
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("kdt_new_panel", "/home/server/kdt_convergence_dev/calibration_kit/panel.py")
NP = _ilu.module_from_spec(_sp); _sp.loader.exec_module(NP)
def _record_panel(i, m):
    """RECORDING ONLY: the new kit's score panel of this call's scored series, and the series itself (sim per call;
    obs and dates once), handed to the recorder through problem.last_record. Never raises."""
    global LAST_SERIES
    rec = None
    try:
        if m is not None and LAST_SERIES is not None:
            dates, sim, obs = LAST_SERIES
            p = NP.panel_from_series(sim, obs, "flow", mean_rule=False)
            sdir = WORK.parent / "series"; sdir.mkdir(exist_ok=True)
            if not (sdir / "obs_and_dates.npz").exists():
                np.savez(sdir / "obs_and_dates.npz", date=dates.astype("U10"), obs=obs)
            np.save(sdir / f"sim_{i:05d}.npy", sim.astype(np.float32))
            rec = {"panel": {"Q": {k: v for k, v in p.items() if not k.startswith("_")}}, "series": f"series/sim_{i:05d}.npy"}
    except Exception as e:
        rec = {"record_error": str(e)[:200]}
    LAST_SERIES = None
    if PROBLEM_REF.get("p") is not None:
        PROBLEM_REF["p"].last_record = rec
def scored_metrics(values, run_start, run_end, score_start, score_end):
    write_params(values)
    write_namelist(run_start, run_end)
    run_binary()
    sim = read_sim()
    df = sim.merge(OBS, on="date", how="inner")
    mask = (df.date >= pd.Timestamp(score_start)) & (df.date <= pd.Timestamp(score_end))
    d = df[mask]
    n, obsv = d["tci"].to_numpy(), d["Q_mmday"].to_numpy()
    global LAST_SERIES                                   # RECORDING ONLY: the scored series of this call
    LAST_SERIES = (d["date"].dt.strftime("%Y-%m-%d").to_numpy(), n.astype(float), obsv.astype(float))
    val_nse, npts = nse(n, obsv)
    return {"nse": val_nse, "kge": kge(n,obsv), "pbias": pbias(n,obsv), "n": npts}

def main():
    global OBS
    t0=time.time()
    setup_dirs()
    fdf = load_forcing()
    OBS = load_obs()
    # write forcing covering cal spinup..cal end (also covers val window since 1980..2008)
    write_forcing_csv(fdf, WORK/"input/forcing"/f"forcing.sacbmi.rates.{BASIN}.csv")
    print(f"forcing rows {len(fdf)} {fdf.date.min().date()}..{fdf.date.max().date()}")
    print(f"obs rows {OBS.Q_mmday.notna().sum()} valid; mean obs {OBS.Q_mmday.mean():.3f} mm/d")

    CAL_RUN=(1997100112, 2008093012); CAL_SCORE=("1999-10-01","2008-09-30")
    VAL_RUN=(1987100112, 1999093012); VAL_SCORE=("1989-10-01","1999-09-30")

    # ---- SCE-UA calibration on CAL window ----
    names = CAL_PARAMS
    lower=[RANGES[n][0] for n in names]; upper=[RANGES[n][1] for n in names]
    neval={"n":0}
    def evaluate(x):
        neval["n"]+=1
        vals={n:float(v) for n,v in zip(names,x)}
        m = None
        try:
            m=scored_metrics(vals, *CAL_RUN, *CAL_SCORE)
            loss=1.0-m["nse"]
        except Exception as e:
            loss=1e30
        _record_panel(neval["n"]-1, m)                    # RECORDING ONLY (does not touch the loss)
        with open(WORK.parent/"eval_history.jsonl","a") as _fh:
            _fh.write(json.dumps({"i": neval["n"]-1, "x": vals, "metrics": m, "loss": loss}, default=str)+"\n")
        return [loss if math.isfinite(loss) else 1e30]
    problem=Problem(names=names, lower=lower, upper=upper,
                    objective_names=["scalar_loss"], evaluate=evaluate, is_multi_objective=False)
    PROBLEM_REF["p"] = problem                           # RECORDING ONLY
    budget=int(sys.argv[1]) if len(sys.argv)>1 else 3000
    print(f"=== SCE-UA start budget={budget} ===", flush=True)
    res=SpotpyBackend("sceua").optimize(problem, budget=budget, seed=0)
    best=dict(zip(names,res.best_x)); best_nse_cal=1.0-res.best_loss[0]
    print(f"SCE-UA done: {res.n_evaluations} evals, best cal NSE={best_nse_cal:.4f}", flush=True)
    print("best params:", {k:round(v,4) for k,v in best.items()})

    # ---- final calibration + validation metrics ----
    cal_m=scored_metrics(best, *CAL_RUN, *CAL_SCORE)
    val_m=scored_metrics(best, *VAL_RUN, *VAL_SCORE)
    out={"basin":"08171300","area_km2":AREA_KM2,
         "optimizer":"SCE-UA (calibration_kit spotpy backend)","objective":"NSE(tci)",
         "n_model_evals":res.n_evaluations,"wallclock_s":round(time.time()-t0,1),
         "calibrated_params":{k:round(v,5) for k,v in best.items()},
         "fixed_params":FIXED,
         "cal_window":CAL_SCORE,"val_window":VAL_SCORE,
         "cal":cal_m,"val":val_m,"pet_method":"Hamon","forcing":"CAMELS Maurer"}
    (WORK.parent/"e2_blanco_result.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
