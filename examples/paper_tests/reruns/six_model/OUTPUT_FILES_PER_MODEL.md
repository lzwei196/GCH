# Six-model campaign: which files hold the simulated series, per model (2026-10-03)

Read-only check. Nothing was run, moved or changed. Only this file was written.

Short names used below:
- `REC` = `/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07`
- `SMF` = `REC/03_supporting_evidence/six_model_families`
- `KIT` = `REC/01_framework/framework_code/calibration_kit` (the paper-frozen kit)
- `MAR` = `REC/03_supporting_evidence/multivariable_archive/cases`
- `KTC` = `/mnt/disk1/Hydrocraft_server/models/ki_tools_common/ki_tools_common/metrics.py` (the live scorer all runners import)
- `<wd>` = the `workdir` the rerun driver gives `calibrate()`. The kit passes this same folder as `--workdir` to every runner call (`KIT/runner.py:59-61`).

## 0. Things that hold for all six

1. **One run folder, reused.** The kit runs the runner as one subprocess per call, always with the same `--workdir <wd>` and the same metrics file `<wd>/calib_metrics.json` (`KIT/runner.py:59-63`, `:74-92`). So any file a runner writes at a fixed name inside `<wd>` is overwritten by the next call.
2. **Repeated points are not re-run.** If a parameter vector + split was already scored, the kit takes the cached metrics and does not run the model (`KIT/evaluator.py:285-293`). Such a call makes no new series. Its series is the same as the first call with that key. So: start each seed in an **empty** `<wd>` (the old runs did not, see the inventory rows 1-5), and log every call's parameter file (`<wd>/kdt_params.json`, written just before each real run, `KIT/evaluator.py:299-301`).
3. **Call number.** The kit sets `KDT_CALIB_EVAL_ID` only for calls that go through `evaluate()` (`KIT/evaluator.py:311-312`). Probe, certification and baseline runs call the runner directly (`KIT/calib.py:245`, `:414`, `:1146`) and do not bump the number. So name copies with the driver's own counter, not with `KDT_CALIB_EVAL_ID`.
4. **A hook without editing the kit.** `calibrate()` takes a `run_model=` argument and only builds its own when that is `None` (`KIT/calib.py:639`, `:659-665`). The rerun driver can build the normal one with `KIT/runner.py:make_run_model` and wrap it: run, then copy the files below, then return the metrics. The wrapper also sees probe, baseline and holdout calls.
5. **The scorer can save the scored pairs by itself.** `all_metrics()` in `KTC:326-371` calls `dump_scored_series()` (`KTC:420-475`). That writes `<label>.NNNN.csv` (columns `date,obs,sim`) plus one line in `manifest.jsonl` (with the NSE/KGE/PBIAS/RMSE/r it computed) into the folder named by the env var `KDT_SERIES_DUMP_DIR` (`KTC:387-393`). If no folder is set it writes nothing. HBV, SUMMA and VIC call `all_metrics()` on their final scored pairs, so **setting `KDT_SERIES_DUMP_DIR` in the driver's environment (per call, through the wrapper) makes these three runners save exactly the obs/sim pairs they scored.** No KI or kit file has to change. Notes: (a) HBV, SUMMA and VIC pass plain arrays, so the `date` column will be 0..n-1, not dates (`KTC:48-57`). That is fine for r, alpha, beta and lnNSE, which only need the pairs. (b) All three use the default label `headline`, so each call writes two files: discharge first, then the second variable (order given per model below). (c) The save is best-effort and silent if it fails (`KTC:367-370`, `:474-475`), so the wrapper must check that two new files appeared. (d) This env var is not part of the cache name (`KIT/evaluator.py:118-148`), so it does not change cache reuse.
6. **Live code outside the frozen record.** Several runners import live code at run time, not the frozen copy: `KTC` (all six), and the live model KI tools for CRHM, SUMMA and WOFOST (see each section). These files may have changed since the paper. `rerun_conv.py` also imports the live kit (`SMF/rerun_conv.py:4-7`), not `KIT`, so the new driver must point `sys.path` at `KIT`. The scratch KI folders it used (`/tmp/.../scratchpad/<dir>/ki`, `SMF/rerun_conv.py:17-19`) are gone. The KI copies to use are `SMF/0N_<MODEL>/ki/`.

Call counts: budget per seed = contract `max_evaluations`: WOFOST 600, HBV 500, CRHM 500, SUMMA 240, MODFLOW6 200, VIC 150 (`SMF/*/ki/calibration.yaml`, lines given below). The old runs added about 10-35 % on top for holdout and probe calls (inventory rows 1-6). Disk numbers below use "budget x 1.15 x 3 seeds" and also give the "600 x 3 = 1,800 calls" figure you asked for.

---

## 1. HBV (Moyie 08NH120): Q + SWE

**1. Variables and kind.** `Q` (discharge, mm/d) = **flow**. `SWE` (basin snowpack `Sp`, mm) = **series** (`SMF/02_HBV/ki/tools/calib_run_moyie_mv.py:14-21`). The cache has, for Q: nse, kge, pbias, rmse, r; for SWE: the same + day_bias (`calib_run_moyie_mv.py:319-324`). Missing: alpha, beta (Q, SWE) and lnNSE (Q).

**2. Runner and per-call output.** Runner is `tools/calib_run_moyie_mv.py`, not `tools/calib_run.py` (`SMF/02_HBV/ki/calibration.yaml:221-228`).
- It deletes and re-creates `<wd>/run/` at the start of every call (`calib_run_moyie_mv.py:348-350`) → **overwritten each call**.
- The model writes `<wd>/run/bmi_out/Q.npy`, `bmi_state.npz` (12 variables, incl. `Q` and `Sp`, one value per day, float64) and `manifest.json` (`tools/run_hbv.py:36-40`, `:144-166`; called at `calib_run_moyie_mv.py:230-231`). Dates are not in these files. They come from the forcing time axis, skipping the first step (`tools/parse_hbv_output.py:50-52`).
- At the end of a good call the runner **deletes `<wd>/run/`** unless `KDT_HBV_KEEP_WORKDIR=1` (`calib_run_moyie_mv.py:404-405`).
- Size: about 12 vars x 4,383 days x 8 B ≈ 0.42 MB for `bmi_state.npz` (worked out, not measured).
- Easier: with `KDT_SERIES_DUMP_DIR` set (section 0.5), each call writes two `headline.NNNN.csv`: first Q (window pairs), then SWE (`calib_run_moyie_mv.py:316-317`). That is about 1,826 Q rows + the paired SWE rows, roughly 0.1-0.15 MB per call.

**3. Observed series.** `obs_08NH120_discharge.csv` (`date,discharge_m3s`, turned into mm/d with the 239 km² area) and `obs_BCE-2C10P_swe.csv` (`date,swe_mm`) (`calib_run_moyie_mv.py:108-110`, `:251-277`). Windows: calibration 2006-01-01..2010-12-31, holdout 2011-01-01..2015-12-31; both variables use the same window (`:128-133`; contract `calibration.yaml:250-251`).

**4. What to copy after each call.** Either (a) the two new `headline.NNNN.csv` + their `manifest.jsonl` lines from the dump folder, or (b) with `KDT_HBV_KEEP_WORKDIR=1`: `<wd>/run/bmi_out/bmi_state.npz` + `<wd>/run/bmi_out/manifest.json`. Always also copy `<wd>/kdt_params.json` and `<wd>/calib_metrics.json`. Disk: (a) ≈ 0.15 MB x 575 x 3 ≈ **0.26 GB** (1,800 calls ≈ 0.27 GB); (b) ≈ 0.43 MB x 1,725 ≈ **0.75 GB** (1,800 calls ≈ 0.8 GB).

**5. Risks.** **Blocker as things stand:** the runner reads its inputs from a fixed folder, `/tmp/claude-1000/-home-server/350a2276-…/scratchpad/hbv_mv/stage` (`calib_run_moyie_mv.py:106-110`), and that folder **no longer exists** (checked). It also checks a sha256 for each file (`:113-115`, `:171-178`). The same three files are kept at `MAR/02_hbv_discharge_swe_moyie/stage/` and their sha256 values match the pinned ones (checked). So they must be copied back to that `/tmp` path before the rerun. That is a data copy, not a KI edit. The cache name is built from `tools/calib_run.py` (`KIT/evaluator.py:127-131`), but HBV's real runner is `calib_run_moyie_mv.py`, so changes to the real runner do not change the cache name. Not a problem if each seed starts with an empty `<wd>`.

---

## 2. CRHM (Blue River 08LB038): basinflow_s + SWE

**1. Variables and kind.** `basinflow_s` (daily discharge, m³/s) = **flow**. `SWE` (sim = HRU-1/HRU-2 blend at the 1,570 m pillow) = **series** (`SMF/06_CRHM/ki/tools/calib_run.py:19-26`, `:583-587`). The cache has, for both: nse, kge, pbias, rmse, r, n_paired; plus `discharge_splits` nse_cal / nse_val / nse_full / pbias_full (`calib_run.py:696-708`).

**2. Runner and per-call output.** `tools/calib_run.py` (`calibration.yaml:339-349`).
- **This runner already saves the series.** It sets `KDT_SERIES_DUMP_DIR=<wd>/_series` itself (`calib_run.py:634`). On every model run it scores discharge on all three windows (`cal`, `val`, `headline`=full) and SWE on the scored window (`SWE_calibration` or `SWE_holdout`) (`:675-683`). Each scoring writes `<label>.NNNN.csv` with real dates (`date,obs,sim`) plus a `manifest.jsonl` line (`:565-575`, `:604-613`; `KTC:432-471`).
- **Not overwritten:** the number NNNN is the next free one (`KTC:432-435`), so files pile up across calls (and across seeds if `<wd>` is reused).
- Raw model output `<wd>/output.txt` and `<wd>/parsed/crhm_results.csv` are at fixed names → overwritten each call (`calib_run.py:653-665`).
- Size per call (measured in `SMF/06_CRHM/_series/`): cal ≈ 104 KB, val ≈ 104 KB, headline ≈ 208 KB, SWE ≈ 63 KB → ≈ **0.47 MB** per call (290 MB for 621 sets).
- Does it cover every call? It covers every real model run that got as far as scoring, including probe and baseline runs. It does **not** cover cache-hit calls (no run). In the old folder: 621 cal/val/headline sets, 540 SWE_calibration, 81 SWE_holdout, against 535 calibration + 81 holdout cache lines (inventory row 2). So the 5 extra sets are runs that are not in the cache (probes / baseline). The files carry no call number. The link to a cache line has to come from the order, or from matching the manifest `metric_values` to the cache scores (e.g. `cal.0000` NSE 0.43331 = the default run).

**3. Observed series.** Discharge: `/mnt/disk1/Hydrocraft_server/outputs/crhm_08lb038_blueriver/obs_08LB038.csv` (`date,Q_m3s`, sha256 pinned) (`calib_run.py:81`, `:109-112`). SWE: `SMF/06_CRHM/ki/calib/swe_obs_BCE-1E02P.csv` (`date,swe_mm`, not daily) (`:75`, `:590-601`). Windows: calibration 2006-2010, holdout 2011-2015, full 2006-2015 (`:102-104`; `calibration.yaml:363-368`). The obs column is also inside every series file.

**4. What to copy after each call.** The new files in `<wd>/_series/` since the last call (normally 4: `cal`, `val`, `headline`, `SWE_<split>`) + the new `manifest.jsonl` lines, + `<wd>/kdt_params.json`, `<wd>/calib_metrics.json`. Moving them out after each call (or just keeping `<wd>/_series/` per seed and tagging with the call counter) both work. Disk: 0.47 MB x 575 x 3 ≈ **0.8 GB** (1,800 calls ≈ 0.85 GB).

**5. Risks.** The runner uses the **live** CRHM KI tools (`run_crhm.py`, `parse_crhm_output.py`) from `/mnt/disk1/Hydrocraft_server/models/CRHM/knowledge_infrastructure/tools` (`calib_run.py:73-74`, `:654-661`), not a frozen copy. Inputs exist (checked: deck, forcing folder, obs csv, `crhm` binary at `calib_run.py:78-82`).

---

## 3. MODFLOW6 (Sagehen 10343500): streamflow + baseflow, scored as flow-duration curves

**1. Variables and kind.** `streamflow` = **flow**, `baseflow` (Lyne-Hollick) = **series**. **But both are scored as 19-point flow-duration curves (FDC), not as dated series** (`SMF/05_MODFLOW6/ki/calibration.yaml:13-19`, `:58-63`; `tools/calib_run.py:97`, `:315-317`, `:420-425`). The model forcing is a repeating yearly cycle, not real dates (`calibration.yaml:15-17`), so day-by-day r / alpha / beta against the gauge do not make sense here. The cache already has, for each: nse, kge, pbias, r, **lognse**, sim_mean_m3s (`calib_run.py:435-441`). **It also has all four FDCs** (sim and obs, streamflow and baseflow, 19 values each, rounded to 5 decimals) in `__kdt__` (`:456-459`).

**2. Runner and per-call output.** `tools/calib_run.py` (`calibration.yaml:165-170`).
- It deletes `<wd>/model/` and copies a fresh model in at the start of each call (`calib_run.py:410-413`) → **overwritten each call**, and the last one stays after the run.
- The simulated daily outflow comes from `<wd>/model/ex-gwf-sagehen.sfr.bud` (binary, `EXT-OUTFLOW`, 399 periods) (`:286-294`). The first 30 days are dropped, which leaves 369 values (`:86`, `:420`).
- Size: `.sfr.bud` ≈ **18.2 MB** and `.hds` ≈ 18.9 MB per call (measured in `MAR/06_modflow6_flow_baseflow_sagehen/calib_work/model/`). The 369-value outflow array alone is about 3 KB (`MAR/06_…/sim_outlet_q.npy` = 3,320 B).

**3. Observed series.** `SMF/05_MODFLOW6/ki/obs/sagehen_baseflow.csv` (`date,total_m3s,baseflow_lyne_hollick_m3s,…`, sha256 pinned) (`calib_run.py:91-92`, `:351-375`). Windows: calibration 1953-10-01..1990-03-12, holdout 1990-03-13..2026-08-22 (`:96`, `:364-371`; `calibration.yaml:193-194`). The obs FDC for the scored window is also in each cache line.

**4. What to copy after each call.** For the panel on the FDC: **nothing extra**. The cache line already has both FDCs, so r, alpha, beta and lnNSE can be worked out from it. Only limit: the stored FDCs are rounded to 5 decimals, so the results will differ slightly from full-precision numbers. If full precision or the 369-day series is wanted: right after each call, read `<wd>/model/ex-gwf-sagehen.sfr.bud` with flopy, the same way the runner does (`calib_run.py:289-291`), and save the outflow as a small `.npy` (≈ 3 KB/call, ≈ 6 MB in total). Copying the raw `.sfr.bud` would cost 18.2 MB x 230 x 3 ≈ **12.5 GB** (1,800 calls ≈ 33 GB). Not worth it.

**5. Risks.** MODFLOW6 does not call `all_metrics()`; it scores with its own `metrics_pair()` (`calib_run.py:377-388`), so `KDT_SERIES_DUMP_DIR` gives nothing here. Its `lognse` uses log10 with values below 1e-4 raised to 1e-4 (`:384-385`). The NSE of log10 values equals the NSE of ln values, so this matches lnNSE except for that floor. Clear decision needed: whether the "series" panel for this model is defined on the FDC (what the paper scored) or not at all.

---

## 4. SUMMA (Moyie 08NH120): scalarTotalRunoff + scalarSWE

**1. Variables and kind.** `scalarTotalRunoff` (scored as `averageRoutedRunoff` x area, daily mean m³/s) = **flow**. `scalarSWE` (HRU nearest 1,940 m, daily mean mm) = **series** (`SMF/03_SUMMA/ki/calibration.yaml:60`, `:89-102`; `tools/calib_run.py:140`, `:508-539`, `:547-576`). The cache has, for both: nse, kge, pbias, r, rmse (`calib_run.py:654-656`). Missing: alpha, beta, lnNSE.

**2. Runner and per-call output.** `tools/calib_run.py` (`calibration.yaml:752-756`).
- Each call runs in its own folder `<wd>/_evals/eval_<KDT_CALIB_EVAL_ID>_<pid>/` (`calib_run.py:730-733`). The SUMMA output is one NetCDF, `…/output/moyie_*.nc` (hourly, 2005-01-01 01:00..2015-12-31 23:00, 5 variables: `averageRoutedRunoff, scalarTotalRunoff, pptrate, scalarTotalET, scalarSWE`) (`:97-99`, `:192-196`, `:410-412`, `:501-506`).
- After a good call the folder is **deleted** unless `KDT_CALIB_KEEP_EVAL=1` (`:734`, `:763-765`). With keep on, folders are not overwritten (unique name), but they pile up.
- Size: not measured (no kept folder on disk). By arithmetic: ≈ 96,400 hours x 13 values x 8 B ≈ **10 MB per call**.
- Easier: with `KDT_SERIES_DUMP_DIR` set (section 0.5), each call writes two `headline.NNNN.csv`: first runoff (1,826 days for either window), then SWE (common days only) (`calib_run.py:645-656`, `:774`, `:777`). About 0.1-0.15 MB per call.

**3. Observed series.** Discharge comes from the HYDAT database `/mnt/disk4/Hydat_sqlite3_20260116/Hydat.sqlite3`, table `DLY_FLOWS`, station 08NH120 (`calib_run.py:87`, `:90`, `:270-291`). SWE comes from CanSWE `/mnt/datasets/数据/CanSWE/CanSWE-CanEEN_1928-2024_v7.nc`, station BCE-2C10P (`:133-134`, `:578-622`). Both exist (checked). Windows: calibration 2006-01-01..2010-12-31, holdout 2011-01-01..2015-12-31 (`:104-107`; `calibration.yaml:73-77`, `:787-788`). There is no ready-made obs CSV, so the dumped files (which carry obs) are the easiest source.

**4. What to copy after each call.** (a) The two new `headline.NNNN.csv` + manifest lines (≈ 0.15 MB x 280 x 3 ≈ **0.13 GB**; 1,800 calls ≈ 0.27 GB), or (b) with `KDT_CALIB_KEEP_EVAL=1`: `<wd>/_evals/eval_*_*/output/moyie_*.nc`, then delete the folder (≈ 10 MB x 840 ≈ **8.4 GB**; 1,800 calls ≈ 18 GB). Plus `<wd>/kdt_params.json`, `<wd>/calib_metrics.json`.

**5. Risks.** The runner calls the **live** SUMMA KI tools (`/mnt/disk1/Hydrocraft_server/models/SUMMA/knowledge_infrastructure`, `calib_run.py:75`) and the case folder `/mnt/disk1/Hydrocraft_server/outputs/moyie_08nh120_summa` (`:82-84`; exists). The contract says dds but the kit switched to NSGA-II (`SMF/03_SUMMA/rerun.log` line 4). About 62 s per call (`calibration.yaml:758`).

---

## 5. VIC Xixian (50225601): OUT_DISCHARGE + OUT_EVAP

**1. Variables and kind.** `OUT_DISCHARGE` (Lohmann-routed daily Q at station XX, m³/s) = **flow**. `OUT_EVAP` (basin-mean ET over 28 cells, **monthly** mean mm/day; 24 months per window) = **series** (`SMF/04_VIC/ki/tools/calib_run.py:13-15`, `:511-533`; `calibration.yaml:89-104`). The cache has, for both (and a top-level copy of discharge): nse, kge, pbias, r, rmse (`calib_run.py:685-690`, `:807-813`). Missing: alpha, beta, lnNSE.

**2. Runner and per-call output.** `tools/calib_run.py` (`calibration.yaml:574-578`).
- It deletes and re-creates `<wd>/run/` at the start of each call (`calib_run.py:730-733`) → **overwritten each call**. It deletes it at the end too, unless `KDT_VIC_KEEP_WORKDIR=1` (`:849-850`).
- Discharge series: `<wd>/run/routing/rout_out/XX.day` (`year month day Q`, 2005-01-01..2010-12-31 = 2,191 rows, about 55 KB) (`:589-603`).
- ET series: `<wd>/run/vic_result/xixian_fluxes_<lat>_<lon>.txt`, 28 files, 27 columns incl. `OUT_EVAP` (`:484-509`, `:511-533`). Measured on a past run of the same domain: ≈ 0.47 MB each, ≈ **13 MB** for 28 (`/mnt/disk1/Hydrocraft_server/outputs/xixian_2005_2010_025deg/vic_result/`). (`<wd>/run/routing/vic_in/fluxes_*` also has evap, but rounded to 4 decimals, `:506-507`; not what was scored.)
- Easier: with `KDT_SERIES_DUMP_DIR` set (section 0.5), each call writes two `headline.NNNN.csv`: first discharge (731 or 730 days), then ET (24 months) (`calib_run.py:775-797`, via `_metric_block` `:685-686`). About 35 KB per call.

**3. Observed series.** Discharge: `/mnt/datasets/china_water_level/淮河txt/息县.txt` (station 50225601, Q ≤ -90 dropped) (`calib_run.py:111`, `:612-683`; `calibration.yaml:35-49`). ET: `SMF/04_VIC/ki/calib/gleam_xixian_monthly.csv` (`date,year,month,gleam_et_mm_month,gleam_et_mm_day`) (`calib_run.py:98`, `:536-560`). Windows: calibration 2007-01-01..2008-12-31, holdout 2009-01-01..2010-12-31; the model always runs 2005-2010 (`calib_run.py:142-146`; `calibration.yaml:595-596`).

**4. What to copy after each call.** (a) The two new `headline.NNNN.csv` + manifest lines (≈ 35 KB x 250 x 3 ≈ **26 MB**; 1,800 calls ≈ 63 MB), or (b) with `KDT_VIC_KEEP_WORKDIR=1`: `XX.day` + the 28 flux files (≈ 13 MB x 750 ≈ **10 GB**; 1,800 calls ≈ 24 GB). Plus `<wd>/kdt_params.json`, `<wd>/calib_metrics.json`.

**5. Risks.** VIC is staged (6 rounds in the old run) and the old run had many holdout calls (128 holdout vs 366 calibration lines, inventory row 5), so expect more calls than 150. The runner uses `/dev/shm/kdt_vic_xixian_calib/` caches for forcing and routing (`calib_run.py:101-102`, `:188-232`), which are shared across calls (fine when run one at a time). The contract's holdout note (copied from the Tangnaihai case) says the holdout split re-runs 2005-2016 (`calibration.yaml:608-612`), but the runner always runs 2005-2010 (`calib_run.py:142-146`). Only a text mismatch. A certification run against `calib/reference_run.json` happens first (`SMF/04_VIC/rerun.log` line 4); it runs the model directly (section 0.3).

---

## 6. WOFOST (KBS maize): TWSO + TAGP, snapshot

**1. Variables and kind.** `TWSO` (final grain yield / 0.85) and `TAGP` (final biomass), one value per year = **snapshot** (`SMF/01_WOFOST/ki/calibration.yaml:9-13`; `tools/calib_run.py:128-136`). The cache has nrmse, pbias, nse for both (`calib_run.py:141-144`). **That already covers the snapshot panel (pbias, nrmse).**

**2. Runner and per-call output.** `tools/calib_run.py` (`calibration.yaml:31-36`). Per year it writes `<wd>/a<year>.yaml` and `<wd>/o<year>.csv` (daily crop output); these are at fixed names → **overwritten each call** (`calib_run.py:119-127`). The per-year sim values that go into the scores are not saved anywhere; only the scores are. Each `o<year>.csv` is small (one row per day of a ~200-day season).

**3. Observed series.** KBS LTER files under `/mnt/datasets/obs/ltar_edi/knb-lter-kbs/`: `20_r58/Agronomic+Yields+-+Anual+Crops.csv` (yield, T1, *Zea mays*) and `19_r85/Annual+Crops+and+Alfalfa+Biomass` (biomass WHOLE x 10) (`calib_run.py:11`, `:20-34`). Split: maize years present in both, taken in turn, even positions = calibration, odd = holdout (`:110-112`). Not a date window.

**4. What to copy after each call.** **Nothing is needed** for the pbias / nrmse panel: the cache line has both. Optional, if the per-year sim values are wanted for checks: the `<wd>/o<year>.csv` files of the call (a few hundred KB per call by estimate; not measured).

**5. Risks.** The runner loads the **live** WOFOST KI tools, including the live `tools/calib_run.py` of the WOFOST KI, from `/mnt/disk1/Hydrocraft_server/models/WOFOST/knowledge_infrastructure/tools` (`calib_run.py:12-13`, `:104-109`). Years where the run fails are silently skipped (`:137-138`), so `n_years` in the cache can change from call to call. Any split value other than `calibration` (also `full` or `_both`) gives the holdout years (`:102`, `:112`).

---

## Summary table

| Model | Var 1 (kind) | Var 2 (kind) | Cache has now | Series file per call (after env/flag) | Overwritten? | Copy per call | Disk, 3 seeds at budget (1,800 calls) | Blocker / risk |
|---|---|---|---|---|---|---|---|---|
| HBV | Q (flow) | SWE (series) | nse kge pbias rmse r (+day_bias) | dump `headline.NNNN.csv` x2, or `<wd>/run/bmi_out/bmi_state.npz` with `KDT_HBV_KEEP_WORKDIR=1` | yes (`run/` wiped) | 2 csv (~0.15 MB) or npz (~0.42 MB) | 0.26 GB (0.27) / 0.75 GB (0.8) | **staged inputs missing at the `/tmp` path**; copies with matching sha256 are in `MAR/02_…/stage/` |
| CRHM | basinflow_s (flow) | SWE (series) | nse kge pbias rmse r n | `<wd>/_series/{cal,val,headline,SWE_<split>}.NNNN.csv` (already written) | no (piles up) | 4 csv (~0.47 MB) | 0.8 GB (0.85) | live CRHM tools; no call id in file names |
| MODFLOW6 | streamflow (flow, FDC) | baseflow (series, FDC) | nse kge pbias r lognse + all 4 FDCs | `<wd>/model/ex-gwf-sagehen.sfr.bud` | yes (`model/` wiped) | nothing (FDCs in cache), or a 3 KB outflow `.npy` | ~0 / 6 MB (raw .bud would be 12.5 GB) | scored on 19-point FDC, not dates; FDCs rounded to 5 dp |
| SUMMA | scalarTotalRunoff (flow) | scalarSWE (series) | nse kge pbias r rmse | dump `headline.NNNN.csv` x2, or `<wd>/_evals/eval_*/output/moyie_*.nc` with `KDT_CALIB_KEEP_EVAL=1` | no (unique folder, but deleted unless kept) | 2 csv (~0.15 MB) or nc (~10 MB) | 0.13 GB (0.27) / 8.4 GB (18) | live SUMMA tools; obs only in HYDAT db + CanSWE nc |
| VIC | OUT_DISCHARGE (flow) | OUT_EVAP monthly (series) | nse kge pbias r rmse | dump `headline.NNNN.csv` x2, or `XX.day` + 28 flux txt with `KDT_VIC_KEEP_WORKDIR=1` | yes (`run/` wiped) | 2 csv (~35 KB) or ~13 MB | 26 MB (63 MB) / 10 GB (24) | more calls than budget (staged + many holdouts) |
| WOFOST | TWSO (snapshot) | TAGP (snapshot) | nrmse pbias nse | none needed | `o<year>.csv` yes | nothing | 0 | live WOFOST tools; failed years silently dropped |
