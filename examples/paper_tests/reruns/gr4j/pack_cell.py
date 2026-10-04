"""Pack one finished GR4J cell's per-call series (run_workdir/qsim_<id>.csv, 0.7 MB each) to save disk: (1) every
CSV goes into run_workdir/qsim_all.tar.gz (all columns kept); (2) Qsim_mm of every call -> qsim_Qsim_mm.npz (float64,
keyed by id, dates once); (3) the CSVs are removed ONLY after the archive lists every file with its exact size and the
npz re-reads equal to the CSVs. Runs only when check.json exists.   usage: python pack_cell.py <tag>"""
import glob, json, os, sys, tarfile
import numpy as np, pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", sys.argv[1])
W = os.path.join(D, "run_workdir")
assert os.path.exists(os.path.join(D, "check.json")), "cell not finished"
files = sorted(glob.glob(os.path.join(W, "qsim_*.csv")))
assert files, "nothing to pack"
tgz = os.path.join(W, "qsim_all.tar.gz")
with tarfile.open(tgz, "w:gz") as t:
    for f in files:
        t.add(f, arcname=os.path.basename(f))
with tarfile.open(tgz, "r:gz") as t:
    got = {m.name: m.size for m in t.getmembers()}
assert got == {os.path.basename(f): os.path.getsize(f) for f in files}, "archive incomplete"
q, dates = {}, None
for f in files:
    df = pd.read_csv(f)
    if dates is None:
        dates = df["Date"].astype(str).to_numpy()
    assert (df["Date"].astype(str).to_numpy() == dates).all()
    q[os.path.basename(f)[5:-4]] = df["Qsim_mm"].to_numpy(float)
np.savez(os.path.join(W, "qsim_Qsim_mm.npz"), dates=dates.astype("U10"), **{"id_" + k: v for k, v in q.items()})
z = np.load(os.path.join(W, "qsim_Qsim_mm.npz"))
assert all(np.array_equal(z["id_" + k], v) for k, v in q.items())
for f in files:
    os.remove(f)
print("packed", sys.argv[1], len(files), "files")
