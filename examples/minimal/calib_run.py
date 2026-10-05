"""Subprocess runner for the synthetic reservoir, with explicit split/read-back."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from calibration_kit.panel import panel_block
from model import simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(Path(os.environ["KDT_CALIB_PARAMS"]).read_text())
    applied = {name: float(raw[name]) for name in ("release", "gain")}
    data = np.genfromtxt(Path(__file__).with_name("observations.csv"), delimiter=",", names=True)
    modeled = simulate(data["forcing"], **applied)
    split = os.environ.get("KDT_CALIB_SPLIT", "calibration")
    if split not in ("calibration", "holdout"):
        raise ValueError(f"Unknown split: {split}")
    mask = slice(180, 240) if split == "holdout" else slice(40, 180)
    sim, obs = modeled[mask], data["observed"][mask]
    reply = {"nse": float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
             "r": float(np.corrcoef(sim, obs)[0, 1])}
    reply.update(panel_block({"Q": (sim, obs)}, {"Q": "flow"}))
    reply["__kdt__"].update(applied_params=applied, split=split, case_id="SITE:synthetic_reservoir")
    index = [str(int(step)) for step in data["step"][mask]]
    values = np.asarray(sim, dtype=np.float64)
    reply["__kdt__"]["target_proofs"] = [{
        "schema": "kdt-target-proof/1", "target": "Q", "unit": "arbitrary flow units",
        "split": split, "window": [index[0], index[-1]], "n": len(values), "dtype": "float64",
        "index_hash": hashlib.sha1("\n".join(index).encode()).hexdigest(),
        "values_hash": hashlib.sha1(values.tobytes()).hexdigest(),
        "values": values.tolist(), "index": index, "source": "linear reservoir raw output",
    }]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("KDT_CALIB_EMIT_SERIES") == "1":
        series = args.out.parent / "synthetic_Q.npz"
        np.savez(series, sim=sim, obs=obs)
        reply["__kdt__"]["series"] = {"Q": str(series.resolve())}
    args.out.write_text(json.dumps(reply, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
