"""(validation_native copy) The run-time dependencies of the built-in-rule validation runs, hashed (codex B2 r2): every .py file of the paper-frozen kit, the
recorder, the SPOTPY source tree the runs import, the installed pymoo, the new kit files the driver and judge load,
and the two scripts. `python deps_manifest.py write` makes DEPS_MANIFEST.json (once, before the first run);
deps_digest() returns the digest of the files as they are NOW (run_val.py refuses to start, and the judge refuses a
run, when it differs from the frozen one)."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = {
    "frozen_kit": "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code/calibration_kit",
    "recorder": os.path.join(os.path.dirname(HERE), "recorder", "frozen_recorder.py"),
    "spotpy_src": "/mnt/disk1/Hydrocraft_server/models/spotpy/source/repo/src/spotpy",
    "pymoo": "/mnt/disk1/Hydrocraft_server/python_env/lib/python3.12/site-packages/pymoo",
    "numpy": "/mnt/disk1/Hydrocraft_server/python_env/lib/python3.12/site-packages/numpy",
    "frozen_rule": os.path.join(os.path.dirname(HERE), "validation", "frozen_rule"),   # the same snapshot
    "scipy": "/mnt/disk1/Hydrocraft_server/python_env/lib/python3.12/site-packages/scipy",   # codex B2 r4 #2
    "numpy_libs": "/mnt/disk1/Hydrocraft_server/python_env/lib/python3.12/site-packages/numpy.libs",   # linked BLAS etc.
    "scipy_libs": "/mnt/disk1/Hydrocraft_server/python_env/lib/python3.12/site-packages/scipy.libs",   # (codex B2 r5)
    "run_native": os.path.join(HERE, "run_native.py"),
    "judge_native": os.path.join(HERE, "judge_native.py"),
    "sceua_recording": os.path.join(HERE, "sceua_recording.py"),
    "deps_manifest": os.path.abspath(__file__),       # the checker is inside its own lock (codex B2 r3 #2)
}


def _files(root):
    if os.path.isfile(root):
        return [root]
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        out += [os.path.join(d, f) for f in sorted(fs) if not f.endswith(".pyc")]   # every file, data too (codex B2 r3 #1)
    return out


def check_loaded(modules):
    """Every loaded module of these packages must come from its hashed root (codex B2 r3 #3). Returns the problems."""
    want = {"calibration_kit": ROOTS["frozen_kit"], "spotpy": ROOTS["spotpy_src"], "pymoo": ROOTS["pymoo"],
            "numpy": ROOTS["numpy"], "scipy": ROOTS["scipy"], "kdt_rule_frozen": ROOTS["frozen_rule"]}
    bad = []
    for name, mod in list(modules.items()):
        top = name.split(".")[0]
        f = getattr(mod, "__file__", None)
        # a module of these packages must come from ONE of the hashed folders (pymoo re-exports numpy under its own
        # names, e.g. pymoo.gradient.toolbox IS numpy)
        if top in want and f and not any(os.path.realpath(f).startswith(os.path.realpath(r) + os.sep)
                                         for r in want.values()):
            bad.append(f"{name} <- {f}")
    return bad


def manifest():
    m = {}
    for name, root in ROOTS.items():
        for f in _files(root):
            m[f] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    return m


def digest(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def deps_digest():
    return digest(manifest())


def frozen_digest():
    return json.load(open(os.path.join(HERE, "DEPS_MANIFEST.json")))["digest"]


if __name__ == "__main__" and sys.argv[1:] == ["write"]:
    p = os.path.join(HERE, "DEPS_MANIFEST.json")
    if os.path.exists(p):
        sys.exit("DEPS_MANIFEST.json exists; never rewritten")
    m = manifest()
    json.dump({"digest": digest(m), "n_files": len(m), "roots": ROOTS, "files": m}, open(p, "w"), indent=0)
    print("written", len(m), "files", digest(m))
