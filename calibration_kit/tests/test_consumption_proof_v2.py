"""Acceptance suite for consumption_proof v2 (plan v2, 2026-09-14). All four fixtures codex
required must pass before the STOP is enabled:
  (a) dead-address knobs -> exactly those UNREACHABLE, live ones ALIVE
  (b) structured dormant_when -> DORMANT / WINDOW_MASKED, never UNREACHABLE
  (c) weak-but-live knob -> ALIVE or INSENSITIVE, never UNREACHABLE
  (d) raw output moves materially, ROUNDED objective identical -> OBJECTIVE_MASKED, never UNREACHABLE
  (e) free-text dormant_when is REFUSED (would otherwise wave through a wrong address)
  (f) a driver that emits no proof -> ok=None, nothing proven, no verdicts
"""
import hashlib, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # this checkout, not the shared folder
import tempfile
import numpy as np
from calibration_kit.evaluator import Evaluator, resolve_train_split

NAMES  = ["gw_coeff", "gw_zmax", "refkdt", "slope", "smcmax_mult", "retdeprtfac", "revap_co", "gw_zinit", "masked"]
DEAD   = {"refkdt", "slope", "smcmax_mult"}      # written to a file the model never opens
WEAK   = {"retdeprtfac"}                          # real but tiny effect
DORM   = {"revap_co", "gw_zinit"}                 # consumed but inactive here / erased by spin-up
N = 100
T = np.linspace(0, 6.28, N)


def fake_series(x):
    """The 'model': a daily series that depends on the LIVE knobs only."""
    x = dict(zip(NAMES, x))
    s = 100.0 + 50.0 * np.sin(T) * x["gw_coeff"] / 44.8 + 10.0 * (x["gw_zmax"] / 250.0)
    s = s + 1e-4 * x["retdeprtfac"]                       # weak: moves raw by ~1e-4
    # 'masked': a zero-MEAN perturbation — raw moves a lot, the objective (mean) does not
    z = np.zeros(N); z[0] = +5.0 * x["masked"]; z[1] = -5.0 * x["masked"]
    return s + z


def proof(vals, split):
    idx = [f"1981-01-{d+1:02d}" for d in range(N)]
    return {"schema": "kdt-target-proof/1", "target": "streamflow", "unit": "m3/s", "split": split,
            "window": [idx[0], idx[-1]], "n": N, "dtype": "float64",
            "index_hash": hashlib.sha1("\n".join(idx).encode()).hexdigest(),
            "values_hash": hashlib.sha1(np.asarray(vals, dtype=np.float64).tobytes()).hexdigest(),
            "values": [float(v) for v in vals], "index": idx, "source": "fake"}


class Fake(Evaluator):
    """Minimal evaluator: objective = ROUNDED mean (4 dp), like the real validators."""
    def __init__(self, emit_proof=True):
        self._last_metrics = {}; self._wrong_case = None; self.emit_proof = emit_proof
        self.objectives = [type("O", (), {"loss": staticmethod(lambda m: -m["score"])})()]
    def evaluate(self, x):
        s = fake_series(x); split = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
        m = {"score": round(float(s.mean()), 4), "__kdt__": {"applied_params": dict(zip(NAMES, x))}}
        if self.emit_proof:
            m["__kdt__"]["target_proofs"] = [proof(s, split)]
        self._last_metrics[split] = m
        return [-m["score"]]


# evidence must resolve to an existing file; the original pointed at a review file that was moved on
# 2026-09-28, which broke this test for a reason unrelated to the code under test
import atexit, shutil
_EVIDENCE_DIR = tempfile.mkdtemp(prefix="kdt_proof_evidence_")
atexit.register(shutil.rmtree, _EVIDENCE_DIR, ignore_errors=True)
_EVIDENCE = os.path.join(_EVIDENCE_DIR, "codex.md")
open(_EVIDENCE, "w").write("declared dormancy evidence (test fixture)\n")
_EVIDENCE_JSON = os.path.join(_EVIDENCE_DIR, "diagnosis.json")
open(_EVIDENCE_JSON, "w").write('{"day1_max_dQ": 2.4, "in_window_max_dQ": 0.0}\n')

XD = [44.8, 250.0, 3.0, 0.1, 1.0, 1.0, 0.02, 25.0, 0.0]
LO = [9.0, 50.0, 0.3, 0.1, 0.85, 0.1, 0.02, 5.0, 0.0]
HI = [225.0, 500.0, 8.0, 1.0, 1.10, 5.0, 0.2, 45.0, 1.0]
PARAMS_OK = [
    {"name": "revap_co", "dormant_when": {"kind": "process_inactive",
        "condition": "aquifer storage < revap_min (750 mm) throughout the scored window",
        "evidence": _EVIDENCE, "declared_by": "codex 2026-09-10"}},
    {"name": "gw_zinit", "dormant_when": {"kind": "window_erased",
        "condition": "6-month spin-up precedes the scored window; day-1 max|dQ| 2.4 m3/s, 0.0 in window",
        "evidence": _EVIDENCE_JSON, "declared_by": "round-4 review"}},
]


def run(params, emit_proof=True):
    return Fake(emit_proof).consumption_proof(XD, LO, HI, names=NAMES, parameters=params)


def test_a_dead_addresses_are_unreachable():
    r = run(PARAMS_OK)
    assert set(r["unreachable"]) == DEAD, r["by_verdict"]
    assert r["ok"] is False
    for live in ("gw_coeff", "gw_zmax"):
        assert live in r["by_verdict"]["ALIVE"], r["by_verdict"]

def test_b_structured_dormancy_is_not_unreachable():
    r = run(PARAMS_OK)
    assert "revap_co" in r["by_verdict"].get("DORMANT", []), r["by_verdict"]
    assert "gw_zinit" in r["by_verdict"].get("WINDOW_MASKED", []), r["by_verdict"]
    assert "revap_co" not in r["unreachable"] and "gw_zinit" not in r["unreachable"]

def test_c_weak_live_knob_never_unreachable():
    r = run(PARAMS_OK)
    v = [p["verdict"] for p in r["params"] if p["param"] == "retdeprtfac"][0]
    assert v in ("ALIVE", "INSENSITIVE"), v
    assert "retdeprtfac" not in r["unreachable"]

def test_d_raw_moves_objective_identical_is_masked_not_unreachable():
    r = run(PARAMS_OK)
    row = [p for p in r["params"] if p["param"] == "masked"][0]
    assert row["targets"]["streamflow"]["material"] is True, row
    assert row["objective_max_delta"] == 0.0, row            # rounded objective unchanged
    assert row["verdict"] == "OBJECTIVE_MASKED", row["verdict"]
    assert "masked" not in r["unreachable"]

def test_e_free_text_dormancy_is_refused():
    bad = [{"name": "revap_co", "dormant_when": "process never activates here"},      # prose
           {"name": "gw_zinit", "dormant_when": {"kind": "window_erased"}}]            # no evidence
    r = run(bad)
    assert "revap_co" in r["unreachable"] and "gw_zinit" in r["unreachable"], r["by_verdict"]

def test_f_no_proof_means_nothing_proven():
    r = run(PARAMS_OK, emit_proof=False)
    assert r["ok"] is None and r["params"] == [], r
    assert "no __kdt__.target_proofs" in r["summary"]


# ---------------------------------------------------------------------------------------------
# Round-6 regression fixtures — one per must-fix from the build review (codex REVISE, kimi A-W-F)
# ---------------------------------------------------------------------------------------------

def test_g_nan_loss_is_unproven_not_unreachable():
    """kimi: a NaN loss read as 'no movement' -> UNREACHABLE -> false STOP. Must be UNPROVEN."""
    class NanOn(Fake):
        def evaluate(self, x):
            r = super().evaluate(x)
            if x[NAMES.index("gw_coeff")] != XD[NAMES.index("gw_coeff")]:
                return [float("nan")]                       # displaced gw_coeff run yields NaN loss
            return r
    r = NanOn().consumption_proof(XD, LO, HI, names=NAMES, parameters=PARAMS_OK)
    v = [p["verdict"] for p in r["params"] if p["param"] == "gw_coeff"][0]
    assert v == "UNPROVEN", v
    assert "gw_coeff" not in r["unreachable"]

def test_h_partial_target_mismatch_is_unproven_not_unreachable():
    """codex: one target identical + another target MISSING in the displaced run used to yield
    UNREACHABLE. Must be UNPROVEN unless a comparable target moved."""
    class TwoTargets(Fake):
        def evaluate(self, x):
            s = fake_series(x); split = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
            displaced = any(a != b for a, b in zip(x, XD))
            proofs = [proof(s, split)]
            # second target 'yield': present in the DEFAULT run, MISSING once anything is displaced
            if not displaced:
                y = proof(np.full(N, 3.0), split); y["target"] = "yield"; proofs.append(y)
            m = {"score": round(float(s.mean()), 4), "__kdt__": {"applied_params": dict(zip(NAMES, x)), "target_proofs": proofs}}
            self._last_metrics[split] = m; return [-m["score"]]
    r = TwoTargets().consumption_proof(XD, LO, HI, names=NAMES, parameters=PARAMS_OK)
    # dead knobs: streamflow identical AND yield missing -> must be UNPROVEN, NOT unreachable
    for d in DEAD:
        v = [p["verdict"] for p in r["params"] if p["param"] == d][0]
        assert v == "UNPROVEN", (d, v)
    assert not r["unreachable"], r["unreachable"]
    # live knobs: streamflow moved -> still ALIVE despite the missing yield target
    assert "gw_coeff" in r["by_verdict"]["ALIVE"]

def test_i_unresolvable_evidence_is_refused():
    """both seats: evidence that does not exist on disk must NOT suppress the STOP."""
    bad = [{"name": "revap_co", "dormant_when": {"kind": "process_inactive",
            "condition": "x", "evidence": "/nonexistent/receipt_that_is_a_typo.json"}}]
    r = run(bad)
    assert "revap_co" in r["unreachable"], r["by_verdict"]

def test_j_corrupt_proof_record_is_unproven():
    """codex: a proof whose hash/length/finiteness do not check out must not be compared."""
    class Corrupt(Fake):
        def evaluate(self, x):
            r = super().evaluate(x); split = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
            pr = self._last_metrics[split]["__kdt__"]["target_proofs"][0]
            if any(a != b for a, b in zip(x, XD)):
                pr["values"][0] = float("nan")             # corrupt the displaced proof
            return r
    r = Corrupt().consumption_proof(XD, LO, HI, names=NAMES, parameters=PARAMS_OK)
    assert all(p["verdict"] == "UNPROVEN" for p in r["params"]), r["by_verdict"]
    assert not r["unreachable"]

def test_k_duplicate_target_names_make_proofs_unusable():
    class Dup(Fake):
        def evaluate(self, x):
            r = super().evaluate(x); split = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
            tp = self._last_metrics[split]["__kdt__"]["target_proofs"]; tp.append(dict(tp[0]))
            return r
    r = Dup().consumption_proof(XD, LO, HI, names=NAMES, parameters=PARAMS_OK)
    assert r["ok"] is None, r


# ---------------------------------------------------------------------------------------------
# Round-2 build-review regression fixtures
# ---------------------------------------------------------------------------------------------

def test_l_mixed_valid_and_invalid_proofs_are_unusable():
    """codex r2: a valid sibling must not be judged alone when another record is corrupt —
    if the sibling is identical and the dropped one would have moved, that is a false STOP."""
    class Mixed(Fake):
        def evaluate(self, x):
            r = super().evaluate(x); split = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
            tp = self._last_metrics[split]["__kdt__"]["target_proofs"]
            bad = dict(tp[0]); bad["target"] = "yield"; bad["values"] = [1.0]      # n mismatch -> invalid
            tp.append(bad); return r
    r = Mixed().consumption_proof(XD, LO, HI, names=NAMES, parameters=PARAMS_OK)
    assert r["ok"] is None, r          # whole set unusable -> nothing proven, no STOP

def test_m_directory_or_empty_file_is_not_evidence(tmp_path=None):
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp()); empty = d / "empty.txt"; empty.write_text("")
    for ev_path in (str(d), str(empty)):
        bad = [{"name": "revap_co", "dormant_when": {"kind": "process_inactive", "condition": "x", "evidence": ev_path}}]
        r = run(bad); assert "revap_co" in r["unreachable"], (ev_path, r["by_verdict"])
    good = d / "receipt.json"; good.write_text('{"kind": "process_inactive", "measured": true}')
    ok = [{"name": "revap_co", "dormant_when": {"kind": "process_inactive", "condition": "x", "evidence": str(good)}}]
    r = run(ok); assert "revap_co" in r["by_verdict"].get("DORMANT", []), r["by_verdict"]
    bad_json = d / "broken.json"; bad_json.write_text("{not json")
    r = run([{"name": "revap_co", "dormant_when": {"kind": "process_inactive", "condition": "x", "evidence": str(bad_json)}}])
    assert "revap_co" in r["unreachable"]

def test_n_force_live_bypasses_resumability_cache():
    """kimi r2: with _force_live the evaluator must NOT serve a cached payload.
    2026-09-16 (checkpoint, both seats C3): the previous body grepped evaluator.py for a substring and
    never ran anything. The behavioural version lives in test_entry_points_checkpoint.py::test_p4 and
    is invoked from here so this suite still fails if that guard breaks."""
    from calibration_kit.tests.test_entry_points_checkpoint import test_p4_force_live_bypasses_preloaded_cache_behaviourally as _t
    _t()


def test_o_extra_target_only_in_displaced_run_is_unproven():
    """codex r3: an EXTRA target appearing only in the displaced run is a mismatch too."""
    class Extra(Fake):
        def evaluate(self, x):
            r = super().evaluate(x); split = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
            if any(a != b for a, b in zip(x, XD)):
                tp = self._last_metrics[split]["__kdt__"]["target_proofs"]
                y = proof(np.full(N, 2.0), split); y["target"] = "yield"; tp.append(y)
            return r
    r = Extra().consumption_proof(XD, LO, HI, names=NAMES, parameters=PARAMS_OK)
    for d in DEAD:
        assert [p["verdict"] for p in r["params"] if p["param"] == d][0] == "UNPROVEN"
    assert not r["unreachable"]

if __name__ == "__main__":
    import traceback
    ok = True
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try: fn(); print(f"  PASS {name}")
            except Exception as e: ok = False; print(f"  FAIL {name}: {e}"); traceback.print_exc()
    print("\nALL PASS" if ok else "\n*** FAILURES ***"); sys.exit(0 if ok else 1)
