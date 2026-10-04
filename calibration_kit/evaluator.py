"""The shared model-agnostic EVALUATOR: parameter vector -> dag-valid metrics.

The ONE per-model integration surface. Reuses the self-improve KI's machinery:
applicator.write_param to set parameters, the KI run path (run_model) to run+score,
and objectives to turn metrics into losses. The optimizer only sees
Problem.evaluate(x) -> losses; everything model-specific lives here. evaluate()
NEVER raises into the optimizer — a failed/infeasible run returns +inf losses.
"""
from __future__ import annotations
import contextlib as _contextlib
import math
import time
import os
import numpy as np
import threading as _threading
from dataclasses import dataclass, field
from pathlib import Path
from .applicator import write_param, read_param
from .objectives import Objective


#: the ONLY authoritative explicit splits — exactly what holdout.py sets for its two comparison runs.
VALID_SPLITS = ("calibration", "holdout")
#: IN-PROCESS split authority (codex round-4). Provenance must NOT itself be an env value: a parent
#: carrying both KDT_CALIB_SPLIT=holdout AND an owner sentinel would be inherited and trusted, letting
#: commissioning/screen/DDS score the VALIDATION window. Authority is therefore a thread-local flag that
#: only the `split_authority()` context manager sets — it cannot cross a process boundary.
_AUTH = _threading.local()


def _authority_active() -> bool:
    return getattr(_AUTH, "active", False)


@_contextlib.contextmanager
def split_authority(split: str):
    """The ONLY legitimate way to direct a run's split (holdout validation, reference replay). Sets
    IN-PROCESS authority + the KDT_CALIB_SPLIT env the subprocess runner reads, and restores both."""
    prev_flag = _authority_active()
    prev_env = os.environ.get("KDT_CALIB_SPLIT")
    _AUTH.active = True
    os.environ["KDT_CALIB_SPLIT"] = split
    try:
        yield
    finally:
        _AUTH.active = prev_flag
        if prev_env is None:
            os.environ.pop("KDT_CALIB_SPLIT", None)
        else:
            os.environ["KDT_CALIB_SPLIT"] = prev_env


def split_is_authoritative(env_value) -> bool:
    """True only inside a `split_authority(...)` block AND for a valid value. An inherited env — even
    KDT_CALIB_SPLIT=holdout with any sentinel — is NOT authoritative across a process boundary."""
    return _authority_active() and (env_value or "").strip().lower() in VALID_SPLITS


def resolve_train_split(env_value) -> str:
    """Resolve the split a run must score. ONLY a holdout-validator-stamped split wins (provenance, not
    just a valid value); unset, blank, 'full', garbage, OR a stale inherited 'calibration'/'holdout'
    without the owner sentinel all fall back to "calibration". Trusting the value alone let a stale env
    make the screen/commissioning/DDS score the FULL record (or worse, the VALIDATION window) — exactly
    the contamination this contract exists to prevent."""
    if split_is_authoritative(env_value):
        return (env_value or "").strip().lower()
    return "calibration"


def decode_value(param: dict, xi, inv_fn) -> object:
    """Single source of truth for search-vector -> model-value decode (codex
    calib.py:198): inverse-transform, then type-quantize (int round / bool
    threshold). Used for BOTH scoring and the final best-apply so the workdir is
    never left at a raw float that was never actually evaluated."""
    v = inv_fn(xi)
    t = param.get("type", "continuous")
    if t == "integer":
        return int(round(v))
    if t == "bool":
        return bool(v > 0.5)
    return v



def _plain(v):
    """JSON-safe scalar for the history log (numpy types, inf/nan -> str)."""
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            v = v.item()
    except Exception:
        pass
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v

def split_echo(metrics):
    """The split a runner says it scored: metrics["__kdt__"]["split"] (a string), else None."""
    try:
        v = ((metrics or {}).get("__kdt__") or {}).get("split")
    except AttributeError:
        return None
    return str(v).strip().lower() if isinstance(v, str) and v.strip() else None


def log_runner_call(workdir, phase: str, i: int, wall_s: float, ok: bool, metrics=None, reason: str = "",
                    split=None):
    """One history record for a runner call made OUTSIDE the evaluator (the certification replay, the
    objective probe, the machine probe's lane runs), so it is counted and timed like every other model run
    (gap 2j). `split` = the split the kit asked for; the echo is read from `metrics`. Never raises."""
    try:
        import json as _json
        rec = {"i": int(i), "t": round(time.time(), 3), "phase": phase, "split": split, "eval_id": None,
               "x": {}, "losses": [], "ok": bool(ok), "cache_hit": False, "wall_s": round(float(wall_s), 4),
               "panel": {}, "outside_evaluator": True}
        if reason:
            rec["reason"] = reason
        if isinstance(metrics, dict) and metrics:
            rec["split_echo"] = split_echo(metrics)
        with open(Path(workdir) / "eval_history.jsonl", "a") as fh:
            fh.write(_json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


_PROV_COUNTS = ("calls", "echo_ok", "echo_missing", "echo_wrong")


def valid_prov_record(d) -> bool:
    """A saved per-phase split record (kdt_budget_plan.json) is usable only if its four counts are whole
    numbers >= 0 and its two index fields are lists — a hand-edited plan must never crash the report."""
    if not isinstance(d, dict):
        return False
    for k in _PROV_COUNTS:
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return False
    if d["calls"] != d["echo_ok"] + d["echo_missing"] + d["echo_wrong"]:     # the counts must add up
        return False
    return isinstance(d.get("missing_at", []), list) and isinstance(d.get("wrong_at", []), list)


def split_provenance(records, claim_phases=("pilot", "search"), saved=None, unchecked=None) -> dict:
    """Which calls echoed the split they were asked for (gap 2p). Per phase: calls with metrics, echo ok,
    echo missing, echo wrong (first few call indices) — every phase, the runs made outside the evaluator
    included. `calibration_only_supported` is True only if every call of `claim_phases` (the phases the
    search's claims rest on: pilot, search, and the objective probe when its payload set pilot decisions)
    that returned metrics echoed its requested split; False if any did not (False beats "not checked");
    None if a claimed phase could not be checked, or if no claimed call returned metrics.

    saved:     {phase: record} — counts saved in the budget plan by the call that ran that phase (a resume
               reuses that pilot and its decisions). They replace this call's records for the phase; this
               call's own records for it are kept under "<phase> (this call)".
    unchecked: {phase: reason} — a claimed phase whose calls cannot be judged (an older plan without the
               record); this call's own records for such a phase are
               kept under "<phase> (this call)" and do not decide the claim. Never raises."""
    try:
        return _split_provenance(records, claim_phases, saved or {}, unchecked or {})
    except Exception as e:                                   # a history helper never breaks a run
        return {"by_phase": {}, "claim_phases": list(claim_phases), "calibration_only_supported": None,
                "text": f"split provenance not computed ({type(e).__name__}: {e})"}


def _split_provenance(records, claim_phases, saved, unchecked) -> dict:
    by: dict = {}
    for r in records or []:
        if "split_echo" not in r:
            continue
        ph = str(r.get("phase"))
        d = by.setdefault(ph, {"calls": 0, "echo_ok": 0, "echo_missing": 0, "echo_wrong": 0,
                               "missing_at": [], "wrong_at": []})
        d["calls"] += 1
        want = r.get("split")
        got = r.get("split_echo")
        if got is None:
            d["echo_missing"] += 1
            if len(d["missing_at"]) < 10:
                d["missing_at"].append(r.get("i"))
        elif want is not None and got != str(want).lower():
            d["echo_wrong"] += 1
            if len(d["wrong_at"]) < 10:
                d["wrong_at"].append([r.get("i"), want, got])
        else:
            d["echo_ok"] += 1
    unchecked = dict(unchecked)
    for ph in unchecked:                     # a phase judged from the saved plan, not from this call
        if ph in by and ph != "search":
            by[f"{ph} (this call)"] = by.pop(ph)
    for ph, rec in saved.items():
        if ph in by:
            by[f"{ph} (this call)"] = by.pop(ph)
        if valid_prov_record(rec):
            by[ph] = dict(rec, source="the saved budget plan (the call that ran it)")
        else:
            unchecked[ph] = f"{ph.replace('_', ' ')} not checked (its saved split record is not usable)"
    rel = {p: by[p] for p in claim_phases if p in by}
    n = sum(d["calls"] for d in rel.values())
    bad = sum(d["echo_missing"] + d["echo_wrong"] for d in rel.values())
    counted = [p.replace("_", "-") if p == "objective_probe" else p for p in rel if rel[p]["calls"] > 0]
    names = (", ".join(counted[:-1]) + " and " + counted[-1]) if len(counted) > 1 else (counted[0] if counted else "")
    notes = [unchecked[p] for p in claim_phases if p in unchecked]
    if bad:                                    # a wrong or missing echo decides, whatever is unchecked
        sup, text = False, (f"{bad} of {n} {names} calls did not echo the requested split (missing or "
                            f"different) — they cannot support a calibration-only claim")
    elif notes:
        sup, text = None, "; ".join(notes) + (f"; {n} other {names} calls echoed their split" if n else "")
    elif n == 0:
        sup, text = None, "no pilot or search call returned metrics"
    else:
        sup, text = True, f"every {names} call echoed the split it was asked for ({n} calls)"
    return {"by_phase": by, "claim_phases": list(claim_phases), "calibration_only_supported": sup, "text": text}


def read_history(workdir) -> list[dict]:
    """Ordered evaluation records from <workdir>/eval_history.jsonl ([] when absent).
    Truncated/corrupt trailing lines are skipped — a log is never allowed to break a reader."""
    out = []
    try:
        with open(Path(workdir) / "eval_history.jsonl") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json as _json
                    rec = _json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):      # a truncated or `null` line is not a record
                    out.append(rec)
    except Exception:
        return []
    return out


def phase_counts(workdir, records=None) -> dict:
    """{phase: {"n", "ok", "cache_hits", "wall_s"}} — WHERE the evaluations went. A single
    total cannot separate search effort from commissioning, the consumption proof and the
    holdout re-runs, which is what made "how many evaluations did the search need" unanswerable.
    Runner calls made outside evaluate() — the certification replay, the objective probe, the machine
    probe's lane runs — are logged by log_runner_call (build step 7) and appear here as `certify`,
    `objective_probe` and `machine_probe`. The machine probe's lane runs overlap in time, so its wall_s
    here is the sum of the runs, not the elapsed time the ledger charges."""
    out = {}
    for r in (records if records is not None else read_history(workdir)):
        d = out.setdefault(str(r.get("phase")), {"n": 0, "ok": 0, "cache_hits": 0, "wall_s": 0.0})
        d["n"] += 1
        d["ok"] += 1 if r.get("ok") else 0
        d["cache_hits"] += 1 if r.get("cache_hit") else 0
        try:
            d["wall_s"] = round(d["wall_s"] + float(r.get("wall_s") or 0.0), 3)
        except Exception:
            pass
    return out


@dataclass
class Evaluator:
    ki_path: str
    workdir: str
    parameters: list[dict]
    objectives: list[Objective]
    transform_inv: dict
    run_model: callable
    constraints_ok: callable | None = None
    base_seed: int = 0
    injection_mode: str = "applicator"   # "applicator" (kit writes via addresses) | "runner" (calib_run.py injects)
    param_rtol: float = 1e-6
    expected_case_id: str | None = None   # if set, the runner's __kdt__.case_id MUST match (fail-closed on a
    #                                       DECLARED-but-mismatched case — the deterministic wrong-gauge guard)
    fixed_params: dict = field(default_factory=dict)  # inactive params frozen at default (staged rounds) —
    #                                                   merged into every candidate so constraints + the runner
    #                                                   payload see them, and they aren't silently unfrozen.
    _originals: dict = field(default_factory=dict)
    _eval_id: int = 0
    _verify_fail: int = 0
    _wrong_case: str | None = None        # set to the runner-declared case_id when it != expected_case_id
    _metrics_cache: dict = field(default_factory=dict)   # (param vector, split) -> metrics (resumability)
    _last_metrics: dict = field(default_factory=dict)     # split -> most-recent full metrics (for the correlation floor)
    # ── PHASE-TAGGED HISTORY (2026-09-27, convergence work) ────────────────────────────────
    # The eval cache is keyed by a HASH and holds only SUCCESSFUL evals, so it cannot answer
    # "where did the search converge?": no order guarantee, no phase, no parameter vector, no
    # failures. This adds an append-only ordered log next to it. It is a SIDE EFFECT ONLY —
    # every write is wrapped, so a logging fault can never change a loss or stop a run.
    # set via phase_as(); tags each eval with what it was for. An eval made OUTSIDE any phase block is
    # "untagged" — never silently counted as search effort (build step 7, gap 2j)
    phase: str = "untagged"
    _hist_n: int = 0
    _hist_file: object = None
    panel_extract: object = None          # callable(metrics) -> {var: {metric: value}}; set by calib.py
    search_seed: object = None            # the seed of the search running now (build step 6); logged per call

    def __post_init__(self):
        # Persistent EVAL CACHE (resumability — time is never a constraint): a long real-binary
        # calibration must survive interruption, so every COMPLETED eval is cached by (param
        # vector, split); a restart fast-forwards over cached evals instead of re-running the
        # model (a deterministic optimizer re-proposes the same vectors -> all hits until it
        # reaches where it stopped). Also dedups repeated candidates within a single run.
        import json as _json, re as _re, hashlib as _hl
        # SCOPE the cache by CASE + SETUP fingerprint so cached metrics are never reused for a
        # different case OR after the driver/contract changed (Codex): case flip (VIC tangnaihai
        # vs harbin) via KDT_CALIB_CASE_TAG; setup drift via a hash of calibration.yaml +
        # tools/calib_run.py contents — edit either and the fingerprint changes -> fresh cache,
        # stale metrics ignored. Absent tag -> single-case default (backward compatible).
        _tag = _re.sub(r"[^A-Za-z0-9]+", "_", os.environ.get("KDT_CALIB_CASE_TAG", "")).strip("_")
        _fp = _hl.sha1()
        # SPLIT-CONTRACT VERSION (codex 2026-07-19): the fingerprint must also namespace the
        # TRAIN-SPLIT semantics, not just the contract files. Before v2 the optimizer ran with
        # KDT_CALIB_SPLIT unset (= the runner's FULL record) while _cache_key() normalized a falsy
        # split to "calibration" — so pre-v2 FULL-period metrics sit under the very key v2 uses for
        # calibration-split metrics and would be replayed as if they were in-window. Bumping this
        # constant retires every pre-v2 cache file instead of silently reusing contaminated entries.
        _fp.update(b"split-contract-v2:train=calibration")
        for _rel in ("calibration.yaml", "tools/calib_run.py"):
            try:
                _fp.update(Path(self.ki_path, _rel).read_bytes())
            except Exception:
                _fp.update(b"\x00")
        # 2026-08-30 (System 1 round 2): a SITE contract (KDT_CALIB_CONTRACT) and a wrapper driver named in
        # runner.command are part of the setup too — a cache keyed only on the KI's own files replayed stale
        # metrics after the driver's scoring changed. Hash the contract file and every existing script path
        # in the runner command; also the declared split sets (they define what a split means).
        try:
            _c = os.environ.get("KDT_CALIB_CONTRACT")
            if _c and Path(_c).is_file():
                _fp.update(Path(_c).read_bytes())
                import yaml as _y
                for _tok in ((_y.safe_load(Path(_c).read_text()).get("runner") or {}).get("command") or []):
                    if isinstance(_tok, str) and _tok.startswith("/") and Path(_tok).is_file() and _tok.endswith(".py"):
                        _fp.update(Path(_tok).read_bytes())
            _fp.update(os.environ.get("KDT_CALIB_SPLIT_SETS", "").encode())
        except Exception:
            _fp.update(b"\x01")
        _setup = _fp.hexdigest()[:10]
        self._cache_file = Path(self.workdir) / (
            f"eval_metrics_cache_{_tag}_{_setup}.jsonl" if _tag else f"eval_metrics_cache_{_setup}.jsonl")
        self._hist_file = Path(self.workdir) / "eval_history.jsonl"
        try:
            if self._cache_file.is_file():
                for line in self._cache_file.read_text().splitlines():
                    try:
                        rec = _json.loads(line)
                        self._metrics_cache[rec["key"]] = rec["metrics"]
                    except Exception:
                        continue
        except Exception:
            pass
        # applicator mode: snapshot original values so restore_originals() can undo candidate writes if the
        # calibration fails. runner mode: no applicator addresses to snapshot — the runner owns injection in
        # a fresh candidate workdir, so there is nothing central to restore.
        if self.injection_mode != "applicator":
            return
        for p in self.parameters:
            try:
                self._originals[p["name"]] = read_param(p["address"], self.workdir)
            except Exception:
                self._originals[p["name"]] = None

    def restore_originals(self):
        if self.injection_mode != "applicator":
            return
        for p in self.parameters:
            v = self._originals.get(p["name"])
            if v is not None:
                try:
                    write_param(p["address"], v, self.workdir)
                except Exception:
                    pass

    def _verify_applied(self, named: dict, metrics: dict):
        """RUNNER mode: the params we requested MUST equal what calib_run.py reports it applied
        (`__kdt__.applied_params`), else the metrics describe a model that never actually changed.
        FAIL-CLOSED: EVERY param we handed the runner (active + any staged-frozen defaults) MUST
        be echoed and match — an incomplete echo could hide a misapplied frozen value, so a
        driver that omits a handed param is rejected (#5; the calib-dev prompt requires echoing
        all of KDT_CALIB_PARAMS). We allow the runner to echo EXTRA keys (ignored). In NON-staged
        mode fixed_params is empty, so `named` is exactly the optimized params. Rejects
        non-finite / clamp-to-0; math.isclose with a tiny rel_tol. (File-level proof; the
        did-the-run-consume-it proof is responsiveness_check.) Returns (ok, reason)."""
        meta = (metrics or {}).get("__kdt__") or {}
        applied = meta.get("applied_params")
        if not isinstance(applied, dict):
            return False, "runner did not echo __kdt__.applied_params"
        missing = set(named) - set(applied)              # every handed param must be echoed
        if missing:
            return False, (f"runner did not echo param(s) {sorted(missing)} it was handed "
                           "(echo ALL of KDT_CALIB_PARAMS in __kdt__.applied_params, incl. frozen)")
        for name, want in named.items():
            got = applied.get(name)
            try:
                fg, fw = float(got), float(want)
                if not (math.isfinite(fg) and math.isfinite(fw)):
                    return False, f"{name} non-finite (applied={got!r}, requested={want!r})"
                if fw != 0.0 and fg == 0.0:     # a non-zero request clamped to 0 is NOT "close" (Codex re-review)
                    return False, f"{name} clamped to 0 (requested {want})"
                if not math.isclose(fg, fw, rel_tol=self.param_rtol, abs_tol=0.0):
                    return False, f"{name} applied {got} != requested {want}"
            except (TypeError, ValueError):
                if str(got) != str(want):
                    return False, f"{name} applied {got!r} != requested {want!r}"
        return True, "ok"

    def _runner_gate(self, named, metrics) -> bool:
        """Runner-mode FAIL-CLOSED gate applied to EVERY trusted metrics payload — a fresh run AND a
        CACHE HIT (codex 2026-07-18: a cache hit must not bypass these checks, else a legacy/wrong-case
        cached payload is reused). (a) every handed param must be echoed in __kdt__.applied_params and
        match; (b) when a case is pinned, __kdt__.case_id MUST be present and equal. Records the failure
        (_verify_fail / _wrong_case) and returns False when the payload can't be trusted."""
        if self.injection_mode != "runner":
            return True
        ok, _reason = self._verify_applied(named, metrics)
        if not ok:
            self._verify_fail += 1
            return False
        if self.expected_case_id:
            _declared = ((metrics or {}).get("__kdt__") or {}).get("case_id")
            if _declared is None:
                self._wrong_case = "<undeclared>"      # pinned case but runner omitted case_id
                return False
            if str(_declared) != str(self.expected_case_id):
                self._wrong_case = str(_declared)
                return False
        return True

    def _named(self, x):
        # Typed decode via the shared helper so scoring and best-apply agree. Frozen inactive params
        # (staged rounds) are merged in FIRST so constraints that reference them and the runner payload
        # both see them; the active candidate values override.
        named = dict(self.fixed_params)
        named.update({p["name"]: decode_value(p, xi, self.transform_inv.get(p["name"], lambda v: v))
                      for p, xi in zip(self.parameters, x)})
        return named

    def _cache_key(self, named, split):
        """Stable key for (param vector, split). 10 sig figs so the round-tripped param
        value hashes identically across runs; split matters (calibration vs holdout metrics)."""
        import hashlib, json as _json
        payload = {"p": {k: (f"{float(v):.10g}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                             else str(v)) for k, v in sorted(named.items())},
                   "split": split or "calibration"}
        return hashlib.sha1(_json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @_contextlib.contextmanager
    def phase_as(self, name: str):
        """Tag every evaluation inside the block with `name` (pilot/commission/screen/search/holdout/...)."""
        prev, self.phase = self.phase, name
        try:
            yield
        finally:
            self.phase = prev

    def _hist(self, *, named=None, losses=None, ok=False, cache_hit=False, wall_s=0.0,
              split=None, metrics=None, reason=""):
        """Append ONE ordered record per evaluate() call. Never raises, never blocks a run."""
        self._hist_n += 1
        try:
            import json as _json
            panel = {}
            if metrics and callable(self.panel_extract):
                try:
                    panel = self.panel_extract(metrics) or {}
                except Exception:
                    panel = {}
            rec = {"i": self._hist_n - 1, "t": round(time.time(), 3), "phase": self.phase,
                   "split": split, "eval_id": self._eval_id,
                   "x": {k: _plain(v) for k, v in (named or {}).items()},
                   "losses": [_plain(l) for l in (losses or [])], "ok": bool(ok),
                   "cache_hit": bool(cache_hit), "wall_s": round(float(wall_s), 4),
                   "panel": panel}
            if reason:
                rec["reason"] = reason
            if metrics:
                # SPLIT PROVENANCE (design §1 "Data within a search", gap 2p): the split the runner says
                # it scored (`__kdt__.split`), next to the split the kit asked for
                rec["split_echo"] = split_echo(metrics)
            if self.search_seed is not None:
                rec["seed"] = self.search_seed         # which seed's search made this call
            with open(self._hist_file, "a") as fh:
                fh.write(_json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def _store_cache(self, key, metrics):
        self._metrics_cache[key] = metrics
        try:
            import json as _json
            with open(self._cache_file, "a") as fh:
                fh.write(_json.dumps({"key": key, "metrics": metrics}, default=str) + "\n")
        except Exception:
            pass

    def evaluate(self, x) -> list[float]:
        """Apply x -> run -> score. Returns one loss per objective (minimize).
        +inf on infeasible/failed (never raises into the optimizer)."""
        _t0 = time.perf_counter()
        _inf = [float("inf")] * len(self.objectives)
        named = None
        _split = None
        try:
            # the split this call is FOR, decided first, so every record carries it — also a call rejected
            # by a constraint or one whose runner raised (Opus 8b/9 r2 #1: the replay needs it on every call)
            _split = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
            named = self._named(x)
            # hard feasibility BEFORE any run (cheap) — codex C5
            if self.constraints_ok and not self.constraints_ok(named):
                self._hist(named=named, losses=_inf, ok=False, split=_split, wall_s=time.perf_counter() - _t0,
                           reason="infeasible")
                return _inf
            # RESUMABILITY: reuse a completed real-model eval for this (param vector, split).
            # A restart of a long calibration fast-forwards over cached evals; also dedups
            # repeated candidates within a run. Only SUCCESSFUL evals are cached (below).
            # TRAIN ON THE CALIBRATION SPLIT (2026-07-19). holdout.py sets KDT_CALIB_SPLIT explicitly for
            # its two comparison runs; NOTHING set it for the screen/DDS, so with it UNSET a runner scored
            # its FULL record — the optimizer was fitting params on data that INCLUDES the holdout years,
            # making the holdout gate contaminated rather than out-of-sample. Default (only when unset) to
            # "calibration" so training and validation are genuinely disjoint.
            # Only holdout.py's explicit "calibration"/"holdout" is authoritative. Unset, blank, or an
            # inherited "full" resolves to "calibration" and WE own the env for this run (codex round-2).
            _env_raw = os.environ.get("KDT_CALIB_SPLIT")
            _split = resolve_train_split(_env_raw)
            _we_own_split = not split_is_authoritative(_env_raw)
            _ck = self._cache_key(named, _split)
            # round-2 (kimi): the consumption gate needs a LIVE default run to validate a cached
            # proof receipt against the model as it is NOW. _metrics_cache is rehydrated from disk
            # at construction, so without this flag that "live" run could be a replay from a
            # previous session and the guard would pass a stale receipt.
            _hit = None if getattr(self, "_force_live", False) else self._metrics_cache.get(_ck)
            if _hit is not None:
                # a cache hit is a trusted metrics payload -> it MUST pass the same runner-mode gate
                # as a fresh run (applied-params echo + pinned-case declaration). A legacy/invalid
                # cached entry fails closed (+inf) instead of silently bypassing the guard.
                if not self._runner_gate(named, _hit):
                    self._hist(named=named, losses=_inf, ok=False, cache_hit=True, split=_split,
                               wall_s=time.perf_counter() - _t0, reason="runner_gate_failed_cached")
                    return _inf
                self._last_metrics[_split] = _hit
                _cl = [o.loss(_hit) for o in self.objectives]
                self._hist(named=named, losses=_cl, ok=all(math.isfinite(l) for l in _cl), cache_hit=True,
                           split=_split, metrics=_hit, wall_s=time.perf_counter() - _t0)
                return _cl
            if self.injection_mode == "runner":
                # DELEGATED injection: hand the candidate vector to calib_run.py (via a params file +
                # env), which injects them the model's own way and echoes __kdt__.applied_params.
                import json
                pf = Path(self.workdir) / "kdt_params.json"
                pf.write_text(json.dumps(named, default=str))
                os.environ["KDT_CALIB_PARAMS"] = str(pf)
            else:
                for p in self.parameters:
                    write_param(p["address"], named[p["name"]], self.workdir)
            # Reproducibility: expose a per-eval seed + id so STOCHASTIC runners/models
            # can be deterministic (codex evaluator.py:31). Subprocess runners inherit
            # the env; python/detached runners can read it.
            # carry the resolved split into the runner subprocess (it scores FULL when unset/blank/'full')
            if _we_own_split:
                os.environ["KDT_CALIB_SPLIT"] = _split
            self._eval_id += 1
            os.environ["KDT_CALIB_EVAL_ID"] = str(self._eval_id)
            os.environ["KDT_CALIB_SEED"] = str(self.base_seed)
            try:
                metrics = self.run_model()
            finally:
                os.environ.pop("KDT_CALIB_PARAMS", None)   # never leak this eval's params into a later run
                if _we_own_split:                          # restore EXACTLY what was there (blank stays
                    if _env_raw is None:                   # blank, 'full' stays 'full')
                        os.environ.pop("KDT_CALIB_SPLIT", None)
                    else:
                        os.environ["KDT_CALIB_SPLIT"] = _env_raw
            if not metrics:
                self._hist(named=named, losses=_inf, ok=False, split=_split,
                           wall_s=time.perf_counter() - _t0, reason="no_metrics")
                return _inf
            # FAIL-CLOSED runner-mode gate (applied-params echo + pinned-case declaration), applied
            # identically here and on cache hits above via _runner_gate.
            if not self._runner_gate(named, metrics):
                self._hist(named=named, losses=_inf, ok=False, split=_split, metrics=metrics,
                           wall_s=time.perf_counter() - _t0, reason="runner_gate_failed")
                return _inf
            self._last_metrics[_split] = metrics   # expose full metrics (incl. r) for the correlation floor
            # CACHE the completed real-model eval (resumability + intra-run dedup) — but ONLY when
            # every objective loss is FINITE (Codex): a truthy-but-non-finite metrics payload
            # (e.g. nan/missing key) would otherwise freeze a transient bad result into the cache
            # and be replayed forever on restart. A failed/non-finite eval must be retried, never cached.
            losses = [o.loss(metrics) for o in self.objectives]
            if all(math.isfinite(l) for l in losses):
                self._store_cache(_ck, metrics)
            self._hist(named=named, losses=losses, ok=all(math.isfinite(l) for l in losses),
                       split=_split, metrics=metrics, wall_s=time.perf_counter() - _t0)
            return losses
        except Exception:
            self._hist(named=named, losses=_inf, ok=False, split=_split, wall_s=time.perf_counter() - _t0,
                       reason="exception")
            return _inf

    # ------------------------------------------------------------------------------------------
    # CONSUMPTION PROOF v2 (2026-09-14) — the calibration-kit gap fix, plan v2, owner-approved.
    #
    # The gap: before spending a search the kit asked "does ANY knob move ANY score?" (the
    # responsiveness_check below). Eight working WRF-Hydro knobs hid three that wrote into a file
    # the model never opens. The check must ask "does EVERY knob move the OUTPUT?"
    #
    # Why v1 (rejected by codex 2026-09-10) was wrong: it compared OBJECTIVES, which are rounded
    # (NSE/KGE/r to 4 dp). A real weak knob can leave a rounded score untouched, so v1 would have
    # failed working setups. v2 compares the RAW scored series the runner emits under
    # __kdt__.target_proofs (kdt-target-proof/1) — the thing itself, not a summary of it.
    #
    # Owner rule: UNREACHABLE STOPS the fit. No auto-repair, no dropping the knob. The KI is the
    # owner's to fix. Everything else continues with a warning.
    # ------------------------------------------------------------------------------------------
    PROOF_SCHEMA = "kdt-target-proof/1"

    @staticmethod
    def _proof_valid(t):
        """A proof record is usable only if it is internally consistent (round-6 review): schema,
        n == len(values) == len(index), recomputed hashes match, every value finite, target named.
        A corrupt or partial record must never be compared — it becomes UNPROVEN, not a verdict."""
        import hashlib
        try:
            if not isinstance(t, dict) or t.get("schema") != Evaluator.PROOF_SCHEMA or not t.get("target"):
                return False
            vals = t.get("values"); idx = t.get("index")
            if not isinstance(vals, list) or not isinstance(idx, list) or not vals:
                return False
            if int(t.get("n", -1)) != len(vals) or len(idx) != len(vals):
                return False
            v = np.asarray(vals, dtype=np.float64)
            if not np.all(np.isfinite(v)):
                return False
            if t.get("values_hash") != hashlib.sha1(v.tobytes()).hexdigest():
                return False
            if t.get("index_hash") != hashlib.sha1("\n".join(str(x) for x in idx).encode()).hexdigest():
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _proofs_of(metrics):
        """The runner's VALID raw target proofs, or None if this driver has not been upgraded.
        Duplicate target names make the whole set unusable (which one would be compared?)."""
        tp = ((metrics or {}).get("__kdt__") or {}).get("target_proofs")
        if not tp or not isinstance(tp, list):
            return None
        # round-2 (codex): if ANY record is invalid the whole set is unusable. Dropping only the
        # bad one and keeping a valid sibling let a run be judged on the sibling alone — if the
        # sibling was identical and the dropped target was the one that moved, that is a false
        # UNREACHABLE at the most expensive action the kit can take.
        if not all(Evaluator._proof_valid(t) for t in tp):
            return None
        names = [t["target"] for t in tp]
        if len(set(names)) != len(names):
            return None
        return list(tp)

    @staticmethod
    def _dormancy(pspec, ki_path=None):
        """A STRUCTURED dormancy declaration, or None. Free text is refused — a sentence must not be
        able to wave through a wrong address (codex, plan v2 #4). Required: kind + condition +
        evidence, and (round-6, both seats) the evidence must RESOLVE TO AN EXISTING FILE — absolute,
        or relative to the KI. DORMANT/WINDOW_MASKED are the verdicts that bypass the owner's STOP,
        so a typo or placeholder string must fall through to UNREACHABLE, not to a free pass."""
        d = (pspec or {}).get("dormant_when")
        if not isinstance(d, dict):
            return None
        kind = str(d.get("kind") or "")
        if kind not in ("process_inactive", "window_erased"):
            return None
        ev = str(d.get("evidence") or "").strip()
        if not ev or not str(d.get("condition") or "").strip():
            return None
        # round-3 (codex): a RELATIVE path resolves against the KI only — never the process cwd,
        # where an unrelated file of the same name could acquit a dead address.
        cands = [Path(ev)] if Path(ev).is_absolute() else ([Path(ki_path) / ev] if ki_path else [])
        resolved = None
        for c in cands:
            try:
                if c.is_file() and c.stat().st_size > 0:      # round-2: a directory or an empty
                    resolved = c; break                        # placeholder must not count
            except OSError:
                continue
        if resolved is None:
            return None            # declaration refused: no non-empty file at that path
        if resolved.suffix.lower() == ".json":
            try:
                import json as _json; _json.loads(resolved.read_text())
            except Exception:
                return None        # a JSON receipt that does not parse is not evidence
        return {"kind": kind, "condition": d["condition"], "evidence": str(resolved),
                "declared_by": d.get("declared_by")}

    def consumption_proof(self, *a, **k):
        """Phase-tagging wrapper (2026-09-27): every eval this probe spends is tagged
        `consume_proof` in eval_history.jsonl, so it is never counted as search effort."""
        with self.phase_as("consume_proof"):
            return self._consumption_proof(*a, **k)

    def _consumption_proof(self, x_default, lower, upper, names=None, parameters=None,
                          raw_abs_min=None, raw_rel_min: float = 1e-3, obj_rel_min: float = 1e-6,
                          ki_path=None):
        """Per parameter, one at a time, displaced to the far bound: does the RAW scored output move?

        Verdicts (plan v2, six + UNPROVEN):
          ALIVE            raw series moved materially (>= raw_abs_min or >= raw_rel_min of the
                           default series' max) AND the objectives moved
          INSENSITIVE      raw series moved, but below the materiality threshold
          OBJECTIVE_MASKED raw series moved materially, objectives (full precision) did NOT
          DORMANT          raw series bitwise identical, but the contract DECLARES this knob may be
                           inactive here (structured dormant_when, kind=process_inactive)
          WINDOW_MASKED    raw series bitwise identical, contract declares kind=window_erased
                           (consumed, but erased before the scored window — e.g. a spin-up)
          UNREACHABLE      applied-parameter echo OK AND raw series bitwise identical AND no
                           structured dormancy declared -> the value never reached the model. STOP.
          UNPROVEN         could not be displaced (zero-width range) or far bound infeasible, or
                           the runner emitted no proof / proofs with mismatched index -> NOT proven
        Returns a dict; `ok` is False iff any UNREACHABLE. `ok` is None when nothing could be proven.
        """
        params = parameters or []
        pspec = {p.get("name"): p for p in params if isinstance(p, dict)}
        l0 = self.evaluate(x_default)
        if not all(math.isfinite(v) for v in l0):      # round-6: isfinite, not isinf — a NaN loss
            return {"ok": None, "params": [], "summary": "default vector infeasible or non-finite — nothing proven",
                    "schema": self.PROOF_SCHEMA}     # must never read as "no movement"
        split0 = resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))
        m0 = self._last_metrics.get(split0) or {}
        p0 = self._proofs_of(m0)
        if p0 is None:
            return {"ok": None, "params": [], "schema": self.PROOF_SCHEMA,
                    "summary": ("runner emits no __kdt__.target_proofs (kdt-target-proof/1) — "
                                "consumption cannot be proven on rounded objectives; upgrade the driver")}
        base = {t["target"]: t for t in p0}
        out = []
        for i in range(len(x_default)):
            nm = names[i] if names and i < len(names) else f"param[{i}]"
            mid = (lower[i] + upper[i]) / 2.0
            xp = list(x_default)
            xp[i] = upper[i] if x_default[i] <= mid else lower[i]
            row = {"param": nm, "displaced_to": xp[i], "targets": {}}
            if xp[i] == x_default[i]:
                row.update(verdict="UNPROVEN", why="zero-width range — cannot be displaced"); out.append(row); continue
            li = self.evaluate(xp)
            if not all(math.isfinite(v) for v in li):
                row.update(verdict="UNPROVEN", why="far bound infeasible, run failed, or non-finite loss"); out.append(row); continue
            # applied echo: evaluate() returns +inf if _runner_gate rejects the echo, so finite
            # losses here mean the runner CONFIRMED it applied this value.
            pi = self._proofs_of(self._last_metrics.get(split0) or {})
            if pi is None:
                row.update(verdict="UNPROVEN", why="displaced run emitted no target proof"); out.append(row); continue
            moved_any = False; material_any = False; mismatch = False
            # round-6 (codex): iterate the DEFAULT run's targets, so a target missing from the
            # displaced run is a mismatch too — not silently ignored.
            disp = {t["target"]: t for t in pi}
            if set(disp) != set(base):        # round-3 (codex): target sets must match in BOTH
                mismatch = True               # directions; an extra target only in the displaced
            for tname, b in base.items():     # run is a mismatch too
                t = disp.get(tname)
                if t is None or b.get("index_hash") != t.get("index_hash") or b.get("n") != t.get("n"):
                    mismatch = True; continue
                v0 = np.asarray(b["values"], dtype=np.float64); v1 = np.asarray(t["values"], dtype=np.float64)
                d = np.abs(v1 - v0); dmax = float(d.max()) if d.size else 0.0
                scale = float(np.max(np.abs(v0))) if v0.size else 0.0
                thr = max(float(raw_abs_min) if raw_abs_min is not None else 0.0, raw_rel_min * scale)
                row["targets"][t["target"]] = {"raw_max_abs_delta": dmax, "raw_scale": scale,
                                               "raw_rel_delta": (dmax / scale if scale else None),
                                               "bitwise_identical": bool(dmax == 0.0),
                                               "material": bool(dmax >= thr and dmax > 0.0)}
                moved_any |= (dmax > 0.0); material_any |= (dmax >= thr and dmax > 0.0)
            if mismatch and not moved_any:
                # round-6 (codex): a partial mismatch must NEVER produce UNREACHABLE. If any comparable
                # target moved, the parameter is acquitted; if none moved and some target could not
                # be compared, we simply do not know.
                row.update(verdict="UNPROVEN", why="one or more target proofs missing or index-mismatched between default and displaced runs; no comparable target moved"); out.append(row); continue
            oscale = max(1.0, max(abs(v) for v in l0 + li))
            omove = max(abs(a - b) for a, b in zip(l0, li))
            row["objective_max_delta"] = omove
            dorm = self._dormancy(pspec.get(nm), ki_path=ki_path)
            if not moved_any:
                if dorm and dorm["kind"] == "window_erased":
                    row.update(verdict="WINDOW_MASKED", dormancy=dorm)
                elif dorm:
                    row.update(verdict="DORMANT", dormancy=dorm)
                else:
                    row.update(verdict="UNREACHABLE", why=(
                        "the runner confirmed it applied this value and the RAW scored output is "
                        "bitwise identical to the default run. The value never reached the model. "
                        "Audit the ADDRESS this parameter is written to (which file/table does this "
                        "build actually read? a build flag can switch it). Not a modelling question "
                        "— a KI defect. The fit will not start until it is fixed."))
            elif material_any and omove > obj_rel_min * oscale:
                row["verdict"] = "ALIVE"
            elif material_any:
                row.update(verdict="OBJECTIVE_MASKED", why="raw output moved materially; the objectives did not — the metric cannot see this parameter")
            else:
                row.update(verdict="INSENSITIVE", why="raw output moved, below the materiality threshold — a weak parameter, a modelling judgement")
            out.append(row)
        by = {}
        for r in out:
            by.setdefault(r["verdict"], []).append(r["param"])
        unreachable = by.get("UNREACHABLE", [])
        return {"ok": (len(unreachable) == 0), "schema": self.PROOF_SCHEMA, "params": out,
                "unreachable": unreachable, "by_verdict": by,
                "default_values_hash": {t["target"]: t.get("values_hash") for t in p0},
                "summary": ", ".join(f"{k}={len(v)}" for k, v in sorted(by.items()))}

    def responsiveness_check(self, *a, **k):
        """Phase-tagging wrapper (2026-09-27): commissioning evals are tagged `commission`."""
        with self.phase_as("commission"):
            return self._responsiveness_check(*a, **k)

    def _responsiveness_check(self, x_default, lower, upper, min_rel_move: float = 1e-6):
        """One-time COMMISSIONING proof (esp. runner mode): round-trip proves the value was WRITTEN;
        this proves the scored run actually CONSUMED it. Perturbs ONE param at a time (a single all-params
        corner false-fails on compensation/saturation/infeasible corners) to an interior point; the model is
        RESPONSIVE if ANY finite perturbation materially moves ANY objective. Returns
        (responsive: bool|None, detail): None = inconclusive (don't hard-fail), not unresponsive."""
        l0 = self.evaluate(x_default)
        if any(math.isinf(v) for v in l0):
            return None, "default vector infeasible — inconclusive"
        n_moved = n_tested = 0
        for i in range(len(x_default)):
            mid = (lower[i] + upper[i]) / 2.0
            xp = list(x_default)
            # perturb to the FAR bound — guarantees a decoded-value change even for bool / small-int params
            # (a half-span step can round back to the same int or bool → false 'unresponsive'; Codex re-review).
            xp[i] = upper[i] if x_default[i] <= mid else lower[i]
            if xp[i] == x_default[i]:            # degenerate (zero-width range) — can't perturb this param
                continue
            li = self.evaluate(xp)
            if any(math.isinf(v) for v in li):
                continue                                   # bad corner for this param → skip (inconclusive)
            n_tested += 1
            scale = max(1.0, max(abs(v) for v in l0 + li))
            if max(abs(a - b) for a, b in zip(l0, li)) > min_rel_move * scale:
                n_moved += 1
        if n_tested == 0:
            return None, "all single-param perturbations infeasible — inconclusive"
        return (n_moved > 0), f"{n_moved}/{n_tested} params moved an objective"
