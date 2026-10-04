"""Calibration kit — main entry. Sibling module to the self-improve KI: it tunes a
correct-running model's PARAMETERS to fit its dag-defined obs. (Data assimilation is
a separate sibling module — sequential state estimation, OpenDA-based.)

Flow: load calibration.yaml + dag.yaml -> objectives from gate-valid metric_families
-> PREFLIGHT every parameter address (exists + round-trips) -> build evaluator
(apply->run->score) -> pick backend by cost_class/objective structure -> optimize
-> apply best params back -> (holdout validation: to build) -> report.
"""
from __future__ import annotations
import ast
import math
import time
import operator
import os
import json
import tempfile
import shutil
from pathlib import Path


def _load_contract(ki_path):
    import yaml
    # 2026-08-30 (System 1): a SITE contract may live outside the KI (models/<model>/calibration_<site>.yaml);
    # KDT_CALIB_CONTRACT names it. Default unchanged: <ki>/calibration.yaml.
    f = Path(os.environ["KDT_CALIB_CONTRACT"]) if os.environ.get("KDT_CALIB_CONTRACT") else Path(ki_path) / "calibration.yaml"
    if not f.is_file():
        raise FileNotFoundError(f"no calibration.yaml at {f} — model has no calibration "
                                f"contract yet (run the calibration-contract campaign)")
    return yaml.safe_load(f.read_text())


def _validate_transforms(params):
    """Build name -> (to_search, from_search) and VALIDATE transform domains
    against the declared range (codex calib.py:33). Returns (fwd, inv) or raises
    ValueError with a precise message — an invalid range must abort, not crash mid-run."""
    fwd, inv = {}, {}
    for p in params:
        n = p["name"]; t = p.get("transform", "identity")
        ptype = p.get("type", "continuous")
        if ptype == "categorical":
            # categorical needs an explicit index<->category encoder/sampler that the
            # current continuous Problem doesn't model — fail fast (codex spotpy:50).
            raise ValueError(f"param {n}: categorical type not yet supported "
                             f"(continuous/integer/bool only)")
        lo, hi = float(p["range"][0]), float(p["range"][1])
        if not (lo < hi):
            raise ValueError(f"param {n}: range lower {lo} must be < upper {hi}")
        if t == "log":
            if lo <= 0:
                raise ValueError(f"param {n}: log transform requires range > 0 (got {lo})")
            fwd[n] = math.log; inv[n] = math.exp
        elif t == "logit":
            if not (0 < lo and hi < 1):
                raise ValueError(f"param {n}: logit transform requires range in (0,1) "
                                 f"(got [{lo},{hi}])")
            fwd[n] = lambda v: math.log(v / (1 - v))
            inv[n] = lambda z: 1 / (1 + math.exp(-z))
        elif t in (None, "identity"):
            fwd[n] = lambda v: v; inv[n] = lambda v: v
        else:
            raise ValueError(f"param {n}: unknown transform {t!r}")
    return fwd, inv


def _preflight_addresses(params, workdir):
    """Enforce schema C1 (address file exists) + C7 (write/read round-trips) on a
    SCRATCH COPY before optimizing (codex calib.py:72). A lossy or mis-targeted
    writer would otherwise silently calibrate the wrong field for every eval.
    Returns a list of problems (empty == ok)."""
    from .applicator import verify_roundtrip
    problems = []
    scratch = Path(tempfile.mkdtemp(prefix="calib_preflight_"))
    try:
        from .applicator import _resolve
        # mirror just the files the addresses touch, resolving the REAL target the
        # SAME way the applicator does (absolute / explicit root / workdir), then
        # round-trip a SCRATCH-RELATIVE copy of the address so preflight can never
        # mutate the live input (codex calib.py:76).
        for p in params:
            addr = p.get("address") or {}
            f = addr.get("file")
            if not f:
                problems.append(f"{p['name']}: address has no 'file'"); continue
            src = _resolve(addr, workdir)            # exact target the run would edit
            if not src.is_file():
                problems.append(f"{p['name']}: address file missing on disk: {src}"); continue
            rel = f"{p['name']}__{Path(src).name}"   # flat unique name in scratch
            dst = scratch / rel
            shutil.copy2(src, dst)
            scratch_addr = dict(addr)                 # rewrite to point at the scratch copy
            scratch_addr["file"] = rel
            scratch_addr.pop("root", None)
            lo, hi = float(p["range"][0]), float(p["range"][1])
            probe = (lo + hi) / 2.0
            try:
                if not verify_roundtrip(scratch_addr, probe, str(scratch)):
                    problems.append(f"{p['name']}: address write/read did NOT round-trip "
                                    f"(would corrupt the search): {addr}")
            except Exception as e:
                problems.append(f"{p['name']}: address not usable ({type(e).__name__}: {e})")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return problems


# Map a DETERMINING metric to the ONE metric_family it natively grades. A determining
# metric overrides ONLY its own family (never a sibling), so we can't clobber another
# family's native metric (e.g. spatial's `csi` or ranking's `spearman`) with a temporal
# `r` — Codex review. objectives_from_dag only applies an override for a family that is
# actually present for the (var, obs_shape), so a metric aimed at a family the case
# doesn't expose is simply a no-op.
_METRIC_NATIVE_FAMILY = {
    "r": "temporal_pattern_match", "nse": "temporal_pattern_match",
    "kge": "temporal_pattern_match", "r2": "temporal_pattern_match",
    "nse_log": "temporal_pattern_match", "rho": "temporal_pattern_match",
    "pearson_r": "temporal_pattern_match",
    "csi": "spatial_pattern_match",
    "spearman": "ranking_quality",
    "pbias": "magnitude_accuracy", "bias": "magnitude_accuracy",
}


def _effective_metric_overrides(contract, determining_metric):
    """The per-obs DETERMINING metric (the gate/model_obs_map headline, threaded by
    readiness) is what the FIELD grades — e.g. groundwater & soil-moisture headline is
    `r`, NOT the family-default `nse`. If the contract didn't declare `metric_overrides`,
    synthesize a SINGLE-family override from the determining metric so BOTH the objective
    and the triage's per-target fit verdict optimize/judge the headline instead of a
    silently-different default (Bug #1: determining_metric was dropped before the engine).

    Explicit contract overrides always win — including an explicit empty `{}` (author's
    intent = keep family defaults), so we test key PRESENCE, not truthiness (Codex review).
    The override is keyed to the metric's OWN native family only (never a sibling), and
    the loss fn for that family is unchanged (only metric_key differs), so it's sound.
    An unknown metric is NOT guessed — keep the family defaults."""
    if "metric_overrides" in contract:
        return contract.get("metric_overrides")          # explicit (incl. {} / null) wins
    dm = str(determining_metric or "").lower()
    fam = _METRIC_NATIVE_FAMILY.get(dm)
    return {fam: dm} if fam else None


def _load_reference_run(ki_path):
    """Durable reference of the VALIDATED KI run (calib/reference_run.json), captured from
    self_improve_runs — NOT the ephemeral real_case_result.json (which later runs overwrite).
    Returns the dict or None (then certification is skipped — backward compatible)."""
    import json
    f = Path(ki_path) / "calib" / "reference_run.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


from .evaluator import split_authority as _ev_split_authority
from .evaluator import phase_counts, read_history


def resolve_reference_run(contract, ki_path, expected_case_id=None):
    """Which validated reference the module certification replays (System 1, 2026-08-29).
    Order: the contract's own `runner.reference_run` (a SITE reference) first, else the KI's
    calib/reference_run.json (the KI's validation case). A reference whose case_id differs from
    the contract's target case is REFUSED explicitly — the KI-level reference (e.g. VIC =
    SITE:tangnaihai) is not something a case-bound driver can or should reproduce, and a silent
    "no metrics" failure hid that. Returns (reference_or_None, refusal_dict_or_None)."""
    import json as _json
    ref_path = (contract.get("runner", {}) or {}).get("reference_run")
    if ref_path:
        try:
            ref = _json.loads(Path(ref_path).read_text())
        except Exception as e:
            return None, {"status": "runner_uncertified",
                          "reason": f"runner.reference_run {ref_path!r} unreadable: {e}"}
    else:
        ref = _load_reference_run(ki_path)
    if ref and expected_case_id and not ref.get("case_id"):
        # codex R1 (2026-08-29): a reference that names no case cannot be matched to the target —
        # refuse it rather than let a legacy wrong-case reference through the mismatch check.
        return None, {"status": "runner_uncertified",
                      "reason": f"reference run names no case_id; the contract targets {expected_case_id!r} — "
                                f"add case_id to the reference (a case-less reference is never trusted)",
                      "validated_run_id": ref.get("validated_run_id")}
    if ref and expected_case_id and ref.get("case_id") and str(ref["case_id"]) != str(expected_case_id):
        return None, {"status": "runner_uncertified",
                      "reason": f"reference run is for case {ref['case_id']!r} but the contract targets "
                                f"{expected_case_id!r} — give the contract its own validated reference "
                                f"(runner.reference_run); certification is never skipped for a case mismatch",
                      "validated_run_id": ref.get("validated_run_id")}
    return ref, None


def certify_runner(contract, run_model, reference, workdir, expected_case_id=None):
    """MODULE-CERTIFICATION gate (2026-07-10, GPT-5.5 design). The framework auto-generates a
    calibration function for a validated, working process-based KI; this proves the generated
    function actually EXECUTES THE KI CORRECTLY by REPRODUCING the validated reference run —
    BEFORE any objective probe / pilot / triage. It injects the reference params, runs
    the module once at the reference split, and requires a finite headline metric within
    tolerance of the validated value (+ run-signature match when both sides report it). A module
    that omits prepare/routing/staging, or drifts from the validated execution, produces None or
    a wrong metric -> NOT certified, stopped CHEAPLY (no screen, no nested repair agent).
    Returns {certified: bool, reason, observed}."""
    import json as _json
    params = reference.get("params") or {}
    headline = str(reference.get("headline_metric") or "nse").lower()
    ref_val = (reference.get("metrics") or {}).get(headline, reference.get("headline_value"))
    tol = reference.get("tolerance") or {}
    split = reference.get("reference_split")
    ref_sig = reference.get("signature") or {}
    inj_mode = (contract.get("injection", {}) or {}).get("mode", "applicator")

    _auth_cm = None                      # in-process split authority for the reference replay
    prev_params = os.environ.get("KDT_CALIB_PARAMS")
    prev_split = os.environ.get("KDT_CALIB_SPLIT")
    try:
        # INJECT the reference param state — FAIL-CLOSED (Codex): certification is worthless if
        # we didn't actually inject the validated params, so ANY setup/write failure (bad address,
        # mkdir/write error, applicator import, malformed contract) => NOT certified, never a
        # falsely-passing run on stale/partial params.
        try:
            if inj_mode == "runner":
                pf = Path(workdir) / "kdt_certify_params.json"
                pf.parent.mkdir(parents=True, exist_ok=True)
                pf.write_text(_json.dumps(params, default=str))
                os.environ["KDT_CALIB_PARAMS"] = str(pf)
            else:
                from .applicator import write_param
                for p in contract.get("parameters", []):
                    if isinstance(p, dict) and p.get("name") in params:
                        write_param(p["address"], params[p["name"]], workdir)   # raises => fail-closed
            # reference replay is a LEGITIMATE split director -> declare in-process authority
            # (codex round-4) instead of raw-setting the env outside holdout._eval_split.
            # NOTE: entered here, EXITED in the outer finally — an unbalanced __enter__ would leave
            # the thread-local authority flag ON and make later TRAINING evals trust a stale env.
            if split and split != "full":
                _auth_cm = _ev_split_authority(split)
                _auth_cm.__enter__()
            else:
                os.environ.pop("KDT_CALIB_SPLIT", None)
        except Exception as e:
            return {"certified": False, "observed": None,
                    "reason": f"could not inject the validated reference param state "
                              f"(cannot certify without it): {e}"}
        try:
            metrics = run_model() or {}
        except Exception as e:
            return {"certified": False, "observed": None,
                    "reason": f"module raised while running at the reference params: {e}"}
    finally:
        if _auth_cm is not None:                  # release in-process split authority (see above)
            try:
                _auth_cm.__exit__(None, None, None)
            except Exception:
                pass
        if prev_params is None:
            os.environ.pop("KDT_CALIB_PARAMS", None)
        else:
            os.environ["KDT_CALIB_PARAMS"] = prev_params
        if prev_split is None:
            os.environ.pop("KDT_CALIB_SPLIT", None)
        else:
            os.environ["KDT_CALIB_SPLIT"] = prev_split

    if not metrics:
        return {"certified": False, "observed": None,
                "reason": "module produced NO metrics at the reference params — likely missing "
                          "prepare/staging/routing; cannot reproduce the validated KI run"}
    # WRONG-GAUGE GUARD (codex round-6): certification is a direct run_model() path, so a runner that
    # scores a DIFFERENT gauge could pass on a coincidentally-matching headline. For a pinned case in
    # runner mode, the module MUST self-declare __kdt__.case_id AND it must match — else NOT certified,
    # for the wrong-gauge reason (before the headline/signature checks can falsely pass it).
    if expected_case_id and inj_mode == "runner":
        _dc = (metrics.get("__kdt__") or {}).get("case_id")
        if _dc is None or str(_dc) != str(expected_case_id):
            return {"certified": False, "observed": metrics,
                    "reason": f"module scored case '{_dc if _dc is not None else '<undeclared>'}' but the "
                              f"target case is '{expected_case_id}' — certified against the WRONG gauge/obs"}
    obs = metrics.get(headline)
    try:
        obs = float(obs)
    except (TypeError, ValueError):
        obs = None
    if obs is None or not math.isfinite(obs):
        return {"certified": False, "observed": metrics,
                "reason": f"module did not emit a finite headline metric {headline!r}"}
    if ref_val is None:
        return {"certified": False, "observed": metrics,
                "reason": f"reference_run.json has no {headline!r} value to certify against"}
    htol = float(tol.get(headline, 0.02))
    if abs(obs - float(ref_val)) > htol:
        return {"certified": False, "observed": metrics,
                "reason": f"headline {headline}={obs:.4f} differs from validated {float(ref_val):.4f} "
                          f"by > tol {htol} — module does not reproduce the validated KI run"}
    sig = metrics.get("__kdt__") or {}
    for k in ("n_paired_days", "n_cells"):
        if k in ref_sig and k in sig:
            try:
                if int(sig[k]) != int(ref_sig[k]):
                    return {"certified": False, "observed": metrics,
                            "reason": f"run-signature {k}={sig[k]} != reference {ref_sig[k]} — the module "
                                      "scores a different window/domain than the validated run"}
            except (TypeError, ValueError):
                pass
    return {"certified": True, "observed": metrics,
            "reason": f"reproduced validated {headline}={obs:.4f} (ref {float(ref_val):.4f}, tol {htol})"}


def _holdout_band_ceilings(objs, headline_objectives):
    """PRAGMATIC convention pass-band -> per-objective HOLDOUT loss ceiling (metric-match,
    unambiguous-only — user's choice 2026-07-09). `headline_objectives` is
    {dag_variable: [{metric, target, direction, ...}]} from the validation convention
    (re-keyed to the model's own output names in the 2026-08-15 obs-map rewire; a direct
    variable->objective match is now possible and can replace this metric-match later).
    Today we still match a convention headline to a calibration objective BY METRIC, and apply
    the band ONLY when that metric has a SINGLE unambiguous numeric target across all the
    convention's quantities. If the metric's target is absent or ambiguous (e.g. discharge
    pbias 25 vs ET pbias 10), we add NO band and fall back to the non-degradation gate — a
    wrong band is worse than none. Returns {objective_name: max_loss}."""
    from .objectives import _FAMILY_LOSS
    if not headline_objectives:
        return {}
    by_metric = {}                                  # metric -> {numeric targets across quantities}
    for _q, entries in (headline_objectives or {}).items():
        for e in (entries or []):
            m = str(e.get("metric") or "").lower()
            t = e.get("target")
            # a finite numeric target only — NaN/inf must NOT synthesize a bogus band
            # (isinstance(float('nan'), float) is True); bool is not a target (Codex review).
            if m and isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t):
                by_metric.setdefault(m, set()).add(float(t))
    ceilings = {}
    for o in objs:
        targets = by_metric.get(str(o.metric_key).lower())
        if not targets or len(targets) != 1:        # absent or ambiguous -> non-degradation only
            continue
        fam = _FAMILY_LOSS.get(o.family)
        if not fam:
            continue
        _key, loss_fn = fam
        try:
            ceilings[o.name] = float(loss_fn(next(iter(targets))))
        except Exception:
            continue
    return ceilings


def _is_multi_objective(contract, objs):
    """Pareto mode ONLY when there's a real trade-off (codex calib.py:92): an explicit
    strategy.multi_objective flag, OR >=2 DISTINCT target VARS. Multiple families of
    the SAME var are scalarized (their weights already split in objectives.py)."""
    # runtime override (operator/testing): KDT_CALIB_MULTIOBJ=1 forces Pareto mode even for a
    # single-var, multi-family case (the families become separate objectives); =0 forces scalar.
    env_mo = (os.environ.get("KDT_CALIB_MULTIOBJ") or "").strip().lower()
    if env_mo in ("1", "true", "yes"):
        return True
    if env_mo in ("0", "false", "no"):
        return False
    strat = contract.get("strategy", {}) or {}
    if "multi_objective" in strat:
        return bool(strat["multi_objective"])
    return len({o.var for o in objs}) >= 2


_SPOTPY_ALGOS = ("dds", "sceua", "dream")     # scalar (single-objective) backends
_PYMOO_ALGOS = ("nsga2", "nsga3", "moead")    # multi-objective (Pareto) backends


def _pick_backend(contract, multi, n_distinct_vars):
    # runtime override (operator/testing): KDT_CALIB_ALGO forces the optimizer without editing the
    # (hash-pinned) calibration.yaml. Resolution order: env override -> yaml default_algorithm ->
    # mode default (nsga2 if multi else dds). An unrecognized env value is ignored.
    env_algo = (os.environ.get("KDT_CALIB_ALGO") or "").strip().lower()
    strat = contract.get("strategy", {}) or {}
    if env_algo in _SPOTPY_ALGOS + _PYMOO_ALGOS:
        algo = env_algo
    else:
        algo = strat.get("default_algorithm") or ("nsga2" if multi else "dds")
    # COMPATIBILITY FENCE (codex): spotpy algos are scalar-only and pymoo algos need >=2 objectives, so
    # an incompatible (algo, multi) pair would raise inside backend.optimize() and crash the run. Coerce
    # to the mode-appropriate default instead — whether the mismatch came from the env override OR a
    # yaml default that disagrees with multi_objective.
    if multi and algo in _SPOTPY_ALGOS:
        print(f"  [calib] algo {algo!r} is scalar-only but multi_objective=True -> using nsga2", flush=True)
        algo = "nsga2"
    elif (not multi) and algo in _PYMOO_ALGOS:
        print(f"  [calib] algo {algo!r} needs >=2 objectives but the problem is scalar -> using dds", flush=True)
        algo = "dds"
    return algo


def _probe_runner_metrics(run_model, inj_mode, params, workdir):
    """Run the runner ONCE at DEFAULT params to see which metrics it actually emits. A RUNNER-mode
    driver REQUIRES KDT_CALIB_PARAMS, so the probe MUST set it to a defaults file — otherwise the
    driver dies, the probe reads {}, the un-producible-objective drop is skipped, and every eval
    scalarizes to +inf because a declared-but-unemitted family (e.g. timing_accuracy/day_bias) stays
    in the objective set (the CaMa screen_failed root cause, 2026-07-19). Returns the metrics dict."""
    import json as _json
    prev = os.environ.get("KDT_CALIB_PARAMS")
    prev_split = os.environ.get("KDT_CALIB_SPLIT")
    try:
        if inj_mode == "runner":
            wd = Path(workdir); wd.mkdir(parents=True, exist_ok=True)
            pf = wd / "kdt_probe_params.json"
            defaults = {p["name"]: p.get("default") for p in params if p.get("default") is not None}
            pf.write_text(_json.dumps(defaults, default=str))
            os.environ["KDT_CALIB_PARAMS"] = str(pf)
        # SCORE THE CALIBRATION WINDOW (2026-07-19): with the split UNSET a runner scores its FULL
        # record, so a judgement of adequacy from it would read data the optimizer will not train on (a
        # SWAT+ case whose calibration window is NSE -0.47 once looked adequate off a full-period NSE
        # 0.585). The probe must score the window being optimized.
        from .evaluator import resolve_train_split, split_is_authoritative
        if not split_is_authoritative(prev_split):      # unset / blank / inherited 'full' -> we own it
            os.environ["KDT_CALIB_SPLIT"] = resolve_train_split(prev_split)
        return run_model() or {}
    except Exception:
        return {}
    finally:
        if prev is None:
            os.environ.pop("KDT_CALIB_PARAMS", None)
        else:
            os.environ["KDT_CALIB_PARAMS"] = prev
        if prev_split is None:
            os.environ.pop("KDT_CALIB_SPLIT", None)
        else:
            os.environ["KDT_CALIB_SPLIT"] = prev_split


def _filter_objectives_by_probe(objs, probe, expected_case_id, inj_mode):
    """Using a probe metrics dict: (1) enforce the pinned case (wrong/undeclared -> error), and
    (2) DROP objectives whose metric the runner didn't produce (so a declared-but-unemitted family
    can't poison every eval to +inf). Returns (kept_objs, error_dict_or_None)."""
    if not probe:
        return objs, None                       # probe couldn't run -> no info, keep all
    if expected_case_id and inj_mode == "runner":
        pcase = (probe.get("__kdt__") or {}).get("case_id")
        if pcase is None or str(pcase) != str(expected_case_id):
            return objs, {"status": "runner_wrong_case", "injection_mode": "runner",
                          "expected_case_id": expected_case_id,
                          "declared_case_id": str(pcase) if pcase is not None else "<undeclared>",
                          "reason": f"objective probe scored case "
                                    f"'{pcase if pcase is not None else '<undeclared>'}' but the target "
                                    f"case is '{expected_case_id}' — capability built for the wrong gauge"}
    def _present(o):
        mm = probe.get(o.var) if isinstance(probe.get(o.var), dict) else probe
        return (mm or {}).get(o.metric_key) is not None
    kept = [o for o in objs if _present(o)]
    dropped = [o.name for o in objs if not _present(o)]
    if dropped and kept:
        print(f"  [calib] objective family probe: keeping {[o.name for o in kept]}, "
              f"dropping (metric not produced by runner) {dropped}", flush=True)
        return kept, None
    if not kept:
        return objs, {"status": "runner_emits_no_objective_metric",
                      "reason": f"runner produced {list(probe.keys())[:8]} but none match "
                                f"the dag objectives {[o.name for o in objs]}"}
    return kept, None


def _select_front_member(ev, objs, result, holdout_spec, band_ceilings,
                         probe_x, baseline_x, cap: int | None = None,
                         spread_floor: float = 1e-9, protect=None):
    """MULTI-OBJECTIVE FRONT SELECTION — gate the Pareto front, don't commit one guess.

    pymoo returns a single representative = the min-normalized-SUM knee. That knee can
    commit a point that sacrifices ONE objective badly while the others are ~perfect
    (observed on CRHM: discharge NSE fell to ~0.24 while SWE calibrated near-perfectly).
    The front, however, CONTAINS balanced members. Instead of gating only the sum-knee,
    rank members by a MINIMAX (Chebyshev) criterion — smallest WORST normalized objective,
    which protects every objective — and validate them on the HELD-OUT split in that order,
    committing the FIRST member that PASSES the holdout on ALL objectives. If none of the
    gated members passes, keep the top-minimax member (a strictly better default than the
    sum-knee) and let the caller report it not-promotable.

    EXHAUSTIVE by default (codex 2026-08-20): every finite front member is gated, because
    truncating could report promotable=False while a lower-ranked member would have passed —
    and time/cost is not a constraint here. `cap` (if set) bounds the gated set and the outcome
    is explicitly flagged TRUNCATED (never silently "none passed"). Objectives whose spread
    across the front is numerically trivial are EXCLUDED from the minimax ranking, so float-noise
    normalization on a near-degenerate objective cannot dominate the worst-objective criterion.

    MUTATES result.best_x / result.best_loss to the chosen member; records the choice in
    result.notes. Returns (chosen_idx, holdout_dict) or (None, None) if there is no usable front.
    """
    import numpy as np
    from .holdout import validate_holdout
    X = result.pareto_x or []
    F = result.pareto_f or []
    if not X or not F or len(X) != len(F):
        return None, None                       # no front (or pre-pareto_f engine) -> caller keeps knee
    Fa = np.asarray(F, float)
    fin = [i for i in range(len(Fa)) if np.all(np.isfinite(Fa[i]))]
    if not fin:
        return None, None
    Fa = Fa[fin]; X = [X[i] for i in fin]; F = [F[i] for i in fin]
    # minimax knee over DISCRIMINATING objectives only: an objective with trivial spread across
    # the front doesn't separate members — including it would let float noise, blown up to [0,1]
    # by normalization, dominate the worst-objective criterion (codex 2026-08-20).
    span = np.ptp(Fa, axis=0)
    scale = np.maximum(np.abs(Fa).max(0), 1e-12)
    disc = (span > spread_floor) & (span > 1e-6 * scale)
    if disc.any():
        Fd = Fa[:, disc]
        Fn = (Fd - Fd.min(0)) / (np.ptp(Fd, axis=0) + 1e-12)
        crit = Fn.max(1)                        # worst discriminating (normalized) objective
    else:
        crit = Fa.sum(1)                        # all objectives degenerate -> fall back to sum
    order = list(np.argsort(crit))              # most-balanced (smallest worst-obj) first
    gated = order if not cap else order[:cap]
    truncated = bool(cap and len(gated) < len(order))
    if truncated:                               # NEVER a silent cap
        print(f"  [calib] front-select: TRUNCATED to {len(gated)}/{len(order)} members "
              f"(cap={cap}); remaining lower-ranked members NOT gated", flush=True)
    top_holdout = None
    holdouts = {}                                   # idx -> holdout dict (for protected fallback)
    # gating a front spends many holdout re-runs; tag them `front_select` so phase_counts keeps
    # them apart from the ONE final holdout gate (2026-09-27).
    _prev_phase = getattr(ev, "phase", None)
    if _prev_phase is not None:
        ev.phase = "front_select"
    try:
        for rank, idx in enumerate(gated):
            hd = validate_holdout(ev, objs, list(X[idx]), holdout_spec, probe_x=probe_x,
                                  band_ceilings=band_ceilings, baseline_x=baseline_x)
            holdouts[idx] = hd
            if rank == 0:
                top_holdout = hd
            if hd.get("passed") is True:
                result.best_x = list(X[idx]); result.best_loss = list(F[idx])
                result.notes = (result.notes or "") + (
                    f" | front-select: committed member {idx} (minimax-rank {rank} of "
                    f"{len(order)}) PASSED holdout; exhaustive={not truncated}")
                print(f"  [calib] front-select: member #{idx} (minimax-rank {rank}) PASSES holdout "
                      f"on all objectives -> committing", flush=True)
                return idx, hd
    finally:
        if _prev_phase is not None:                 # the FINAL holdout gate must not inherit this tag
            ev.phase = _prev_phase

    # PROTECTED FALLBACK (2026-08-25). No member passed the full gate on ALL objectives.
    # The raw minimax knee protects the WORST normalized objective, but on a strongly-conflicting
    # front that worst objective can be one the caller cares about keeping — CRHM @ Blue River: the
    # knee sacrificed discharge TIMING (holdout r fell from the uncalibrated 0.82 toward 0) to fit
    # SWE near-perfectly. When the caller names PROTECTED objectives, prefer a member that keeps
    # every protected objective admissible on the held-out split and is best on the FREE
    # (non-protected) objectives, instead of that knee. Three admissibility tiers, tried best-first:
    #   A  protected objective PASSES its own holdout gate (ok=True);
    #   B  protected objective BEATS the uncalibrated baseline out-of-sample;
    #   C  protected objective degrades no more than `protect_tol` beyond the baseline holdout loss.
    # Tier C matters when the UNCALIBRATED model is already best on a protected objective (its
    # baseline is unbeatable, so A/B are infeasible) yet a small, bounded give-back there buys a
    # large gain on the free objective — the real discharge/SWE knee. Among admissible members, rank
    # by a MINIMAX over the free objectives' held-out losses (best worst-free = best SWE here). If no
    # tier has a member, protection is INFEASIBLE (the conflict is fundamental) — say so and keep the
    # minimax knee. This never promotes: a protected pick is still reported not-promotable.
    prot = {p for p in (protect or set()) if any(o.name == p for o in objs)}
    if protect and not prot:                        # all named objectives unknown (typo/stale) ->
        _ignored = sorted(set(protect))             # do NOT let a bad env silently re-rank the front
        print(f"  [calib] front-select[protect]: IGNORING unknown protected objective(s) {_ignored} "
              f"(none match this run's objectives); protection not applied", flush=True)
        result.notes = (result.notes or "") + (
            f" | front-select[protect]: ignored unknown protected objective(s) {_ignored}")
    if prot:
        free_names = [o.name for o in objs if o.name not in prot]
        protect_tol = float((holdout_spec or {}).get("protect_tol", 0.25))
        protect_abs = float((holdout_spec or {}).get("protect_abs_tol", 0.05))
        def _po_map(hd):
            return {p.get("objective"): p for p in (hd.get("per_objective") or [])}
        def _num(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) else None
        def _admissible(hd, tier):
            m = _po_map(hd)
            for nm in prot:
                p = m.get(nm)
                if not p:
                    return False
                hl = _num(p.get("holdout_loss"))
                if hl is None:
                    return False
                if tier == "A" and p.get("ok") is not True:
                    return False
                if tier == "B" and p.get("beats_baseline") is not True:
                    return False
                if tier == "C":
                    bl = _num(p.get("baseline_holdout_loss"))
                    if bl is None:                    # no baseline to bound against -> can't claim C
                        return False
                    if hl > bl * (1.0 + protect_tol) + protect_abs:
                        return False
            return True
        def _free_vec(hd):
            m = _po_map(hd)
            out = []
            for nm in free_names:
                v = _num((m.get(nm) or {}).get("holdout_loss"))   # 0.0 is valid; only None -> inf
                out.append(v if v is not None else float("inf"))
            return out
        for tier in ("A", "B", "C"):
            cand = [i for i in gated if _admissible(holdouts.get(i, {}), tier)]
            if not cand:
                continue
            # rank admissible members by a MINIMAX over the FREE objectives' holdout losses. Only
            # members whose free losses are ALL finite can be ranked; if none are, fall back to the
            # first admissible member (gated preserves minimax order = most-balanced first). This is
            # fail-closed: a missing free loss can never win the ranking via inf->nan normalization.
            if free_names:
                vecs = {i: _free_vec(holdouts[i]) for i in cand}
                rankable = [i for i in cand if all(math.isfinite(x) for x in vecs[i])]
                if rankable:
                    M = np.asarray([vecs[i] for i in rankable], float)
                    Mn = (M - M.min(0)) / (np.ptp(M, axis=0) + 1e-12)
                    crit2 = Mn.max(1)                # best worst-free objective (best SWE)
                    pick = rankable[int(np.argmin(crit2))]
                else:
                    pick = cand[0]                   # no rankable free info -> most-balanced admissible
            else:
                pick = cand[0]
            result.best_x = list(X[pick]); result.best_loss = list(F[pick])
            _tname = {"A": "A:protected-all-pass", "B": "B:protected-beats-baseline",
                      "C": f"C:protected-within-{protect_tol:.0%}-of-baseline"}[tier]
            result.notes = (result.notes or "") + (
                f" | front-select[protect]: committed member {pick} protecting {sorted(prot)} "
                f"[tier {_tname}], best on free {free_names}; still not-promotable")
            print(f"  [calib] front-select[protect]: member #{pick} keeps {sorted(prot)} admissible "
                  f"(tier {_tname}); committing as protected knee (not-promotable)", flush=True)
            return pick, holdouts[pick]
        result.notes = (result.notes or "") + (
            f" | front-select[protect]: INFEASIBLE — no front member keeps {sorted(prot)} "
            f"admissible on holdout (even within {protect_tol:.0%} of baseline); kept minimax knee")
        print(f"  [calib] front-select[protect]: INFEASIBLE — no front member keeps {sorted(prot)} "
              f"admissible on the held-out split even within {protect_tol:.0%} of baseline "
              f"(conflict is fundamental); keeping minimax knee", flush=True)

    idx = order[0]
    result.best_x = list(X[idx]); result.best_loss = list(F[idx])
    status = "TRUNCATED-none-in-cap-passed" if truncated else "exhaustive-none-passed"
    result.notes = (result.notes or "") + (
        f" | front-select: NO member passed holdout ({status}, {len(gated)}/{len(order)} gated); "
        f"kept minimax knee {idx}")
    print(f"  [calib] front-select: no member passed holdout ({status}); keeping minimax knee "
          f"#{idx} (better-balanced than sum-knee, still not promotable)", flush=True)
    return idx, top_holdout


def _consumption_code_key(contract, ki_path):
    """The CODE half of the consumption-proof cache identity (round-6 fix, both seats).

    Hashes: the contract text; EVERY .py under the KI's tools dir (the driver AND the helpers it
    imports — codex: a driver that imports helper modules was invisible to the old key); the
    runner command's .py tokens resolved against runner.cwd and ki_path; the kit's own scoring +
    proof code (evaluator/objectives/holdout/calib); the proof-schema version (kimi: plan v2 item
    5 required it — a schema bump would otherwise replay stale receipts); the split; and the
    values of env vars the contract DECLARES as setup-selecting (runner.env keys plus any listed
    under strategy.cache_env). What this cannot see, the DATA half below catches."""
    import hashlib
    from .evaluator import Evaluator
    h = hashlib.sha1(); h.update(b"kdt-consumption-proof-cache/3")
    h.update(Evaluator.PROOF_SCHEMA.encode())
    cpath = os.environ.get("KDT_CALIB_CONTRACT") or str(Path(ki_path) / "calibration.yaml")
    try: h.update(Path(cpath).read_bytes())
    except OSError: h.update(b"<no-contract>")
    runner = contract.get("runner") or {}
    rcwd = Path(str(runner.get("cwd") or ki_path))
    for tok in (runner.get("command") or []):
        for cand in (Path(str(tok)), rcwd / str(tok), Path(ki_path) / str(tok)):
            if cand.suffix == ".py" and cand.exists():
                h.update(cand.read_bytes()); break
    tools = Path(ki_path) / "tools"
    if tools.is_dir():
        # round-2 (kimi/codex): EVERY file under tools/, not only .py — a namelist template, a
        # .json config, a table or a rebuilt binary there can change the address mapping while the
        # default output stays bit-identical. Backups, caches and outputs excluded.
        for f in sorted(tools.rglob("*")):
            if not f.is_file() or ".bak" in f.name or "__pycache__" in f.parts or f.suffix in (".pyc", ".log", ".out"):
                continue
            try:
                h.update(f.relative_to(tools).as_posix().encode()); h.update(f.read_bytes())
            except OSError:
                h.update(f"<unreadable:{f.name}>".encode())
    kit = Path(__file__).parent
    for f in ("evaluator.py", "objectives.py", "holdout.py", "calib.py", "runner.py"):
        try: h.update((kit / f).read_bytes())
        except OSError: pass
    h.update(str(os.environ.get("KDT_CALIB_SPLIT") or "calibration").encode())
    env_keys = sorted(set((runner.get("env") or {}).keys())
                      | set((contract.get("strategy") or {}).get("cache_env") or []))
    for k in env_keys:
        h.update(f"|{k}={os.environ.get(k, '')}".encode())
    return h.hexdigest()


def _consumption_gate(ev, contract, ki_path, workdir, x_default, lower, upper, names, params):
    """Run (or reuse) the per-parameter consumption proof; STOP the fit on UNREACHABLE.

    KDT_CALIB_C8: 'enforce' (default) | 'off'. 'off' exists for launchers that ALREADY ran an
    equivalent hand probe against the current setup and gate on it themselves (System 1's
    noop_probe) — it must be a deliberate, recorded choice, never a silent default.

    CACHE REUSE (round-6 fix): a cached receipt is trusted only if (a) its CODE key matches the
    current code key, AND (b) the DEFAULT run, re-executed NOW (one eval, not N+1), produces the
    SAME raw-output hash the receipt recorded. The old version validated a receipt against the
    hash stored inside the receipt itself, which could never detect the setup drifting — both
    seats caught it. Re-running the default is the only check that looks at the model.
    Returns {} to continue, or a terminal status dict to stop."""
    import json as _json
    mode = os.environ.get("KDT_CALIB_C8", "enforce").strip().lower()
    if mode == "off":
        print("  [calib] consumption proof: SKIPPED (KDT_CALIB_C8=off — the launcher must have gated "
              "on an equivalent probe against the current setup)", flush=True)
        return {"consumption_proof": {"ok": None, "skipped": True, "summary": "KDT_CALIB_C8=off",
                                      "params": [], "by_verdict": {}}}
    cache_dir = Path(workdir) / "kdt_consumption_proof"; cache_dir.mkdir(parents=True, exist_ok=True)
    code_key = _consumption_code_key(contract, ki_path)
    cp = None
    # round-2 (codex): reuse is OPT-IN. The code key cannot see compiled binaries, shared
    # libraries, external data, or undeclared env; a contract that opts in accepts that residual.
    # Default is to re-prove every time — the direction that looks at the model.
    _reuse_ok = bool((contract.get("strategy") or {}).get("consumption_cache_reuse", False))
    cands = [f for f in sorted(cache_dir.glob("*.json"))] if _reuse_ok else []
    if cands:
        # one LIVE default evaluation to learn the CURRENT raw-output hash before trusting anything.
        # round-2 (kimi): the evaluator rehydrates its resumability cache from disk, so without
        # _force_live this "live" run could be a replay from a previous session.
        _prev = getattr(ev, "_force_live", False)
        ev._force_live = True
        try:
            with ev.phase_as("proof"):                 # gap 2j: never counted as search effort
                l0 = ev.evaluate(x_default)
        finally:
            ev._force_live = _prev
        from .evaluator import resolve_train_split
        m0 = ev._last_metrics.get(resolve_train_split(os.environ.get("KDT_CALIB_SPLIT"))) or {}
        p0 = ev._proofs_of(m0) or []
        now_hash = {t["target"]: t.get("values_hash") for t in p0}
        for f in cands:
            try:
                rec = _json.loads(f.read_text())
            except Exception:
                continue
            if (rec.get("code_key") == code_key and now_hash and
                    rec.get("default_values_hash") == now_hash and
                    all(math.isfinite(v) for v in l0)):
                print(f"  [calib] consumption proof: REUSED {f.name} — code key matches AND the default "
                      f"run re-executed now gives the same raw output ({rec.get('summary')})", flush=True)
                cp = rec; break
        if cp is None:
            print("  [calib] consumption proof: cached receipts present but NONE valid for the current "
                  "code + default output — re-proving", flush=True)
    if cp is None:
        # round-3 (both seats): the proof's OWN default + N displaced evals must be live too, not
        # only the receipt-validating one — otherwise a pre-drift displaced run replayed from the
        # disk cache acquits a knob that drift has since deadened, and the receipt is then frozen
        # under a code key that DOES cover the drift. Restore whatever the flag was, not False.
        _prev = getattr(ev, "_force_live", False)
        ev._force_live = True
        try:
            cp = ev.consumption_proof(x_default, lower, upper, names=names, parameters=params, ki_path=ki_path)
        finally:
            ev._force_live = _prev
        cp["code_key"] = code_key
        if cp.get("ok") is not None:            # never cache "nothing proven"
            (cache_dir / f"{code_key[:16]}.json").write_text(_json.dumps(cp, indent=1, default=str))
    print(f"  [calib] consumption proof: {cp.get('summary')}", flush=True)
    if cp.get("ok") is None:
        print("  [calib] consumption proof: NOTHING PROVEN — " + str(cp.get("summary")) + ". Continuing "
              "because there is nothing sound to enforce; this box is UNPROVEN, not proven.", flush=True)
        # 2026-09-16 (checkpoint, codex must-fix 4): the proof result travels into the FINAL report even
        # when the fit continues, so "nothing proven" is visible there and cannot read as a pass.
        return {"consumption_proof": cp}
    by = cp.get("by_verdict") or {}
    for v in ("OBJECTIVE_MASKED", "INSENSITIVE", "WINDOW_MASKED", "DORMANT", "UNPROVEN"):
        if by.get(v):
            print(f"  [calib] consumption proof {v}: {by[v]}", flush=True)
    if cp.get("ok") is False:
        rows = [r for r in cp["params"] if r["verdict"] == "UNREACHABLE"]
        return {"status": "param_unreachable", "injection_mode": "runner",
                "unreachable": cp["unreachable"], "consumption_proof": cp,
                "reason": ("FIT REFUSED. " + str(len(rows)) + " parameter(s) never reach the model: "
                           + ", ".join(f"{r['param']} (displaced to {r['displaced_to']}, raw output "
                                       f"bitwise identical)" for r in rows)
                           + ". This is a KI defect at the parameter's ADDRESS, not a modelling "
                             "question, and it is the owner's to fix. The kit will not search a box "
                             "with a dead dimension, and it will not drop the dimension for you.")}
    return {"consumption_proof": cp}


def consumption_coverage(cp: dict | None, n_params: int) -> dict:
    """What the final report says about the proof (checkpoint 2026-09-16, codex must-fix 4): how many
    parameters got a verdict, which did not, and the verdict counts. A fit that continued with
    nothing proven says so HERE — 'promotable' is a holdout statement, never a consumption one."""
    cp = cp or {}
    rows = cp.get("params") or []
    proven = [r["param"] for r in rows if r.get("verdict") not in (None, "UNPROVEN")]
    unproven = [r["param"] for r in rows if r.get("verdict") in (None, "UNPROVEN")]
    if not rows:                                   # skipped, or the driver emits no proof
        return {"n_params": n_params, "n_proven": 0, "n_unproven": n_params,
                "unproven": [], "proven": [], "by_verdict": {}, "ok": cp.get("ok"),
                "summary": cp.get("summary"), "note": "no per-parameter verdicts — nothing proven"}
    return {"n_params": n_params, "n_proven": len(proven), "n_unproven": len(unproven) + max(0, n_params - len(rows)),
            "proven": proven, "unproven": unproven, "by_verdict": cp.get("by_verdict") or {},
            "ok": cp.get("ok"), "summary": cp.get("summary")}


def _band_override(stop_translated, var):
    """An old strategy.stop floor band is the standards band of its main_stat (design §5.1b), for the
    floor's own dag_variable only when it named one. ONE rule for the triage and the search's standards."""
    if not (stop_translated and stop_translated.get("band")):
        return None
    if (stop_translated.get("dag_variable")
            and str(stop_translated["dag_variable"]).lower() != str(var).lower()):
        return None
    return {stop_translated["main_stat"]: stop_translated["band"]}


#: proof verdicts that PROVE a parameter, moved alone from the defaults to its far bound, leaves the
#: objectives unmoved (design §1 triage route `not_calibratable`)
_NOT_MOVING = ("OBJECTIVE_MASKED", "DORMANT", "WINDOW_MASKED")


def triage_route(pilot, proof, searched) -> dict:
    """TRIAGE BEFORE THE SEARCH (design §1, Leo 2026-09-28; gap 2x). The kit is there to calibrate, so it
    always calibrates unless the run cannot be calibrated. First match wins:
      1. `no_baseline`       the pilot's default run gives no finite objective metric -> no search;
      2. `not_calibratable`  the consumption proof ran and gave a verdict for EVERY searched parameter,
                             and every verdict is OBJECTIVE_MASKED, DORMANT or WINDOW_MASKED -> no search;
      3. `calibrate`         everything else (proof off / not run / empty, UNPROVEN or INSENSITIVE
                             parameters included) -> the search runs.
    How good the fit is at the defaults is never a reason to skip. "sensitivity not proven" whenever no
    parameter is ALIVE. `searched` = the names of the parameters the search moves."""
    p = pilot if isinstance(pilot, dict) else {}
    nf = p.get("default_n_finite")
    nf = nf if (isinstance(nf, int) and not isinstance(nf, bool) and nf >= 0) else None
    rows = [r for r in ((proof or {}).get("params") or []) if isinstance(r, dict) and r.get("param") is not None]
    by = {str(r["param"]): str(r.get("verdict")) for r in rows}
    alive = sorted(n for n, v in by.items() if v == "ALIVE")
    searched = [str(n) for n in (searched or [])]
    proof_summary = {"ran": bool(rows), "verdicts": by, "alive": alive,
                     "summary": (proof or {}).get("summary") if isinstance(proof, dict) else None}
    out = {"proof": proof_summary, "sensitivity_not_proven": not alive, "notes": []}
    if nf == 0:
        why = "the pilot's default run gives no finite objective metric — there is no baseline to calibrate from"
        if p.get("reason"):
            why += f" ({p.get('reason')})"
        return dict(out, route="no_baseline", reason=why)
    if nf is None:
        out["notes"].append("baseline not checked (no default-run count in the pilot summary)")
    if searched and rows and all(n in by for n in searched) and all(by[n] in _NOT_MOVING for n in searched):
        return dict(out, route="not_calibratable",
                    reason=("every searched parameter was proven not to move the objectives, each parameter "
                            "moved alone from the defaults to its far bound: "
                            + "; ".join(f"{n}: {by[n]}" for n in searched)))
    return dict(out, route="calibrate",
                reason="the run can be calibrated" + ("; sensitivity not proven (no parameter is ALIVE in the "
                                                      "consumption proof)" if not alive else ""))


#: a pilot record with no call that returned metrics (saved so a resume knows the pilot WAS checked)
_EMPTY_PROV = {"calls": 0, "echo_ok": 0, "echo_missing": 0, "echo_wrong": 0, "missing_at": [], "wrong_at": []}


def effective_budget_mode(budget_spec: dict | None) -> str:
    """`measured` | `fixed`, decided in ONE place. An `allowance` IS a request for a measured cap,
    with or without an explicit `mode` — reading the raw field in several places once made two
    planners disagree (codex round 2)."""
    b = budget_spec or {}
    m = str(b.get("mode") or ("measured" if b.get("allowance") not in (None, "") else "fixed")).strip().lower()
    return m if m in ("measured", "fixed") else "fixed"


def calibrate(ki_path: str, workdir: str, obs_shape_by_var: dict,
              run_model=None, budget: int | None = None, seed: int = 0,
              determining_metric=None, headline_objectives=None,
              expected_case_id=None, reply_gate=None):
    """Top-level calibration. `run_model` (callable -> metrics dict) is the KI's
    programmatic run+score path; if None it is built from the contract's `runner`.
    `obs_shape_by_var` comes from the model_obs_map join. `determining_metric` (the
    per-obs headline metric from readiness/the convention) overrides the family-default
    metric when the contract didn't; `headline_objectives` (convention pass-bands) gate
    the holdout on the absolute field band. Every call searches ALL declared parameters and runs its own
    pilot (the caller-subset `active_idx` and caller-given `pilot` arguments of staged calibration were
    removed in build step 8b). `reply_gate` (Leo 2026-10-02, the runner reply upgrade): True — stop right
    after the pilot with status "reply_incomplete" when the runner's reply lacks what the convergence check needs
    (a required watched score, a 2n mismatch, the scored series); "check" — always stop there with status
    "reply_checked". The gate acts only before any search evaluation exists in the work folder; the plan and
    the pilot stay saved (tied to the runner's code) and are reused by the next call. Returns a report dict."""
    # ONE calibrate() per workdir at a time (Opus 6b r1): an exclusive lock held for this call — and, through
    # the inherited file, by any forked lane — so a resume can never run beside lanes of a killed run. Taken
    # here and closed in a finally (Opus 6b r2 #6: a kept exception must not keep the lock). Its OWN file
    # name: KI runners (HBV, Noah-MP) already lock `<workdir>/.kdt_calib.lock` per evaluation — the kit holding
    # that one would make every model run wait (found by the HBV tracer bullet).
    _wd_lock = None
    try:
        import fcntl as _fcntl
        Path(workdir).mkdir(parents=True, exist_ok=True)
        _wd_lock = open(Path(workdir) / ".kdt_calibrate.lock", "a")
        _fcntl.flock(_wd_lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except BlockingIOError:
        if _wd_lock is not None:
            _wd_lock.close()
        return {"status": "workdir_busy", "certified": None,
                "reason": f"another calibration (or a lane of one) is running in {workdir} — wait for it or "
                          f"end it; two runs in one workdir would mix their calls and caches"}
    except (OSError, ImportError):
        if _wd_lock is not None:
            _wd_lock.close()
        _wd_lock = None                         # no locking on this filesystem: carry on as before
    _state: dict = {"certified": None}
    try:
        _rep = _calibrate_locked(ki_path, workdir, obs_shape_by_var, run_model=run_model, budget=budget,
                                 seed=seed, determining_metric=determining_metric,
                                 headline_objectives=headline_objectives, expected_case_id=expected_case_id,
                                 _state=_state, reply_gate=reply_gate)
        # every report says whether the runner was certified against a validated reference run (the
        # contract's runner.reference_run or the KI's calib/reference_run.json): True / False (no reference)
        # / None (the call ended before certification); the agent reads it (Opus 8b/9 r1 #11)
        if isinstance(_rep, dict):
            _rep.setdefault("certified", _state.get("certified"))
        return _rep
    finally:
        if _wd_lock is not None:
            _wd_lock.close()


def _calibrate_locked(ki_path: str, workdir: str, obs_shape_by_var: dict,
                      run_model=None, budget: int | None = None, seed: int = 0,
                      determining_metric=None, headline_objectives=None,
                      expected_case_id=None, _state: dict | None = None, reply_gate=None):
    """`calibrate()` under its workdir lock (see calibrate). `_state` collects what the wrapper adds to
    every report (whether the runner was certified)."""
    _state = {} if _state is None else _state
    from . import objectives as O
    from .backends.base import Problem
    import yaml as _yaml

    contract = _load_contract(ki_path)
    dag = _yaml.safe_load((Path(ki_path) / "dag.yaml").read_text())
    params = contract["parameters"]
    # strategy.convergence.window: "auto" or a positive whole number of calls — checked at LOAD,
    # before any model run (a refusal after the pilot would leave applicator inputs changed)
    _conv0 = (contract.get("strategy", {}) or {}).get("convergence")
    if _conv0 is not None and not isinstance(_conv0, dict):
        return {"status": "invalid_contract",
                "reason": f"strategy.convergence must be a mapping; got {type(_conv0).__name__}"}
    # the mode ("keep_going" default, "stop"), and the old strategy.stop block translated (§1.6, §5.1b)
    from .rule import resolve_mode as _resolve_mode
    try:
        _cmode, _cwarn, _stop_translated = _resolve_mode(contract)
    except ValueError as _e:
        return {"status": "invalid_contract", "reason": str(_e)}
    for _w in _cwarn:
        print(f"  [calib] {_w}", flush=True)
    _wspec0 = ((_conv0 or {}).get("window", "auto"))
    if _wspec0 not in (None, "auto"):
        if isinstance(_wspec0, bool) or not isinstance(_wspec0, int) or _wspec0 < 1:
            return {"status": "invalid_contract",
                    "reason": f"strategy.convergence.window must be 'auto' or a positive whole number of "
                              f"calls; got {_wspec0!r}"}
    # strategy.budget is checked at LOAD, so a typo is not paid for with commissioning, the proof and
    # the pilot (Opus rounds 3-5). The block and its mode are always checked (a mistyped mode would
    # silently drop the ledger and the live check even under a caller's budget=); the allowance is
    # skipped when the caller passes its own budget=.
    _bspec0 = (contract.get("strategy", {}) or {}).get("budget") or {}
    if not isinstance(_bspec0, dict):
        return {"status": "invalid_contract",
                "reason": f"strategy.budget must be a mapping (e.g. {{mode: measured, allowance: 72h}}); "
                          f"got {_bspec0!r}"}
    if _bspec0.get("mode") not in (None, "") and \
            str(_bspec0.get("mode")).strip().lower() not in ("measured", "fixed"):
        # a mistyped mode used to become fixed mode silently, ignoring the allowance (Opus round 4)
        return {"status": "invalid_contract",
                "reason": f"strategy.budget.mode {_bspec0.get('mode')!r} is not measured or fixed"}
    if isinstance(_bspec0, dict) and effective_budget_mode(_bspec0) == "measured" and not budget:
        from .budget import parse_allowance as _parse_a0
        _a0 = _bspec0.get("allowance")
        if _a0 in (None, ""):
            return {"status": "invalid_contract",
                    "reason": "strategy.budget.mode is measured but no allowance is given (e.g. allowance: 72h)"}
        _a0s = _parse_a0(_a0)
        if not (math.isfinite(_a0s) and _a0s > 0):
            import re as _re0
            _readable = (isinstance(_a0, (int, float)) and not isinstance(_a0, bool)) or bool(
                _re0.fullmatch(r"-?[0-9]*\.?[0-9]+[smhd]?", str(_a0).strip().lower().replace(" ", "")))
            return {"status": "invalid_contract",
                    "reason": (f"strategy.budget.allowance {_a0!r} must be a positive time" if _readable else
                               f"strategy.budget.allowance {_a0!r} could not be read "
                               f"(use a number with s, m, h or d, e.g. 90m, 1.5h, 3d)")}
    _seeds0 = _bspec0.get("seeds", 3)
    if isinstance(_seeds0, bool) or not isinstance(_seeds0, int) or _seeds0 < 1:
        return {"status": "invalid_contract",
                "reason": f"strategy.budget.seeds must be a positive whole number; got {_seeds0!r}"}
    if effective_budget_mode(_bspec0) == "measured":
        # defaults that break the constraints are a contract error the kit sees WITHOUT a model run;
        # in measured mode the pilot would stop on it anyway, after commissioning, the proof and the
        # pilot (Opus round 5 note b). Fixed mode keeps running (the pilot only warns there).
        try:
            _cok0 = _make_constraints(contract)
            # the default AS THE RUN SEES IT: mapped into the search box and decoded back (integers
            # rounded, bools made true/false) — the same path as the pilot's default (Opus round 6)
            from .evaluator import decode_value as _dv0
            _fwd0, _inv0 = _validate_transforms(params)
            _named0 = {p["name"]: _dv0(p, _fwd0[p["name"]](float(p.get("default", (float(p["range"][0]) +
                                                                                 float(p["range"][1])) / 2.0))),
                                       _inv0[p["name"]]) for p in params}
            if _cok0 is not None and not _cok0(_named0):
                return {"status": "invalid_contract",
                        "reason": "the default parameters break the contract's constraints — fix the defaults "
                                  "or the constraints before searching"}
        except Exception:
            pass                        # a constraint error is reported by the evaluator, as before
    metric_overrides = _effective_metric_overrides(contract, determining_metric)
    _inj_mode = (contract.get("injection", {}) or {}).get("mode", "applicator")   # needed early (probe gate)

    # the runner calls made outside the evaluator (certification, objective probe) are logged too
    # (gap 2j); this call's records start here
    from .evaluator import read_history as _read_history0
    _hist_start0 = len(_read_history0(workdir))
    _n_outside = 0
    _rm_from_contract = run_model is None       # a lane can then build its own runner (step 6b)
    if run_model is None:
        spec = contract.get("runner")
        if not spec:
            return {"status": "no_runner",
                    "reason": "no run_model passed and no `runner` in calibration.yaml"}
        from .runner import make_run_model
        run_model = make_run_model(spec, ki_path, workdir)

    # MODULE CERTIFICATION (top-level only; opt-in via calib/reference_run.json). Prove the
    # auto-generated calibration function reproduces the validated KI run BEFORE any probe /
    # pilot / triage — a module that doesn't execute the KI correctly (missing
    # prepare/routing/staging) is caught here in ONE run, not after a pilot and a search.
    # System 1 (2026-08-29): a SITE contract may carry its own validated reference run
    # (`runner.reference_run: <path>`) — the KI-level calib/reference_run.json is the KI's
    # validation CASE (e.g. VIC = SITE:tangnaihai), which a case-bound driver cannot and must
    # not reproduce. Resolution: contract reference first, else the KI's. A reference whose
    # case_id differs from the contract's target case is refused EXPLICITLY (not a silent
    # "no metrics" failure): the fix is to give the contract its own reference, never to skip.
    _ref, _refuse = resolve_reference_run(contract, ki_path, expected_case_id)
    if _refuse:
        return _refuse
    _state["certified"] = False if not _ref else None
    if _ref:
        from .evaluator import log_runner_call as _log_rc
        _t_c = time.perf_counter()
        _cert = certify_runner(contract, run_model, _ref, workdir, expected_case_id=expected_case_id)
        _ref_split = _ref.get("reference_split")
        _log_rc(workdir, "certify", _n_outside, time.perf_counter() - _t_c, bool(_cert.get("certified")),
                metrics=(_cert.get("observed") if isinstance(_cert.get("observed"), dict) else None),
                reason=str(_cert.get("reason") or "")[:200],
                split=(str(_ref_split).lower() if _ref_split and str(_ref_split).lower() != "full" else "full"))
        _n_outside += 1
        print(f"  [calib] runner certification: {'PASS' if _cert.get('certified') else 'FAIL'} "
              f"— {_cert.get('reason')}", flush=True)
        _state["certified"] = bool(_cert.get("certified"))
        if not _cert.get("certified"):
            return {"status": "runner_uncertified", "certification": _cert,
                    "reason": _cert.get("reason"),
                    "validated_run_id": _ref.get("validated_run_id")}

    # STAGED CALIBRATION IS REMOVED (design §1 "Data within a search", gap 2x; build step 8): which
    # parameters to calibrate is decided once, by the agent from the KI. A contract that still says
    # `staged: true` runs ALL its declared parameters in one search, with a warning.
    _staged_warning = None
    if (contract.get("strategy", {}) or {}).get("staged"):
        _staged_warning = ("strategy.staged is removed (build step 8): all declared parameters are "
                           "calibrated in one search; the Morris screen and escalation rounds no longer run")
        print(f"  [calib] WARNING {_staged_warning}", flush=True)

    objs = O.objectives_from_dag(dag, contract.get("targets"), obs_shape_by_var,
                                metric_overrides=metric_overrides)
    if not objs:
        return {"status": "no_objectives",
                "reason": "no gate-valid metric_families for the targets/obs_shapes "
                          "(check obs_shape_by_var has an entry per target var)"}
    # DECLARED variables (design §1.4), fixed BEFORE the objective probe: the contract's targets
    # that are dag outputs, or (targets omitted) the variables that get an objective. A variable
    # whose objectives the probe drops stays declared and monitored (reported as having none).
    _dag_vars = {o.get("var") for o in (dag.get("outputs") or []) if isinstance(o, dict)}
    if contract.get("targets"):
        _declared_vars = list(dict.fromkeys(str(t["var"]) for t in contract["targets"]
                                            if isinstance(t, dict) and t.get("var") in _dag_vars))
    else:
        # targets omitted: only outputs with their OWN obs entry are declared, and only they get
        # objectives — the one-obs_shape fallback is for variables NAMED in targets (design §1.4)
        objs = [o for o in objs if o.var in (obs_shape_by_var or {})]
        if not objs:
            return {"status": "no_objectives",
                    "reason": "targets omitted and no dag output has its own obs_shape entry"}
        _declared_vars = list(dict.fromkeys(o.var for o in objs))

    # FAMILY AUTO-SELECTION (codex SWAT+ #1): a dag output may declare more gate-valid
    # families than the runner can emit (e.g. flo_out -> temporal+magnitude+TIMING, but
    # the runner only writes nse/pbias, not day_bias). Requiring an un-produced metric
    # makes EVERY eval score +inf. Probe the runner once at defaults and keep only the
    # objectives whose metric the runner actually produces. Skippable via
    # strategy.probe_objectives: false (then all declared families are required).
    # FAMILY AUTO-SELECTION: probe the runner AT DEFAULTS (runner mode: with KDT_CALIB_PARAMS set)
    # and drop objectives whose metric it doesn't emit — else a declared-but-unemitted family scores
    # every eval +inf. Also fail-closed on a wrong/undeclared pinned case seen in the probe.
    _probe = None
    if (contract.get("strategy", {}) or {}).get("probe_objectives", True):
        from .evaluator import log_runner_call as _log_rc
        _t_p = time.perf_counter()
        _probe = _probe_runner_metrics(run_model, _inj_mode, params, workdir)
        from .evaluator import resolve_train_split as _rts0
        _log_rc(workdir, "objective_probe", _n_outside, time.perf_counter() - _t_p, bool(_probe),
                metrics=_probe if isinstance(_probe, dict) else None,
                split=_rts0(os.environ.get("KDT_CALIB_SPLIT")))
        _n_outside += 1
        objs, _perr = _filter_objectives_by_probe(objs, _probe, expected_case_id, _inj_mode)
        if _perr:
            return _perr

    try:
        fwd, inv = _validate_transforms(params)
    except ValueError as e:
        return {"status": "invalid_contract", "reason": str(e)}

    # C1/C7 preflight — abort on any mis-targeted/lossy address (codex :72). APPLICATOR mode only:
    # runner mode has no central addresses to verify (calib_run.py owns injection + its own readback), so a
    # runner KI with optional/no addresses or a runner-only kind must NOT hard-fail here.
    if _inj_mode == "applicator":
        addr_problems = _preflight_addresses(params, workdir)
        if addr_problems:
            return {"status": "address_preflight_failed", "problems": addr_problems}

    # every declared parameter is searched (the caller-subset search of staged calibration is removed,
    # build step 8b); nothing is frozen
    _params = params
    _fixed = {}

    lower = [fwd[p["name"]](float(p["range"][0])) for p in _params]
    upper = [fwd[p["name"]](float(p["range"][1])) for p in _params]
    multi = _is_multi_objective(contract, objs)

    from .evaluator import Evaluator
    ev = Evaluator(ki_path=ki_path, workdir=workdir, parameters=_params,
                   objectives=objs, transform_inv=inv, run_model=run_model,
                   constraints_ok=_make_constraints(contract), base_seed=seed,
                   injection_mode=_inj_mode, fixed_params=_fixed,
                   expected_case_id=expected_case_id)
    # the evaluation log only grows (earlier attempts in this workdir stay in it): the ledger charges
    # the overhead of THIS call only (design §1.3 "charged once")
    from .evaluator import read_history as _read_history, split_provenance as _split_prov
    _hist_start = _hist_start0                      # includes the certification / probe runs above
    ev._hist_n = _n_outside                         # one numbering for this call's records

    # ── METRIC PANEL (2026-09-27) ───────────────────────────────────────────────────────
    # Every evaluation's history record carries the panel — r, alpha, beta, pbias, nse, kge,
    # lnnse — so "where did the fit stop moving?" can be answered per metric afterwards, not
    # only for the headline one. Wired before commissioning so the pilot and the proof runs are
    # recorded the same way as the search.
    # Each variable's KIND (flow / series / snapshot / categorical, design §1.4) decides which
    # metrics it records and requires: stated in strategy.convergence.variables.<var>.kind, else
    # taken from the matched dag entry. The KI's validation convention decides whether a runner's
    # plain `pbias` is the kit's percent PBIAS (KI files are read, never changed).
    from .panel import Panel as _Panel, setup_kinds as _setup_kinds
    _cspec = (contract.get("strategy", {}) or {}).get("convergence") or {}
    _pvars = list(dict.fromkeys(_declared_vars + [o.var for o in objs]))
    _vars_without_objective = [v for v in _pvars if v not in {o.var for o in objs}]
    _conv_yaml = Path(ki_path) / "docs" / "validation_convention.yaml"
    try:
        _conv = _yaml.safe_load(_conv_yaml.read_text()) if _conv_yaml.exists() else None
    except Exception:
        _conv = None
    try:
        _kset = _setup_kinds(_pvars, dag, obs_shape_by_var, contract, _conv)
    except ValueError as _e:
        return {"status": "invalid_contract", "reason": str(_e)}
    for _w in _kset["warnings"]:
        print(f"  [calib] kind: {_w}", flush=True)
    if _cspec.get("panel"):
        print("  [calib] strategy.convergence.panel is ignored: each variable's panel comes from "
              "its kind (design §1.4)", flush=True)
    _panel = _Panel(_pvars, _cspec.get("panel"), kinds=_kset["kinds"],
                    pbias_percent=_kset["pbias_percent"], pbias_unit=_kset["pbias_unit"])
    ev.panel_extract = _panel.extract
    # strategy.protect load check (design §1.4): a declared target, and metrics its kind records
    from .rule import check_protect as _check_protect
    try:
        _protect_spec = _check_protect((contract.get("strategy", {}) or {}).get("protect"), _panel,
                                       declared=_declared_vars)
    except ValueError as _e:
        return {"status": "invalid_contract", "reason": str(_e)}

    # COMMISSIONING (runner mode): round-trip proves the value was written; this proves the delegated
    # injection actually reaches the SCORED run. Default vs a clearly-different vector — the objective must
    # move, else calib_run.py isn't injecting into the file the model reads. (applicator mode is already
    # round-trip-verified by _preflight_addresses.)
    _smoke, _cproof = None, None      # recorded in the final report (checkpoint 2026-09-16)
    if _inj_mode == "runner":
        _xd = [fwd[p["name"]](float(p.get("default", (float(p["range"][0]) + float(p["range"][1])) / 2.0)))
               for p in _params]
        _resp, _detail = ev.responsiveness_check(_xd, lower, upper)
        # WRONG-GAUGE GUARD (codex 2026-07-18): commissioning ran the runner at defaults, so if it
        # self-declared a case_id different from the target, abort NOW — never optimize the wrong site.
        if ev._wrong_case:
            return {"status": "runner_wrong_case", "injection_mode": "runner",
                    "expected_case_id": expected_case_id, "declared_case_id": ev._wrong_case,
                    "reason": f"runner scored case '{ev._wrong_case}' but the target case is "
                              f"'{expected_case_id}' — capability was built for the wrong gauge/obs"}
        # 2026-09-16 (plan checkpoint, codex+kimi C1): this objective-only check is a SMOKE TEST, never
        # a veto (plan v2 #7; plan v1 "does not fail on rounded scores. Ever."). It used to abort here,
        # BEFORE the raw-series gate below — so a box whose raw output moves but whose ROUNDED
        # objectives do not was killed as "unresponsive" and OBJECTIVE_MASKED never got a turn. The
        # raw-series gate is the evidence; this result is recorded in the report and only warned on.
        _smoke = {"responsive": _resp, "detail": _detail}
        if _resp is False:
            print(f"  [calib] WARNING objective smoke test: no objective moved for any single-parameter "
                  f"displacement ({_detail}). Not a stop — the raw-series consumption proof decides.", flush=True)

        # CONSUMPTION PROOF v2 (2026-09-14, plan v2, owner-approved). The check above passes if ANY
        # parameter moves ANY objective, so eight working knobs hide three dead ones. This one asks
        # EVERY parameter, against the RAW scored series the runner emits (kdt-target-proof/1), so
        # rounded metrics cannot hide a dead address or fake one.
        # OWNER RULE: an UNREACHABLE parameter STOPS the fit. No auto-repair. The KI is fixed by hand.
        _gate = _consumption_gate(ev, contract, ki_path, workdir, _xd, lower, upper,
                                  [p["name"] for p in _params], _params)
        if _gate.get("status") == "param_unreachable":
            return _gate
        _cproof = _gate.get("consumption_proof")

    # ── PILOT: measure the model before deciding how long to search it (§5.3) ───────────
    # Ten timed evaluations. They pay for themselves three times over: the run time that turns a
    # wall-clock allowance into an evaluation cap, the stability check that catches a broken
    # range before a long search, and the paired series the metric tolerances are measured from.
    # Cached like any other evaluation, so the search re-uses the points.
    from .budget import estimate_overhead_evals, plan_budget
    from .compute import probe_lanes
    from .pilot import pilot_tolerances, run_pilot
    _bspec = (contract.get("strategy", {}) or {}).get("budget") or {}
    # SEEDS (design §1.7, build step 6): 3 slots by default
    _n_seeds = int(_bspec.get("seeds", 3))
    _x_default = [fwd[p["name"]](float(p.get("default",
                      (float(p["range"][0]) + float(p["range"][1])) / 2.0))) for p in _params]
    # (Since build step 5 a legacy contract runs the pilot too — design gap 2k. Its ten extra live
    # evaluations advance KDT_CALIB_EVAL_ID, so a stochastic runner's search differs from the runs that
    # contract produced with kit 0.3.0; the CHANGELOG says so.)
    # PILOT ALWAYS ON (design §1.3, gap 2k): the first pilot run is the default parameter set, which
    # anchors protection and tolerances, so every calibration has one — caller-given budgets too.
    # 10 runs by default; 3 when the user gave a typical run time (strategy.budget.run_time).
    from .budget import parse_allowance as _parse_t
    _run_time_s = _parse_t(_bspec.get("run_time")) or None
    _pilot_n = int(_bspec.get("pilot_runs", 3 if _run_time_s else 10))
    _budget_warn: list = []
    if _bspec.get("run_time") is not None and not _run_time_s:
        _budget_warn.append(f"strategy.budget.run_time {_bspec.get('run_time')!r} could not be read "
                            f"(use e.g. 90s, 2m, 1.5h) — ignored; the pilot's own timing is used")
    if _pilot_n < 1:
        _budget_warn.append(f"strategy.budget.pilot_runs = {_pilot_n}: the pilot is always on (its default run "
                            f"anchors protection and tolerances) — running the default run only")
        _pilot_n = 1
    _pilot = None
    # RESUME (2026-09-27): a restarted calibration must NOT re-pay the pilot, and — more
    # importantly — must not get a different cap. DDS schedules its perturbation probability
    # against the declared budget, so a resumed run with a re-measured cap would be a different
    # algorithm mid-search. The first run writes its plan next to the cache; a resume reuses it.
    _plan_file = Path(workdir) / "kdt_budget_plan.json"
    # the plan is only reusable for the SAME setup: a changed contract, case, parameter subset or
    # algorithm choice is a different search and must be re-planned (codex review 2026-09-27)
    def _plan_identity():
        import hashlib as _hl
        h = _hl.sha1()
        # the CONTENTS of the contract actually in force, not the path it came from: a site
        # contract edited in place used to keep its identity and reuse an obsolete cap (codex r2)
        try:
            h.update(json.dumps(contract, sort_keys=True, default=str).encode())
        except Exception:
            h.update(b"\x00")
        h.update(str(os.environ.get("KDT_CALIB_CASE_TAG", "")).encode())
        h.update(str(expected_case_id or "").encode())
        h.update(",".join(p["name"] for p in _params).encode())
        h.update(str(_inj_mode).encode())
        # the number of seeds changes the cap per seed (and the default went 1 -> 3 in build step 6):
        # a plan saved for another count must be re-planned, never resumed (Opus step 6 r1)
        h.update(f"seeds={_n_seeds}".encode())
        # the RUNNER's own code (3c, 2026-10-02): a pilot measured with one runner is never reused for another —
        # after a runner is updated (the reply upgrade) the next call plans afresh; with the same runner, the
        # pilot of the reply gate / check call is reused by the search instead of being run again
        try:
            _rs = contract.get("runner") or {}
            _rcwd = str(_rs.get("cwd") or "{ki_path}").replace("{ki_path}", str(ki_path))
            for _tok in (_rs.get("command") or []):
                _t = str(_tok).replace("{ki_path}", str(ki_path))
                if _t.endswith(".py"):
                    _f = Path(_t) if os.path.isabs(_t) else Path(_rcwd) / _t
                    h.update(b"runner:" + (_f.read_bytes() if _f.is_file() else b"<missing>"))
        except Exception:
            h.update(b"\x01")
        return h.hexdigest()
    _plan_id = _plan_identity()
    _saved_plan = None
    if _plan_file.exists():
        try:
            _cand = json.loads(_plan_file.read_text())
            if isinstance(_cand, dict) and _cand.get("setup_id") == _plan_id:
                # a hand-edited or truncated plan must re-plan, never crash the run
                _cand["cap_per_seed"] = int(_cand.get("cap_per_seed") or 0)
                if _cand.get("effective_cap") is not None:
                    _cand["effective_cap"] = int(_cand["effective_cap"])
                if not isinstance(_cand.get("pilot_summary"), dict):
                    _cand["pilot_summary"] = None      # anything else is not a pilot measurement
                if not isinstance(_cand.get("warnings"), list):
                    _cand["warnings"] = []
                _cand["allowance_s"] = float(_cand.get("allowance_s") or 0.0)
                _t90 = float(_cand.get("t_p90_s") or 0.0)
                _cand["t_p90_s"] = _t90
                _eff_cap = _cand.get("effective_cap")
                _dnf = (_cand.get("pilot_summary") or {}).get("default_n_finite")
                if (_cand["cap_per_seed"] <= 0 or not math.isfinite(_t90) or _t90 < 0
                        or (_eff_cap is not None and int(_eff_cap) <= 0)
                        # a plan whose pilot default gave no finite objective (triage no_baseline) is never
                        # reused: the runner may have been fixed since (Opus step 8a r1)
                        or _dnf == 0):
                    print(f"  [calib] budget plan in {_plan_file.name} is not usable "
                          + ("(its pilot's default run gave no finite objective metric) " if _dnf == 0 else
                             f"(cap {_cand['cap_per_seed']}, t_p90 {_t90}) ")
                          + "— re-planning", flush=True)
                else:
                    _saved_plan = _cand
            else:
                print(f"  [calib] budget plan in {_plan_file.name} is for a DIFFERENT setup "
                      f"(contract/case/parameters changed) — re-planning", flush=True)
        except Exception:
            _saved_plan = None
    # where the pilot this search rests on came from (split provenance, build step 7)
    _pilot_source = "plan" if _saved_plan else "here"
    if _saved_plan:
        _pilot = _saved_plan.get("pilot_summary")
        print(f"  [calib] budget plan REUSED from {_plan_file.name} (resume): cap "
              f"{_saved_plan.get('cap_per_seed')} — the pilot is not re-run and the cap does not move",
              flush=True)
    elif _pilot_n > 0:
        _pilot = run_pilot(ev, lower, upper, _x_default, n=_pilot_n, seed=seed)
        if _pilot.get("status") == "pilot_unstable":
            # More than a third of the pilot failed. In MEASURED mode that is a stop: the cap is
            # derived from these timings and a search would spend the allowance failing. In fixed /
            # legacy mode it is a WARNING and the run continues — a contract that searched fine for
            # months must not start refusing because a Latin-hypercube corner is infeasible
            # (codex review 2026-09-27). The default vector failing is reported either way.
            if effective_budget_mode(_bspec) == "measured":
                ev.restore_originals()
                return {"status": "pilot_unstable", "reason": _pilot.get("reason"),
                        "pilot": {k: v for k, v in _pilot.items() if k != "records"},
                        "phase_counts": phase_counts(workdir)}
            print(f"  [calib] WARNING pilot: {_pilot.get('reason')} — continuing because this "
                  f"contract declares a fixed budget (the cap does not depend on the pilot)",
                  flush=True)
            _budget_warn.append(f"pilot: {_pilot.get('reason')} — continuing (fixed budget)")

    _eval_losses: list = []            # per-objective loss vector per SEARCH eval, in order —
    #                                    the backend history only keeps a scalar, and a
    #                                    multi-objective rule needs the vector (rule.py)
    def _evaluate(x):
        losses = ev.evaluate(x)
        _eval_losses.append([float(l) for l in losses])
        if not multi:                  # scalarize same-target families
            tot = sum(o.weight for o in objs) or 1.0
            return [sum(o.weight * l for o, l in zip(objs, losses)) / tot]
        return losses

    problem = Problem(
        names=[p["name"] for p in _params], lower=lower, upper=upper,
        objective_names=[o.name for o in objs] if multi else ["scalar_loss"],
        evaluate=_evaluate, is_multi_objective=multi)

    algo = _pick_backend(contract, multi, len({o.var for o in objs}))
    from .verdict import TRACKED_BACKENDS as _TRACKED
    if algo not in _TRACKED and _n_seeds > 1:
        # the seed rules read each seed's calls through the rule; a backend that does not report each
        # call cannot be compared or picked on them (design §1.8 "convergence not tracked") — one seed
        print(f"  [calib] {algo}: convergence not tracked, so seeds are not compared — running ONE seed "
              f"instead of {_n_seeds}", flush=True)
        _seeds_note = f"{algo} does not report each call: one seed run instead of {_n_seeds} (seeds not compared)"
        _n_seeds = 1
    else:
        _seeds_note = None
    # ── BUDGET: a CAP derived from the pilot, the machine and the user's allowance (§5.5) ──
    # Precedence: an explicit budget= from the caller (a launcher that knows what it is doing)
    # wins; then a measured cap from strategy.budget.allowance; then the contract's declared
    # max_evaluations, which is what every pre-2026-09 contract has and keeps.
    _declared_max = (contract.get("strategy", {}) or {}).get("max_evaluations", 200)
    # An `allowance` with no `mode` used to be silently ignored — the contract asked for a measured
    # budget and got the hand-written one (kimi review 2026-09-27). An allowance IS the request.
    _bmode = effective_budget_mode(_bspec)
    # the split record of this call (build step 7); defined before the triage, which may return early
    _probe_used_for_pilot = False
    _panel_setup_reused = False

    def _prov_now():
        """Split provenance of this call's records (design §1 "Data within a search", gap 2p). A resumed
        search rests on the saved pilot (and on the objective probe, when its payload set the saved
        pilot decisions): their saved records are used, or the claim is None."""
        _saved, _unchecked = {}, {}
        _probe_claim = _probe_used_for_pilot
        if _pilot_source == "plan":
            _sp = _saved_plan if isinstance(_saved_plan, dict) else {}
            if "pilot_split_provenance" in _sp:
                _saved["pilot"] = _sp.get("pilot_split_provenance")
            else:
                _unchecked["pilot"] = "pilot not checked (resumed from a saved plan that has no split record for it)"
            _flag = _sp.get("probe_used_for_pilot")
            _probe_claim = _flag is True
            if not _panel_setup_reused:
                # the decisions were re-derived in THIS call (from the saved pilot and this call's
                # probe): this call's flag and this call's probe records are the ones that count
                _probe_claim = _probe_used_for_pilot
            elif not isinstance(_flag, bool):
                # an older or hand-edited plan: whether the probe set the decisions is not known
                _probe_claim = True
                _unchecked["objective_probe"] = "objective probe not checked (the saved plan does not say " \
                                                "whether its payload set the pilot decisions)"
            elif _probe_claim:
                if "probe_split_provenance" in _sp:
                    _saved["objective_probe"] = _sp.get("probe_split_provenance")
                else:
                    _unchecked["objective_probe"] = "objective probe not checked (no saved split record)"
        _cp = ("pilot", "search") + (("objective_probe",) if _probe_claim else ())
        return _split_prov(_read_history(workdir)[_hist_start0:], claim_phases=_cp, saved=_saved,
                           unchecked=_unchecked)

    # ── TRIAGE (design §1 "Triage before the search"; gap 2x; build step 8) ──────────────────────────
    # always calibrate unless the run cannot be calibrated. The default run's standards (§2.9) and the fit
    # verdict per target variable are reported as INFORMATION only — never a reason to skip.
    _triage = triage_route(_pilot, _cproof, [p["name"] for p in _params])
    _triage["per_target"] = {}
    try:
        from .verdict import standards as _standards_t
        _dm_t = (_pilot or {}).get("default_metrics") if isinstance(_pilot, dict) else None
        for _v in _panel.variables:
            # an old strategy.stop floor band is the standards band for its variable (design §5.1b), as in
            # the search's own standards check
            _bov = _band_override(_stop_translated, _v)
            _fams = [o.family for o in objs if o.var == _v]
            _mv = ((_dm_t or {}).get(_v) if isinstance((_dm_t or {}).get(_v), dict) else (_dm_t or {}))
            _fv = O.calibration_verdict(_fams, _mv, metric_overrides=metric_overrides) if _fams else None
            _triage["per_target"][_v] = {
                "fit_verdict": ((_fv or {}).get("decision") if _fv else "no objective"),
                # INFORMATION ONLY: the search runs whatever this says (design §1)
                "fit_why": (("information only (never a reason to skip the search): " + str(_fv.get("why")))
                            if (_fv or {}).get("why") else None),
                "default_standards": _standards_t(_conv, _v, (_kset.get("shapes") or {}).get(_v), _dm_t,
                                                  len(_panel.variables) == 1, _bov)}
    except Exception as _te:
        _triage["per_target_error"] = f"{type(_te).__name__}: {_te}"
    print(f"  [calib] triage: {_triage['route']} — {_triage['reason']}", flush=True)
    if _triage["route"] != "calibrate":
        ev.restore_originals()
        # before the machine probe, the budget plan and its file: nothing of a no-search run is saved
        # for a resume, so a fixed runner re-plans and re-runs the pilot (Opus step 8a r1)
        return {"status": _triage["route"], "route": _triage["route"], "reason": _triage["reason"],
                "triage": _triage, "algorithm": algo, "staged_warning": _staged_warning,
                "pilot": ({k: v for k, v in _pilot.items() if k != "records"} if isinstance(_pilot, dict) else None),
                "phase_counts": phase_counts(workdir), "split_provenance": _prov_now(),
                "consumption_proof": _cproof, "consumption_coverage": consumption_coverage(_cproof, len(_params)),
                "objective_smoke_test": _smoke}

    # MACHINE PROBE (gap 2m): L and e are MEASURED only for a runner whose copies are clearly
    # separate — a `subprocess` runner in runner-mode injection whose contract says
    # parallel_safe: true. Anything else stays at 1 lane (design: "1 until measured").
    _clone_eval = None
    _psafe = _bspec.get("parallel_safe") is True
    # the probe runs extra model runs: only when lanes could be used at all (codex 6b r1 #1) — several seeds, a
    # runner the kit builds from the contract as a subprocess, a tracked backend, fork, one Python thread, and not
    # a daemonic worker. Otherwise it is skipped and the seeds run one after another.
    _no_lane_why = None
    if _psafe and _inj_mode == "runner" and not _saved_plan:
        import threading as _thrp
        import sys as _sysp
        import multiprocessing as _mpp
        if _n_seeds <= 1:
            _no_lane_why = "one seed"
        elif not _rm_from_contract:
            _no_lane_why = "the caller passed its own run_model"
        elif str((contract.get("runner") or {}).get("kind")) != "subprocess":
            _no_lane_why = "the runner is not a subprocess"
        elif algo not in _TRACKED:
            _no_lane_why = f"backend {algo} does not report each call"
        elif not hasattr(os, "fork"):
            _no_lane_why = "no fork on this system"
        elif max(_thrp.active_count(), len(_sysp._current_frames())) > 1:
            _no_lane_why = "this process has other Python threads"
        elif _mpp.current_process().daemon:
            _no_lane_why = "this process is a daemonic worker"
        if _no_lane_why:
            _budget_warn.append(f"machine probe skipped ({_no_lane_why}): lanes cannot be used, so no extra model "
                                f"runs are spent measuring them — 1 lane")
    if _psafe and _inj_mode == "runner" and not _saved_plan and not _no_lane_why:
        try:
            from .compute import make_clone_and_eval as _mk_clone
            _named_default = ev._named(_x_default)          # the real run's types and fixed params
            _clone_raw = _mk_clone(contract.get("runner") or {}, ki_path, workdir, _named_default)
            if _clone_raw is not None:
                # every lane run is a model run: logged under `machine_probe` (build step 7, gap 2j);
                # the lanes run at once, so the numbering takes a lock
                import threading as _thr
                from .evaluator import log_runner_call as _log_mp
                _mp_lock = _thr.Lock()

                def _clone_eval(i, _raw=_clone_raw):
                    _t = time.perf_counter()
                    _ok = False
                    try:
                        _w = _raw(i)
                        _ok = True
                        return _w
                    finally:
                        with _mp_lock:
                            _log_mp(workdir, "machine_probe", ev._hist_n, time.perf_counter() - _t, _ok,
                                    split="calibration", reason="lane run (metrics not read)")
                            ev._hist_n += 1
                _clone_eval.cleanup = _clone_raw.cleanup
        except Exception as _e:
            _budget_warn.append(f"machine probe not possible ({type(_e).__name__}: {_e}) — 1 lane")
    try:
        _lanes = probe_lanes(_pilot or {}, parallel_safe=_psafe,
                             runner_mode=str((contract.get("runner") or {}).get("kind", "")),
                             clone_and_eval=_clone_eval,
                             # also measure at the seed count, so the plan's e is the one the seeds run at
                             ks=tuple(sorted({1, 2, 4} | ({int(_n_seeds)} if _n_seeds > 1 else set()))))
    finally:
        if _clone_eval is not None and hasattr(_clone_eval, "cleanup"):
            _clone_eval.cleanup()
    _overhead = estimate_overhead_evals(len(_params),
                                        runner_mode=_inj_mode, pilot_n=(_pilot or {}).get("n", 0),
                                        consumption_proof=bool(_cproof))
    # SHARED OVERHEAD, measured (design §1.3): the time already spent on commissioning, the
    # consumption proof and the pilot (from the evaluation log), plus the machine probe. Charged once.
    _pc_now = phase_counts(workdir, records=_read_history(workdir)[_hist_start:])
    _shared_parts = {ph: float((_pc_now.get(ph) or {}).get("wall_s") or 0.0)
                     for ph in ("certify", "objective_probe", "commission", "consume_proof", "proof", "pilot")}
    _shared_parts["machine_probe"] = float(((_lanes.get("measured") or {}).get("probe_wall_s")) or 0.0)
    # the seeds run ONE AFTER ANOTHER in this kit version (parallel seed lanes are build step 6b), so
    # the plan counts one lane for them — a plan that assumed parallel seeds would overrun the allowance
    # SEED LANES (design §1.3, §1.7; build step 6b): seeds run min(S, L) at a time in separate processes on
    # cloned workdirs — only when the lanes were MEASURED (a parallel_safe subprocess runner), the runner can
    # be rebuilt per lane from the contract, and the backend reports each call. Otherwise one at a time.
    _L = int(_lanes.get("lanes") or 1)
    if isinstance(_saved_plan, dict):                # a resume does not re-probe: the saved lanes count
        try:
            _L = max(_L, int(_saved_plan.get("seeds_parallel") or 1))
        except (TypeError, ValueError):
            pass
    _seed_par = (min(_n_seeds, _L) if (_n_seeds > 1 and _L > 1 and _rm_from_contract
                                        and str((contract.get("runner") or {}).get("kind")) == "subprocess"
                                        and algo in _TRACKED and hasattr(os, "fork")) else 1)
    _lane_notes: list = []
    if _seed_par > 1:
        # a forked lane copies every lock as it was at the fork: with other Python threads running (an
        # orchestrator's thread pool printing), a lane can hang forever on one (Opus 6b r1). Lanes only in a
        # one-thread process.
        import threading as _thr0
        import sys as _sys0
        _nthr = max(_thr0.active_count(), len(_sys0._current_frames()))
        if _nthr > 1:
            _lane_notes.append(f"this process has {_nthr} Python threads — forking lanes is not safe; the seeds "
                               f"run one after another")
            _seed_par = 1
        else:
            import multiprocessing as _mp0
            if _mp0.current_process().daemon:        # e.g. a multiprocessing.Pool worker: it may not have children
                _lane_notes.append("this process is a daemonic worker (e.g. multiprocessing.Pool) — it cannot start "
                                   "lanes; the seeds run one after another")
                _seed_par = 1
    _eff_map = ((_lanes.get("measured") or {}).get("efficiency") or {})
    _eff_at = _eff_map.get(_seed_par, _eff_map.get(str(_seed_par)))
    if _seed_par > 1 and _eff_at is None:
        _lane_notes.append(f"efficiency was not measured at {_seed_par} lanes; the plan uses e measured at "
                           f"{_L} lanes ({_lanes.get('efficiency')})")
    _seed_eff = (float(_eff_at if _eff_at is not None else (_lanes.get("efficiency") or 1.0))
                 if _seed_par > 1 else 1.0)
    budget_plan = plan_budget(_bspec.get("allowance"), _pilot or {}, lanes=_seed_par,
                              efficiency=_seed_eff, n_params=len(_params),
                              n_seeds=_n_seeds,
                              overhead_evals=_overhead["total"], hard_max=_bspec.get("hard_max"),
                              # measured: max_evaluations is an OPTIONAL ceiling — only a declared one
                              # applies (the 200 fallback is for fixed mode only)
                              mode=_bmode, max_evaluations=(_declared_max if _bmode != "measured" else
                                                            (contract.get("strategy", {}) or {}).get("max_evaluations")),
                              shared_overhead_s=sum(_shared_parts.values()), run_time_s=_run_time_s)
    budget_plan["ledger"]["shared_parts_s"] = dict(_shared_parts)
    budget_plan["ledger"]["shared_parts_note"] = ("this call's records only; the report's phase_counts "
                                                  "covers the whole evaluation log of the workdir")
    if (_pilot or {}).get("warning"):
        _budget_warn.append(_pilot["warning"])
    if _run_time_s and (_pilot or {}).get("max_s") and float(_run_time_s) > 2.0 * float(_pilot["max_s"]):
        _budget_warn.append(f"the given run time ({_run_time_s:.1f} s) is more than twice the slowest pilot run "
                            f"({float(_pilot['max_s']):.1f} s); the plan uses the given time (conservative)")
    budget_plan["warnings"] = _budget_warn + budget_plan["warnings"]
    budget_plan["machine"] = _lanes["machine"]
    budget_plan["lane_notes"] = list(dict.fromkeys(list(_lanes["notes"]) + _lane_notes))
    budget_plan.setdefault("warnings", []).extend(_lane_notes)
    budget_plan["lane_probe"] = _lanes.get("measured")      # per-k efficiency and any probe errors
    # an old evaluation-count estimate, kept for reading only: the ledger charges MEASURED time
    budget_plan["overhead"] = dict(_overhead, note="estimate only — not charged; see ledger.shared_parts_s")
    if _saved_plan:
        budget_plan = dict(_saved_plan)
        budget_plan["reused_from_workdir"] = str(_plan_file)
        # the saved plan's lane count is what the cap was planned with; say so when this call runs fewer
        # lanes at once (e.g. a threaded host), and why (Opus 6b r2 #3)
        _rn = list(_lane_notes)
        try:
            _sp_saved = int(_saved_plan.get("seeds_parallel") or 1)
        except (TypeError, ValueError):
            _sp_saved = 1
        if _sp_saved != _seed_par:
            _rn.append(f"this resume runs {_seed_par} seed(s) at a time; the saved plan (and its cap) counted "
                       f"{_sp_saved}")
        if _rn:
            budget_plan["warnings"] = list(budget_plan.get("warnings") or []) + _rn
            budget_plan["lane_notes"] = list(dict.fromkeys(list(budget_plan.get("lane_notes") or []) + _rn))
    else:
        budget_plan["pilot_summary"] = {k: v for k, v in (_pilot or {}).items()
                                        if k not in ("records",)}
        budget_plan["setup_id"] = _plan_id
        # written AFTER the caller's override is resolved, below, so a resume reuses the cap the
        # search was actually run with — DDS schedules against that number.
    # CAP PRECEDENCE, resolved ONCE (codex round 2: the effective cap was reported before the
    # saved plan overrode it, so a resume reported one number and searched with another):
    #   1. an explicit budget= from the caller;
    #   2. the cap this workdir's plan was already RUN with (a resume must not move it);
    #   3. the cap measured now;
    #   4. the contract's declared max_evaluations (fixed / legacy).
    if budget:
        budget_plan["overridden_by_caller"] = int(budget)
        if budget_plan.get("refuse"):
            budget_plan["refuse"] = False
            budget_plan["warnings"].append(f"the measured plan would have refused ({budget_plan.get('refuse_reason')}); "
                                           f"the caller's budget of {int(budget)} is used instead")
    elif _saved_plan and _saved_plan.get("effective_cap"):
        budget = int(_saved_plan["effective_cap"])
        budget_plan["resumed_cap"] = int(budget)
    elif budget_plan.get("refuse"):
        # design §1.3: a measured cap under 20 calls is refused, and says why
        ev.restore_originals()
        return {"status": "budget_exhausted", "algorithm": algo,
                "reason": budget_plan.get("refuse_reason"),
                "budget_plan": budget_plan, "phase_counts": phase_counts(workdir)}
    elif budget_plan["cap_per_seed"] > 0:
        budget = budget_plan["cap_per_seed"]
    elif _bmode == "measured":
        # the allowance did not cover even a minimal search of this model. Falling back to the
        # declared max_evaluations would launch a search the user never granted time for, and would
        # also sail past hard_max (codex review 2026-09-27) — refuse and say what is needed.
        ev.restore_originals()
        return {"status": "budget_exhausted", "algorithm": algo,
                "reason": (f"the granted allowance ({budget_plan['allowance_s']:.0f} s) does not "
                           f"cover a search of this model at {budget_plan['t_p90_s']:.1f} s per "
                           f"evaluation plus {_overhead['total']} overhead evaluations"),
                "budget_plan": budget_plan, "phase_counts": phase_counts(workdir)}
    else:
        budget = _declared_max
        budget_plan["warnings"].append(f"no usable cap computed — using the contract's "
                                       f"max_evaluations {_declared_max}")
    budget_plan["effective_cap"] = int(budget)     # what the search below actually uses
    if not _saved_plan:
        try:
            _plan_file.write_text(json.dumps(budget_plan, indent=2, default=str))
        except Exception:
            pass
    backend = _make_backend(algo)
    if backend is None or not backend.available():
        ev.restore_originals()          # the pilot may have written pilot values into the inputs
        return {"status": "backend_unavailable", "algorithm": algo,
                "reason": f"{algo} backend/library not importable in this env"}

    # what a backend may use beyond the Problem (all optional; spotpy ignores them): the full metrics
    # of an evaluation (agentic rung needs the diagnosis), the native parameter specs + transforms
    # (so an LLM reasons in mm / 1/d while the kit searches its transformed box), and the workdir.
    def _metrics_of(_i, _ev=ev):
        for _sp in ("calibration", None):
            _mt = getattr(_ev, "_last_metrics", {}).get(_sp)
            if _mt:
                return _mt
        return None
    _bk = dict(metrics_of=_metrics_of, param_specs=_params, transforms=(fwd, inv), workdir=workdir)
    # DDS can start at the model's own defaults instead of a random draw — but that CHANGES the
    # search trajectory, so it is opt-in (strategy.budget.start_at_default). Passing it by default
    # would make an existing contract's run differ from every run it has already produced at the
    # same seed and budget (codex review 2026-09-27).
    if bool(_bspec.get("start_at_default", False)):
        _bk["x_initial"] = _x_default
    # pymoo's own rule as shipped (period 50, n_skip 5; 2026-10-04) unless the contract sets it
    if _cspec.get("front_period") is not None:
        _bk["front_period"] = int(_cspec["front_period"])
    if _cspec.get("front_ftol") is not None:
        _bk["front_ftol"] = float(_cspec["front_ftol"])

    # ── CONVERGENCE (design §1-§2; build steps 1-4) ─────────────────────────────────────────────
    # SCE-UA settings per mode, and what SPOTPY is told as repetitions (the kit's call hook enforces
    # the cap; design §1.6, gap 2b/2h). DDS and DREAM get exactly the cap.
    from .rule import STOPPABLE as _STOPPABLE, sceua_settings as _sceua_settings
    if _cmode == "stop" and algo in ("dds", "dream"):
        _cwarn.append(f"mode stop does not end {algo}: " + (
            "DDS's search is planned around its budget (Tolson & Shoemaker 2007); for a search that can "
            "stop at convergence use a smaller DDS budget or SCE-UA / NSGA-II" if algo == "dds"
            else "DREAM's own R-hat test decides (design §5.1b)") + "; our stop point is recorded only")
        print(f"  [calib] {_cwarn[-1]}", flush=True)
    if algo == "sceua":
        _sce = _sceua_settings(_cmode, int(budget), len(_params),
                               rel_gain=float(_cspec.get("rel_gain", 0.005)))
        _bk.update({k: _sce[k] for k in ("ngs", "kstop", "pcento", "peps", "repetitions")})
    elif algo in ("dds", "dream"):
        _bk["repetitions"] = int(budget)
    # Tolerances (design §2.11) and the pilot decisions (design §1.4) come from the pilot's
    # DEFAULT run, once: flow->series when the log is undefined, "mean near zero" metrics set
    # missing by the kit. Every variable gets a tolerance record, measured or labelled fixed.
    from .panel import FIXED_FALLBACK_TOL, pilot_decisions as _pilot_decisions
    from .pilot import pilot_series as _pilot_series
    _tol, _tol_src, _no_metrics, _tol_rec, _pilot_notes = ({}, "none", [], {}, [])
    _default_panel, _default_panel_known, _anchor_step_reached = {}, False, False
    _probe_used_for_pilot = False
    _panel_setup_reused = False              # True: the saved pilot decisions (and probe flag) were reused
    _default_panel_why = "no pilot default run to anchor it"
    _cross_2n: dict = {}                     # gap 2n: the runner's panel vs the kit's recomputation
    _reply_state = None                      # does the runner's reply give the convergence check what it needs
    try:
        _saved_ps = (_saved_plan or {}).get("panel_setup") if isinstance(_saved_plan, dict) else None
        if isinstance(_saved_ps, dict) and _saved_ps.get("variables") == list(_panel.variables):
            # RESUME: the frozen pilot decisions and tolerances are reused, never re-derived, so
            # the required set and the tolerances cannot change between the halves of one search.
            # Marked FIRST: once saved decisions may have been applied, the saved flag is the one
            # that describes them (even if this branch fails part-way; Opus step 7 r5)
            _panel_setup_reused = True
            _pilot_notes = _panel.apply_pilot({"kinds": _saved_ps.get("kinds") or {},
                                               "kit_missing": _saved_ps.get("kit_missing") or {},
                                               "notes": _saved_ps.get("pilot_notes") or []})
            _reply_state = _saved_ps.get("reply")
            if "cross_check_2n" in _saved_ps:
                _cross_2n = _saved_ps.get("cross_check_2n") or {}
            else:                                # a plan saved before build step 9 (Opus 8b/9 r1 #6)
                _cross_2n = {v: {"status": "not checked (the saved plan predates the 2n check)"}
                             for v in _panel.variables}
                _pilot_notes = list(_pilot_notes) + [
                    "2n cross-check not run: this resume reuses a plan saved before it, so the runner's metric "
                    "values are used unchecked"]
            _tol_rec = _saved_ps.get("tolerance_records") or {}
            _tol_src = _saved_ps.get("tolerance_source") or "none"
            _anchor_step_reached = True
            if "default_panel" in _saved_ps:
                _default_panel, _default_panel_known = (_saved_ps.get("default_panel") or {}), True
            else:
                _default_panel_why = "resumed from a plan saved without a default panel"
            print(f"  [calib] panel setup REUSED from {_plan_file.name} (resume)", flush=True)
        else:
            _why_no = {}
            _ser = _pilot_series(_pilot, _panel.variables, _why_no) if _pilot else {}
            _ser_have, _ser_why0 = set(_ser), dict(_why_no)       # before a `tolerance: fixed` contract clears them
            if not _pilot:
                _why_no = {v: "no pilot run" for v in _panel.variables}
            _pilot_notes = _panel.apply_pilot(_pilot_decisions(_ser, _panel.kinds))
            # gap 2n (Leo, 2026-09-29): the runner's own panel values at the default run against the
            # kit's recomputation from the SAME series; a difference above the recorded precision
            # (+1e-6) makes that metric missing for the whole search, with the reason
            from .panel import cross_check_2n as _xc2n
            # each metric's recorded precision is read over the pilot's runs (Leo, 2026-09-30): the values
            # the panel took from the runner at THIS pilot's calls — a pilot run here is this call's records, never
            # an earlier setup's in the same workdir (Opus 8b/9 r3 #3); a pilot reused from the plan: the log
            _prec_from: dict = {}
            _hist_all = _read_history(workdir)
            for _hr in (_hist_all[_hist_start0:] if _pilot_source == "here" else _hist_all):
                if _hr.get("phase") == "pilot":
                    for _pv, _pm in ((_hr.get("panel") or {}).items() if isinstance(_hr.get("panel"), dict) else []):
                        for _mk, _mv in ((_pm or {}).items() if isinstance(_pm, dict) else []):
                            _prec_from.setdefault(_pv, {}).setdefault(_mk, []).append(_mv)
            _x2n = _xc2n(_ser, (_pilot or {}).get("default_metrics") if isinstance(_pilot, dict) else None,
                         _panel.kinds, _panel.variables, pbias_percent=dict(_panel.pbias_percent),
                         precision_from=_prec_from, no_series_why=_why_no)
            _pilot_notes = _pilot_notes + _panel.apply_pilot(_x2n)
            # a block without nse / kge / r takes the runner's plain ones; with no series they are NOT checked —
            # say so (Opus 8b/9 r6 #1)
            from .panel import _num as _num0
            _dm0 = (_pilot or {}).get("default_metrics") if isinstance(_pilot, dict) else None
            _blk0 = (((_dm0 or {}).get("__kdt__") or {}).get("panel") or {}) if isinstance(_dm0, dict) else {}
            for _v in _panel.variables:
                _b = _blk0.get(_v)
                if isinstance(_b, dict) and str((_x2n["checks"].get(_v) or {}).get("status", "")).startswith("not checked"):
                    _bl = {str(k).lower() for k, x in _b.items() if _num0(x) is not None}   # keys holding a number
                    _pl0 = (_dm0.get(_v) if isinstance(_dm0.get(_v), dict) else
                            (_dm0 if len(_panel.variables) == 1 else {}))
                    _pl0 = {str(k).lower(): x for k, x in (_pl0 or {}).items() if not str(k).startswith("__")}
                    _took = [h for h in ("nse", "kge", "r") if h not in _bl and _num0(_pl0.get(h)) is not None
                             and h in _panel.recorded(_v)]                       # only what the panel really took
                    if _took:
                        _pilot_notes = _pilot_notes + [
                            f"{_v}: {', '.join(_took)} taken from the runner's plain keys (the reserved section lacks "
                            f"them) — not checked against the kit's formula (no series)"]
            # a series the workflow DECLARED but the kit cannot read: say so (Opus 8b/9 r4 #2) — the tolerances
            # fall back to fixed values and the 2n check does not run
            for _v, _w in (_why_no or {}).items():
                if str(_w).startswith("no series (the runner could not save it"):
                    _pilot_notes = _pilot_notes + [
                        f"{_v}: {_w} — tolerances use fixed values and the 2n check did not run"]
                elif str(_w).startswith("series unreadable"):
                    _pilot_notes = _pilot_notes + [
                        f"{_v}: the default run's series could not be read — {_w}; tolerances use fixed values and "
                        f"the 2n check did not run (save sim / obs / date as plain numpy arrays, not object arrays)"]
            _cross_2n = _x2n["checks"]
            # variables without a series: a block's own "mean near zero" (block _why, if a workflow wrote one) from the
            # run at the defaults — decided once, here, like the series-based decision
            from .panel import pilot_decisions_from_payload as _pd_payload
            _dblk = (_pilot or {}).get("default_panel_block") if isinstance(_pilot, dict) else None
            _payload = ({"__kdt__": {"panel": _dblk}} if _dblk
                        else (_probe if isinstance(_probe, dict) else None))
            _pd_dec = _pd_payload(_payload, _panel.kinds, [v for v in _panel.variables if v not in _ser])
            # the objective probe's payload set pilot decisions only if it decided something: then its
            # split echo counts in the claim (and is saved for a resume, which reuses those decisions)
            _probe_used_for_pilot = bool(not _dblk and isinstance(_probe, dict) and _probe
                                         and any((_pd_dec or {}).get("kit_missing", {}).values()))
            _panel.apply_pilot(_pd_dec)
            if _cspec.get("tolerance", "bootstrap") == "fixed":
                _ser, _why_no = {}, {v: "contract sets tolerance: fixed" for v in _panel.variables}
            # the protection anchor: the pilot DEFAULT run's panel (design §2.4) — read before the
            # tolerance work, so a failure there cannot hide it
            _anchor_step_reached = True
            _dm = (_pilot or {}).get("default_metrics") if isinstance(_pilot, dict) else None
            if _dm:
                _default_panel, _default_panel_known = _panel.extract(_dm), True
            elif _pilot and "default_metrics" not in _pilot:
                _default_panel_why = "the pilot summary (saved by an older kit) has no default-run metrics"
            elif _pilot and ((_pilot.get("records") or [{}])[0] or {}).get("rejected_by_constraints"):
                _default_panel_why = "the default parameters break the contract's constraints (never run)"
            elif _pilot:
                _default_panel_why = "the pilot's default run gave no metrics"
            if _default_panel_known:
                # THE REPLY (Leo 2026-10-02): what the convergence check needs from the runner, judged on the
                # pilot's default run after the pilot decisions and the 2n check — a required watched score not
                # reported, a score the 2n check rejected, the scored series not saved (kinds that record something)
                _r_mis = {v: _panel.missing_required(v, _default_panel.get(v)) for v in _panel.variables}
                _r_mm = {v: sorted(m for m, w in (_panel.kit_missing.get(v) or {}).items()
                                   if str(w).startswith(("2n mismatch", "worked out from PBIAS")))
                         for v in _panel.variables}
                _r_ser = {v: (v in _ser_have) for v in _panel.variables if _panel.recorded(v)}
                _reply_state = {"missing": {v: m for v, m in _r_mis.items() if m},
                                "mismatch": {v: m for v, m in _r_mm.items() if m},
                                "series": _r_ser,
                                "series_why": {v: _ser_why0.get(v) for v, ok_ in _r_ser.items() if not ok_},
                                "complete": (not any(_r_mis.values()) and not any(_r_mm.values())
                                             and all(_r_ser.values()))}
            _tol_rec, _tol_src = pilot_tolerances(None, _panel.variables, _panel.kinds,
                                                  series=_ser, why=_why_no)
            try:
                _pl = json.loads(_plan_file.read_text()) if _plan_file.exists() else dict(budget_plan)
                # the pilot's split echoes, so a resume (which reuses this pilot) can still judge them
                _this = _read_history(workdir)[_hist_start0:]
                if _pilot_source == "here":
                    _pl["pilot_split_provenance"] = _split_prov(
                        [r for r in _this if r.get("phase") == "pilot"])["by_phase"].get("pilot") or _EMPTY_PROV
                # the decisions were derived NOW (also on a resume whose saved setup could not be
                # reused): the probe flag and record describe THIS derivation (Opus step 7 r4 note 10)
                _pl["probe_used_for_pilot"] = _probe_used_for_pilot
                _pl.pop("probe_split_provenance", None)
                if _probe_used_for_pilot:
                    _pl["probe_split_provenance"] = _split_prov(
                        [r for r in _this if r.get("phase") == "objective_probe"]
                    )["by_phase"].get("objective_probe") or _EMPTY_PROV
                _pl["panel_setup"] = {"variables": list(_panel.variables),
                                      "kinds": dict(_panel.kinds),
                                      "kit_missing": {v: dict(m) for v, m in _panel.kit_missing.items() if m},
                                      "pilot_notes": _pilot_notes,
                                      "cross_check_2n": _cross_2n, "reply": _reply_state,
                                      "tolerance_records": _tol_rec, "tolerance_source": _tol_src,
                                      **({"default_panel": _default_panel} if _default_panel_known else {})}
                _plan_file.write_text(json.dumps(_pl, indent=2, default=str))
            except Exception as _pe:
                print(f"  [calib] WARNING could not save the panel setup in {_plan_file.name}: {_pe}",
                      flush=True)
        for _n in _pilot_notes:
            print(f"  [calib] pilot: {_n}", flush=True)
        # THE REPLY GATE: only BEFORE any search evaluation exists in this work folder — a search that has begun
        # keeps its runner (report only). The plan (cap, pilot, panel setup) STAYS saved: it is tied to this
        # runner's code, so the next call with the same runner reuses these first runs instead of repeating them,
        # and a call with an updated runner plans afresh (3c).
        _search_begun = int((phase_counts(workdir).get("search") or {}).get("n") or 0) > 0
        if (reply_gate and not _search_begun and isinstance(_reply_state, dict)
                and (reply_gate == "check" or not _reply_state.get("complete"))):
            ev.restore_originals()
            _st = "reply_checked" if reply_gate == "check" else "reply_incomplete"
            print(f"  [calib] reply gate: {_st} — {json.dumps(_reply_state, default=str)[:400]}", flush=True)
            return {"status": _st, "reply": _reply_state, "algorithm": algo, "triage": _triage,
                    "objectives": [o.name for o in objs],
                    "default_reply": ((_pilot or {}).get("default_metrics") if isinstance(_pilot, dict) else None),
                    "pilot": ({k: v for k, v in _pilot.items() if k != "records"} if isinstance(_pilot, dict) else None),
                    "pilot_notes": _pilot_notes, "cross_check_2n": _cross_2n,
                    "phase_counts": phase_counts(workdir), "split_provenance": _prov_now()}
        # the required-metric tolerances (recorded-only metrics — NSE, KGE, PBIAS, NRMSE — and kit-set
        # missing ones are never tested for settling, design §1.4); the full records are in the report
        _tol = {_v: {_m: _t for _m, _t in (_r.get("tol") or {}).items() if _m in _panel.required(_v)}
                for _v, _r in _tol_rec.items()}
        _tol = {_v: _mm for _v, _mm in _tol.items() if _mm}
        _no_metrics = [v for v in _panel.variables if not _panel.recorded(v)]
        if _no_metrics:
            print(f"  [calib] convergence: {_no_metrics} have no panel metric (categorical) and are "
                  f"judged on their own loss only", flush=True)
    except Exception as _e:
        _tol, _tol_src, _tol_rec = ({}, f"unavailable: {type(_e).__name__}", {})
        if not _default_panel_known and not _anchor_step_reached:
            _default_panel_why = f"panel setup failed ({type(_e).__name__}) before the anchor was read"
        print(f"  [calib] WARNING convergence tolerances unavailable ({_e}) — the rule will have no "
              f"panel tolerances, so no variable can settle (verdicts read unknown)", flush=True)
    from .rule import design_window as _design_window
    _win_spec = _cspec.get("window", "auto")
    # the design window (§2.3); a number in the contract fixes it
    _rule_win = _design_window(len(_params), int(budget)) if (_win_spec in (None, "auto")) else int(_win_spec)
    # THE RULE (design §2.2-§2.6, build step 2): per call — incumbent/compromise, loss test, panel
    # test, front test, stop point, first settle points. In stop mode its stop point ends SCE-UA /
    # NSGA-II / NSGA-III / MOEA/D through the hook below (build step 4).
    from .rule import ConvergenceRule as _Rule, resolve_protect as _resolve_protect
    _protect_active, _protect_dropped = _resolve_protect(
        _protect_spec, _panel, (_default_panel if _default_panel_known else None),
        no_default_reason=_default_panel_why)
    for _d in _protect_dropped:
        print(f"  [calib] protect: {_d['var']}:{_d['metric']} dropped — {_d['reason']}", flush=True)
    # ── ONE SEED (design §1.7, §2.7; build step 6) ──────────────────────────────────────────────
    # The search and its verdicts, per seed: a fresh rule, a fresh backend and a fresh call log each
    # time; the pilot, tolerances, protection anchor, cap and panel are shared (charged once, §1.3).
    def _run_seed(_seed_no):
        _eval_losses.clear()
        backend = _make_backend(algo)
        _rule_tol = {_v: dict(_r.get("tol") or {}) for _v, _r in (_tol_rec or {}).items()}
        _rule = _Rule([(o.name, o.var) for o in objs], _panel, _rule_tol, _rule_win, trade_off=multi,
                      weights=[o.weight for o in objs],
                      rel_gain=float(_cspec.get("rel_gain", 0.005)),
                      eps_front=float(_cspec.get("eps_front", 0.01)),
                      protect=_protect_active, default_panel=_default_panel)
        _rule_err: list = []
        _end = {"reason": None}          # set when the kit's hook ends the search (cap)
        # THE CONVERGENCE RULE (owner decision 2026-10-04; validated in part C): SCE-UA = SPOTPY's own loop-end test
        # on the objective and the watched scores of the best point (classic settings); NSGA-II / NSGA-III / MOEA-D =
        # pymoo's own default termination (inside the backend); DDS / DREAM / others = no rule, the settle point only.
        # In stop mode the rule ends the search at a loop / generation end; in keep_going it is recorded only.
        from .native_rules import SpotpyTest as _SpotpyTest
        from .settle import SettlePoint as _SettlePoint
        _ntest = _SpotpyTest(watch_scores=True, variables=list(_rule.variables)) if algo == "sceua" else None
        _settle = None if multi else _SettlePoint(_panel, _rule_tol, rel_gain=float(_cspec.get("rel_gain", 0.005)),
                                                  variables=list(_rule.variables))
        _native = {"stopped": False, "errors": []}

        def _on_loop_end(_loop, _gnrng, _bestf, _calls):
            try:
                _sc_at = _rule.X[_calls - 1] if 0 < _calls <= len(_rule.X) else {}
                _fired = _ntest.add_loop(_loop, _bestf, _gnrng, _sc_at)
            except Exception as _e:              # the rule never fails a run; the fault is reported
                if len(_native["errors"]) < 5:
                    _native["errors"].append(f"{type(_e).__name__}: {_e}")
                return False
            if _fired and _cmode == "stop":
                _native["stopped"] = True
                return True
            return False

        def _on_generation_end(_gen, _n_eval, _fired_now, _converged):
            if _converged and _cmode == "stop":
                _native["stopped"] = True
                return True
            return False
        # LIVE TIME CHECK (design §1.3): the kit keeps timing calls and warns once when the projected
        # finish passes this seed's share of the allowance (an estimate, never a deadline)
        _live = {"t0": None, "warned": None}
        try:                             # a hand-edited saved plan must never crash the run
            _seed_time_s = float(((budget_plan.get("ledger") or {}) if isinstance(budget_plan, dict)
                                  else {}).get("per_seed_search_s"))
            _seed_time_s = _seed_time_s if math.isfinite(_seed_time_s) else None
        except (TypeError, ValueError, AttributeError):
            _seed_time_s = None
        _inc_metrics: dict = {}          # call -> the runner's own metrics (numbers only), for §2.9
        import numbers as _numbers

        def _numbers_only(d):
            out = {}
            for _k, _v in (d or {}).items():
                if isinstance(_v, bool):
                    continue
                if isinstance(_v, _numbers.Real):
                    out[_k] = float(_v)
                elif isinstance(_v, dict) and _k != "__kdt__":
                    _sub = _numbers_only(_v)
                    if _sub:
                        out[_k] = _sub
            return out
        def _hook(history):
            if _lane_parent[0] is not None and os.getppid() != _lane_parent[0]:
                os.kill(os.getpid(), _signal0.SIGTERM)     # parent gone: the lane's handler ends its group
            try:
                _now = time.perf_counter()
                _done = len(history)
                if _seed_time_s is not None and _live["warned"] is None and _live["t0"] is not None:
                    _el = _now - _live["t0"]
                    _proj = _el / max(1, _done) * int(budget)
                    if float(_seed_time_s) <= 0.0 or (_done >= 5 and _proj > float(_seed_time_s)):
                        _live["warned"] = ((f"this seed's share of the allowance is {float(_seed_time_s):.0f} s — the "
                                            f"shared and final overhead used it all; the search runs past the allowance")
                                           if float(_seed_time_s) <= 0.0 else
                                           (f"after {_done} calls the projected search time is {_proj:.0f} s, past "
                                            f"this seed's share of the allowance ({float(_seed_time_s):.0f} s)"))
                        print(f"  [calib] WARNING {_live['warned']}", flush=True)
            except Exception:
                pass
            try:
                _i = len(history) - 1
                _l = _eval_losses[_i] if _i < len(_eval_losses) else None
                # per component: a non-finite loss is None (B_k still counts the finite ones, §2.2)
                _lv = ([float(x) if math.isfinite(float(x)) else None for x in _l] if _l is not None else None)
                _ok = _lv is not None and all(x is not None for x in _lv)
                _sc = None
                if not multi and _ok:
                    _tot = sum(o.weight for o in objs) or 1.0
                    _sc = sum(o.weight * x for o, x in zip(objs, _lv)) / _tot
                _any = _lv is not None and any(x is not None for x in _lv)
                # a call with no finite loss carries no panel: the evaluator keeps the LAST GOOD run's metrics
                _raw = (_metrics_of(_i) or {}) if _any else {}
                _rule.add(_lv, _sc, _panel.extract(_raw) if _any else {})
                if _settle is not None:
                    try:
                        _settle.add(_sc if _ok else None, _panel.extract(_raw) if _any else {})
                    except Exception as _se:
                        if len(_rule_err) < 5:
                            _rule_err.append(f"settle: {type(_se).__name__}: {_se}")
                if _any:
                    # every call with a finite loss: a trade-off compromise can move back to an OLDER
                    # archive member (at t0, or under protection), so "incumbent on arrival" is not enough
                    _inc_metrics[len(_rule.F) - 1] = _numbers_only(_raw)
            except Exception as _e:              # the rule never fails a run; the fault is reported
                if len(_rule_err) < 5:
                    _rule_err.append(f"{type(_e).__name__}: {_e}")
                try:                             # keep the rule's call index aligned with the search
                    if len(_rule.F) < len(history):
                        _rule.add(None, None, {})
                except Exception:
                    pass
            # ONE termination policy (design §2.8, gap 2a/2b): the cap is counted in calls here, for every
            # backend; in stop mode our stop point ends SCE-UA / NSGA-II / NSGA-III / MOEA/D (never DDS,
            # whose search is planned around the budget, nor DREAM, where R-hat decides)
            # (the per-call step rule is recorded only; since 2026-10-04 the optimizers' own rules stop a search,
            # at a loop / generation end, through the backends' loop / generation hooks)
            if len(history) >= int(budget):
                _end["reason"] = "cap"
                # pymoo ends at its own n_eval limit and already stops calling the model past the cap;
                # raising EarlyStop there would throw away its final population (and its front)
                return algo not in ("nsga2", "nsga3", "moead")
            return False

        # every optimizer eval is tagged `search` in eval_history.jsonl — commissioning, the
        # consumption proof and the holdout gate carry their own phase, so "how many evaluations did
        # the SEARCH take" is answerable from the log instead of inferred from a total (2026-09-27).
        with ev.phase_as("search"):
            _live["t0"] = time.perf_counter()          # the clock starts with the search's first call
            ev.search_seed = _seed_no
            try:
                _bk_seed = dict(_bk)
                if algo == "sceua":
                    _bk_seed["on_loop_end"] = _on_loop_end
                elif algo in ("nsga2", "nsga3", "moead"):
                    _bk_seed["on_generation_end"] = _on_generation_end
                result = backend.optimize(problem, budget=budget, seed=_seed_no, on_eval=_hook, **_bk_seed)
            except Exception as _ce:          # a CRASHED seed (design §1.7): its slot may be re-run once
                # the model calls this attempt made (a crash can come after a call and before the rule saw it)
                return {"crashed": True, "seed": _seed_no, "calls": max(len(_rule.F), len(_eval_losses)),
                        "error": f"{type(_ce).__name__}: {_ce}"}
            finally:
                ev.search_seed = None
        # WHAT ENDED THE SEARCH, AND WHY (design §2.8): three separate facts — our stop point, the call at
        # which the search ended, and the reason.
        _term = getattr(result, "termination", None) or {}
        _n_calls = int(result.n_evaluations)
        if _term.get("status") == "stopped_by_kit_rule":
            _reason = ("SCE-UA own test (kit)" if algo == "sceua" else "pymoo own rule (kit)")
        elif _end["reason"]:
            _reason = _end["reason"]
        elif algo == "sceua" and _term.get("status") in ("population_converged", "improvement_below_pcento",
                                                           "max_loops", "max_trials"):
            _reason = {"population_converged": "SCE-UA peps", "improvement_below_pcento": "SCE-UA kstop/pcento",
                       "max_loops": "SCE-UA max loops", "max_trials": "SCE-UA max trials"}[_term["status"]]
        elif algo == "dream" and _term.get("rhat_reached") and _n_calls < int(budget):
            _reason = "DREAM convergence"
        elif _n_calls >= int(budget):
            _reason = "cap"
        else:
            _reason = "optimizer ended on its own"
        # the backends' own "stopped early" flags cannot tell the cap from our rule (the kit's hook ends
        # both); the kit knows which, so it sets them (a cap end is never an early stop)
        _early = _reason in ("SCE-UA own test (kit)", "pymoo own rule (kit)")
        if _early and isinstance(_term, dict) and _term.get("status") in ("max_evals",):
            _term["status"] = "stopped_by_kit"     # our rule ended it, even on the last allowed call
        try:
            result.stopped_early = _early
            if isinstance(_term, dict) and "early_stopped_by_rule" in _term:
                _term["early_stopped_by_rule"] = _early
            if isinstance(result.notes, str):
                result.notes = result.notes.replace("; stopped early by the stop rule", "; stopped by the kit's rule" if _early else "")
        except Exception:
            pass
        # the SCE-UA loop / pymoo generation of EVERY call (design §2.8 "always recorded"), kept as the
        # first call of each loop / generation, so a replay can map any call to it
        try:
            for _key, _name in (("loop", "loop_starts"), ("gen", "generation_starts")):
                _tags = [h.get(_key) for h in (result.history or [])]
                if any(t is not None for t in _tags):
                    _starts, _prev = [], object()
                    for _i, _t in enumerate(_tags):
                        if _t != _prev:
                            _starts.append([_i, _t])
                            _prev = _t
                    _term[_name] = _starts
        except Exception:
            pass
        convergence = {"mode": _cmode, "warnings": _cwarn}
        if _live.get("warned"):
            budget_plan.setdefault("warnings", []).append(f"seed {_seed_no}: {_live['warned']}")
        if _stop_translated:
            convergence["translated_from_strategy_stop"] = _stop_translated
        convergence["rule"] = _rule.summary()
        # VERDICTS (design §2.7, §2.9, §2.10; build step 3): "rule triggered" when our rule ended the search;
        # otherwise §2.10 or seed rule 3. DREAM: R-hat decides.
        from .verdict import seed_verdict as _seed_verdict, final_value_settle_points as _fvsp, \
            standards as _standards
        try:
            _vd = _seed_verdict(_rule, ended_at_stop=False,
                                dream=({"rhat_recorded": bool(_term.get("rhat_recorded")),
                                        "rhat_reached": bool(_term.get("rhat_reached"))}
                                       if algo == "dream" else None),
                                algorithm=algo)
            _vd["final_value_settle"] = _fvsp(_rule)
            if algo == "dream":                  # R-hat decides a DREAM seed: kept per slot for the replay
                convergence["dream_rhat"] = {"rhat_recorded": bool(_term.get("rhat_recorded")),
                                             "rhat_reached": bool(_term.get("rhat_reached"))}
        except Exception as _e:
            _vd = {"verdict": "unknown", "how": f"verdict failed: {type(_e).__name__}: {_e}"}
        try:                                     # a separate try: a bad convention never loses the verdict
            _single_var = len(_panel.variables) == 1
            # an old strategy.stop floor band becomes the band for its main_stat (design §5.1b) — for the
            # floor's own dag_variable only, when it named one
            def _band_ov_for(_var):
                return _band_override(_stop_translated, _var)
            _fin_inc = _rule.inc[-1] if _rule.inc else None
            _stop_inc = _rule.inc[_rule.stop_point] if _rule.stop_point is not None else None
            _vd["standards"] = {}
            for _v in _panel.variables:
                _shape = (_kset.get("shapes") or {}).get(_v)
                _vd["standards"][_v] = {
                    "obs_shape": _shape,
                    "calibration": _standards(_conv, _v, _shape, _inc_metrics.get(_fin_inc), _single_var, _band_ov_for(_v)),
                    "at_stop_point": (_standards(_conv, _v, _shape, _inc_metrics.get(_stop_inc), _single_var, _band_ov_for(_v))
                                      if _stop_inc is not None else None),
                    "assessed_call": _fin_inc, "validation": "no validation"}
        except Exception as _e:
            _vd["standards"] = f"standards not assessed: {type(_e).__name__}: {_e}"
        convergence["verdict"] = _vd
        _sp = _rule.stop_point
        _how = str(_vd.get("how", ""))
        if _n_calls == 0:
            _text = "no call was made"
        elif _how.startswith("convergence not tracked"):
            _text = f"{_how}; the search ended at call {_n_calls - 1} ({_reason})"
        elif algo == "dream" and _term.get("rhat_reached"):
            _c = _term.get("rhat_converged_at_call")
            _text = (f"DREAM reached R-hat < 1.2 at call {_c}; "
                     + ("ended by cap during DREAM's post-convergence calls" if _reason == "cap"
                        else f"ended at call {_n_calls - 1} ({_reason})"))
        elif _sp is not None:
            _text = f"stop point at call {_sp}; the search ended at call {_n_calls - 1} ({_reason})"
        elif _reason == "cap" and str(_vd.get("verdict")) == "unknown" and "not recorded" in str(_vd.get("how", "")):
            _text = "ran to the cap; convergence unknown (" + str(_vd.get("how")).split("only because ")[-1] + ")"
        elif _reason == "cap":
            _text = "ran to the cap without converging"
        else:
            _text = f"ended by {_reason} at call {_n_calls - 1} before our rule fired"
        # the optimizer's own rule (the kit's convergence rule) and the settle point
        if _ntest is not None:
            _nat = _ntest.summary()
        elif algo in ("nsga2", "nsga3", "moead"):
            _nv = (_term.get("native_verdict") or {}) if isinstance(_term, dict) else {}
            _nat = {k: _nv.get(k) for k in ("rule", "settings", "generations_seen", "fired_at", "verdict")}
        else:
            _nat = {"rule": None, "verdict": f"{algo} has no convergence rule of its own; see the settle point"}
        _nat.update(mode=_cmode, stopped_the_search=bool(_native["stopped"]), errors=_native["errors"])
        convergence["native"] = _nat
        if _settle is not None:
            convergence["settle"] = _settle.summary()
        _fa = (_nat or {}).get("fired_at") or {}
        _step = _fa.get("loop", _fa.get("n_gen"))
        _unit = "loop" if algo == "sceua" else "generation"
        if _reason in ("SCE-UA own test (kit)", "pymoo own rule (kit)"):
            _text = (f"converged at {_unit} {_step} by {'SCE-UA' if algo == 'sceua' else 'pymoo'}'s own rule"
                     f"{' on the objective and watched scores' if algo == 'sceua' else ''}; the search stopped there "
                     f"(call {_n_calls - 1})")
        elif _step is not None:
            _text += f"; {'SCE-UA' if algo == 'sceua' else 'pymoo'}'s own rule fired at {_unit} {_step} (recorded only)"
        # THE SEED'S VERDICT is the kit's convergence rule (2026-10-04); the per-call step rule's verdict is kept as
        # step_rule_verdict (recorded only, not the kit's answer)
        from .verdict import CONVERGED as _CV, NOT_CONVERGED as _NCV, UNKNOWN as _UNK
        _nword = {"converged": _CV, "not converged within the budget": _NCV}.get(str((_nat or {}).get("verdict")), _UNK)
        _rname = (_nat or {}).get("rule") or f"{algo}: no convergence rule of its own"
        if _nword == _CV:
            _vhow = f"{_rname}: fired at {_unit} {_step}"
        elif _nword == _NCV:
            _vhow = f"{_rname}: did not fire within the budget"
        elif (_nat or {}).get("rule") is None:
            _vhow = (f"{algo} has no convergence rule of its own"
                     + (f"; {convergence['settle']['wording']}" if convergence.get("settle") else ""))
        else:
            _vhow = f"{_rname}: {(_nat or {}).get('verdict')}"
        convergence["step_rule_verdict"] = dict(_vd, note="the per-call step rule, recorded only; not the kit's "
                                                         "convergence answer (see verdict)")
        _rule_used = (_nat or {}).get("rule")
        if str(_vd.get("how", "")).startswith("convergence not tracked"):
            # a backend that does not report each call: nothing can be judged, by any rule
            _nword, _vhow, _rule_used = _UNK, str(_vd.get("how")), None
        elif algo == "dream":
            # DREAM: its own rule is R-hat < 1.2 (design §2.7; the decided method) — the step verdict already says it
            _nword, _vhow, _rule_used = _vd.get("verdict", _UNK), str(_vd.get("how")), "DREAM R-hat < 1.2"
        _vd_kit = {"verdict": _nword, "how": _vhow, "rule": _rule_used,
                   "rule_verdict": (_nat or {}).get("verdict"),
                   "variables": {v: {"verdict": _nword} for v in _panel.variables},
                   "standards": _vd.get("standards")}
        convergence["verdict"] = _vd_kit
        convergence["ended"] = {"stop_point": _sp, "ended_at_call": (_n_calls - 1 if _n_calls else None), "calls": _n_calls,
                                "reason": _reason, "text": _text, "cap": int(budget)}
        convergence["rule"].update({"protect_requested": _protect_spec, "protect_dropped": _protect_dropped,
                                    "calls_with_runner_metrics": len(_inc_metrics),
                                    "errors": _rule_err, "state": _rule.state()})
        convergence.update({"tolerances": _tol, "tolerance_source": _tol_src,
                            # per variable and metric: bootstrap or fixed_fallback, and why (§2.11)
                            "tolerance_records": _tol_rec,
                            "variables_without_panel": _no_metrics,
                            "kinds": dict(_panel.kinds),
                            "pbias_percent": dict(_panel.pbias_percent),
                            "variables_without_objective": _vars_without_objective,
                            "required_panel": {v: list(_panel.required(v)) for v in _panel.variables},
                            "kit_missing": {v: dict(m) for v, m in _panel.kit_missing.items() if m},
                            "pilot_notes": _pilot_notes, "cross_check_2n": _cross_2n, "reply": _reply_state,
                            "kind_warnings": _kset["warnings"],
                            "panel_flags": dict(_panel.flags),
                            "panel": list(_panel.metrics),
                            # the optimizer's native verdicts (SCE-UA test + loop count; pymoo native
                            # verdict and its ftol part; DREAM R-hat), recorded as they are (§3)
                            "optimizer_termination": _term,
                            "seed": _seed_no})
        if _panel.flags:
            convergence["panel_flags"] = _panel.flags
        if _rule.stop_point is not None:
            from .convergence import marker_evidence as _param_width
            convergence["param_width_at_stop_point"] = _param_width(
                result.history, _rule.stop_point, lower=lower, upper=upper,
                names=[p["name"] for p in _params])
        return {"crashed": False, "seed": _seed_no, "result": result, "convergence": convergence,
                "rule": _rule, "end": _end, "term": _term, "reason": _reason, "vd": _vd_kit, "vd_step": _vd,
                "inc_metrics": _inc_metrics, "n_calls": _n_calls}

    # ── SEEDS (design §1.7, §2.7): every slot tracked; a crashed seed replaced ONCE by a new number ──
    from .seeds import (seed_agreement as _seed_agreement, pick_returned_seed as _pick_seed,
                        param_spread as _param_spread, across_seeds as _across_seeds)
    def _merge_cache_file(_lf):
        """Complete cache lines of a lane's cache file (the parent's file name) into the main cache and the
        in-memory cache; returns how many were new."""
        _m = 0
        try:
            if Path(_lf).exists():
                with open(_lf) as _src, open(ev._cache_file, "a") as _dst:
                    for _line in _src:
                        if not _line.endswith("\n"):
                            continue                         # a torn last line of a killed lane
                        try:
                            _cr = json.loads(_line)
                        except Exception:
                            continue
                        if isinstance(_cr, dict) and "key" in _cr and _cr["key"] not in ev._metrics_cache:
                            ev._metrics_cache[_cr["key"]] = _cr.get("metrics")
                            _dst.write(_line)
                            _m += 1
        except Exception as _ce:
            budget_plan.setdefault("warnings", []).append(f"lane cache not merged ({_ce})")
        return _m

    import signal as _signal0
    _lane_parent: list = [None]            # set in a forked lane: the parent's pid (the hook checks it)
    _pcache_name = Path(ev._cache_file).name
    # leftovers of a killed earlier call (the workdir lock guarantees none still runs): keep their complete
    # cache lines, then remove them — on EVERY call, whether or not this one uses lanes (Opus 6b r2 #3)
    import glob as _glob0
    for _old in sorted(Path(workdir).parent.glob(f"{_glob0.escape(Path(workdir).name)}_seed_lanes*")):
        if not _old.is_dir():                    # (the name is escaped: `run[1]` must not match `run1`)
            continue
        _nl = sum(_merge_cache_file(_oc) for _oc in _old.rglob(_pcache_name))
        shutil.rmtree(_old, ignore_errors=True)
        budget_plan.setdefault("warnings", []).append(
            f"lane folder {_old.name} of an earlier, killed calibration: {_nl} finished calls kept, "
            + ("folder removed" if not _old.exists() else "the folder could NOT be removed"))

    class _LanesUnavailable(Exception):
        """The lane root could not be made: the seeds run one after another instead."""

    def _parallel_seeds():
        """Seeds in parallel lanes (build step 6b; Opus 6b r1). Each attempt runs in a FORKED process on its own
        cloned workdir with its own evaluator (same cache file name as the parent's) and runner (built from the
        contract), from a snapshot of the evaluation cache. At most `_seed_par` lanes run at once; a slot whose
        attempt crashed is re-run once as soon as a lane is free. A lane closes the pipe ends it inherited, dies
        with its parent, and flushes its output before it exits; the parent waits on the result AND the lane's
        exit (a dead lane is a crashed attempt), ends a lane that makes no call for a long time, merges each
        lane's call log (renumbered; `lane` = the lane it ran in, `slot`) and cache, and — in a finally — ends
        any live lane and removes this call's lane folder. A lane that cannot be SET UP (clone, cache, evaluator)
        is not a crash: its slot runs in this process, one after another. Returns (slots, search wall time)."""
        import multiprocessing as _mpx
        import multiprocessing.connection as _mpc
        import signal as _signal
        import sys as _sys
        from .compute import clone_workdir as _clone
        from .runner import make_run_model as _mk_rm
        from .evaluator import Evaluator as _Ev2, read_history as _rh
        _ctx = _mpx.get_context("fork")
        _parent_pid = os.getpid()
        try:
            _root = Path(tempfile.mkdtemp(prefix=f"{Path(workdir).name}_seed_lanes_", dir=str(Path(workdir).parent)))
        except OSError as _re:
            raise _LanesUnavailable(f"the lane folder could not be made ({_re})")
        _open_reads: list = []                 # the parent's read ends — every new lane closes its copies

        def _child(sno, k, lane, conn, eid_offset, inherited):
            nonlocal ev, _metrics_of
            _setup_done = False
            try:
                for _c in inherited:                   # the pipe ends this lane inherited (earlier lanes' + its own)
                    try:
                        _c.close()
                    except Exception:
                        pass
                # the lane leads its OWN process group, so ending it also ends the model runs it started; its
                # SIGTERM handler (replacing any the host set) kills that group. SIGTERM comes from the parent
                # (an ended lane) or, through PR_SET_PDEATHSIG, from the kernel when the parent dies (Opus 6b r2 #4)
                os.setpgid(0, 0)

                def _end_group(*_a):
                    _g = os.getpgrp()
                    if _g == os.getpid() and _g > 1:           # never a group this lane does not lead
                        os.killpg(_g, _signal.SIGKILL)
                    os._exit(1)
                _signal.signal(_signal.SIGTERM, _end_group)
                # the parent decides when a lane ends; a no-op HANDLER (not SIG_IGN, which exec keeps) so the
                # model runs this lane starts still get the default SIGINT (Opus 6b r3 #2)
                _signal.signal(_signal.SIGINT, lambda *_x: None)
                for _sg in (_signal.SIGTSTP, _signal.SIGCONT):   # the parent's forwarding handlers are not ours
                    _signal.signal(_sg, _signal.SIG_DFL)
                # a lane is a BACKGROUND group for the terminal: with `stty tostop` its first print would stop it
                # (and the run would wait forever); ignoring SIGTTOU lets it write (Opus 6b r4 #2)
                _signal.signal(_signal.SIGTTOU, _signal.SIG_IGN)
                # the parent blocked Ctrl-C / Ctrl-Z around the fork; this lane's own handlers are set now
                _signal.pthread_sigmask(_signal.SIG_UNBLOCK, {_signal.SIGINT, _signal.SIGTSTP})
                try:                                   # die with the parent (Linux)
                    import ctypes as _ct
                    _ct.CDLL("libc.so.6", use_errno=True).prctl(1, int(_signal.SIGTERM))
                except Exception:
                    pass
                _lane_parent[0] = _parent_pid
                if os.getppid() != _parent_pid:
                    _end_group()
                _clone(workdir, lane)
                for _cf in Path(workdir).glob("eval_metrics_cache_*"):
                    shutil.copy2(_cf, Path(lane) / _cf.name)            # resume: earlier calls replay
                _eid = ev._eval_id
                ev = _Ev2(ki_path=ki_path, workdir=str(lane), parameters=_params, objectives=objs,
                          transform_inv=inv, run_model=_mk_rm(contract.get("runner") or {}, ki_path, str(lane)),
                          constraints_ok=_make_constraints(contract), base_seed=seed,
                          injection_mode=_inj_mode, fixed_params=_fixed, expected_case_id=expected_case_id)
                # the PARENT's cache file name (its setup fingerprint), so the lane reads the snapshot and its
                # entries merge back even if a KI file changed since the parent's evaluator was built
                ev._cache_file = Path(lane) / _pcache_name
                try:
                    for _line in ev._cache_file.read_text().splitlines():
                        _cr = json.loads(_line)
                        if isinstance(_cr, dict) and "key" in _cr:
                            ev._metrics_cache.setdefault(_cr["key"], _cr.get("metrics"))
                except Exception:
                    pass
                ev.panel_extract = _panel.extract
                ev._eval_id = _eid + eid_offset                 # a KDT_CALIB_EVAL_ID stream no other lane uses

                def _mo(_i, _ev=ev):
                    for _sp in ("calibration", None):
                        _mt = getattr(_ev, "_last_metrics", {}).get(_sp)
                        if _mt:
                            return _mt
                    return None
                _metrics_of = _mo
                _bk["metrics_of"] = _mo
                _bk["workdir"] = str(lane)
                _setup_done = True
                _w0 = len(budget_plan.get("warnings") or [])
                _r = _run_seed(sno)
                _r["new_warnings"] = list((budget_plan.get("warnings") or [])[_w0:])
                conn.send(_r)
            except BaseException as _e:                               # the lane crashed, not the run
                try:
                    conn.send({"crashed": True, "seed": sno, "calls": None, "setup_failed": not _setup_done,
                               "error": f"lane process: {type(_e).__name__}: {_e}"})
                except BaseException:
                    pass
            finally:
                for _s in (_sys.stdout, _sys.stderr):
                    try:
                        _s.flush()
                    except BaseException:
                        pass
                try:
                    conn.close()
                except BaseException:
                    pass
                os._exit(0)

        def _merge(lane, pos, k):
            """This lane's call log and cache into the main workdir; returns (search calls, cache lines merged)."""
            _n = 0
            _m = _merge_cache_file(Path(lane) / _pcache_name)       # the cache FIRST (a resume replays it)
            try:
                with open(Path(workdir) / "eval_history.jsonl", "a") as _fh:
                    for _rec in _rh(lane):
                        _rec["i"] = ev._hist_n
                        ev._hist_n += 1
                        _rec["lane"] = pos
                        _rec["slot"] = k + 1
                        _n += 1 if _rec.get("phase") == "search" else 0
                        _fh.write(json.dumps(_rec, default=str) + "\n")
            except Exception as _me:
                budget_plan.setdefault("warnings", []).append(f"slot {k + 1}: call log not merged ({_me})")
            shutil.rmtree(lane, ignore_errors=True)                  # only once both are merged
            return _n, _m

        def _kill_group(pid):
            """SIGKILL the process group a lane leads (the lane and the model runs it started). Safe while the
            lane is NOT reaped: its pid, and so a group with that id, cannot belong to anyone else; before the
            lane's setpgid there is no such group and this is a no-op (Opus 6b r3 #1)."""
            if not pid or pid <= 1 or pid == os.getpgrp():
                return
            try:
                os.killpg(pid, _signal.SIGKILL)
            except OSError:
                pass
            try:
                os.kill(pid, _signal.SIGKILL)
            except OSError:
                pass

        def _dead(p):
            """Has the lane exited? (read from its sentinel — never reaps it, unlike .exitcode)"""
            try:
                return bool(_mpc.wait([p.sentinel], timeout=0))
            except Exception:
                return True

        # Ctrl-Z / fg (Opus 6b r3 #4): the lanes lead their own groups, so the terminal's SIGTSTP reaches only
        # this process — pass it (and the SIGCONT) on to every lane group while lanes run
        def _each_group(sig):
            for _v in list(_running.values()):
                _pid = getattr(_v[4], "pid", None)
                if _pid and _pid > 1 and _pid != os.getpgrp():
                    try:
                        os.killpg(_pid, sig)
                    except OSError:
                        pass
                    try:                                   # a lane that has not yet made its group (r4 #4)
                        os.kill(_pid, sig)
                    except OSError:
                        pass

        def _on_tstp(*_x):
            _each_group(_signal.SIGSTOP)
            _signal.signal(_signal.SIGTSTP, _signal.SIG_DFL)
            os.kill(os.getpid(), _signal.SIGTSTP)                   # stop as the terminal asked
            # here once this process was continued — or at once, when the kernel DROPPED the stop (an orphaned
            # process group: setsid, `ssh -t host cmd`, `docker exec -it`): either way the lanes go on too, so a
            # dropped stop never freezes the lanes with nothing to continue them (Opus 6b r4 #1)
            _signal.signal(_signal.SIGTSTP, _on_tstp)
            _each_group(_signal.SIGCONT)

        def _on_cont(*_x):
            _signal.signal(_signal.SIGTSTP, _on_tstp)
            _each_group(_signal.SIGCONT)

        # NO stall bound (Opus 6b r2 #1): a model run is never ended for being slow (runner.py), and one lane
        # waits for any call; a DEAD lane is seen at once through its sentinel
        _t = time.perf_counter()
        _res: dict = {}
        _setup_failed: list = []
        _pending = [(k, int(seed) + k, False) for k in range(_n_seeds)]
        _running: dict = {}                            # conn -> [k, sno, is_replacement, lane, proc, pos]
        _started = 0
        _spans: list = []                              # (start, end) of every lane that was set up and searched
        _old_sig = {}
        try:
            import threading as _thr1
            _cur_t, _cur_c = _signal.getsignal(_signal.SIGTSTP), _signal.getsignal(_signal.SIGCONT)
            # only in the main thread, only when the caller's handlers can be put back (not one set from C,
            # which reads as None), and never when the caller IGNORES SIGTSTP (it must not be stopped) (r4 #6)
            if (_thr1.current_thread() is _thr1.main_thread() and _cur_t is not None and _cur_c is not None
                    and _cur_t is not _signal.SIG_IGN):
                _old_sig = {_signal.SIGTSTP: _signal.signal(_signal.SIGTSTP, _on_tstp),
                            _signal.SIGCONT: _signal.signal(_signal.SIGCONT, _on_cont)}
        except (ValueError, OSError):
            _old_sig = {}
        try:
            while _pending or _running:
                while _pending and len(_running) < _seed_par:
                    # Process.start() reaps every finished child: a lane that exited since the last wait must be
                    # handled (its group killed, then reaped) FIRST, or its pid could be reused (Opus 6b r4 #3)
                    if any(getattr(v[4], "pid", None) and _dead(v[4]) for v in _running.values()):
                        break
                    k, sno, _is_rep = _pending.pop(0)
                    _free = sorted(set(range(1, _seed_par + 1)) - {v[5] for v in _running.values()})
                    _pos = _free[0]
                    _started += 1
                    print(f"  [calib] seed slot {k + 1}/{_n_seeds}: seed {sno} (lane {_pos} of {_seed_par})",
                          flush=True)
                    _lane = _root / f"slot{k + 1}_seed{sno}"
                    _a, _b = _ctx.Pipe(duplex=False)
                    _p = _ctx.Process(target=_child,
                                      args=(sno, k, _lane, _b, 1_000_000 * _started, list(_open_reads) + [_a]))
                    _running[_a] = [k, sno, _is_rep, _lane, _p, _pos]     # tracked BEFORE the fork (r3 #5c)
                    _mask = _signal.pthread_sigmask(_signal.SIG_BLOCK, {_signal.SIGINT, _signal.SIGTSTP})
                    try:
                        _p.start()
                    except OSError as _fe:                     # no fork (memory, process limit): not a crash
                        del _running[_a]
                        for _c in (_a, _b):
                            _c.close()
                        _setup_failed.append((k, sno, _is_rep, f"fork failed: {_fe}"))
                        continue
                    finally:                                   # the caller's mask back on EVERY way out (r5 #1)
                        _signal.pthread_sigmask(_signal.SIG_SETMASK, _mask)   # a pending Ctrl-C / Ctrl-Z runs now
                    _b.close()
                    _open_reads.append(_a)
                    _running[_a].append(time.monotonic())      # [6] when this lane started
                _ready = _mpc.wait(list(_running) + [v[4].sentinel for v in _running.values()], timeout=5.0)
                for _a in list(_running):
                    _running_row = _running[_a]
                    k, sno, _is_rep, _lane, _p, _pos = _running_row[:6]
                    _r = None
                    if _a in _ready or _p.sentinel in _ready:
                        try:
                            if _a.poll():
                                _r = _a.recv()
                        except (EOFError, OSError):
                            _r = None
                        if _r is None and not _dead(_p) and _a not in _ready:
                            continue                           # sentinel raced ahead of the data: next loop
                        if _r is None:
                            _r = {"crashed": True, "seed": sno, "calls": None,
                                  "error": "lane process ended without a result"}
                    else:
                        continue
                    # a lane that has EXITED (sent its result, or died — e.g. the OOM killer): its group is killed
                    # before it is reaped, so no model run it started outlives it; a lane that sent its result but
                    # is still flushing gets 30 s to exit, then is killed with its group (Opus 6b r3 #1)
                    if not _dead(_p):
                        try:
                            _mpc.wait([_p.sentinel], timeout=30)
                        except Exception:
                            pass
                    _kill_group(_p.pid)
                    _p.join(timeout=30)
                    del _running[_a]
                    try:
                        _a.close()
                    except Exception:
                        pass
                    if _a in _open_reads:
                        _open_reads.remove(_a)
                    for _w in _r.pop("new_warnings", None) or []:
                        budget_plan.setdefault("warnings", []).append(_w)
                    _ns, _nm = _merge(_lane, _pos, k)
                    if _r.get("crashed") and _r.get("calls") is None:
                        _r["calls"] = _ns
                    if _ns and not _nm and not _r.get("crashed"):
                        budget_plan.setdefault("warnings", []).append(
                            f"slot {k + 1}: the lane made {_ns} search calls but had no cache lines to merge")
                    if _r.get("setup_failed"):
                        _setup_failed.append((k, sno, _is_rep, _r.get("error")))
                        continue
                    if len(_running_row) > 6:
                        _spans.append((_running_row[6], time.monotonic()))
                    if _r.get("crashed") and not _is_rep:
                        _rno = int(seed) + _n_seeds + k
                        _msg = (f"seed {sno} crashed after {_r.get('calls')} calls ({_r['error']}) — slot {k + 1} "
                                f"re-run once with seed {_rno}; the replacement is a full search"
                                + (", so the run can pass the allowance" if (budget_plan or {}).get("allowance_s") else
                                   ", so the run makes more search calls than seeds x cap"))
                        print(f"  [calib] WARNING {_msg}", flush=True)
                        budget_plan.setdefault("warnings", []).append(_msg)
                        _res[k] = {"first": _r}
                        _pending.append((k, _rno, True))
                        continue
                    if _is_rep:
                        _first = _res[k]["first"]
                        _r["replaced"] = {"seed": _first["seed"], "error": _first["error"], "calls": _first.get("calls")}
                        if _r.get("crashed"):
                            print(f"  [calib] WARNING replacement seed {sno} also crashed ({_r['error']}) — slot "
                                  f"{k + 1} is missing", flush=True)
                    _res[k] = _r
        finally:
            # an error or an interrupt (Ctrl-C) here: no lane is left behind, and the calls the live lanes
            # finished are KEPT — merged into the main log and cache like a finished lane's (Opus 6b r2 #2)
            for _sg, _h in _old_sig.items():           # the caller's job-control handlers back
                try:
                    _signal.signal(_sg, _h)
                except (ValueError, OSError, TypeError):
                    pass
            for _a, _v in list(_running.items()):
                try:
                    _kill_group(getattr(_v[4], "pid", None))   # the lane and its model runs, from here (r3 #1)
                    if getattr(_v[4], "pid", None):
                        _v[4].join(timeout=30)
                except BaseException:
                    pass
                try:
                    _merge(_v[3], _v[5], _v[0])
                except BaseException:
                    pass
            # remove the lane root only when every lane folder in it was merged (and removed by _merge); a folder
            # an interrupt left unmerged stays for the next call's leftover merge (r3 #5b)
            try:
                _root.rmdir()
            except OSError:
                if _root.exists():
                    budget_plan.setdefault("warnings", []).append(
                        f"lane folder {_root.name} left for the next calibration here to merge (interrupted)")
        # a lane that could not be SET UP: that slot runs here, one after another (not a crash)
        for k, sno, _is_rep, _err in _setup_failed:
            _note = f"slot {k + 1}: its lane could not be set up ({_err}) — the seed ran in this process instead"
            print(f"  [calib] WARNING {_note}", flush=True)
            budget_plan.setdefault("warnings", []).append(_note)
            _r = _run_seed(sno)
            if _r.get("crashed") and not _is_rep:             # replaced once, as in a lane (Opus 6b r2 #5a)
                _rno = int(seed) + _n_seeds + k
                _msg = (f"seed {sno} crashed after {_r.get('calls')} calls ({_r['error']}) — slot {k + 1} "
                        f"re-run once with seed {_rno}; the replacement is a full search")
                print(f"  [calib] WARNING {_msg}", flush=True)
                budget_plan.setdefault("warnings", []).append(_msg)
                _res[k] = {"first": _r}
                _r, _is_rep = _run_seed(_rno), True
            if _is_rep:
                _first = _res[k]["first"]
                _r["replaced"] = {"seed": _first["seed"], "error": _first["error"], "calls": _first.get("calls")}
                if _r.get("crashed"):
                    print(f"  [calib] WARNING replacement seed {_r.get('seed')} also crashed ({_r['error']}) — "
                          f"slot {k + 1} is missing", flush=True)
            _res[k] = _r
        # what really ran at once (r3 #5d, r4 #5): the most lanes that were set up and searched at the same time
        _ev = sorted([(a, 1) for a, _ in _spans] + [(b, -1) for _, b in _spans], key=lambda t: (t[0], t[1]))
        _cur = _most = 0
        for _, _d in _ev:
            _cur += _d
            _most = max(_most, _cur)
        _lanes_used[0] = max(1, _most)
        return [dict(_res[k], slot=k + 1) for k in range(_n_seeds)], time.perf_counter() - _t

    _slots: list = []
    _search_wall = 0.0                     # every attempt's search time, crashed ones included
    _lanes_used = [_seed_par]              # what really ran at once (1 when no lane could be set up)
    if _seed_par > 1:
        try:
            _slots, _search_wall = _parallel_seeds()
        except _LanesUnavailable as _lu:
            _note = f"{_lu} — the seeds run one after another"
            print(f"  [calib] WARNING {_note}", flush=True)
            budget_plan.setdefault("warnings", []).append(_note)
            _lanes_used[0] = 1
    for _slot in (range(_n_seeds) if _lanes_used[0] <= 1 and not _slots else ()):
        _sno = int(seed) + _slot
        if _n_seeds > 1:
            print(f"  [calib] seed slot {_slot + 1}/{_n_seeds}: seed {_sno}", flush=True)
        _t_att = time.perf_counter()
        _sr = _run_seed(_sno)
        _search_wall += time.perf_counter() - _t_att
        if _sr.get("crashed"):
            _rno = int(seed) + _n_seeds + _slot
            _msg = (f"seed {_sno} crashed after {_sr.get('calls')} calls ({_sr['error']}) — slot {_slot + 1} "
                    f"re-run once with seed {_rno}; the replacement is a full search"
                    + (", so the run can pass the allowance" if (budget_plan or {}).get("allowance_s") else
                       ", so the run makes more search calls than seeds x cap"))
            print(f"  [calib] WARNING {_msg}", flush=True)
            budget_plan.setdefault("warnings", []).append(_msg)
            _first = _sr
            _t_att = time.perf_counter()
            _sr = _run_seed(_rno)
            _search_wall += time.perf_counter() - _t_att
            _sr["replaced"] = {"seed": _sno, "error": _first["error"], "calls": _first.get("calls")}
            if _sr.get("crashed"):
                print(f"  [calib] WARNING replacement seed {_rno} also crashed ({_sr['error']}) — slot "
                      f"{_slot + 1} is missing", flush=True)
        _sr["slot"] = _slot + 1                    # slots are numbered from 1 everywhere in the report
        _slots.append(_sr)
    # a hand-edited or truncated plan must never crash the run (the plan loader's rule): read it safely
    try:
        _search_s = float(((budget_plan.get("ledger") or {}) if isinstance(budget_plan, dict) else {}).get("search_s"))
        _search_s = _search_s if math.isfinite(_search_s) else None
    except (TypeError, ValueError, AttributeError):
        _search_s = None
    if _search_s is not None and _search_wall > _search_s > 0:
        budget_plan.setdefault("warnings", []).append(
            f"the seeds' searches took {_search_wall:.0f} s in all, past the search allowance "
            f"({_search_s:.0f} s)")

    def _incumbent(_sr):
        """The seed's final CALIBRATION incumbent (§1.5 option a: the reported result): call index, x,
        loss vector, scalar, panel, admissibility — read from the rule, with the x from the history."""
        if _sr.get("crashed"):
            return None
        _r, _res = _sr["rule"], _sr["result"]
        _i = _r.inc[-1] if _r.inc else None
        _hist = list(getattr(_res, "history", None) or [])
        if _i is None or len(_r.F) != len(_hist) or _i >= len(_hist) or (_hist[_i] or {}).get("x") is None:
            return None
        _f = _r.F[_i]
        if not _f or any(x is None for x in _f):
            return None
        return {"slot": _sr["slot"], "seed": _sr["seed"], "call": _i, "x": list(_hist[_i]["x"]),
                "losses": [float(x) for x in _f],
                "scalar": (_r.S[_i] if not multi else None),
                "panel": _r.P[_i] or {}, "admissible": not bool(_r.infeasible[-1]) if _r.infeasible else True,
                "t0": _r.t0, "z": _r.z, "s": _r.s}
    _incs = [_incumbent(_sr) for _sr in _slots]
    _completed = [c for c in _incs if c is not None]
    _ri, _rhow = _pick_seed(_completed, trade_off=multi)
    _ret = _completed[_ri] if _ri is not None else None
    _required = {v: list(_panel.required(v)) for v in _panel.variables}
    # why a variable is not compared: no panel at all (categorical), or every required metric set
    # missing by the kit (§1.4) — two different statements
    _nc_why = {v: ("its required metrics are set missing by the kit: "
                   + ", ".join(f"{m} ({w})" for m, w in sorted((_panel.kit_missing.get(v) or {}).items()))
                   if _panel.kit_missing.get(v) else "no panel metric")
               for v in _panel.variables if not _required[v]}
    _agree = _seed_agreement([(c or {}).get("panel") if c is not None else None for c in _incs],
                             _required, {v: dict((r or {}).get("tol") or {}) for v, r in (_tol_rec or {}).items()},
                             not_compared_why=_nc_why)
    _slot_vd = [(None if _sr.get("crashed") or _incs[k] is None else
                 (_sr["vd"] or {}).get("verdict")) for k, _sr in enumerate(_slots)]
    _slot_vv = [(None if _sr.get("crashed") or _incs[k] is None else
                 {v: (vv or {}).get("verdict") for v, vv in ((_sr["vd"] or {}).get("variables") or {}).items()})
                for k, _sr in enumerate(_slots)]
    _across = _across_seeds(_slot_vd, _slot_vv, _agree)
    _slot_vd_step = [(None if _sr.get("crashed") or _incs[k] is None else
                      (_sr.get("vd_step") or {}).get("verdict")) for k, _sr in enumerate(_slots)]
    _slot_vv_step = [(None if _sr.get("crashed") or _incs[k] is None else
                      {v: (vv or {}).get("verdict") for v, vv in ((_sr.get("vd_step") or {}).get("variables") or {}).items()})
                     for k, _sr in enumerate(_slots)]
    _across_step = _across_seeds(_slot_vd_step, _slot_vv_step, _agree)
    _named_incs = [ev._named(c["x"]) if c is not None else None for c in _incs]
    _seeds_block = {
        "n_slots": _n_seeds, "base_seed": int(seed),
        "run_one_after_another": _lanes_used[0] <= 1, "seeds_parallel": _lanes_used[0],
        "slots": [{"slot": k + 1, "seed": _sr.get("seed"),
                   "replaced": _sr.get("replaced"),
                   "crashed": bool(_sr.get("crashed")), "error": _sr.get("error"),
                   "calls": (_sr.get("n_calls") if not _sr.get("crashed") else _sr.get("calls")),
                   "verdict": _slot_vd[k],
                   "how": (None if _sr.get("crashed") else (_sr["vd"] or {}).get("how")),
                   "step_rule_verdict": _slot_vd_step[k],
                   "step_rule_how": (None if _sr.get("crashed") else (_sr.get("vd_step") or {}).get("how")),
                   "step_rule_variables": _slot_vv_step[k],
                   "ended": (None if _sr.get("crashed") else _sr["convergence"].get("ended")),
                   # what replay.replay_native needs for THIS seed (codex A3d #4): the kit rule's result and the
                   # optimizer's own step records (loop / generation starts, SPOTPY's loop record, pymoo's per
                   # generation record)
                   "native": (None if _sr.get("crashed") else _sr["convergence"].get("native")),
                   "settle": (None if _sr.get("crashed") else _sr["convergence"].get("settle")),
                   "optimizer_termination": (None if _sr.get("crashed") else
                                             {k: (_sr["convergence"].get("optimizer_termination") or {}).get(k)
                                              for k in ("status", "loop_starts", "generation_starts", "loop_record",
                                                        "native_verdict", "stopped_by_kit_at_loop",
                                                        "stopped_by_kit_at_generation")}),
                   "dream": (None if _sr.get("crashed") else _sr["convergence"].get("dream_rhat")),
                   "variables": _slot_vv[k],
                   "incumbent_call": (_incs[k] or {}).get("call"),
                   "incumbent_losses": (_incs[k] or {}).get("losses"),
                   # the loss the pick compares (single objective: the weighted scalar the optimizer minimized)
                   "incumbent_scalar": (_incs[k] or {}).get("scalar"),
                   # trade-off: was the seed's compromise admissible under protection (the pick puts
                   # admissible seeds first, design §1.7)
                   "incumbent_admissible": ((_incs[k] or {}).get("admissible") if multi and _incs[k] is not None
                                            else None),
                   # trade-off: the slot's frozen offset and scale — the pick uses the LOWEST slot's that
                   # reached t0 for every seed (design §1.7), so a reader can redo the pick from the report
                   "frozen_t0": ((_incs[k] or {}).get("t0") if multi and _incs[k] is not None else None),
                   "frozen_offset": ((_incs[k] or {}).get("z") if multi and _incs[k] is not None else None),
                   "frozen_scale": ((_incs[k] or {}).get("s") if multi and _incs[k] is not None else None),
                   "incumbent_panel": {v: {m: ((_incs[k] or {}).get("panel") or {}).get(v, {}).get(m)
                                           for m in _required.get(v, [])}
                                       for v in _required if _required[v]} if _incs[k] is not None else None,
                   "incumbent_params": _named_incs[k]}
                  for k, _sr in enumerate(_slots)],
        "agreement": _agree,
        "returned": ({"slot": _ret["slot"], "seed": _ret["seed"], "why": _rhow} if _ret else {"why": _rhow}),
        "note": _seeds_note,
        "run_verdict": _across["run_verdict"], "how": _across["how"],
        "variables_across_seeds": _across["variables"],
        "step_rule_run_verdict": _across_step["run_verdict"], "step_rule_how": _across_step["how"],
        "step_rule_variables_across_seeds": _across_step["variables"],
        "param_spread": _param_spread([d for d in _named_incs if d is not None]),
        # every attempt's calls, crashed ones included (= this call's search rows in eval_history.jsonl;
        # phase_counts covers the whole log, so on a resume it also holds the earlier attempts)
        "total_search_calls": sum(int((_sr.get("n_calls") if not _sr.get("crashed") else _sr.get("calls")) or 0)
                                  + int((_sr.get("replaced") or {}).get("calls") or 0) for _sr in _slots),
        "search_wall_s": round(_search_wall, 3)}
    if not any(not _sr.get("crashed") for _sr in _slots):
        ev.restore_originals()
        return {"status": "search_crashed", "algorithm": algo,
                "reason": "every seed's search crashed (each slot re-run once)",
                "seeds": _seeds_block, "budget_plan": budget_plan, "phase_counts": phase_counts(workdir),
                "split_provenance": _prov_now(), "triage": _triage}
    # the RETURNED seed's search is the one reported below (the first completed slot when no seed has a
    # finite incumbent — the finite check below then reports the failure)
    _pick = next(_sr for _sr in _slots if not _sr.get("crashed")) if _ret is None else _slots[_ret["slot"] - 1]
    result, convergence, _end = _pick["result"], _pick["convergence"], _pick["end"]
    if _ret is not None:
        # the reported result is the returned seed's CALIBRATION incumbent (design §1.5 option a) —
        # for a trade-off search the rule's compromise, never the backend's own pick
        result.best_x = list(_ret["x"])
        # the same FORM as before: one loss per objective for a trade-off search, [the scalar] for a
        # single-objective one (the per-objective vector is in best_losses_per_objective)
        result.best_loss = list(_ret["losses"]) if multi else [float(_ret["scalar"])]
    convergence["seeds"] = _seeds_block
    convergence["seeds_agree"] = _agree.get("agree")
    convergence["run_verdict"] = {"verdict": _across["run_verdict"], "how": _across["how"],
                                  "variables": _across["variables"]}
    convergence["step_rule_run_verdict"] = {"verdict": _across_step["run_verdict"], "how": _across_step["how"],
                                            "variables": _across_step["variables"],
                                            "note": "the per-call step rule, recorded only"}
    convergence["per_seed"] = _seeds_block["slots"]

    # All evaluations failed/infeasible -> NOT a successful calibration (codex :123).
    # Restore the original inputs (the evaluator wrote candidates as it went) so the
    # workdir isn't left at a garbage vector, and surface a failure status.
    finite_best = bool(result.best_x) and all(
        math.isfinite(v) for v in (result.best_loss or [float("inf")]))
    if not finite_best:
        ev.restore_originals()
        return {"status": "failed_no_finite_solution", "algorithm": algo,
                "n_evaluations": result.n_evaluations,
                "reason": "no evaluation produced a finite loss (all runs failed/infeasible)",
                "backend_notes": result.notes, "budget_plan": budget_plan,
                "convergence": convergence, "phase_counts": phase_counts(workdir),
                "split_provenance": _prov_now(), "triage": _triage}

    # ---- HOLDOUT INPUTS (shared by front-selection and the final gate) ----
    holdout_spec = (contract.get("strategy", {}) or {}).get("holdout")
    _front_holdout = None
    if holdout_spec:
        band_ceilings = _holdout_band_ceilings(objs, headline_objectives)
        probe_x = [(lo + hi) / 2.0 for lo, hi in zip(lower, upper)]   # off-optimal probe
        baseline_x = [fwd[p["name"]](float(p.get("default",
                          (float(p["range"][0]) + float(p["range"][1])) / 2.0))) for p in _params]
        # The old opt-in front selection BY VALIDATION (KDT_CALIB_FRONT_SELECT=1) is no longer used
        # to pick the member: the reported result is the rule's calibration compromise (design §1.5
        # option a, build step 6); validation is read once, afterwards, only to assess it. Protection
        # is `strategy.protect` in the contract (KDT_CALIB_PROTECT is not read either).
        if os.environ.get("KDT_CALIB_FRONT_SELECT") == "1" and multi:
            convergence.setdefault("warnings", []).append(
                "KDT_CALIB_FRONT_SELECT=1 is ignored: the member is never chosen on validation data "
                "(design §1.5 option a); the reported result is the calibration compromise")
            print(f"  [calib] WARNING {convergence['warnings'][-1]}", flush=True)

    # Leave inputs at the BEST params (TYPED-decoded, identical to what was scored —
    # codex calib.py:198), not raw floats / last-tried.
    from .applicator import write_param
    from .evaluator import decode_value
    def _apply_best():
        out = {}
        for p, xi in zip(_params, result.best_x):
            val = decode_value(p, xi, inv[p["name"]])
            out[p["name"]] = val
            if _inj_mode == "applicator":            # runner mode: best params are case-scoped only
                try:                                 # (no central address to write; calib_run.py injects)
                    write_param(p["address"], val, workdir)
                except Exception as _e:
                    out[p["name"]] = f"<apply failed: {_e}>"
        return out
    best_named = _apply_best()

    # Capture the BEST candidate's METRICS for the record (the human-readable "after": nse/pbias/...).
    # Already computed during optimization -> read from the eval cache by best_x BEFORE the holdout
    # re-runs change the active split (2026-07-19: these were None in the DB because only best_loss was
    # surfaced). Try the optimization split, then common fallbacks.
    _train_metrics = None
    try:
        _bn = ev._named(result.best_x)
        # never trust a stale env split here (codex round-4): training resolved to "calibration"
        for _sp in ("calibration", None):
            _hit = ev._metrics_cache.get(ev._cache_key(_bn, _sp))
            if _hit:
                _train_metrics = _hit
                break
    except Exception:
        _train_metrics = None

    # HOLDOUT GATE (codex C6) — validate the best params on held-out data before
    # calling the calibration promotable. A calibration that doesn't generalize is
    # NOT promotable to the canonical KI no matter how good its calibration loss.
    # (holdout_spec / band_ceilings / probe_x / baseline_x were computed above so the
    # optional front-selection could reuse them.)
    holdout = None
    promotable = True
    if holdout_spec:
        from .holdout import validate_holdout
        # Reuse the holdout the front-selection already computed for the committed member
        # (else validate the sum-knee best_x as before).
        holdout = _front_holdout if _front_holdout is not None else validate_holdout(
            ev, objs, result.best_x, holdout_spec, probe_x=probe_x,
            band_ceilings=band_ceilings, baseline_x=baseline_x)
        _apply_best()   # re-apply best params after the holdout re-runs wrote best_x
        # §2.9 on validation: the reported result's holdout metrics (the returned seed's calibration
        # incumbent — for a trade-off search the rule's compromise; build step 6)
        try:
            from .verdict import standards as _standards_v
            _hm = ev._metrics_cache.get(ev._cache_key(ev._named(result.best_x), "holdout"))
            _stds = (convergence.get("verdict") or {}).get("standards")
            for _v, _st in (_stds.items() if isinstance(_stds, dict) else []):
                _st["validation"] = (_standards_v(_conv, _v, _st.get("obs_shape"), _hm, len(_panel.variables) == 1,
                                                  _band_override(_stop_translated, _v))
                                     if _hm else
                                     (f"not assessed: holdout inconclusive ({holdout.get('reason')})"
                                      if holdout.get("inconclusive") else "holdout metrics not available"))
                _st["validation_member"] = "the reported result (the returned seed's calibration incumbent)"
                if holdout.get("inconclusive"):
                    _st["validation_note"] = f"holdout inconclusive: {holdout.get('reason')}"
        except Exception as _e:
            _stds = (convergence.get("verdict") or {}).get("standards")
            for _st in (_stds.values() if isinstance(_stds, dict) else []):
                _st["validation"] = f"not assessed ({type(_e).__name__})"
        # promotable only if holdout explicitly PASSED (None/inconclusive => gate it)
        promotable = (holdout.get("passed") is True)
    else:
        # No holdout declared -> NOT promotable (C6 makes it mandatory for promotion).
        promotable = False

    return {"status": "completed", "algorithm": algo, "multi_objective": multi,
            "best_x": result.best_x, "best_params": best_named,
            "best_loss": result.best_loss, "train_metrics": _train_metrics,
            "best_losses_per_objective": (_ret["losses"] if _ret is not None else None),
            "pareto_x": result.pareto_x, "pareto_f": result.pareto_f, "n_evaluations": result.n_evaluations,
            "holdout": holdout,
            "holdout_validated": (holdout.get("passed") if holdout else None),
            "promotable": promotable,    # gate for writing best params to CANONICAL
            "objectives": [o.name for o in objs],
            "posterior": result.posterior,      # DREAM per-parameter posterior summary (None otherwise)
            "stopped_early": bool(getattr(result, "stopped_early", False)),   # the kit's convergence rule ended it
            "backend_notes": result.notes,
            # checkpoint 2026-09-16: consumption evidence travels with the result. n_unproven > 0 means
            # this box was fitted WITHOUT every knob proven to reach the model — visible, not implied.
            "consumption_proof": _cproof,
            "consumption_coverage": consumption_coverage(_cproof, len(_params)),
            "objective_smoke_test": _smoke,
            # ── the 2026-09-27 convergence work: three blocks that make a budget accountable ──
            # budget_plan: where the cap came from (pilot timing, machine, allowance, kappa).
            # convergence: where the search stopped improving on the whole metric panel, and how
            #              much it still moved after that point.
            # phase_counts: where the evaluations went — a total alone cannot separate search
            #              effort from commissioning, the consumption proof and the holdout gate.
            "budget_plan": budget_plan,
            "convergence": convergence,
            # which calls echoed the split they were asked for (design §1 "Data within a search", gap 2p)
            "split_provenance": _prov_now(),
            # triage before the search: route, reason, proof summary, per-target fit (information only)
            "triage": _triage, "staged_warning": _staged_warning,
            "budget_used": {"cap": int(budget), "n_evaluations": result.n_evaluations,
                            "stopped_early": bool(getattr(result, "stopped_early", False)),
                            # the cap is PER SEED; n_evaluations is the returned seed's search
                            "seeds": _n_seeds,
                            "total_search_calls_all_seeds": convergence["seeds"]["total_search_calls"]},
            "phase_counts": phase_counts(workdir)}


def _make_backend(algo):
    if algo in ("dds", "sceua", "dream"):
        from .backends.spotpy_backend import SpotpyBackend
        return SpotpyBackend(algorithm=algo)
    if algo == "madr":                                  # agentic rung (2026-08-30): MADR proposes, the kit scores
        from .backends.madr_backend import MadrBackend
        _m = (os.environ.get("KDT_MADR_SPEC") or "{}")
        try:
            _spec = json.loads(_m)
        except Exception:
            _spec = {}
        return MadrBackend(**{k: v for k, v in _spec.items()
                              if k in ("provider", "model", "model_key", "mc_candidates", "mc_every", "patience_rounds", "max_rounds", "timeout_sec")})
    if algo in ("nsga2", "nsga3", "moead"):
        from .backends.pymoo_backend import PymooBackend
        return PymooBackend(algorithm=algo)
    if algo in ("surrogate", "botorch", "smt", "bo"):
        from .backends.surrogate_backend import SurrogateBackend
        return SurrogateBackend(library="smt" if algo == "smt" else "botorch")
    if algo in ("pestpp_ies", "pestpp_glm", "ies", "glm"):
        from .backends.pestpp_backend import PestppBackend
        return PestppBackend(variant="glm" if algo in ("pestpp_glm", "glm") else "ies")
    return None


# ---- safe constraint expression evaluator (codex calib.py:163) -------------
# raw eval() is unsafe even with __builtins__ cleared. Parse a restricted AST over
# the named scalar parameters only — arithmetic + comparison + boolean, no calls,
# no attribute access, no names beyond the parameters.
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_BOOL = {ast.And: all, ast.Or: any}


def _safe_eval(node, names):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"unknown name in constraint: {node.id}")
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_safe_eval(node.left, names), _safe_eval(node.right, names))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _safe_eval(node.operand, names)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left, names); ok = True
        for op, comp in zip(node.ops, node.comparators):
            right = _safe_eval(comp, names)
            if type(op) not in _CMP or not _CMP[type(op)](left, right):
                ok = False; break
            left = right
        return ok
    if isinstance(node, ast.BoolOp):
        # real Python short-circuit (codex calib.py:288) — don't eval dead branches
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not _safe_eval(v, names):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if _safe_eval(v, names):
                    return True
            return False
    raise ValueError(f"disallowed expression element: {ast.dump(node)[:60]}")


def _make_constraints(contract):
    exprs = contract.get("constraints") or []
    if not exprs:
        return None
    trees = [ast.parse(e, mode="eval") for e in exprs]   # parse once; reject bad syntax early

    # WRF-Hydro 2026-09-09: a constraint naming a parameter that is NOT in the box used to make
    # _ok() raise NameError on every candidate, which `except Exception: return False` turned into
    # "every candidate is infeasible" — silently. A search would burn its whole budget, accept
    # nothing, and look like a modelling failure rather than a one-line contract error. Our own
    # contract carried `gw_zinit <= gw_zmax` after gw_zinit had been dropped from the box.
    # Resolve names ONCE, up front, and fail loudly instead.
    declared = {q["name"] for q in (contract.get("parameters") or []) if isinstance(q, dict) and "name" in q}
    used = set()
    for t in trees:
        for node in ast.walk(t):
            if isinstance(node, ast.Name):
                used.add(node.id)
    unknown = sorted(used - declared)
    if unknown:
        dropped = {d.get("name") for d in (contract.get("dropped_parameters") or []) if isinstance(d, dict)}
        hint = ""
        if set(unknown) & dropped:
            hint = (f" {sorted(set(unknown) & dropped)} appear in dropped_parameters — a constraint "
                    f"on a dropped parameter must be removed with it.")
        raise ValueError(
            f"constraint references parameter(s) not in the box: {unknown}. Declared: "
            f"{sorted(declared)}. Every candidate would be marked infeasible and the search would "
            f"return nothing, with no error naming the cause.{hint}")

    def _ok(named: dict) -> bool:
        try:
            return all(bool(_safe_eval(t, named)) for t in trees)
        except Exception:
            return False
    return _ok
