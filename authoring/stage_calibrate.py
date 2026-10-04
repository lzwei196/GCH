"""stage_calibrate — the self-improve KI <-> calibration-kit integration.

Runs AFTER a model has been made to RUN CORRECTLY and validated (the loop's job).
This stage tunes the model's PARAMETERS to fit its dag-defined obs:

  A. DEVELOP calibration capability (if missing): a calib-dev agent authors, into
     the model's KI, a `calibration.yaml` contract (params + addresses + ranges +
     runner + holdout) and a `tools/calib_run.py` (programmatic run-with-current-
     params -> metrics dict, honoring KDT_CALIB_SPLIT for holdout). This is "the
     calibration dev capability added to the model's KI". The new tool is gated by
     the SAME independent codex review as any other tool-build.
  B. RUN the calibration engine (calibration_kit.calib.calibrate): the contract's
     runner is the per-eval run+score; an optimizer (DDS default / NSGA-II multi-obj
     / BoTorch surrogate for expensive) searches; best params are applied back.
  C. GATE on the holdout: only a holdout-PASS calibration is promotable; the report
     carries best_params + improved metric + holdout verdict.

Opt-in via env KDT_CALIBRATE=1 so it never disturbs the existing pipeline. Lazy
imports of orchestrator helpers avoid a circular import.
"""
from __future__ import annotations
import os
import sys
import json
import shutil
import datetime
from pathlib import Path


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _agent_timeout(default_s: int = 7200) -> int:
    """Wall-clock cap for a capability-dev / repair AGENT. TIME IS NOT A CONSTRAINT when developing a
    KI's calibration capability (user directive): a complex model (SWAT+ TxtInOut/HRUs, coupled chains)
    legitimately needs longer than the old hard-coded 1800s, and a timeout there throws away a nearly
    finished authoring pass (observed: 1.8 MB of stream discarded at the 30-min wall, then a full retry
    from scratch). Default 2 h; override with KDT_CALIB_AGENT_TIMEOUT_S (0/unset-invalid -> default)."""
    v = os.environ.get("KDT_CALIB_AGENT_TIMEOUT_S", "")
    try:
        n = int(v)
        return n if n > 0 else default_s
    except (TypeError, ValueError):
        return default_s

# make the sibling calibration_kit importable
_KDT_ROOT = str(Path(__file__).resolve().parent.parent)
if _KDT_ROOT not in sys.path:
    sys.path.insert(0, _KDT_ROOT)


def enabled() -> bool:
    return os.environ.get("KDT_CALIBRATE", "") in ("1", "true", "yes")


CALIB_DEV_PROMPT = """# Calibration-Capability Dev Agent: {model_id}

The model already RUNS CORRECTLY and was validated (metric below). Your job is to
add CALIBRATION capability to its KI so a generic optimizer can tune its parameters
to better fit the dag-defined obs. Author TWO files into the KI and nothing else.

## TARGET CASE — you MUST build for THIS gauge/obs (not another site)
{case_block}
CRITICAL: the KI's SKILL.md / docs may showcase a DIFFERENT validated site (a worked
example on another basin). You must calibrate the case named above, scoring ITS gauge/obs.
Find this case's OWN run recipe by matching the case_id/obs to the model's run scripts and
staged outputs (e.g. grep the model dir for the case name; the case's already-produced
upstream inputs — forcing / land-model runoff / delineation — are typically staged under
`outputs/<case>*/`). If you cannot locate this case's inputs/recipe, return a BLOCKER that
names exactly what's missing — do NOT silently fall back to a different site that happens to
be runnable. Scoring the wrong gauge is a FAILED capability, not a success.

## Validated run (what calibration must reproduce + improve)
- KI: {ki_path}
- dag.yaml outputs + the obs it was scored against (read dag.yaml + SKILL.md).
- Prior validated metric: {prior_metric}
- The run+score recipe the loop already used: read real_case_result.json / the KI's
  run tool(s). Calibration must run the SAME model+obs, just with varied parameters.

## 1) `{ki_path}/calibration.yaml` — the contract
Follow calibration_kit/CALIBRATION_YAML_SCHEMA.md EXACTLY. Choose the CALIBRATABLE parameters ONCE,
from the KI — the engine searches EXACTLY the parameters you list, all together, in every seed (no
screen prunes them afterwards, so every extra parameter costs search calls). Start from the DAG's
influence edges (`{ki_path}/dag.yaml` → influence.edges): the edges with `sensitivity_grade` HIGH/MEDIUM
are the params the model's OWN sensitivity-analysis literature identified as controlling — carry them +
their `citations` into each param's `cite`. Keep the parameters that could control the model's error
mode (e.g. if a volume bias is the issue, include the recharge/baseflow lever even if it lives in a
different input file); leave out LOW-graded or redundant ones, and say in `cite` why each kept one is in.

### PARAMETER RANGES MUST BE GROUNDED IN THE MODEL'S OWN DOCUMENTATION / LITERATURE — not guessed
A guessed-wide range is a main cause of a failed calibration: the pilot, the consumption proof and the
search all sample across the box, and a bound outside the model's physically-valid region makes the model
CLAMP the value (e.g. a depth/width multiplier that pushes a channel below the model's HMIN/WMIN floor, or
a Manning's n past its stable band). A clamped value ≠ the requested value, so the runner's read-back
guard rejects the eval → those calls give no finite metrics (and a parameter proven not to move the
objective can make the run `not_calibratable`). DERIVE each range from sources, in this order:
  1. The model's OFFICIAL documentation/manual — read the KI's `docs/REFERENCES.md` for the manual URL +
     key papers, any manual shipped on disk (e.g. under the model package's `doc/`), and `docs/
     gathered_papers.json` / `docs/papers_index.md`. These give the parameter's PHYSICAL bounds + the
     model's internal floors/ceilings (HMIN, WMIN, valid Manning band, empirical W=WC·Q^WP / H=HC·Q^HP
     coefficients, etc.).
  2. The dag.yaml param `notes` + `citations` — they often already state the default, the floor, and the
     coefficient the model uses.
  3. Peer calibration studies (the KI's gathered papers) — the range other studies actually VARIED the
     parameter over is the calibration range; use it.
Set `range: [lo,hi]` = the LITERATURE calibration range, kept STRICTLY INSIDE the model's clamp-free,
physically-valid region (so the value applies EXACTLY at both bounds). Put the source in each param's `cite`.
If the literature gives no range for a lever, bound it conservatively around the documented default so it
never trips the model's floors. VERIFY in COMMISSIONING (below) that the param reads back == requested at
BOTH range bounds — a bound that clamps is WRONG; tighten it.

For each parameter capture:
  name, type, units, range [lo,hi] (literature-grounded + clamp-free, see above), default, transform,
  scope (+ zone_key if not global). In APPLICATOR mode also give each param an edit
  `address` (yaml_path / json_path / ini_key / text_token / table_cell / namelist —
  the GENERIC formats; pointing at the EXACT token; may use different files per param).
  In RUNNER mode `address` is OPTIONAL documentation — calib_run.py injects via
  `KDT_CALIB_PARAMS` (use runner mode for model-specific formats: modflow_pkg / swmm_inp /
  api, or when the KI regenerates its inputs each run).
  `targets` = the dag output var(s) you calibrate toward (the variable the FORCED obs
  of THIS run measures — score THAT gauge/site, not a different validated one).
  `runner`: kind=subprocess, command runs `tools/calib_run.py` (below) -> metrics JSON.
  `strategy`: cost_class, a MANDATORY `holdout` (years|sites|regions, fraction), and Constraints if
  params have ordering relations. WHICH parameters to calibrate is YOUR decision, made ONCE from the
  KI: the engine calibrates exactly the parameters you list, in one search (there is no Morris screen
  and no staged escalation — do NOT write `staged`).
  `strategy.budget` — ASK the user three things first (when a user is present; if nobody answers,
  use the defaults below):
    1. "Roughly how long does one model run take, if you know?" -> `run_time: <e.g. 90s>` (a 3-run
       pilot then confirms it; without it the engine times the model in a 10-run pilot).
    2. "How long can this run take? For example, 3 days on this server." -> `mode: measured`,
       `allowance: <that time>` (s / m / h / d, e.g. 72h). The engine computes the cap from the
       measured run time — do not guess a budget from the model's reputation. With no allowance:
       `mode: fixed` with `max_evaluations` AND a rationale for that number.
    3. "When the search converges, should it stop there to save time, or keep going to the budget to
       confirm the result held?" -> `strategy.convergence.mode: stop | keep_going`. Explain: stopping
       saves compute; continuing shows whether convergence was real and what stopping early would
       have cost. DEFAULT when nobody answers: keep_going.
    DDS cannot stop early — its search is planned around its budget (Tolson & Shoemaker 2007). If the
    user wants to stop at convergence, offer a shorter DDS search (a smaller budget, planned as such)
    or an optimizer that can stop (SCE-UA, NSGA-II).
    The cap is PER SEED; the engine runs 3 seeds by default (`seeds`). Write `parallel_safe: true`
    ONLY if two copies of calib_run.py in different `--workdir`s share NOTHING (no shared output file,
    no global state, no lock one copy waits on) — otherwise leave it out.
  `strategy.convergence.variables.<var>.kind` for EVERY target: `flow` (a non-negative discharge-type
  series), `series` (any other time series), `snapshot` (e.g. yield at harvest), `categorical` (class
  or event comparisons: flood extent, fire maps, event classes). Leave the other convergence settings
  at their defaults unless the KI gives a reason to change them; the engine records where the search
  converges on a PANEL of metrics (never NSE alone) and reports it.
  OPTIMIZER — you must REASON this choice from THIS problem; do NOT default. Set BOTH
  `default_algorithm` AND a 1-2 sentence `algorithm_rationale` that justifies it from the
  parameter dimensionality (count), the `cost_class` (cheap/moderate/expensive per eval),
  the objective structure implied by your `targets` (single- vs multi-objective), and whether
  posterior UNCERTAINTY / parameter intervals are the goal. Options:
    - `dds`   — cheap, smooth / low-gradient SINGLE-objective search (few params, tight budget)
    - `sceua` — rugged / MULTIMODAL single-objective response surface
    - `dream` — when posterior UNCERTAINTY / parameter intervals are wanted (single-objective, Bayesian)
    - `nsga2` — MULTI-OBJECTIVE: when `targets` span >=2 distinct variables (Pareto trade-off)
  The `algorithm_rationale` is a REQUIRED audit record of your reasoning — write why THIS
  optimizer fits THIS problem, not a generic description. The choice must be yours, grounded
  in the above — never a fixed default.
  NOTE: the kit auto-keeps only the metric families your runner actually emits, so
  it's fine if the dag declares an extra family (e.g. timing) you don't compute.

## 2) `{ki_path}/tools/calib_run.py` — programmatic run+score (NOT an agent)
`python calib_run.py --workdir <wd> --out <metrics.json>`. It MUST run the model via the KI's
EXISTING run tools (reuse them; NEVER reimplement the model), compute the gate-valid metrics
(nse/kge/pbias — the same the dag gates on; ki_tools_common.metrics.all_metrics if available), and
write them as JSON to --out. On any failure: write nothing / exit nonzero (a missing metrics file =
+inf, never a fake pass). Honor env KDT_CALIB_SPLIT ("calibration"|"holdout"): score exactly the
requested subset, and ECHO the subset you scored as metrics["__kdt__"]["split"]. When KDT_CALIB_SPLIT is
unset, score the full record and echo "full". If the model truly cannot score the requested subset, score
the full record and echo "full" (the kit then reports that the run cannot support a calibration-only
claim). Never echo a subset you did not score. Keep
it FAST — called 100s of times in a FRESH per-candidate workdir; do NOT re-download/re-setup per eval
(that is a one-time prepare step).

### Scores the kit watches (convergence check)
Besides your own scores (keep them as the KI defines them), put the watched scores for EACH target in the kit's
reserved section of the output JSON, under the target's `var` name:
    out["__kdt__"]["panel"] = {{"<var>": {{"r": ..., "alpha": ..., "beta": ..., "pbias": ..., "nse": ..., "kge": ...,
                                         "lnnse": ..., "nrmse": ...}},
                               "<second var>": {{...}}}}
ADD it to your existing `__kdt__` block (keep applied_params, case_id, split). Compute each on the SAME (sim, obs)
pairs you scored for that target (pairs where both are finite; float64; population std):
    r      = Pearson correlation of sim and obs (at least 3 pairs)
    alpha  = std(sim) / std(obs)
    beta   = mean(sim) / mean(obs)
    pbias  = 100 * (mean(sim) - mean(obs)) / mean(obs)      (negative = too little)
    nse    = 1 - sum((sim - obs)^2) / sum((obs - mean(obs))^2)
    kge    = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    lnnse  = NSE of log(x + 0.01*mean(obs)) for sim and obs (leave it out if mean(obs) <= 0 or any value <= -0.01*mean(obs))
    nrmse  = 100 * RMSE / |mean(obs)|
Use these exact lower-case keys. Leave out any score that is undefined (never write NaN). For a single-value target (e.g. one regional yield) only
`pbias` and `nrmse` apply. However you score (a time series, grid cells, a region), report these the same way.
Without them the kit cannot judge convergence and says "unknown".
When the environment has `KDT_CALIB_EMIT_SERIES=1` (the kit sets it on ONE run, its pilot's default run), ALSO
save the (sim, obs) pairs you scored — the same pairs, with their dates when there are any — as
`<workdir>/kdt_series_<target var>.npz` with numpy `savez`: float64 arrays `sim` and `obs` and, if dated, `date`
as ISO text "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM" in a PLAIN numpy string array (NOT an object array — pandas gives
object arrays), e.g. `np.asarray(idx.strftime("%Y-%m-%d"), dtype="U16")`; check it by reading EVERY array back:
`with np.load(path, allow_pickle=False) as z: [z[k] for k in z.files]` (opening the file alone does not check it).
Then ADD the paths to your existing `__kdt__` block (keep applied_params, case_id, split), one entry per target:
`out["__kdt__"]["series"] = {{"<var>": "<its path>", "<second var>": "<its path>"}}`. If saving fails, write the
reason as `out["__kdt__"]["series_error"]` and delete any half-written file. The kit measures from it how much change is real
(otherwise it uses fixed values). Never let this step fail the run: if it cannot be written, skip it.
Compute all scores in float64.

### Parameter injection depends on `injection.mode` in calibration.yaml:
- **applicator mode (default):** the kit has ALREADY written this candidate's params into the inputs
  (via each param's `address`). calib_run.py just READS the current inputs and runs. Use this ONLY when
  the model consumes stable files directly.
- **runner mode (required if the KI REGENERATES its inputs each run, or params live in a model-specific
  format — MODFLOW/SWMM/API):** the kit did NOT write params. calib_run.py MUST:
    1. read the candidate vector from the JSON at env `KDT_CALIB_PARAMS` ({{"name": value, ...}});
    2. INJECT each value the model's OWN way (native lib — flopy for MODFLOW, pyswmm for SWMM — or KI
       tools), AFTER any input-regeneration step so it isn't overwritten;
    3. read each value BACK from the effective artifact the model consumes; if any can't be written+read,
       exit nonzero with NO metrics;
    4. include a reserved block in the output JSON:
       `"__kdt__": {{"applied_params": {{"name": <value you actually applied>, ...}},
                    "case_id": "<the TARGET CASE case_id above>",
                    "scored_obs": "<the gauge/obs id + file you scored against>"}}` — the kit diffs
       requested vs applied EVERY eval and scores +inf on any mismatch (a non-injecting runner is never trusted).
       ECHO EVERY param you were handed — i.e. ALL keys present in the KDT_CALIB_PARAMS JSON, including any held
       at their default; omitting a handed key fails the eval closed.
       The `case_id` / `scored_obs` fields SELF-DECLARE which gauge this eval scored — they MUST name the TARGET
       CASE gauge above, so scoring the wrong site is detectable (a mismatch is a FAILED capability).
       CRITICAL — the declaration must be HONEST, not a hardcoded label:
         * PIN the scored gauge/obs to the TARGET CASE. Do NOT read the gauge/obs/lat/lon/area from
           overridable env vars that could redirect scoring to a DIFFERENT gauge. If you must read a
           location from the environment, VALIDATE it equals the target case and exit nonzero otherwise.
         * DERIVE `scored_obs` (and `case_id`) from the SAME resolved gauge/obs the run actually scored —
           set them from the variables you used to read the obs + extract the model cell, so the declaration
           CANNOT diverge from what was scored. A hardcoded literal that a code path can contradict is a
           REJECTED capability (a review will look for exactly this divergence).

### DRIVER ROBUSTNESS CONTRACT — a review REQUIRES all of the below (recurring REQUEST_CHANGES):
  * ENFORCE WHATEVER THE OBS CONTRACT EXPLICITLY DECLARES — and only those, domain-appropriate fields; do not
    just read a value column, and do not invent/require fields the contract does not declare. For every
    obs-envelope field the TARGET CASE / calibration.yaml explicitly states (e.g. for a groundwater-level
    series: station identity, reference surface elevation, the exact month/date set, the observed
    min/max/range/mean), the obs loader MUST read and ASSERT it (exit nonzero, no metrics, on mismatch) so a
    changed or corrupted series for the SAME station cannot be silently scored. Reading only the value column +
    identity when MORE is declared is a REJECTED capability.
  * FAIL-CLOSED RUN-HEALTH. For any run-health diagnostic the model actually produces/claims (e.g. MODFLOW's
    `convergence_failures` / `max_budget_error_pct` / normal-termination), a MISSING or None value must FAIL the
    eval closed — never default it to a passing value via `or 0`. Do not fabricate diagnostics a model doesn't
    emit; but a diagnostic that is required/claimed and then absent/None = fail closed (no metrics, nonzero exit).
  * OPEN READ-ONLY SOURCES READ-ONLY. Any immutable obs/catalog source (an sqlite DB, a data file) MUST be opened
    read-only so scoring works on a read-only mount and in a read-only review sandbox: sqlite ->
    `sqlite3.connect(f"file:{{path}}?mode=ro", uri=True)`; text files -> open(path, "r"), binary/library-backed
    data -> "rb" or the library's own read-only mode. Opening an obs catalog in
    sqlite's default read/write/create mode fails on a read-only source (`unable to open database file`) — REJECTED.

COMMISSIONING (MANDATORY before you finish): run at the default vector, then a clearly-perturbed vector,
in separate workdirs; show applied_params and that the metric MOVES. If nothing moves, the injection isn't
reaching the scored run — return a BLOCKER, not "authored".
ALSO run ONCE exactly as the kit's pilot runs the default vector — with `KDT_CALIB_SPLIT=calibration` AND
`KDT_CALIB_EMIT_SERIES=1` set — and show that the output still carries applied_params / case_id / split, the
watched scores, and a `__kdt__.series` file whose arrays ALL read back with `allow_pickle=False`
(`with np.load(path, allow_pickle=False) as z: [z[k] for k in z.files]`).
ALSO — RANGE-BOUND read-back check (this is what makes the search SUCCEED, not just move): for EACH
parameter, run once at its range `lo` and once at its `hi` and confirm `__kdt__.applied_params[name]`
reads back EQUAL to the requested bound (within the runner's own read-back tolerance). A bound where the
model CLAMPS the value (applied ≠ requested) makes every call near that bound fail closed —
if you see that, TIGHTEN the range to the clamp-free region and re-test, do NOT ship the wide range. Only
finish when every parameter applies exactly at BOTH of its bounds.

Your FINAL message is a JSON object: {{"status":"authored","calibration_yaml":true,"calib_run":true,
"injection_mode":"applicator|runner","params":[...names...],"notes":"..."}}
"""


def _has_contract(ki_path) -> bool:
    return (Path(ki_path) / "calibration.yaml").is_file() \
        and (Path(ki_path) / "tools" / "calib_run.py").is_file()


def _capability_case_path(ki_path) -> Path:
    return Path(ki_path) / "calib" / "capability_case.json"


def _file_sha(path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def _contract_shas(ki_path) -> dict:
    return {"calibration_yaml_sha": _file_sha(Path(ki_path) / "calibration.yaml"),
            "calib_run_sha": _file_sha(Path(ki_path) / "tools" / "calib_run.py")}


def _capability_matches_case(ki_path, case) -> bool:
    """A PRESENT contract is reusable ONLY if it was authored AND codex-APPROVED for THIS case
    AND the on-disk files are byte-identical to what was approved. File existence alone is not
    enough (codex 2026-07-18): a Bengbu-authored contract must not be reused for a Jinghong run,
    a rejected/unreviewed contract must never be reused, and a driver MUTATED after approval
    (e.g. by the driver-repair agent) must be re-reviewed, not silently trusted. Provenance +
    content hashes live in calib/capability_case.json, written ONLY on APPROVE."""
    if not case or not case.get("case_id"):
        return True                       # no case pinned -> don't force redev on identity
    p = _capability_case_path(ki_path)
    if not p.is_file():
        return False                      # unknown provenance -> redevelop (fail closed)
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return False
    if rec.get("approved") is not True or str(rec.get("case_id")) != str(case.get("case_id")):
        return False
    # EXECUTED-OBS DRIFT (codex 2026-08-02): the review is now authoritative on the executed obs /
    # integrity flag. If either changed for this case since approval (e.g. a mislabel was backfilled),
    # the prior approval predates the executed-obs-aware review — re-review rather than reuse.
    if str(rec.get("resolved_obs") or "") != str((case or {}).get("resolved_obs") or "") or \
       str(rec.get("integrity_note") or "") != str((case or {}).get("integrity_note") or ""):
        return False
    # content must match what was approved (a post-approval edit invalidates reuse). BOTH hashes are
    # MANDATORY — a legacy/partial sidecar missing either hash can't prove the files are unchanged, so
    # it must redevelop (codex round-3: an absent hash previously fell through to reuse). Fail closed.
    cur = _contract_shas(ki_path)
    for k in ("calibration_yaml_sha", "calib_run_sha"):
        if not rec.get(k) or not cur.get(k) or rec.get(k) != cur.get(k):
            return False
    return True


def _write_capability_case(ki_path, case, vdict) -> None:
    """Record case provenance + approved-content hashes so the contract is reusable ONLY for this
    case AND only while the files are unchanged. Called ONLY on codex APPROVE."""
    try:
        p = _capability_case_path(ki_path); p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"case_id": (case or {}).get("case_id"), "obs_id": (case or {}).get("obs_id"),
               "resolved_obs": (case or {}).get("resolved_obs"),
               "integrity_note": (case or {}).get("integrity_note"),
               "approved": True, "verdict": vdict, "authored_at": _now()}
        rec.update(_contract_shas(ki_path))
        p.write_text(json.dumps(rec, indent=2, default=str))
    except Exception:
        pass


def _archive_capability(ki_path, tag) -> str | None:
    """Move a not-to-be-reused capability (rejected / unreviewed / wrong-case / post-approval-edit)
    out of the KI so a later run can't bypass review by finding stale files. Reversible (never
    deletes). The archive dir is UNIQUE (pid + microseconds) so concurrent archivals don't collide;
    a rename failure is surfaced (printed) rather than silently swallowed. Returns the archive path,
    or None if nothing was moved."""
    import os as _os
    ki = Path(ki_path)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    bk = ki / f".{tag}_{stamp}_p{_os.getpid()}"
    (bk / "tools").mkdir(parents=True, exist_ok=True)
    moved, failed = [], []
    for src, dst in ((ki / "calibration.yaml", bk / "calibration.yaml"),
                     (ki / "tools" / "calib_run.py", bk / "tools" / "calib_run.py"),
                     (_capability_case_path(ki), bk / "capability_case.json")):
        if not src.is_file():
            continue
        try:
            src.rename(dst); moved.append(src.name)
        except Exception as e:
            failed.append(f"{src.name}: {e}")
    if failed:      # surface, never silently proceed as if the KI is clean
        print(f"  [calib] WARNING: could not archive {failed} to {bk} — stale files may remain", flush=True)
    return str(bk) if moved else None


def _case_block(case) -> str:
    """Human-readable TARGET CASE block for the calib-dev prompt. Without this the agent
    has no signal which gauge/obs to score and may build for a different validated site
    the KI documents (the Bengbu-instead-of-Jinghong bug, 2026-07-18)."""
    if not case:
        return ("- (no explicit case supplied) — score the gauge/obs the KI's validated "
                "run_and_score recipe used for its most recent real case; if ambiguous, BLOCK.")
    c = dict(case)
    osv = c.get("obs_shape_by_var") or {}
    lines = [f"- case_id: {c.get('case_id')}",
             f"- obs / gauge id: {c.get('obs_id')}",
             f"- target quantity: {c.get('target_quantity')}  (dag var/obs_shape: "
             f"{', '.join(f'{k}->{v}' for k, v in osv.items()) or 'see dag.yaml'})",
             f"- determining metric to optimize: {c.get('determining_metric')}",
             f"- provenance of the validated coupling: {c.get('validated_run_id')}"]
    # AUTHORITATIVE executed obs (record-layer resolved evidence). When present it is what the validated
    # run ACTUALLY scored and OVERRIDES the label above if they disagree — build for the EXECUTED obs.
    if c.get("resolved_obs"):
        lines.append(f"- EXECUTED obs (AUTHORITATIVE — what the validated run actually scored): {c.get('resolved_obs')}")
    if c.get("integrity_note"):
        lines.append(f"- ⚠ INTEGRITY: {c.get('integrity_note')} Build for the EXECUTED obs, and set "
                     f"__kdt__.scored_obs/case_id from what you actually score.")
    return "\n".join(lines)


# Shared review payload + focus (codex 2026-08-02): the EXECUTED obs is authoritative over a stale
# target_quantity/obs LABEL. Both the calib-dev AND the driver-repair review MUST use these, or a
# mislabeled case that routes through a no-search route (no_baseline / not_calibratable) hits the same
# stale-label rejection.
_REVIEW_FOCUS_EXEC = (
    "CONFIRM the driver scores the case's gauge/obs; FLAG if it hardcodes a DIFFERENT validated site. "
    "AUTHORITY: when `executed_obs` is present it is what the validated run ACTUALLY scored and OVERRIDES "
    "the target_quantity/obs LABEL. If `integrity_note` flags a label-vs-executed mismatch, the LABEL is "
    "stale — APPROVE a driver that correctly scores the EXECUTED obs; do NOT reject it for matching the "
    "executed obs over the stale label.")


def _review_target_case(case) -> dict:
    _rc = case or {}
    return {"case_id": _rc.get("case_id"), "obs_id": _rc.get("obs_id"),
            "target_quantity": _rc.get("target_quantity"),
            "determining_metric": _rc.get("determining_metric"),
            "executed_obs": _rc.get("resolved_obs"),
            "integrity_note": _rc.get("integrity_note")}


REVISION_PROMPT = """# Calibration-Capability REVISION Agent: {model_id}

Your previously-authored `{ki_path}/calibration.yaml` + `{ki_path}/tools/calib_run.py` for the TARGET CASE
below were codex-reviewed and got {verdict}. FIX EXACTLY the issues listed by EDITING THE EXISTING two files
in place. Do NOT rewrite from scratch and do NOT change what already works — the reviewer accepted the rest,
so a regression will fail the re-review.

## TARGET CASE (unchanged — keep scoring THIS)
{case_block}

## Reviewer issues to fix (address ALL of them)
{issues}

Keep every existing guarantee: reuse the KI's OWN run tools (never reimplement the model); the honest
`__kdt__` case_id/scored_obs declaration derived from what you actually score; the DRIVER ROBUSTNESS CONTRACT
(assert the declared obs envelope; fail-closed run-health; open read-only sources read-only); and
literature-grounded, clamp-free parameter ranges (verify read-back at both bounds). Re-run your COMMISSIONING
checks after editing. Final message: {{"status":"revised","calib_run":true,"notes":"what you changed"}}.
"""


def develop_calibration_capability(model_id, ki_path, prior_metric, case=None, max_review_rounds=3):
    """Author calibration.yaml + tools/calib_run.py, codex-review, and — on REQUEST_CHANGES — feed the
    reviewer's issues back for up to `max_review_rounds` REVISION rounds (2026-08-03). The reviewer is
    strict and a complex driver (e.g. DSSAT's fixed-width FileX, MODFLOW's flopy round-trip) rarely lands
    on the first try; a one-shot gate recorded those as `capability_unavailable` even though a single fix
    round would pass. Attests + reuses ONLY on APPROVE. Returns (ok, verdict).
    `case` is threaded into the prompt so the agent scores the RIGHT gauge, not a different documented site."""
    import orchestrator as O
    from tool_reviewer import review_tool

    def _review():
        # Review BOTH attested files (codex 2026-08-03): _write_capability_case pins hashes of
        # calibration.yaml AND calib_run.py, so BOTH must be reviewed or an unreviewed/bad calibration.yaml
        # (wrong param ranges, wrong obs envelope) could be attested as approved.
        cyaml = (Path(ki_path) / "calibration.yaml").read_text()
        crun = (Path(ki_path) / "tools" / "calib_run.py").read_text()
        diff = ("--- /dev/null\n+++ b/calibration.yaml\n"
                + "".join("+" + l for l in cyaml.splitlines(keepends=True))
                + "--- /dev/null\n+++ b/tools/calib_run.py\n"
                + "".join("+" + l for l in crun.splitlines(keepends=True)))
        v = review_tool(diff=diff[:200_000],   # match review_tool's own 200k cap; 64k truncated large drivers
                        tool_summary={"tool": "calibration.yaml + tools/calib_run.py",
                                      "summary": "calibration contract + run/score driver",
                                      "origin": "calib_dev", "target_case": _review_target_case(case),
                                      "review_focus": _REVIEW_FOCUS_EXEC},
                        ki_path=ki_path, run_id=f"{model_id}_calibdev")
        return {"verdict": v.verdict, "issues": list(v.issues[:5]), "reviewer": v.reviewer}

    try:
        from ki_snapshot import take_snapshot
        take_snapshot(Path(ki_path), label=f"pre_calibdev_{model_id}")
    except Exception:
        pass

    vdict = {"verdict": "REJECT", "issues": [], "reviewer": "self"}
    # round 0 = initial authoring; rounds 1..N-1 = revisions fed the reviewer's own issues
    for rnd in range(max(1, int(max_review_rounds))):
        if rnd == 0:
            prompt = CALIB_DEV_PROMPT.format(model_id=model_id, ki_path=str(ki_path),
                                             case_block=_case_block(case),
                                             prior_metric=json.dumps(prior_metric, default=str)[:400])
        else:
            prompt = REVISION_PROMPT.format(
                model_id=model_id, ki_path=str(ki_path), case_block=_case_block(case),
                verdict=vdict.get("verdict"),
                issues="\n".join(f"  {i+1}. {x}" for i, x in enumerate(vdict.get("issues") or [])) or "  (see prior review)")
        result, output = O.run_claude_resilient(prompt, timeout=_agent_timeout(), max_turns=80)
        (Path(O.WORK_DIR) / model_id / (f"calib_dev_output{'' if rnd == 0 else f'_rev{rnd}'}.txt")).write_text(output or "")
        if not _has_contract(ki_path):
            return False, {"verdict": "REJECT", "reason": "calibration.yaml / calib_run.py not authored",
                           "reviewer": "self", "round": rnd}
        try:
            vdict = _review()
        except Exception as e:
            arch = _archive_capability(ki_path, "unreviewed_calibdev")
            return False, {"verdict": "REJECT", "reason": f"review failed: {e}", "archived_to": arch}
        vdict["round"] = rnd
        try:   # persist the latest codex verdict so a not-APPROVE outcome is recoverable
            (Path(O.WORK_DIR) / model_id / "calib_dev_review.json").write_text(json.dumps(vdict, indent=2, default=str))
        except Exception:
            pass
        if vdict["verdict"] == "APPROVE":
            _write_capability_case(ki_path, case, vdict)   # case + approved-content hashes
            return True, vdict
        if vdict["verdict"] == "WAIT":
            # WAIT = STAGED, pending HUMAN review (tool_reviewer escalation) — NOT a fixable review issue.
            # Do NOT revise or archive: leave the files staged and surface a pending verdict (codex 2026-08-03).
            vdict["pending_human"] = True
            return False, vdict
        # NOT approved (REQUEST_CHANGES / REJECT): leave the files in place so the NEXT revision round can EDIT
        # them (do not archive between rounds — the agent needs the current draft + the reviewer's issues).
        print(f"  [{model_id}] calib-dev review round {rnd}: {vdict['verdict']} ({len(vdict.get('issues') or [])} "
              f"issue(s)) — {'revising' if rnd < max_review_rounds - 1 else 'rounds exhausted'}", flush=True)

    # exhausted all rounds without APPROVE -> archive so nothing rejected/unreviewed lingers for reuse
    vdict["archived_rejected_to"] = _archive_capability(ki_path, "rejected_calibdev")
    return False, vdict


FIX_DRIVER_PROMPT = """# Calibration-Driver Repair Agent: {model_id}

{route_text}
Triage detail:
  default-run metrics (the pilot's first run): {baseline}
  reason: {reason}
  per-target detail and consumption-proof verdicts: {family_verdict}
  prior validated metric: {prior_metric}

## Your job (make the DEFAULT run run and score — fix the DRIVER, not the science)
1. Read `{ki_path}/calibration.yaml` (parameters, defaults, `targets`, the objectives' metric families) and
   `{ki_path}/tools/calib_run.py`; run calib_run.py ONCE at the default parameters yourself, EXACTLY as the kit's
   pilot does (with `KDT_CALIB_SPLIT=calibration` and `KDT_CALIB_EMIT_SERIES=1`), and read its output and log.
2. Find WHY that run gives no finite objective metric. Usual causes:
   - the model run itself fails (missing input/forcing, wrong executable or path, a default outside the
     model's valid region, the run dir not prepared inside `--workdir`);
   - the run succeeds but the driver scores nothing: wrong output file or variable, the scored window has
     no overlap with the obs, obs not found / all missing, a unit or date alignment that leaves 0 pairs;
   - it writes metrics under keys the objectives do not read (see CALIBRATION_YAML_SCHEMA.md: the
     objectives read the metric families' keys; the convergence check reads the watched scores in `__kdt__.panel`);
   - it exits early or writes NaN/inf on a path that should have scored.
3. FIX `tools/calib_run.py` (and, only if a default/address is wrong, `calibration.yaml`) so the default
   run scores finite metrics on the TARGET CASE gauge.
4. PROVE it: run calib_run.py at the defaults AND at 3 random in-range vectors; show every run gives finite
   metrics and that the metrics differ between vectors. Do NOT fake this — if you cannot make the default
   run score, say so and explain the blocker.

## MANDATORY requirements (a prior repair was REJECTED by codex for missing these):
1. HONOR `--workdir`: run the model inside the per-candidate `--workdir` the calibration
   kit passes (copy the canonical TxtInOut into `<workdir>/run/` and run there). Do NOT
   rewrite a single shared run dir — concurrent/retried candidates MUST be isolated.
2. STALE-OUTPUT SAFETY: delete the `--out` metrics file BEFORE running, AND ensure the
   exception/early-exit paths never leave a previous candidate's metrics behind. A failed
   run must write NOTHING (so the kit reads +inf), never a stale success artifact.
3. Prove (1) by running two candidates concurrently in different workdirs and showing
   they don't clobber each other; prove metric responsiveness by showing two different
   param vectors give two different metrics.

Edit ONLY `tools/calib_run.py` and `calibration.yaml`. Your FINAL message is JSON:
{{"status":"fixed"|"could_not_fix","what_was_wrong":"...","verification":"<the two runs + their differing metrics + the isolation proof>","files_changed":[...]}}
"""

DIAGNOSE_SETUP_PROMPT = """# Calibration Setup-Diagnosis Agent: {model_id}

{route_text}
This is a SETUP problem, not a tuning one.
Triage detail:
  default-run metrics (the pilot's first run): {baseline}
  reason: {reason}
  per-target detail and consumption-proof verdicts: {family_verdict}
  prior validated metric (elsewhere): {prior_metric}

{setup_job}3. Write findings to `{work_dir}/setup_diagnosis.md` (root cause + evidence paths +
   the concrete change that would fix the setup) and, if a learning-proposals queue is
   available, queue it as an out-of-scope proposal.

Your FINAL message is JSON:
{{"root_cause":"...","evidence":["path:line", ...],"recommended_fix":"...","in_scope_for_loop":true|false}}
"""


#: the setup-diagnosis job, by the triage route that sent it (Opus 8b/9 r2 #3): not_calibratable, and a
#: CERTIFIED module's no_baseline (its driver reproduced the validated run, so the setup is what differs)
_SETUP_JOB = {
    "not_calibratable": """## Your job (INVESTIGATE — do not fabricate, do not "fix" by changing metrics)
The search would move parameters the objectives cannot see.
1. Read `{ki_path}/calibration.yaml` (the parameters, `targets`, objectives and scored window),
   `{ki_path}/tools/calib_run.py`, the KI's run recipe and dag.yaml influence edges, and the obs the run
   is compared against (model_obs_map / the obs file the driver reads).
2. For EACH parameter, use its consumption-proof verdict to find why it cannot move the objectives:
   - DORMANT: the process the parameter controls is inactive in this case (e.g. a snow parameter in a
     snow-free basin, an irrigation lever with no irrigation) → the WRONG PARAMETERS were chosen: name the
     parameters of the processes that ARE active here (from dag.yaml influence edges and the KI docs);
   - OBJECTIVE_MASKED: the parameter changes the model output but not the scored metric → EITHER the scorer
     does not read the output this run produced (a fixed / cached / canonical output file, a run dir other
     than `--workdir`, an interior gauge scored against the outlet — check this FIRST: every parameter
     OBJECTIVE_MASKED usually means this), OR the metric family cannot see that part of the output (e.g. a
     volume lever scored only on timing) → name the fix or the metric family that would see it;
   - WINDOW_MASKED: the effect falls outside the scored window (spin-up, season, holdout split) → name the
     window that would show it.
""",
    "no_baseline": """## Your job (INVESTIGATE — do not fabricate, do not "fix" by changing metrics)
This module is CERTIFIED: its driver reproduced the validated reference run. Yet the calibration's run at the
contract's DEFAULT parameters gives no finite objective metric — so what differs is the calibration SETUP, not
the driver. No parameter was searched or proven (the proof detail may be empty).
1. Read `{ki_path}/calibration.yaml`, `{ki_path}/tools/calib_run.py`, the reference run the certification
   used (the contract's `runner.reference_run`, else `{ki_path}/calib/reference_run.json`) and the obs file; run
   calib_run.py at the defaults EXACTLY as the kit's pilot does (`KDT_CALIB_SPLIT=calibration`,
   `KDT_CALIB_EMIT_SERIES=1`) — that run, not a plain one, is the one that failed.
2. Compare the calibration's default run with the reference run, one difference at a time:
   - the contract's parameter DEFAULTS vs the reference run's parameter values (a default outside the
     model's valid region, a transform applied twice);
   - the calibration SPLIT / scored window vs the obs period (a window with no obs → 0 pairs);
   - the case's site / gauge / obs file vs the reference's (a different case with no prepared inputs);
   - the metric keys calib_run.py writes at the CALIBRATION split vs the keys the objectives read.
""",
}


def _watched_scores_text() -> str:
    """The contract prompt's "Scores the kit watches" section, as plain text (ONE source for both prompts)."""
    i = CALIB_DEV_PROMPT.index("### Scores the kit watches (convergence check)")
    j = CALIB_DEV_PROMPT.index("### Parameter injection depends on", i)
    return CALIB_DEV_PROMPT[i:j].replace("{{", "{").replace("}}", "}").strip()


REPLY_UPGRADE_PROMPT = """# Calibration-Runner Reply Upgrade Agent: {model_id}

The calibration kit's pilot ran `{ki_path}/tools/calib_run.py` at the default parameters (with
KDT_CALIB_SPLIT=calibration and KDT_CALIB_EMIT_SERIES=1), and the reply does not give the convergence check what it
needs — `missing`: required watched scores not reported; `mismatch`: scores that disagree with the kit's own
recomputation from the saved series (they were not computed on the saved pairs, or not with the kit's formulas);
`series`: false where the scored series was not saved (reason in `series_why`):
{check}

## Your job — change ONLY the runner's REPLY
Edit `{ki_path}/tools/calib_run.py` so its output JSON follows the rules below. Do NOT change how the model is run,
which case / gauge / window / split it scores, how parameters are injected, or the KI's own scores (its plain keys
stay exactly as they are). Add the watched scores computed on the SAME (sim, obs) pairs the runner already scores,
and the series step. Keep every existing `__kdt__` key (applied_params, case_id, split, …).

{rules}

## Prove it
Run calib_run.py once at the defaults exactly as the pilot does — with `KDT_CALIB_SPLIT=calibration` and
`KDT_CALIB_EMIT_SERIES=1` — and once without `KDT_CALIB_EMIT_SERIES`. Show that both replies keep the KI's own scores
unchanged, carry `__kdt__.panel.<var>` for every target with the required scores, and that the first one's
`__kdt__.series` file reads back with `with np.load(path, allow_pickle=False) as z: [z[k] for k in z.files]`.
Do NOT fake this — if you cannot, say why.

## Where you work
`{ki_path}` is this PROJECT's calibration folder. `tools/calib_run.py` in it is the project's own file — the one file
you edit. Every other entry there is a LINK into the model's KI: read them, never edit, replace or add anything
there (no `sed -i` on a helper, no new helper file). A change to anything but `tools/calib_run.py` fails the
upgrade. For your proof runs use the scratch folder `{scratch}` as `--workdir` and for every file you write
(parameter file, output JSON); start the runner from `{ki_path}` exactly as the command above does.

Edit ONLY `tools/calib_run.py`. A change to the KI's own scores at the default run is rejected automatically.
Your FINAL message is JSON:
{{"status":"upgraded"|"could_not_upgrade","what_changed":"...","verification":"<both runs and what they showed>"}}
"""


# ── the reply upgrade: the PROJECT's runner is edited, the KI is only read (Leo 2026-10-02) ─────────────────
# The workflow (calibration.yaml + tools/calib_run.py) belongs to the project: calibration_kit.project_workflow
# gives the project a view of the KI whose workflow files are the project's own. The upgrade edits ONE file there.
_REPLY_DIR = "reply_upgrade"                 # under the project folder
_REPLY_MARKER = "IN_PROGRESS.json"           # exists from just before the agent until the final keep / put-back
_REPLY_FAILED = "failed.json"                # {runner sha256: reason} — the RUNNER's own failures; not retried
_REPLY_KI_FP = "ki_before.json"              # the KI's fingerprint taken before the agent's session
_RUNNER_REL = "tools/calib_run.py"
_OWN_FILES = ("tools/calib_run.py", "calibration.yaml", "calib/capability_case.json")


def _sha_full(p) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _own_state(view) -> dict:
    """{rel: sha256 or None (absent)} of the project's own workflow files."""
    return {r: (_sha_full(Path(view) / r) if (Path(view) / r).is_file() else None) for r in _OWN_FILES}


def _backup_own(view, bdir) -> dict:
    bdir = Path(bdir)
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True)
    state = _own_state(view)
    for r, h in state.items():
        if h is not None:
            (bdir / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(view) / r, bdir / r)
    (bdir / "STATE.json").write_text(json.dumps(state))
    return state


def _put_back_own(view, bdir, only=None) -> list:
    """Make the project's own workflow files equal to the backup again (all, or just `only`). Returns the ones that
    had to be put back. Each file is written to a temp name and moved in, so a put-back cut short leaves the file
    whole. Raises if a file still differs afterwards."""
    view, bdir = Path(view), Path(bdir)
    want = json.loads((bdir / "STATE.json").read_text())
    done = []
    # the view itself must be the project's real folder: through a link (to the KI, say) every write below would
    # land elsewhere. Nothing is written then — the caller stops.
    if view.is_symlink() or not view.is_dir():
        raise RuntimeError(f"{view} is no longer the project's own folder (a link, or gone); nothing was put back")
    # the folders that hold the own files must be the project's REAL folders: if one was swapped for a link (into
    # the KI, say), a file written "into" it would land wherever the link points — drop the link itself first
    for dname in sorted({str(Path(r).parent) for r in _OWN_FILES if "/" in r}):
        dpath = view / dname
        if dpath.is_symlink() or (os.path.lexists(dpath) and not dpath.is_dir()):
            dpath.unlink()
        dpath.mkdir(parents=True, exist_ok=True)
        if os.path.realpath(dpath) != os.path.join(os.path.realpath(view), dname):
            raise RuntimeError(f"{dpath} does not lie in the project's view")
    for r in (only or _OWN_FILES):
        f = view / r
        if f.is_symlink() or (f.exists() and not f.is_file()):      # replaced by a link or a folder
            f.unlink() if (f.is_symlink() or f.is_file()) else shutil.rmtree(f)
        have = _sha_full(f) if f.is_file() else None
        if have == want.get(r):
            continue
        if want.get(r) is None:
            f.unlink()
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + f".kdt_putback_{os.getpid()}")
            shutil.copyfile(bdir / r, tmp)
            os.replace(tmp, f)
        if (_sha_full(f) if f.is_file() else None) != want.get(r):
            raise RuntimeError(f"could not put back {f}")
        done.append(r)
    return done


def _own_places_ok(view) -> str | None:
    """None when the view root, tools/, calib/ are the project's real folders and no own file is a link; else why.
    Checked without following links, before anything is written into the view."""
    view = Path(view)
    for p in (view, view / "tools", view / "calib"):
        if p.is_symlink() or not p.is_dir():
            return f"{p} is a link or not a folder"
    for r in _OWN_FILES:
        f = view / r
        if f.is_symlink() or (os.path.lexists(f) and not f.is_file()):
            return f"{f} is a link or not a plain file"
    return None


def recover_interrupted_reply_upgrade(project_dir, lock_held: bool = False) -> dict | None:
    """A stage killed during a reply upgrade leaves the marker: put the project's own workflow files back to the
    backup. Only those files are touched — never the KI, never anything else in the project. A marker whose owner
    process is still alive is left alone."""
    d = Path(project_dir) / _REPLY_DIR
    mk = d / _REPLY_MARKER
    if not mk.exists():
        return None
    try:
        info = json.loads(mk.read_text() or "{}")
    except Exception:
        info = {}
    pid = info.get("pid")
    # An upgrade always runs under the project's lock. A caller that HOLDS that lock knows the marker's owner is
    # gone (a process number may have been given to some other program since). Without the lock, a live process
    # with that number is taken for the owner.
    if not lock_held and isinstance(pid, int) and pid != os.getpid() and info.get("host") == os.uname().nodename:
        alive = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            pass
        if alive:
            return {"status": "owner_alive", "pid": pid,
                    "to_go_on": f"process {pid} seems to be running this upgrade; if it is not, delete {mk} after "
                                f"checking {d}"}
    def _ki_since():
        """what changed in the KI since the record taken before the agent's session ([] when there is no record)"""
        from calibration_kit import project_workflow as PW
        try:
            rec = json.loads((d / _REPLY_KI_FP).read_text())
        except FileNotFoundError:
            return []                                        # killed before the agent was started
        return PW.ki_changes(rec["fingerprint"], json.loads(json.dumps(PW.ki_fingerprint(rec["ki_path"]))))

    try:
        back = _put_back_own(Path(project_dir) / "ki_view", d / "backup")
    except Exception as e:
        out = {"status": "recovery_failed", "error": str(e)}
        try:                                                 # still say what happened to the KI
            ch = _ki_since()
            if ch:
                out["ki_changed"] = ch[:200]
        except Exception as e2:
            out["ki_check_error"] = str(e2)
        print(f"  CALIBRATE: could not recover an interrupted reply upgrade ({e}) — check {d}"
              f"{'; the KI changed too: ' + str(out['ki_changed'][:8]) if out.get('ki_changed') else ''}", flush=True)
        return out
    try:
        # the agent may have changed the KI through the view's links: compare with the record taken before its
        # session. A change is for a person to look at — the marker stays until they have (or the KI is as before).
        ch = _ki_since()
        if ch:
            print(f"  CALIBRATE: an unfinished reply upgrade was found; the project's runner was put back, but the "
                  f"KI changed since before that upgrade ({len(ch)} entries: {ch[:8]}). Look at them; then delete "
                  f"{mk} to go on.", flush=True)
            return {"status": "ki_changed", "ki_changed": ch[:200], "put_back": back,
                    "to_go_on": f"check the KI entries, then delete {mk}"}
        mk.unlink()
        (d / _REPLY_KI_FP).unlink(missing_ok=True)
        print(f"  CALIBRATE: an interrupted reply upgrade was found — the project's runner was put back {back}",
              flush=True)
        return {"status": "recovered", "put_back": back}
    except Exception as e:
        print(f"  CALIBRATE: could not recover an interrupted reply upgrade ({e}) — check {d}", flush=True)
        return {"status": "recovery_failed", "error": str(e)}


def _plain_numbers(reply) -> dict:
    """The model's own scores in a reply: plain numeric keys, top level and one level under a variable name."""
    out = {}
    for k, v in (reply or {}).items():
        if str(k).startswith("__"):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    out[f"{k}.{kk}"] = float(vv)
    return out


def _same_own_scores(before, after) -> list:
    """How the model's own scores differ between two replies of the same run: a changed value, a score that is
    gone, and a NEW plain score (the engine could start to optimise on it). NaN equals NaN."""
    import math
    a, b = _plain_numbers(before), _plain_numbers(after)
    diff = []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            diff.append(f"{k}: gone")
        elif k not in a:
            diff.append(f"{k}: new")
        elif not (a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k]))
                  or abs(a[k] - b[k]) <= 1e-9 * max(1.0, abs(a[k]))):
            diff.append(f"{k}: {a[k]!r} -> {b[k]!r}")
    return diff


def upgrade_project_runner(model_id, ki_path, view, project_dir, reply_state, first, calibrate_check, case=None,
                           lock_held: bool = False):
    """Upgrade the reply of the PROJECT's runner (`<view>/tools/calib_run.py`) before the search (Leo, 2026-10-02).
    One file is at stake: it is backed up (with the project's contract and approval record), an agent edits it, and
    it is kept only when ALL of these hold — the KI did not change (fingerprint before / after); the project's
    view is still a view (open_view again); the contract and approval record are as before; codex approves the
    whole new file; the engine's own pilot (`calibrate_check()`, reply_gate="check") finds the reply complete, with
    the same objectives and the model's own default-run scores unchanged. Anything else puts the one file back.

    Returns {"status": ...}: upgraded | unchanged | restored (the old runner is back; the search may run on it) |
    stop (the KI changed, or the project's files could not be put back: the search must NOT run) |
    not_attempted. `transient` is True when the failure was not the runner's (agent or reviewer unavailable).
    Never raises, except to pass on an interrupt after putting the file back."""
    import orchestrator as O
    from calibration_kit import project_workflow as PW
    view, project_dir, ki_path = Path(view), Path(project_dir), str(ki_path)
    d = project_dir / _REPLY_DIR
    bdir, mk, failed_f, run_py = d / "backup", d / _REPLY_MARKER, d / _REPLY_FAILED, view / _RUNNER_REL
    scratch = d / "agent_scratch"
    try:
        d.mkdir(parents=True, exist_ok=True)
        if mk.exists():
            rec = recover_interrupted_reply_upgrade(project_dir, lock_held=lock_held)
            if mk.exists():                                  # owner alive, or the recovery failed: never go on
                return {"status": "stop", "reason": f"an earlier reply upgrade is unfinished: {rec}"}
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        fp0 = json.loads(json.dumps(PW.ki_fingerprint(ki_path)))        # as it is read back from the file below
        before = _backup_own(view, bdir)
        (d / _REPLY_KI_FP).write_text(json.dumps({"ki_path": ki_path, "fingerprint": fp0}))
        sha0 = before[_RUNNER_REL]
        mk.write_text(json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "started": _now(),
                                  "ki_path": ki_path, "runner_sha256": sha0}))
    except Exception as e:
        return {"status": "not_attempted", "reason": f"could not prepare the upgrade: {type(e).__name__}: {e}"}

    kept, res = False, {"status": "restored", "transient": False}
    try:
        _chk = json.dumps({k: reply_state.get(k) for k in ("missing", "mismatch", "series", "series_why")},
                          default=str, indent=1)
        rules = _watched_scores_text()
        prompt = REPLY_UPGRADE_PROMPT.format(model_id=model_id, ki_path=str(view), check=_chk, rules=rules,
                                             scratch=str(scratch))
        print(f"  [{model_id}] CALIBRATE: the runner's reply lacks what the convergence check needs — spawning the "
              f"reply-upgrade agent on the project's runner...", flush=True)
        _res, output = O.run_claude_resilient(prompt, timeout=_agent_timeout(), max_turns=50)
        (d / "agent_output.txt").write_text(output or "")
        # 1. the KI must be as it was (the agent could reach it through the view's links)
        ki_ch = PW.ki_changes(fp0, json.loads(json.dumps(PW.ki_fingerprint(ki_path))))
        if ki_ch:
            res.update(status="stop", ki_changed=ki_ch,
                       reason=f"the agent changed the KI ({len(ki_ch)} entries: {ki_ch[:8]}); look at them, put "
                              f"them right, then run again — nothing in the KI was put back automatically")
            return res
        # 2. only the runner may differ among the project's own files
        if view.is_symlink() or any((view / dn).is_symlink() or not (view / dn).is_dir() for dn in ("tools", "calib")):
            res.update(status="stop", transient=True, view_broken="tools/ or calib/ is no longer the project's own "
                       "folder", reason=f"the agent replaced the project's tools/ or calib/ folder; delete {view} "
                                        f"to start from the KI's workflow again, then run again")
            return res
        other = _put_back_own(view, bdir, only=[r for r in _OWN_FILES if r != _RUNNER_REL])
        res["other_edits_put_back"] = other
        # 3. the view must still be a view of the KI (a helper link swapped for a real file, an added helper ...)
        try:
            PW.open_view(ki_path, project_dir)
        except PW.ViewError as e:
            # the runner goes back (finally), but the folder itself is wrong (a helper link swapped for a real
            # file, an added helper): a search from it would use files that are neither the KI's nor approved
            res.update(status="stop", view_broken=str(e), transient=True,
                       reason=f"the agent left the project's folder in a state that is no longer a view of the KI: "
                              f"{e} — put it right (or delete {view} to start from the KI's workflow again), then "
                              f"run again")
            return res
        if not run_py.is_file() or _sha_full(run_py) == sha0:
            res.update(status="unchanged", reason="the agent did not change the runner",
                       transient=(_res is None or not (output or "").strip() or "ENV_ABORT" in (output or "")))
            return res
        # 4. the review: the whole new file, with the kit's rules
        import difflib
        old_src, new_src = (bdir / _RUNNER_REL).read_text(), run_py.read_text()
        if len(new_src) > 180_000:
            res["reason"] = f"the new runner is too long to review whole ({len(new_src)} characters)"
            return res
        diff = "".join(difflib.unified_diff(old_src.splitlines(keepends=True), new_src.splitlines(keepends=True),
                                            f"a/{_RUNNER_REL}", f"b/{_RUNNER_REL}", n=100000))
        review = {"verdict": "SKIPPED"}
        try:
            from tool_reviewer import review_tool
            v = review_tool(diff=diff,
                            tool_summary={"tool": _RUNNER_REL, "summary": "runner reply upgrade (watched scores + "
                                          "series for the calibration kit's convergence check)",
                                          "origin": "calib_reply_upgrade", "target_case": _review_target_case(case),
                                          "review_focus": "ONLY the reply may change: the model run, the scored case "
                                          "/ obs / window, the calibration AND the holdout split and the model's own "
                                          "scores must be untouched on every split; the watched scores and the saved "
                                          "series must be the SAME (sim, obs) pairs the model's own scores use; the "
                                          "watched scores must follow EXACTLY these rules:\n"
                                          + rules + "\n" + _REVIEW_FOCUS_EXEC},
                            ki_path=str(view), run_id=f"{model_id}_replyupgrade")
            review = {"verdict": v.verdict, "issues": v.issues[:3], "reviewer": v.reviewer}
        except Exception as e:
            review = {"verdict": "REVIEW_ERROR", "reason": str(e)}
        res["review"] = review
        if review.get("verdict") != "APPROVE":
            res["reason"] = f"review {review.get('verdict')}"
            res["transient"] = review.get("verdict") in ("REVIEW_ERROR", "SKIPPED", "WAIT")
            return res
        # 5. the engine's own pilot on the new runner
        reviewed_sha = _sha_full(run_py)
        chk = calibrate_check()
        res["check_after"] = ({k: chk.get(k) for k in ("status", "reply", "objectives")}
                              if isinstance(chk, dict) else None)
        if not (isinstance(chk, dict) and chk.get("status") == "reply_checked"):
            res["reason"] = f"the check could not run ({(chk or {}).get('status') if isinstance(chk, dict) else chk})"
            res["transient"] = True
            return res
        if not (chk.get("reply") or {}).get("complete"):
            res["reason"] = "the reply is still incomplete after the upgrade"
            return res
        moved = _same_own_scores(first.get("default_reply"), chk.get("default_reply"))
        if moved:
            res["reason"] = f"the upgrade changed the model's own default-run scores: {moved[:6]}"
            return res
        if first.get("objectives") != chk.get("objectives"):
            res["reason"] = (f"the upgrade changed what the engine optimises: {first.get('objectives')} -> "
                             f"{chk.get('objectives')}")
            return res
        bad = _own_places_ok(view)                           # the check run executed the new runner in this folder
        if not bad and _own_state(view) != dict(before, **{_RUNNER_REL: reviewed_sha}):
            bad = ("the runner is no longer the file that was reviewed and checked, or the project's contract or "
                   "approval record changed during the check run")
        if bad:
            res.update(status="stop", transient=True, view_broken=bad,
                       reason=f"after the check run the project's folder is damaged ({bad}); nothing was written; "
                              f"delete {view} to start from the KI's workflow again")
            return res
        kept = True
        res.update(status="upgraded", reason=None)
        return res
    except BaseException as e:
        res["reason"] = f"{type(e).__name__}: {e}"
        res["transient"] = True
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return res
    finally:
        try:                                                 # on EVERY way out: is the KI as it was?
            if res.get("status") != "stop" or not res.get("ki_changed"):
                _ch = PW.ki_changes(fp0, json.loads(json.dumps(PW.ki_fingerprint(ki_path))))
                if _ch:
                    kept = False
                    res.update(status="stop", ki_changed=_ch[:200], transient=True,
                               reason=f"the KI changed during the reply upgrade ({len(_ch)} entries: {_ch[:8]}); "
                                      f"look at them, put them right, then run again — nothing in the KI was put "
                                      f"back automatically" + (f" [{res.get('reason')}]" if res.get("reason") else ""))
        except Exception as _ke:
            kept = False
            res.update(status="stop", transient=True, reason=f"the KI could not be checked after the upgrade: {_ke}")
        try:
            if kept:
                _write_capability_case(str(view), case, res.get("review") or {})   # the project's approval record
            else:
                res["put_back"] = _put_back_own(view, bdir)
                if res.get("status") not in ("stop", "unchanged") and not res.get("transient"):
                    try:
                        known = json.loads(failed_f.read_text()) if failed_f.exists() else {}
                    except Exception:
                        known = {}
                    known[sha0] = str(res.get("reason"))[:300]       # the RUNNER's own failure: not tried again
                    failed_f.write_text(json.dumps(known, indent=1))
            if not res.get("ki_changed"):
                mk.unlink()
                (d / _REPLY_KI_FP).unlink(missing_ok=True)
            # else: the marker and the KI's record STAY — whether this call returns or passes on an interrupt, the
            # next call compares again and stops until the KI is as before or a person has looked (deleting the
            # marker); a changed KI never becomes the new starting point without a word
            shutil.rmtree(scratch, ignore_errors=True)
        except BaseException as _fe:                         # the one file could not be put back: never search on it
            res.update(status="stop", cleanup_error=f"{type(_fe).__name__}: {_fe}",
                       reason=f"the project's runner could not be put back ({_fe}); the marker stays: {mk}")
        print(f"  [{model_id}] CALIBRATE: reply upgrade {res.get('status')}"
              f"{' — ' + str(res.get('reason')) if res.get('reason') else ''}", flush=True)


class _EnvSet:
    def __init__(self, env):
        self.env, self.old = dict(env), {}

    def __enter__(self):
        for k, v in self.env.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def calibrate_with_reply_upgrade(model_id, ki_path, work_dir, case=None, **calib_kw):
    """Run the calibration engine from the PROJECT's folder (`work_dir`): the project gets its own copy of the
    workflow and a view of the KI (project_workflow.open_view); the engine runs on the view. If the engine's own
    pilot finds the runner's reply incomplete, the project's runner is upgraded first. One lock is held for the
    whole sequence. A workflow that cannot run from a project folder (see open_view) is calibrated in the KI as
    before, without an upgrade. KDT_CALIB_REPLY_UPGRADE=0 turns the upgrade (not the project folder) off."""
    import fcntl
    from calibration_kit import calib
    from calibration_kit import project_workflow as PW
    ki_path, project = str(ki_path), Path(work_dir)
    project.mkdir(parents=True, exist_ok=True)
    lock = open(project / ".kdt_project.lock", "a+")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"status": "workdir_busy", "reason": f"another calibration holds {project / '.kdt_project.lock'}"}
        # first of all: an upgrade that was killed left the project's runner half-edited — put it back
        rec = recover_interrupted_reply_upgrade(project, lock_held=True)
        if rec and rec.get("status") != "recovered":
            return {"status": "reply_upgrade_unfinished", "reason": str(rec),
                    "project_workflow": {"used": True, "view": str(project / "ki_view")}}
        try:
            got = PW.open_view(ki_path, project)
        except PW.WorkflowNotSupported as e:
            pass_e = e
        except PW.ViewError as e:
            # the project's folder is damaged (a helper link swapped for a real file, left-over files ...): never
            # fall back to the KI's workflow without a word — stop and say what to put right
            return {"status": "project_workflow_broken", "reason": f"{e} — put it right, or delete "
                    f"{project / 'ki_view'} to start from the KI's workflow again",
                    "project_workflow": {"used": False, "view": str(project / "ki_view")}}
        else:
            pass_e = None
        if pass_e is not None:
            e = pass_e
            print(f"  [{model_id}] CALIBRATE: this workflow cannot run from a project folder ({e}); calibrating in "
                  f"the KI as before, no reply upgrade", flush=True)
            rep = calib.calibrate(ki_path=ki_path, workdir=str(work_dir), **calib_kw)
            if isinstance(rep, dict):
                rep["project_workflow"] = {"used": False, "why": str(e)}
            return rep
        view, run_dir = got["view"], got["workdir"]
        fp_start = PW.ki_fingerprint(ki_path)                # the runner works through links into the KI

        def _ki_note(rep):
            """A runner that writes into the KI through the view's links (an output folder, a prepared input it
            rewrites) is as it was before project folders — but it is now SEEN and said, in the report."""
            if isinstance(rep, dict):
                ch = PW.ki_changes(fp_start, PW.ki_fingerprint(ki_path))
                if ch:
                    rep.setdefault("project_workflow", {})["ki_changed_during_run"] = ch[:200]
                    print(f"  [{model_id}] CALIBRATE: WARNING the runs changed {len(ch)} entries in the KI (through "
                          f"the project folder's links): {ch[:8]} — the runner writes into the KI; its results "
                          f"depend on, and alter, the KI's state", flush=True)
            return rep
        with _EnvSet(got["env"]):
            def _cal(gate=None):
                return calib.calibrate(ki_path=view, workdir=run_dir, reply_gate=gate, **calib_kw)
            off = str(os.environ.get("KDT_CALIB_REPLY_UPGRADE", "1")).strip().lower() in ("0", "false", "no", "off")
            gate, skipped = (not off), ("switched off" if off else None)
            failed_f = project / _REPLY_DIR / _REPLY_FAILED
            if gate and failed_f.exists():
                try:
                    if _sha_full(Path(view) / _RUNNER_REL) in json.loads(failed_f.read_text()):
                        gate, skipped = False, "an upgrade of this runner failed before"
                except Exception:
                    pass
            own0 = _own_state(view)                          # the project's workflow as this call found it
            first = _cal(True if gate else None)
            own1 = _own_state(view)
            if own1 != own0:
                # a runner must never rewrite its own workflow (the contract, itself, the approval record): what
                # would be searched, backed up or upgraded next is no longer what the project held
                changed = sorted(r for r in own0 if own0[r] != own1.get(r))
                rep = {"status": "project_workflow_broken", "project_workflow": {"used": True, "view": view},
                       "reason": f"the runs changed the project's own workflow files {changed}; put them right, or "
                                 f"delete {view} to start from the KI's workflow again"}
                if isinstance(first, dict) and first.get("status") not in ("reply_incomplete", "reply_checked"):
                    rep["search_report"] = first             # the search already ran: keep what it said
                return _ki_note(rep)
            up = None
            if isinstance(first, dict) and first.get("status") == "reply_incomplete":
                up = upgrade_project_runner(model_id, ki_path, view, project, first.get("reply") or {}, first,
                                            lambda: _cal("check"), case=case, lock_held=True)
                if up.get("status") == "stop":
                    return _ki_note({"status": "reply_upgrade_stopped", "reason": up.get("reason"),
                                     "reply_upgrade": up, "project_workflow": {"used": True, "view": view}})
                # the check run executed the (new) runner: the folder must STILL be a view before the search
                try:
                    PW.open_view(ki_path, project)
                except PW.ViewError as e:
                    return _ki_note({"status": "project_workflow_broken", "reply_upgrade": up,
                                     "reason": f"after the reply upgrade's runs the project's folder is no longer a "
                                               f"view of the KI: {e} — put it right, or delete {view} to start from "
                                               f"the KI's workflow again",
                                     "project_workflow": {"used": True, "view": view}})
                report = _cal(None)
            else:
                report = first
            if isinstance(report, dict):
                report["project_workflow"] = {"used": True, "view": view, "workdir": run_dir,
                                              "re_pointed": got["report"].get("re_pointed")}
                if up is not None:
                    report["reply_upgrade"] = dict(up, reply_before=first.get("reply"))
                elif skipped:
                    report["reply_upgrade"] = {"status": "skipped", "reason": skipped}
                elif gate and ((report.get("convergence") or {}).get("reply") or {}).get("complete") is False:
                    # the gate fires only on a fresh pilot: a saved search is continued with the runner it began
                    # with (a runner is never changed under a running search)
                    report["reply_upgrade"] = {"status": "skipped", "reason": "a saved search in this project's run "
                                               "folder is being continued; the reply is upgraded on a fresh search"}
            return _ki_note(report)
    finally:
        lock.close()


#: the engine's no-search triage routes (design §1, build step 8) and the old status that means the same
NO_SEARCH_ROUTES = ("no_baseline", "not_calibratable")


def triage_route_of(report) -> str | None:
    """The triage route of an engine report: `route` (no-search reports), else `triage.route`; the older
    early stop `runner_emits_no_objective_metric` is a no-baseline case too."""
    if not isinstance(report, dict):
        return None
    r = report.get("route") or (report.get("triage") or {}).get("route")
    if not r and report.get("status") == "runner_emits_no_objective_metric":
        r = "no_baseline"
    return r


_ROUTE_TEXT = {
    "no_baseline": ("The model's run at the DEFAULT parameters gives NO finite objective metric (route=no_baseline):\n"
                    "the runner fails, emits no metric the objectives read, or scores the wrong window / obs. There is\n"
                    "no baseline, so calibration cannot start until `{ki_path}/tools/calib_run.py` runs and scores the\n"
                    "defaults."),
    "not_calibratable": ("The consumption proof moved EACH searched parameter alone from the defaults to its far bound,\n"
                         "and NONE moved the objectives (route=not_calibratable; the per-parameter verdicts are in the\n"
                         "triage detail below): the objective cannot see the parameters — the metric / window / obs\n"
                         "cannot resolve them (OBJECTIVE_MASKED, WINDOW_MASKED), the process is inactive in this case\n"
                         "(DORMANT), or the setup scores something the parameters do not reach."),
}


def act_on_calibration_blocker(model_id, ki_path, work_dir, report, prior_metric, case=None):
    """Turn a no-search triage route into an ACTION (design §1 routes, build step 8b):
      no_baseline      -> spawn a driver-repair agent (the default run gives no finite objective), then
        codex-review the patched tools/calib_run.py (it runs the model 100s of times);
      not_calibratable -> spawn a setup-diagnosis agent (no parameter moves the objectives) that writes
        findings (no fabrication).
    `action_route` overrides which action runs (a CERTIFIED module's no_baseline goes to setup diagnosis).
    Returns an action dict. Never raises (caller wraps), but guards internally too."""
    import orchestrator as O
    route = triage_route_of(report)
    action = report.get("_action_route") or route
    ki_path = str(ki_path)
    _tr = report.get("triage") or {}
    _pilot = (report.get("pilot") or (report.get("budget_plan") or {}).get("pilot_summary") or {})
    # a CERTIFIED module's no_baseline goes to setup diagnosis: its route text must not tell that agent the
    # runner is broken (Opus 8b/9 r3 nit)
    _rt = ("The model's run at the DEFAULT parameters gives NO finite objective metric (route=no_baseline), although\n"
           "this module's driver reproduced its validated reference run (certified)."
           if route == "no_baseline" and action == "not_calibratable" else
           _ROUTE_TEXT.get(route, f"Calibration stopped before the search (route={route}).").format(ki_path=ki_path))
    fmt = dict(model_id=model_id, ki_path=ki_path, work_dir=str(work_dir), route=route,
               route_text=_rt,
               baseline=json.dumps(_pilot.get("default_metrics"), default=str)[:300],
               reason=str(report.get("reason") or _tr.get("reason"))[:500],
               family_verdict=json.dumps({"per_target": _tr.get("per_target"),
                                          "proof": (_tr.get("proof") or {}).get("verdicts")}, default=str)[:600],
               prior_metric=json.dumps(prior_metric, default=str)[:200])
    fmt["setup_job"] = _SETUP_JOB["no_baseline" if route == "no_baseline" else "not_calibratable"].format(
        ki_path=ki_path)

    if action == "no_baseline":
        prompt = FIX_DRIVER_PROMPT.format(**fmt)
        print(f"  [{model_id}] TRIAGE-ACTION: spawning driver-repair agent ({route})...", flush=True)
        try:
            _res, output = O.run_claude_resilient(prompt, timeout=_agent_timeout(), max_turns=70)
        except Exception as e:
            return {"route": route, "action": "driver_repair", "status": "agent_error", "reason": str(e)}
        (Path(work_dir) / "calib_driver_fix_output.txt").write_text(output or "")
        # independent codex review of the patched driver
        review = {"verdict": "SKIPPED"}
        try:
            from tool_reviewer import review_tool
            # review BOTH attested files (codex 2026-08-03): the repair may edit calibration.yaml too, and
            # _write_capability_case pins both hashes — reviewing only calib_run.py could attest an
            # unreviewed calibration.yaml.
            cyaml = (Path(ki_path) / "calibration.yaml").read_text()
            crun = (Path(ki_path) / "tools" / "calib_run.py").read_text()
            diff = ("--- a/calibration.yaml\n+++ b/calibration.yaml\n"
                    + "".join("+" + l for l in cyaml.splitlines(keepends=True))
                    + "--- a/tools/calib_run.py\n+++ b/tools/calib_run.py\n"
                    + "".join("+" + l for l in crun.splitlines(keepends=True)))
            v = review_tool(diff=diff[:200_000],   # match review_tool's own 200k cap (was 64k, truncated large drivers)
                            tool_summary={"tool": "calibration.yaml + tools/calib_run.py",
                                          "summary": "calibration contract + driver (repaired)",
                                          "origin": "calib_driver_fix",
                                          "target_case": _review_target_case(case),
                                          "review_focus": "The repaired driver must STILL score the case's "
                                          "gauge/obs. " + _REVIEW_FOCUS_EXEC},
                            ki_path=ki_path, run_id=f"{model_id}_driverfix")
            review = {"verdict": v.verdict, "issues": v.issues[:3], "reviewer": v.reviewer}
        except Exception as e:
            review = {"verdict": "REVIEW_ERROR", "reason": str(e)}
        # the repair MUTATED calib_run.py, so any prior approved sidecar no longer matches its hash
        # (_capability_matches_case already forces re-review). On APPROVE re-record the sidecar (same case,
        # new hash). WAIT = STAGED pending HUMAN review — leave staged, do NOT archive. Otherwise ARCHIVE
        # so nothing rejected/unreviewed lingers.
        if review.get("verdict") == "APPROVE":
            _write_capability_case(ki_path, case, review)
        elif review.get("verdict") == "WAIT":
            review["pending_human"] = True
        else:
            review["archived_to"] = _archive_capability(ki_path, "rejected_driverfix")
        print(f"  [{model_id}] TRIAGE-ACTION: driver-repair done; codex={review.get('verdict')}", flush=True)
        return {"route": route, "action": "driver_repair", "status": "ran",
                "review": review, "output_tail": (output or "")[-600:]}

    if action == "not_calibratable":
        prompt = DIAGNOSE_SETUP_PROMPT.format(**fmt)
        print(f"  [{model_id}] TRIAGE-ACTION: spawning setup-diagnosis agent...", flush=True)
        try:
            _res, output = O.run_claude_resilient(prompt, timeout=_agent_timeout(), max_turns=60)
        except Exception as e:
            return {"route": route, "action": "setup_diagnosis", "status": "agent_error", "reason": str(e)}
        (Path(work_dir) / "calib_setup_diagnosis_output.txt").write_text(output or "")
        # best-effort: queue an out-of-scope proposal so the finding isn't lost
        queued = False
        try:
            from learning_proposals import propose_out_of_scope_finding
            propose_out_of_scope_finding(
                model_id=model_id, run_id=f"{model_id}_calib_setup",
                diagnosis={"fix_class": "requires_setup_fix", "in_scope": False,
                           "diagnosis": report.get("reason"),
                           "fix_description": (output or "")[-1500:]},
                test_result={"calibration_triage": _tr})
            queued = True
        except Exception as e:
            print(f"  [{model_id}] TRIAGE-ACTION: proposal queue failed (non-fatal): {e}", flush=True)
        print(f"  [{model_id}] TRIAGE-ACTION: setup-diagnosis done; proposal_queued={queued}", flush=True)
        return {"route": route, "action": "setup_diagnosis", "status": "ran",
                "proposal_queued": queued, "output_tail": (output or "")[-600:]}

    return {"route": route, "action": "none", "status": "no_action_for_route"}


def run_stage(model_id, ki_path, work_dir, obs_shape_by_var, prior_metric=None,
              budget=None, determining_metric=None, headline_objectives=None, case=None):
    """Develop-if-missing + run calibration + gate. Returns a report dict (or a
    skip dict). Never raises into the pipeline. `determining_metric` /
    `headline_objectives` (from readiness/the convention) let the engine optimize +
    triage the field HEADLINE metric instead of the family default."""
    if not enabled():
        return {"status": "skipped", "reason": "KDT_CALIBRATE not set"}
    ki_path = str(ki_path)
    try:
        print(f"  [{model_id}] CALIBRATE: tuning parameters to fit dag obs "
              f"(prior metric: {prior_metric})", flush=True)
        # A. ensure calibration capability exists AND was approved for THIS case (develop if not).
        # File existence alone is unsafe (codex 2026-07-18): a contract authored/approved for a
        # DIFFERENT gauge must be archived + redeveloped, never reused for this case.
        _has = _has_contract(ki_path)
        if _has and not _capability_matches_case(ki_path, case):
            arch = _archive_capability(ki_path, "mismatched_case_calibdev")
            print(f"  [{model_id}] CALIBRATE: existing capability was authored for a DIFFERENT case "
                  f"(archived → {arch}) — redeveloping for {(case or {}).get('case_id')}", flush=True)
            _has = False
        if not _has:
            print(f"  [{model_id}] CALIBRATE: no matching calibration.yaml — developing "
                  f"calibration capability for case {(case or {}).get('case_id')} "
                  f"(agent + codex review)...", flush=True)
            ok, verdict = develop_calibration_capability(model_id, ki_path, prior_metric, case=case)
            if not ok:
                return {"status": "calibration_capability_unavailable",
                        "review": verdict}
            print(f"  [{model_id}] CALIBRATE: capability APPROVED "
                  f"({verdict.get('reviewer')})", flush=True)

        # B. run the calibration engine (pass the target case_id so the engine can FAIL-CLOSED
        # if the runner self-declares a different gauge — the deterministic wrong-gauge guard). The engine's own
        # pilot checks the runner's REPLY first; an old runner's reply is upgraded before the search (option 1)
        _expected_case = (case or {}).get("case_id")
        report = calibrate_with_reply_upgrade(model_id, ki_path, work_dir, case=case,
                                              obs_shape_by_var=obs_shape_by_var, budget=budget,
                                              determining_metric=determining_metric,
                                              headline_objectives=headline_objectives,
                                              expected_case_id=_expected_case)
        # C. report + gate
        status = report.get("status")
        promotable = report.get("promotable")
        # TRIAGE (design §1, build step 8): the engine searches unless the run CANNOT be calibrated —
        # no_baseline (the default run gives no finite objective) or not_calibratable (the proof shows
        # no parameter moves the objectives). Fit quality is never a route; it is in report["triage"].
        route = triage_route_of(report)
        if route in NO_SEARCH_ROUTES:
            banner = {"no_baseline": "NO BASELINE — the default run gives no finite objective metric",
                      "not_calibratable": "NOT CALIBRATABLE — no parameter moves the objectives"}[route]
            print(f"  [{model_id}] CALIBRATE TRIAGE -> {route}: {banner}\n"
                  f"      {report.get('reason')}", flush=True)
            # a CERTIFIED module reproduced the validated run (certify_runner, before the pilot), so its
            # driver provably runs the KI: a no_baseline there is a setup / defaults problem — setup
            # diagnosis, not a driver repair. The engine says whether it certified (`certified`, from the
            # contract's runner.reference_run or the KI's calib/reference_run.json — System 1's site
            # contracts use the first; Opus 8b/9 r1 #11); an older report falls back to the KI file.
            _cf = report.get("certified")
            _certified = (bool(_cf) if _cf is not None
                          else (Path(ki_path) / "calib" / "reference_run.json").is_file())
            if _certified and route == "no_baseline":
                report["_action_route"] = "not_calibratable"
                print(f"  [{model_id}] CALIBRATE: module is CERTIFIED — no_baseline goes to setup "
                      f"diagnosis, not a driver repair.", flush=True)
            # every no-search route gets its action (driver repair or setup diagnosis)
            try:
                report["triage_action"] = act_on_calibration_blocker(
                    model_id, ki_path, work_dir, report, prior_metric, case=case)
            except Exception as _ae:
                report["triage_action"] = {"status": "action_error", "reason": str(_ae)}
                print(f"  [{model_id}] CALIBRATE triage-action error (non-fatal): {_ae}", flush=True)
            # LAND THE REPAIR (2026-06-28): if the driver-repair was codex-APPROVED,
            # re-run calibration ONCE so a now-isolated runner actually calibrates.
            # Single-shot (no loop) — guarded by the APPROVE verdict.
            _ta = report.get("triage_action") or {}
            if (route == "no_baseline"
                    and _ta.get("action") == "driver_repair"
                    and (_ta.get("review") or {}).get("verdict") == "APPROVE"):
                print(f"  [{model_id}] CALIBRATE: driver repair APPROVED → re-running "
                      f"calibration once on the fixed runner...", flush=True)
                try:
                    from calibration_kit import calib
                    report2 = calib.calibrate(ki_path=ki_path, workdir=str(work_dir),
                                              obs_shape_by_var=obs_shape_by_var, budget=budget,
                                              determining_metric=determining_metric,
                                              headline_objectives=headline_objectives,
                                              expected_case_id=_expected_case)
                    report["recalibration"] = report2
                    print(f"  [{model_id}] CALIBRATE (post-repair): status={report2.get('status')} "
                          f"route={triage_route_of(report2)} best_loss={report2.get('best_loss')} "
                          f"holdout={report2.get('holdout_validated')} "
                          f"promotable={report2.get('promotable')}", flush=True)
                    # if the re-run actually calibrated, surface THAT as the report
                    if triage_route_of(report2) not in NO_SEARCH_ROUTES:
                        report = {**report2, "driver_was_repaired": True,
                                  "triage_action": _ta}
                except Exception as _re:
                    report["recalibration"] = {"status": "error", "reason": str(_re)}
                    print(f"  [{model_id}] CALIBRATE post-repair re-run error: {_re}", flush=True)
            try:
                (Path(work_dir) / "calibration_report.json").write_text(
                    json.dumps(report, indent=2, default=str))
            except Exception:
                pass
            return report
        print(f"  [{model_id}] CALIBRATE: {status} | algo={report.get('algorithm')} "
              f"| best_loss={report.get('best_loss')} | holdout="
              f"{report.get('holdout_validated')} | promotable={promotable}", flush=True)
        try:
            (Path(work_dir) / "calibration_report.json").write_text(
                json.dumps(report, indent=2, default=str))
        except Exception:
            pass
        return report
    except Exception as e:
        print(f"  [{model_id}] CALIBRATE: stage error (non-fatal): {e}", flush=True)
        return {"status": "error", "reason": str(e)}
