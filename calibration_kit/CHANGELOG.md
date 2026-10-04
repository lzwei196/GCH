# calibration_kit — changelog

Versioning added 2026-09-02. **Why:** the kit is edited while it is in use, by more than one
project. A calibration number is only comparable to another one if both were produced by the same
engine, so every result should record `calibration_kit.__version__`, the git commit, and whether the
tree was dirty. Bump the version on any change to scoring, stopping, holdout or objective behaviour.

## 0.4.0-dev — 2026-09-29 (branch convergence-panel-2026-09; worktree kdt_convergence_dev)
Convergence rebuilt to design HANDOFF_CONVERGENCE_2026-09-27_v2.md (reviewed build steps 1-4):
- panel per variable KIND, kit metric block, tolerances by bootstrap with labels, alpha reconstruction;
- the rule (window, per-objective loss test, full-window panel test, eps front test, compromise with
  protection), verdicts (seed / variable / run, unknown cap, safety, standards);
- ONE termination policy: cap counted in calls for every backend; modes keep_going (default) / stop;
  `strategy.stop` and the old marker removed from the calibration path (old blocks translated on load);
  pymoo's own test recorded, never ending the search; SCE-UA settings per mode; DREAM R-hat recorded.
Stopping behaviour changed -> results are not comparable with 0.3.0 runs without saying so.
Build step 5 (budget, design §1.3; gaps 2k/2m/2i/2v):
- the pilot is always on — also for a contract with no `budget` block. Its extra live runs advance
  KDT_CALIB_EVAL_ID, so a stochastic runner's search differs from the same contract under 0.3.0;
- t_slow = 1.2 x p90 (10-run pilot) or 1.2 x max(budget.run_time, slowest of a 3-run pilot);
- one allowance ledger: measured shared overhead (this call's pilot, machine probe, commissioning,
  proof) and a final overhead of 4 x t_slow, charged once; a measured cap under 20 is refused;
- the machine probe runs real copies of a subprocess runner, only with `parallel_safe: true`;
- a live time check warns once when the projected search time passes the seed's share;
- a measured budget with no allowance, or a zero/negative/non-finite or unreadable one, is refused at
  LOAD (`invalid_contract`); before step 5 it fell back to max_evaluations with a warning. A caller's
  explicit budget= skips this allowance check. A non-empty `budget` that is not a mapping or an
  unknown `budget.mode` is refused even then (an unknown mode used to become fixed silently); in measured mode,
  defaults that break the constraints are refused at load. `allowance: 0` with no mode is now a
  measured request (refused), as `"0s"` already was;
- pilot points the contract's constraints reject never run: they are replaced with fresh draws and are
  not failed runs; a default that breaks the constraints makes the pilot `pilot_unstable` at any size.

Build step 6 (seeds, design §1.7, §2.7; gap 2l):
- `strategy.budget.seeds` defaults to **3** (was 1). The cap is PER SEED, so a contract with no budget
  block now spends 3 × max_evaluations search calls (plus one pilot, commissioning and proof). Set
  `seeds: 1` for the old single search;
- a crashed seed is re-run once with a new number; seeds agree on their final CALIBRATION incumbents;
  the returned seed = lowest calibration loss, whatever the agreement; run verdict across seeds;
- the reported result is the returned seed's calibration incumbent (a trade-off search: the rule's
  compromise, no longer the backend's own pick); the holdout is read once, on it;
  `KDT_CALIB_FRONT_SELECT=1` (validation-based member choice) is ignored with a warning;
- every search call in eval_history.jsonl carries its `seed`; a run where every seed crashed returns
  `search_crashed` (it used to raise);
- seeds run one after another (parallel seed lanes: a later step), and the plan counts one lane;
- backends that do not report each call (surrogate, PEST++, MADR) run ONE seed; `best_loss` keeps its
  form ([scalar] single-objective), the per-objective vector is `best_losses_per_objective`; a plan
  saved for another number of seeds (including every plan saved before this step) is re-planned.

Build step 7 (phase tags and split provenance, gaps 2j, 2p):
- the evaluator's default phase is `untagged` (was `search`): a call outside any phase block is never
  counted as search effort; the consumption proof's receipt check is tagged `proof`;
- the certification replay and the objective probe are logged in eval_history.jsonl (`certify`,
  `objective_probe`, `outside_evaluator: true`), timed, and charged as shared overhead;
- every call with metrics records `split_echo` (`__kdt__.split`); the report has `split_provenance` with
  `calibration_only_supported`. A runner that does not echo its split is reported, not refused;
- the machine probe's lane runs are logged too (`machine_probe`); all runs before the search are shared
  overhead;
- the claim covers the pilot and search calls, plus the objective probe when its payload decided a pilot
  decision; a wrong echo gives False, an unchecked phase None. kdt_budget_plan.json saves
  `pilot_split_provenance` (and `probe_used_for_pilot` / `probe_split_provenance`) so a resume judges the
  pilot it reuses; an older plan gives None ("pilot not checked"). A resume whose saved decisions cannot be
  reused derives them again and judges this call's probe instead. The failed and crashed reports carry
  `split_provenance` too;
- the calibration-agent prompt no longer tells a runner to "score both" subsets when it cannot split: score
  exactly the requested subset and echo it; unset split or no subset possible → score the full record and
  echo "full"; never echo a subset that was not scored.

Build step 8a (triage, staged removal, StopRule removal, untracked backends; gaps 2x, 2w, 2s):
- triage before the search (design §1): `no_baseline` / `not_calibratable` / `calibrate`, first match wins;
  a default run with no finite objective metric now STOPS the run (`no_baseline`) — also in fixed mode,
  where a failed default used to be only a pilot warning; the report's `triage` block (route, reason, proof
  summary, per-target default standards and fit verdict — information only);
- `strategy.staged` removed: `calibrate_staged`, the Morris screen (sensitivity.py) and the old triage
  (calibratability.py, routes already_adequate / diagnose_setup / calibrate_caution / undetermined /
  fix_driver / screen_failed) are gone; `staged: true` runs one search over all parameters, with a warning;
- `StopRule` removed from stop.py (convention_floor kept); `kit_report["stop"]` no longer exists;
- the pilot summary records `default_n_finite` (finite objective losses of the default run);
- surrogate / PEST++ / MADR: one seed, "convergence not tracked";
- the triage runs right after the pilot — before the machine probe, the budget plan, its file and the
  refusals; its no-search report has no `budget_plan` and carries `pilot`, phase_counts, split_provenance,
  consumption_proof / coverage and the smoke test; a saved plan whose default gave no finite metric is
  re-planned;
- a 3-run measured pilot no longer stops on a failed default: it ends as `no_baseline` (no finite
  objective) or searches (a partly finite default), where it used to end as `pilot_unstable`;
- budget.estimate_overhead_evals lost its `staged` / `morris_R` arguments.

Build step 6b (parallel seed lanes, design §1.3, §1.7):
- seeds run min(S, L) at a time in forked processes on cloned workdirs when the lanes were measured
  (parallel_safe subprocess runner, runner built from the contract, a backend that reports each call);
  the plan counts those lanes at their measured efficiency; each lane's call log (`lane` tag) and cache
  are merged back; a crashed slot is replaced as soon as a lane is free; a resume uses the saved lane count;
- lanes only in a one-thread process; lanes close inherited pipes, die with the parent, flush output; the
  parent waits on result and exit, ends a stalled lane, cleans up in a finally; unique lane folders, leftover
  lane caches kept; lanes use the parent's cache file name; eval-id streams counted over all lanes; a lane that
  cannot be set up runs its seed in-process; the probe also measures at the seed count;
- every calibrate() holds an exclusive workdir lock (`.kdt_calibrate.lock` — not the runners' own `.kdt_calib.lock`): a second one there returns
  `workdir_busy` (new status).
- review round 2: no stall bound (a slow model run never ends its lane); a lane leads its own process group and
  ends it (with its model runs) on SIGTERM, whatever handler the host set; an interrupt or error in the parent
  merges the live lanes' finished calls before cleaning up; the leftover merge runs on every call; a crashed
  seed in the in-process fallback is replaced once; no lane folder -> seeds one after another; the workdir
  lock is closed in a finally (a kept exception no longer keeps it).

Build step 8b (agent side and argument clean-up, gaps 2r, 2w, 2x):
- `calibrate()` no longer takes `active_idx` (caller subset search) or `pilot` (caller-given pilot), the
  arguments of the removed staged calibration: every call searches all declared parameters with its own
  pilot, seeds and plan file. A caller that still passes them gets a TypeError;
- the agent prompt, route actions, verdict mapping and DB verdict list follow the new engine
  (auto_dissect_multi_agent; SYSTEM1_MIGRATION.md lists what moves with the kit).
Build step 9 (replay and the runner/kit cross-check, gaps 2u, 2n):
- `replay.py`: a recorded search is fed through the same rule and verdicts, with an evidence manifest;
  Level A only for a complete history with every required metric and a complete manifest, otherwise Level
  B "on available evidence: …"; on a history this kit wrote it reproduces each seed's stop point and verdict;
  `tools/replay_convergence.py` is superseded;
- 2n: at the pilot's default run the kit recomputes the panel from the runner's series and compares the
  runner's own values (its kit block, or without one only NSE / KGE / r); a difference above half a unit of
  the runner's last recorded decimal + 1e-6 makes that metric missing for the search ("2n mismatch: runner
  …, kit …"). Reported as `convergence.cross_check_2n`, saved for a resume. Other KI metrics are listed
  "not checked (the KI may define it differently)". No KI runner writes a series or the kit block yet;
- the final overhead is 5 x t_slow (was 4): the holdout gate also runs the DEFAULT parameters on the
  holdout split (found by the HBV tracer bullet, 2026-09-30), so caps are one t_slow smaller.

Review rounds after the first 8b/9 and 6b passes (2026-09-30):
- 6b round 3: the parent kills a lane's whole process group (also a lane that died on its own); model runs no
  longer inherit an ignored SIGINT; the leftover-folder search escapes the workdir name; Ctrl-Z / fg reach the
  lanes; a lane folder is removed only after its cache and log were merged; `seeds_parallel` = lanes that
  really ran at once.
- 8b/9 round 2: every evaluator record carries the split it was for (also constraint-rejected calls and runner
  exceptions); the 2n precision is per metric over the pilot's runs, and beta goes with a failed PBIAS when the
  runner gives none (Leo); a certified module's no_baseline gets its own setup job; the not_calibratable job
  names the decoupled scorer; the contract prompt makes runners score through `ki_tools_common.kdt_panel`.
- `ki_tools_common.kdt_panel` (staged in `ki_runner_panel_staging/`, not yet installed): the runner-side panel
  helper, tied to the kit by `test_ki_tools_common_panel.py`.

Leo's decisions after the review rounds (2026-09-30, design rev 29):
- the runner-side helper file is DROPPED; each workflow (runner) reports the watched scores itself in its reply's
  reserved section `__kdt__.panel.<var>` (formulas in the contract prompt), apart from its own KI scores, and saves
  its scored series once when the kit asks; the kit reports a declared series it cannot read, with the reason;
- the shared scorer `ki_tools_common.metrics` gains alpha, beta, lnnse, nrmse and `all_metrics(..., extended=True)`
  (live; the default output unchanged); HBV's workflow reports them (tracer bullets 3 and 4);
- a daemonic process (multiprocessing.Pool worker) runs its seeds one after another; the signal mask around a
  lane's fork is restored on every way out (6b round 5).

Review round 5 (8b/9) and round 2 (scorer/HBV), 2026-10-01: the reserved section carries nse and kge too (and the
panel takes the plain nse / kge / r when a section lacks them, so a protected NSE is never lost); section keys are
read case-insensitively; series dates must be ISO (else refused with the reason); a two-target runner's undeclared
series file is picked up; a runner's `series_error` is reported and no half-written file is searched for; the
prompt's read-back check loads every array. HBV v2 is live (Leo: keep it). Step 6b PASSED (round 6).

Reviews of 2026-10-01: step 8b PASSED (Opus r6, codex r1); step 6b PASSED (Opus r6, codex r2: the machine probe
is skipped when lanes cannot be used; a test makes start() raise); the scorer + HBV reply PASSED (Opus r3, codex r2:
HBV refuses fewer than 3 pairs, as r needs 3). Step 9 rounds 6-7: a plain nse / kge / r taken beside a section is a
panel value only (marked; never also "KI headline" or "not checked"); a pilot note when it is unchecked (no series)
and only for what the panel really took; a runner's series_error gets a note; the stale-file check compares the
series file with a marker file touched at the call's start (one clock); an exact lower-case key wins.

Step 9 rounds 8-9 (2026-10-01): the plain nse / kge / r fall-back applies only to scores the target's kind records
(panel, 2n, note); a 2n mismatch on a score the kind never uses is reported only ("not used for this kind"); tests
for HBV's six-key shape, the NaN key, the stale-file check on the file system's clock. Step 9 PASSED (Opus r9).

codex on step 9 (2026-10-01): replay Level A also needs an unbroken, ordered call log (call numbers without gaps,
repeats or reorders); a value a workflow reports for a score the kit's formula leaves undefined on the series (e.g. r
with constant obs) is a 2n mismatch, missing for the search.

- **The workflow belongs to the project; old runners are updated there (2026-10-02, Leo).**
  - `calibrate(reply_gate=...)`: the engine's own pilot judges the runner's reply (required watched scores reported,
    the 2n check agrees, the scored series saved) and reports it as `convergence.reply`. `reply_gate=True` returns
    `status: reply_incomplete` right after the pilot when the reply is not complete; `"check"` always returns
    `reply_checked`; neither saves anything for a resume; a continued search only reports.
  - `project_workflow.py` (new): `build_ki_view` / `open_view` give a project its own copy of the workflow
    (`calibration.yaml`, `tools/calib_run.py`, `calib/capability_case.json`) inside a KI-shaped folder whose other
    entries are links into the KI; the engine is called with `ki_path = <project>/ki_view`, `workdir =
    <project>/run`. One shape of runner section is allowed (python + tools/calib_run.py, injection mode runner);
    anything else is refused with the reason. `ki_fingerprint` / `ki_changes` notice a change in the KI.
  - The calibration agent (`stage_calibrate.calibrate_with_reply_upgrade`) runs the engine from the project
    folder under one lock. An old runner whose reply is incomplete is updated IN THE PROJECT by an agent; the new
    file is kept only if the KI is unchanged, the folder is still a view, codex approves the whole file, and the
    engine's pilot finds the reply complete with the same objectives and the model's own scores unchanged.
    Otherwise the one file is put back. A changed KI, or a file that cannot be put back, stops the call (no
    search); a kill is recovered at the next call. The KI is never written. A workflow that cannot run from a
    project folder is calibrated in the KI as before. `KDT_CALIB_REPLY_UPGRADE=0|false|no|off` turns the update off.
  - Known cost, to be removed next: the first runs (probe, proof, pilot) are repeated for the gate, the check and
    the search.

## 0.3.0 — 2026-09-02 (working tree; not yet a commit)
Features added for System 1's Huai work, all currently uncommitted on top of `8c0f3af`:
- **Declared per-year holdout criteria** (`strategy.holdout.per_year`): a contract can require an
  operational bar in *every* held-out year plus a yield-slope floor, not just an average. Fail
  closed — declared but no `__kdt__.per_year` payload from the runner is inconclusive, never a pass.
- **Convention-band floors**: `per_year: {band: good}` resolves the numeric bar from the model KI's
  own `docs/validation_convention.yaml` instead of a hard-coded number, so the bar is a cited
  convention rather than an invented one.
- **Composite objective via explicit family weights**: a target may weight the dag-valid families it
  declares, e.g. `{temporal_pattern_match: 1.0, magnitude_accuracy: 0.5, timing_accuracy: 0.6}`.
  Only families the dag declares can be weighted; unknown families are refused.
- **MADR backend** (`backends/madr_backend.py`, nested leader/worker), last touched 2026-09-02.
- Scoring-critical modules `calib.py`, `holdout.py`, `objectives.py`, `evaluator.py` last changed
  2026-08-29 — i.e. before the System 1 W4 fits of 2026-08-30/31 and unchanged since, which is what
  makes those 14 structure fits mutually comparable and comparable with later structures.

## 0.2.x — 2026-08-18 (`8c0f3af`)
- DREAM backend fix: sample toward good fits and surface the posterior (codex approved).

## 0.1.x — 2026-08-16 (`0343d64`)
- Initial snapshot of the KI-grounded autonomous calibration framework.

## Known debt
The 0.3.0 feature set lives in the working tree, not in a commit. Until it is committed, results
reference a content hash rather than a revision. Committing should be coordinated — other sessions
edit this tree (the MADR backend changed on 2026-09-02).
