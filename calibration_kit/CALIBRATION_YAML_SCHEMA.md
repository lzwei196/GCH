# `calibration.yaml` — the per-model calibration contract

The calibration analog of `dag.yaml`. Where `dag.yaml` declares *what to compare and
which metric is gate-valid*, `calibration.yaml` declares *which parameters may be
tuned, where they live in the model's inputs, and how to vary them*. One file per
model KI, at `<ki_path>/calibration.yaml`.

Per codex's design review (2026-06-26), a contract of only `name/range/default/
transform` does NOT generalize across crop config files, MODFLOW packages, and SWMM
`.inp` sections. It must also capture **type, units, spatial scope + tying, activation
conditions, inter-parameter constraints, and an edit `address`**. The `address` is the
crux — it is how a generic optimizer writes a parameter value back into a
heterogeneous model input.

---

## Top-level shape

```yaml
template_version: "calib-0.1"
model_id: PCSE
identity:
  domain: crop
  binding: Wofost72_WLP_CWB        # the specific model/engine variant these params apply to

# Optional: which dag outputs this calibration targets (objectives derive from the
# dag's gate-valid metric_families for these vars; see objectives.py). If omitted,
# all observable dag outputs with a mapped obs are used.
targets:
  - var: TWSO                       # must exist in dag.outputs
    weight: 1.0                     # relative weight if scalarized

parameters:
  - name: TSUM1                     # human/optimizer-facing id (unique within file)
    description: "Temperature sum emergence->anthesis (°C·d)"
    type: continuous                # continuous | integer | categorical | bool
    units: "degC.day"
    default: 1050.0
    range: [800.0, 1300.0]          # required for continuous/integer
    # categories: [...]             # required for categorical
    transform: identity             # identity | log | logit  (search-space transform)

    # --- spatial scope + tying (REQUIRED for generalization) ---
    scope: global                   # global | per-crop | per-soil-class | per-HRU |
                                    # per-layer | per-subbasin | per-cell | per-zone
    tie: null                       # null = free; or "<other param name>" / a tie-group
                                    # id so several locations move together
    # zone_key: "crop"             # when scope != global: the grouping key in the model

    # --- when this parameter is even active ---
    activation: always              # always | "<expr over other params/config>"
                                    # e.g. "binding == 'Wofost72_WLP_CWB'"

    # --- WHERE to write it (the hard part) ---
    address:
      kind: yaml_path               # see "Address kinds" below
      file: "work/grid/maize.crop"  # relative to ki_path or run workdir
      path: "TSUM1"                 # kind-specific locator

  - name: SPAN
    description: "Life span of leaves at 35°C (d)"
    type: continuous
    units: "day"
    default: 33.0
    range: [25.0, 45.0]
    transform: identity
    scope: global
    address:
      kind: yaml_path
      file: "work/grid/maize.crop"
      path: "SPAN"

# --- cross-parameter constraints (hard feasibility, checked before a run) ---
constraints:
  - "TSUM1 + 200 <= TSUM2"          # python-eval expr over parameter names; False => infeasible

# --- calibration strategy hints (optional; the kit picks defaults from model class) ---
# --- HOW to run+score one candidate, programmatically (NOT a claude agent) ---
# Calibration does 100s of evals, so the run must be a cheap repeatable script that
# runs the model with the CURRENT inputs (applicator already wrote this candidate's
# params) and returns the gate-valid metrics dict. See runner.py.
runner:
  kind: subprocess                  # python | subprocess | detached
  # subprocess: command writes a JSON metrics dict; {workdir}/{metrics_json} substituted
  command: ["./venv/bin/python", "tools/calib_run.py", "--workdir", "{workdir}", "--out", "{metrics_json}"]
  metrics_file: "{metrics_json}"
  timeout: 900
  # python:   callable: "tools.calib_run:run_and_score"   # def run_and_score(workdir)->dict
  # detached: runner: "run_and_score.py"  interpreter: "venv/bin/python"  # poll result.json

strategy:
  cost_class: expensive             # cheap | moderate | expensive  (drives algo + subset/surrogate)
  default_algorithm: dds            # dds | sceua | nsga2 | nsga3 | dream | pestpp_ies | surrogate
  max_evaluations: 300              # a CAP. With `budget.mode: measured` it is an optional CEILING.
  # --- how long to search: MEASURED from the model, not guessed (2026-09-27) -------------
  budget:
    mode: measured                  # measured | fixed   (absent => measured if an allowance is given,
                                    # else fixed, using max_evaluations)
    allowance: 72h                  # measured mode: the wall-clock the user granted (s/m/h/d)
    seeds: 3                        # independent searches (slots), default 3; the cap is PER SEED. 1 =
                                    # one search, reported as "seeds: not checked". See C11.
    run_time: 90s                   # optional: the user's own time for ONE model run. Given => a 3-run
                                    # pilot confirms it and t_slow = 1.2 x max(run_time, slowest pilot run);
                                    # absent => a 10-run pilot and t_slow = 1.2 x its p90
    pilot_runs: 10                  # timed runs before the search (default 3 with run_time, else 10).
                                    # The pilot is ALWAYS on: a value under 1 is raised to 1 with a warning
    hard_max: 20000                 # optional absolute ceiling on the derived cap
    parallel_safe: true             # must be exactly true before the kit runs copies side by side (the
                                    # machine probe; subprocess runners only). Absent/false => 1 lane
  # --- convergence (design HANDOFF_CONVERGENCE_2026-09-27_v2.md) ---------------------------------
  convergence:
    mode: keep_going                # keep_going (default): the convergence rule is recorded only | stop: it
                                    # ends the search at a loop / generation end. The rule (2026-10-04) is each
                                    # optimizer's OWN: SCE-UA = SPOTPY's loop-end test on the objective AND the
                                    # best point's watched scores (classic kstop 10, pcento 0.1, peps 0.001);
                                    # NSGA-II / NSGA-III / MOEA-D = pymoo's DefaultMultiObjectiveTermination
                                    # (recorded; not validated as a stop on our tests); DDS = no rule, the
                                    # settle point "settled by run N of M"; DREAM = R-hat. Old names
                                    # observe/enforce are read with a warning.
    variables:                      # per target: its kind decides its required metrics (§1.4)
      Q: {kind: flow}               # flow | series | snapshot | categorical (default from the dag entry)
    tolerance: bootstrap            # watched-score tolerances (settle point; the recorded per-call step rule):
                                    # bootstrap (measured from the pilot's default run) | fixed
    window: auto                    # the recorded per-call step rule's window (calls): auto | a whole number
    rel_gain: 0.005                 # settle point / step rule: a loss change below this (relative, floor 1e-3)
                                    # counts as flat
    eps_front: 0.01                 # front test (trade-off searches)
    front_period: 50                # pymoo's own stopping rule window, in generations (pymoo's shipped default)
  protect: {Q: [r, beta, lnnse]}    # trade-off searches: metrics no worse than at the default run
  holdout:                          # MANDATORY for credibility (codex pitfall #1)
    kind: years                     # years | sites | regions | random
    fraction: 0.3
  subset:                           # for expensive models: calibrate on a stratified subset
    kind: stratified_cells
    n: 40
    stratify_by: ["agro_zone"]
```

---

## Address kinds (how a value is written back)

The applicator (`applicator.py`) dispatches on `address.kind`. Each kind needs a
distinct locator so a generic optimizer can edit any model input format:

| kind          | locator fields                          | example target |
|---------------|-----------------------------------------|----------------|
| `yaml_path`   | `file`, `path` (dotted/bracket path)    | PCSE `.crop`/`.site` YAML, config.yaml |
| `json_path`   | `file`, `path`                          | JSON config |
| `namelist`    | `file`, `group`, `key`                  | Fortran `&PARAM var=` (wflow, CLM, SWAT old) |
| `ini_key`     | `file`, `section`, `key`                | INI/TOML configs |
| `text_token`  | `file`, `pattern` (regex, 1 capture grp)| free-form text decks; replace capture group |
| `table_cell`  | `file`, `row` (0-indexed), `col` (0-indexed), `delimiter` (REQUIRED for writes; whitespace read-only) | CSV/delimited tables, DSSAT *.SOL |
| `fixed_width` | `file`, `line` (1-indexed), `col_start`, `col_end` (1-indexed inclusive) | fixed-width Fortran decks (APEX/EPIC) |
| `modflow_pkg` | `file`, `package`, `record`, `field`    | MODFLOW package records (via flopy) |
| `swmm_inp`    | `file`, `section`, `obj_id`, `field`    | SWMM `.inp` `[SUBCATCHMENTS]` etc. |
| `api`         | `setter` (python "module:callable")     | in-process override (PCSE ParameterProvider, pyswmm) |

The applicator MUST be idempotent and round-trippable: write value V, read it back,
get V. Each kind has a paired reader so the contract can be self-tested
(`applicator.verify_roundtrip`).

---

## Rules (mirroring dag.yaml's R-rules)

- **C1 Membership.** Every `address.file` must exist (after a model run sets up its
  workdir). Validate at load; a dangling address is a contract error, not a silent skip.
- **C2 Observable target.** `targets[].var` must be a real `dag.outputs` var with a
  mapped obs — you can only calibrate toward something you can score.
- **C3 Scope honesty.** `scope != global` REQUIRES a `zone_key` the model actually
  groups by; otherwise the param is mis-tied and the search is meaningless.
- **C4 Identifiability.** Prefer FEW free parameters (codex pitfall: over-parameterization
  / equifinality). Use `tie` and hierarchical `scope` to reduce dimensionality before
  freeing more.
- **C5 Feasibility first.** `constraints` are hard — an infeasible vector is rejected
  WITHOUT a model run (cheap), never scored.
- **C6 Holdout mandatory.** `strategy.holdout` is required; a calibration that doesn't
  validate on held-out years/sites/regions is not promotable (over-fit risk).
- **C9 Budget accountability.** *(2026-09-27)* A cap is only meaningful next to the run time it
  was derived from. *(Build step 5, 2026-09-29.)* The engine always runs a pilot (10 runs, or 3
  when `budget.run_time` is given), also for a contract with no `budget` block, and the report of
  every run that searched carries `budget_plan` (a no-search triage report does not: C13) —
  pilot timing, machine, lanes, κ = cap/(d+1) — and ONE allowance ledger
  (`budget_plan.ledger`): cap = floor((allowance − shared − final) / waves × e / t_slow), where the
  shared overhead is the MEASURED time of this call's pilot, machine probe, commissioning and
  consumption proof, and the final overhead is 5 × t_slow for the validation of the one reported
  result (split probe on both splits, the result on both splits, the defaults on the holdout split) — each
  charged once, not per seed. In measured mode a cap under 20 calls is refused
  (`budget_exhausted`, with the reason; when `max_evaluations` or `hard_max` cut it, the reason names
  that ceiling). While the search runs the kit warns once if the projected finish passes the seed's
  share of the allowance (a warning, never a deadline). `phase_counts` separates search evaluations
  from the other phases. With no `budget` block `max_evaluations` is still the cap (200 when absent);
  in measured mode `max_evaluations` applies only when declared. In a pilot of 6 runs or more,
  more than 30% failing makes it `pilot_unstable`; in a smaller pilot (the 3-run one) only every
  non-default run failing does — other failures are a warning. A failed DEFAULT run is the triage's to
  judge (C13: `no_baseline` when it gave no finite objective metric).
  `pilot_unstable` stops the run in measured mode; in fixed mode it is a warning. Pilot points that
  break the contract's `constraints` never run and are not failures: they are replaced by fresh draws
  and counted in `pilot_summary.n_rejected_by_constraints`; defaults that break the constraints make the
  pilot `pilot_unstable` at every pilot size. A measured budget with no allowance, or a zero, negative,
  non-finite or unreadable one (the kit takes s, m, h or d: `90m`, not `90min`), is refused at load,
  before any run, unless the caller passes its own budget=. A non-empty `budget` that is not a
  mapping, or a `budget.mode` other than measured / fixed, is refused at load even then (an empty
  value counts as no block). In measured mode, defaults that break the constraints are also refused
  at load.
- **C10 Convergence uses optimizer-native rules.** *(updated 2026-10-05 to describe the
  2026-10-04 implementation)* The report's `convergence.verdict` uses optimizer loop or generation
  evidence; per-call panel checks remain diagnostics under `step_rule_verdict`. SCE-UA converges when
  its normalized geometric population range falls below 0.001, OR when the objective and watched
  scores change by at most 0.1% over ten evolution loops. The population-range branch can fire earlier;
  these are classic SCE-UA settings, not SPOTPY's shipped defaults. For pymoo optimizers, the native
  design-space or objective-space termination test supplies the verdict; evaluation and generation
  caps are budget limits, not convergence. Default `mode: keep_going` records this evidence without
  ending search through the GCH convergence hook; explicit `mode: stop` enables stopping at the
  supported optimizer's loop or generation boundary. DDS retains its budget-dependent schedule and
  reports a descriptive settle point; DREAM uses its own R-hat criterion. `ended` records why execution
  ended, and scientific acceptance against standards is assessed separately. Short or incomplete
  evidence can remain unknown. The legacy `strategy.stop` block is translated on load (mode stop;
  its floor band becomes the standards band). See [current usage](../USAGE.md#budgets-and-convergence)
  for settings, report fields and validation limits.
- **C11 Seeds must agree.** *(build step 6, 2026-09-30; design §1.7, §2.7)* `strategy.budget.seeds`
  (default 3, a positive whole number) searches run from seeds `seed`, `seed+1`, … on the same pilot,
  tolerances, protection anchor and cap per seed; commissioning, the proof and the pilot run once. A
  seed whose search crashes is re-run once with a new number (`seed + seeds + slot`); if that also
  crashes the slot is missing. The report's `convergence.seeds` block lists every slot, the
  **agreement** of the seeds' final CALIBRATION incumbents on each variable's required panel (max − min
  ≤ the pilot's tolerance; a categorical variable is "not compared"; missing evidence → unknown, never
  agreement), the **returned seed** (lowest calibration loss, whatever the agreement; trade-off
  searches: admissible under protection first, then the smallest worst normalized objective on the
  frozen scale of the lowest slot that reached t0), the **run verdict** and each variable's verdict
  across seeds, and the parameter spread (reported, never a gate). The **reported result**
  (`best_x`, `best_params`, `best_loss`) is the returned seed's calibration incumbent — for a trade-off
  search the rule's compromise — and validation (the holdout) is read once, on it. The old
  `KDT_CALIB_FRONT_SELECT=1` validation-based member choice is ignored with a warning. *(Build step
  6b)* Seeds run min(seeds, L) at a time in PARALLEL LANES when the machine probe measured L > 1 lanes
  (a `parallel_safe: true` subprocess runner), the runner is built from the contract (no caller-given
  `run_model`) and the backend reports each call: each attempt runs in a forked process on a cloned
  workdir with its own evaluator and runner, starting from a snapshot of the evaluation cache; its call
  log (tagged `lane`) and cache are merged into the main workdir and the lane workdir is removed; a
  crashed attempt is replaced as soon as a lane is free (at most min(seeds, L) run at once). The budget
  plan counts min(seeds, L) lanes at the efficiency measured for that many (the machine probe also
  measures at the seed count; otherwise the plan says which e it used); a resume uses the saved lane
  count. Lanes are used only in a process with ONE Python thread (a forked lane can hang on a lock another
  thread held); a lane closes the pipe ends it inherited, leads its own process group (ending a lane also
  ends the model runs it started), dies with its parent whatever SIGTERM handler the host set, and flushes
  its output; the calling process ends a lane by killing its whole group (a lane that died — e.g. to the OOM
  killer — takes its model runs with it) and passes Ctrl-Z / fg (SIGTSTP / SIGCONT) on to the lanes — not when the caller
  ignores SIGTSTP or set a handler Python cannot restore; in an orphaned group (setsid, `ssh -t`, `docker exec -it`)
  the kernel drops the stop and the lanes keep running. Lanes and their model runs ignore SIGTTOU (they write to the
  terminal even with `stty tostop`); model runs keep the default Ctrl-C behaviour. A daemonic process (e.g. a
  `multiprocessing.Pool` worker) runs its seeds one after another. No lane is ended for being slow (a model run is never killed for its run time); a lane that
  dies is seen at once. On an interrupt or error in the calling process the live lanes are ended and the
  calls they finished are merged into the main log and cache. A lane that cannot be set up (or a failed
  fork) runs its seed in the calling process instead, where a crash is still replaced once; if no lane
  folder can be made at all, the seeds run one after another, with a note. Each call's log record carries `lane` (the lane
  it ran in) and `slot`. Lane folders are unique per call; every call (with or without lanes) first keeps the cache lines
  of a killed call's leftover lane folders and removes them; a resume that runs fewer lanes at once than its
  saved plan says so. A calibration holds an exclusive lock on its workdir (`.kdt_calibrate.lock`,
  inherited by its lanes, closed when the call ends even by an exception): a second calibration there
  returns `workdir_busy`. Otherwise the seeds run one
  after another (`seeds.seeds_parallel` = 1). The results are the same searches either way. Each lane is a
  FULL copy of the workdir (L copies on disk at once); the copy time is not in the budget plan. A backend that does not report each call (surrogate, PEST++, MADR)
  runs ONE seed ("convergence not tracked", seeds not compared). `best_loss` keeps its form ([the
  scalar] for a single-objective search, one loss per objective for a trade-off search); the
  per-objective losses are in `best_losses_per_objective`. A crashed attempt's calls count in
  `seeds.total_search_calls`, and a replacement warns that the run can pass the allowance (a fixed
  budget: that it makes more search calls than seeds × cap). Trade-off slot records carry
  `incumbent_admissible` and the slot's `frozen_t0` / `frozen_offset` / `frozen_scale`, so the pick can be
  redone from the report; when no seed passes protection the reason says so and never calls it protected. A saved
  budget plan for another number of seeds is re-planned, never resumed.
- **C12 Every run is counted; every call echoes its split.** *(build step 7, 2026-09-30; gaps 2j, 2p;
  design §1 "Data within a search")* `eval_history.jsonl` tags each model run with what it was for:
  `certify`, `objective_probe` and `machine_probe` (runner calls outside the evaluator, marked
  `outside_evaluator`; the machine probe's lane runs overlap in time), `commission`, `consume_proof`,
  `proof` (the proof's receipt check), `pilot`, `search` (with its `seed`), `holdout`; an evaluation made
  outside any phase is `untagged`, never search effort. The runs before the search (certify,
  objective_probe, machine_probe, commission, consume_proof, proof, pilot) are charged as shared overhead;
  the search is what the cap per seed pays for, and the holdout is the final overhead. Every
  call whose metrics the kit reads records `split_echo`, the split the runner says it scored
  (`__kdt__.split`), next to the split the kit asked for (the machine probe's lane runs are timed only;
  their metrics are not read). The report's `split_provenance` counts, per phase, the calls whose echo was
  missing or different, and says whether the calls the search rests on — pilot, search, and the objective
  probe when its payload decided a pilot decision ("mean near zero") — can support a calibration-only
  claim (`calibration_only_supported`; a wrong or missing echo gives False, an unchecked phase None). A
  resumed search rests on the saved pilot and decisions: the pilot's (and, when it decided, the probe's)
  counts are saved in the budget plan (`pilot_split_provenance`, `probe_used_for_pilot`,
  `probe_split_provenance`) and reused; a plan without them, or an unusable record, gives None ("not
  checked"). (A caller can no longer hand in a pilot, build step 8b.) A resume whose saved decisions cannot
  be reused (e.g. the panel variables changed) derives them again from the saved pilot and this call's
  probe, judges this call's probe, and saves this call's flag (and, when its probe decided, its
  record). The failed and crashed reports carry
  `split_provenance` too. Runners must echo
  `__kdt__.split` ("calibration" / "holdout"; "full" if they cannot score a subset or the split is unset).
- **C13 Triage before the search; one search.** *(build step 8, 2026-09-30; design §1, Leo 2026-09-28;
  gaps 2x, 2w, 2s)* The kit always calibrates unless the run cannot be calibrated. Before the search, first
  match wins: `no_baseline` — the pilot's default run gives no finite objective metric (status
  `no_baseline`, no search); `not_calibratable` — the consumption proof gave a verdict for EVERY searched
  parameter and every one is OBJECTIVE_MASKED, DORMANT or WINDOW_MASKED (status `not_calibratable`, no
  search; the reason names each parameter's verdict, "each parameter moved alone from the defaults");
  `calibrate` — everything else, including a proof that was off, not run, empty, or with UNPROVEN or
  INSENSITIVE parameters ("sensitivity not proven" when no parameter is ALIVE). The report's `triage` block
  has the route, reason, proof summary and, per target, the default run's standards and fit verdict —
  information only: how good the fit is at the defaults is never a reason to skip. The old routes
  (`already_adequate`, `diagnose_setup`, `calibrate_caution`, `undetermined`, `fix_driver`,
  `screen_failed`) are gone. `strategy.staged` is removed: a `staged: true` contract runs all its declared
  parameters in one search, with a warning (`staged_warning`). `StopRule` is removed (`stop.py` keeps
  `convention_floor`); `kit_report["stop"]` is not produced — read `convergence`. Backends that do not
  report each call (surrogate, PEST++, MADR) run one seed and report "convergence not tracked". The triage
  runs right after the pilot, before the machine probe, the budget plan and the refusals; a no-search
  report carries the route, `triage`, `pilot` (summary), phase_counts, split_provenance and the proof
  evidence (consumption_proof / coverage, objective_smoke_test), and no `budget_plan`.
- **C14 The workflow reports the scores the kit watches.** *(build step 9, 2026-09-30; gap 2n, Leo 2026-09-29/30)*
  The convergence check watches, per target, the scores its kind requires (flow: r, alpha, beta, lnnse; series:
  r, alpha, beta; snapshot: pbias, nrmse). The workflow (runner) puts them in the kit's reserved section of its
  reply, per target: `"__kdt__": {"panel": {"<var>": {"r", "alpha", "beta", "pbias", "nse", "kge", "lnnse", "nrmse"}}}`
  (lower-case keys; read case-insensitively; when the section lacks nse / kge / r the plain ones are taken, if the
  target's kind records them) — its
  own KI scores stay plain keys, as the KI defines them (so a KI's own PBIAS sign or NRMSE never stands in for the
  kit's). Formulas (float64, pairs where both are finite, population std): r = Pearson (n >= 3); alpha =
  std(sim)/std(obs); beta = mean(sim)/mean(obs); pbias = 100 (mean(sim) - mean(obs))/mean(obs); nse = 1 -
  sum((sim-obs)^2)/sum((obs-mean(obs))^2); kge = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2); lnnse = NSE of
  log(x + 0.01 mean(obs)) (left out if mean(obs) <= 0 or any value <= -0.01 mean(obs)); nrmse = 100 RMSE/|mean(obs)|.
  Undefined scores are left out. Without them the verdict is "unknown". (A reply without the section is still read
  from plain keys for r, alpha, beta, nse, kge, lnnse and pbias-in-percent — never plain nrmse.) When the kit asks
  (`KDT_CALIB_EMIT_SERIES=1`, its pilot's default run) the workflow also saves the paired series it scored as
  `<workdir>/kdt_series_<var>.npz` (float64 `sim`, `obs`, a plain numpy string `date` in ISO form, YYYY-MM-DD or
  YYYY-MM-DDTHH:MM — other forms are refused as unreadable) and ADDS `__kdt__.series.<var>` (one entry per target; a
  file not declared but written by this run is also picked up); if it cannot save it, it writes
  `__kdt__.series_error` (the kit then reports that reason and searches for no file); the kit measures its tolerances from it (else fixed values, labelled; an unreadable series
  is reported with its reason) and recomputes the scores once (the 2n check): every value the panel reads is
  compared, within half a unit of the finest decimal the workflow wrote for THAT metric over this pilot's runs,
  +1e-6; a larger difference — or a value for a score the kit's formula leaves undefined on that
  series — makes that metric MISSING for the search ("2n mismatch: runner …, kit …"), and when
  that is PBIAS and there is no beta, beta too. `convergence.cross_check_2n` holds every comparison; a resume
  reuses it.
- **C7 Round-trip.** Every `address` must pass `verify_roundtrip` before the param is
  used — a write/read that doesn't agree corrupts the search silently.
  **C7 proves the value was WRITTEN. It does NOT prove the model READ it.** See C8.
- **C8 Consumption.** *(ENFORCED by default from 2026-09-14 — `KDT_CALIB_C8=off` only for a
  launcher that has already gated on an equivalent probe against the current setup.)*
  Before any search, EVERY declared parameter is displaced one at a time to its far bound and
  the **raw scored output series** — not the objective — must move. The runner supplies that
  series under `__kdt__.target_proofs` (schema `kdt-target-proof/1`: target, split, window, n,
  dtype, `index_hash`, `values_hash`, `values`, `index`). Objectives are rounded (NSE to 4 dp), so
  they can neither reveal a dead address nor be trusted to acquit a weak one; the raw series can.
  Verdicts, per parameter:
  - `ALIVE` — raw output moved materially and the objectives moved.
  - `INSENSITIVE` — raw output moved, below threshold. Warning. A weak parameter, a modelling call.
  - `OBJECTIVE_MASKED` — raw output moved materially, objectives did not. Warning: the metric
    cannot see this parameter.
  - `DORMANT` — raw output bitwise identical, and the contract carries a **structured**
    `dormant_when` (below) with `kind: process_inactive`. Recorded, not a fault.
  - `WINDOW_MASKED` — as DORMANT with `kind: window_erased` (consumed, erased before the scored
    window — a spin-up). Recorded.
  - `UNREACHABLE` — the runner CONFIRMED it applied the value (`applied_params` echo) and the raw
    output is bitwise identical, and no structured dormancy is declared. **The fit STOPS.** No
    auto-repair, no dropping the parameter. It is a KI defect at the parameter's address and the
    owner fixes it.
  - `UNPROVEN` — zero-width range, infeasible far bound, or the runner emitted no proof. Warning:
    NOT proven, never counted as proven.
  A driver that emits no `target_proofs` yields `ok = null`: nothing is proven and nothing is
  enforced — the box is explicitly UNPROVEN.
  **Cached** per (contract text + driver code + kit scoring code + default-run `values_hash` +
  split). Not per setup fingerprint alone: changing the driver from writing one file to another
  leaves the setup untouched and moves every live address.
  **Target naming (pinned 2026-09-15, both seats):** exactly ONE proof per calibration target,
  named by the contract's `targets[].var`, carrying the FULL scored series concatenated in window
  order (for year-set drivers: every scored year, post any loss term, in year order). Do NOT invent
  segment names (`Q:1991`) — the validator accepts any string, so a driver that did would silently
  get per-segment verdicts the contract never declared. A `segment` field may be added to the
  schema later, additively, when a driver needs segment-level diagnosis.
  Acceptance suite: `tests/test_consumption_proof_v2.py` — dead addresses, structured dormancy,
  weak-live, and raw-moves-but-rounded-objective-identical must all classify correctly.

  > **Why C8 exists.** WRF-Hydro declared `refkdt`, `slope`, `smcmax_mult` at
  > `DOMAIN/soil_properties.nc` — a file the KI's own build tool CREATES by copying values out of
  > `SOILPARM.TBL`, and which that build never opens (no `-DSPATIAL_SOIL`). We calibrated the copy;
  > the model read the original. The file existed (C1 ✓), the values read back (C7 ✓), a July test
  > asserted read-back and passed, and the old responsiveness check passed because the other eight
  > knobs worked. **Eight live knobs hid three dead ones for two months.**
  >
  > The general rule: **before writing a check, ask what you would see if the thing were broken.
  > If the answer is "the same thing I see now", it is not a check.**

## `dormant_when` — declaring that a parameter can legitimately not move here

Structured only. Free text is refused by the kit (it would let a sentence wave through a wrong
address). All of `kind`, `condition`, `evidence` are required:

```yaml
- name: revap_co
  dormant_when:
    kind: process_inactive          # or window_erased
    condition: "aquifer storage stays below revap_min (750 mm) throughout the scored window"
    evidence: "reviews/2026-10_SWATPLUS_REVAP/aquifer_output.json"   # a receipt, not a claim
    declared_by: "codex 2026-10-01"
```
The kit cannot evaluate `condition` itself — it checks that evidence is NAMED. A declaration
without evidence counts as no declaration, and the parameter is judged on the raw output alone.

---

# v2 (2026-07-08) — `injection.mode`, provenance, and the runner contract  (READ THIS; supersedes pre-v2 runner semantics)

## `injection.mode: applicator | runner`  (top-level, default `applicator`)
Existing YAMLs are unchanged (default = `applicator`). The mode is per-KI, NOT per-parameter.

**`applicator` mode (default):** the KIT writes each param value into the model's input files via its
`address` (round-trip verified, C7). Valid ONLY when the model consumes stable files DIRECTLY. Address kinds:
- GENERIC FORMATS (central, reusable): `yaml_path` `json_path` `ini_key` `text_token` `namelist`
  `fixed_width` `table_cell`.
- MODEL/MECHANISM-specific (`modflow_pkg` `swmm_inp` `api`) are NOT applicator kinds — they raise a
  runner-mode error. Use `runner` mode for those (a central MODFLOW handler would reimplement flopy).

**`runner` mode:** the KIT does NOT write params. It writes the candidate vector to `kdt_params.json` in the
eval workdir and sets `KDT_CALIB_PARAMS` to its path; your `tools/calib_run.py` INJECTS them the model's own
way (e.g. `import flopy` / `pyswmm`), runs, and scores. **Required** when the KI regenerates its inputs each
run (a value written before the run would be overwritten). In runner mode `address` is optional documentation.

### Runner-mode output contract (mandatory — else the run is rejected, not trusted)
`tools/calib_run.py` MUST emit, in its metrics JSON, a reserved block:
```json
{ "...metrics...": ..., "__kdt__": { "applied_params": { "<name>": <value>, ... } } }
```
- The kit DIFFS `requested` vs `applied_params` on EVERY eval; any mismatch/missing → that eval scores +inf
  (fail-closed: a silently-non-injecting runner is never trusted).
- Before optimization, the kit runs a one-time objective SMOKE TEST (default vs each single-parameter
  displacement). Since 2026-09-16 it never aborts — objectives are rounded, so a flat objective is not
  evidence. It is recorded in the report as `objective_smoke_test`. The evidence is the C8 consumption
  proof (below), which compares the RAW scored series and is the only check that can STOP a fit.
- The final report always carries `consumption_proof` and `consumption_coverage` (n_proven / n_unproven /
  per-verdict lists). `promotable` is a holdout statement; a box can be promotable with knobs UNPROVEN, and
  the coverage block is where that shows.

## Parameter ranges — split soft prior from hard limit
- `hard_bounds: [lo, hi]` — true physical/numerical limits (constrain the search).
- `prior` / `expected_sensitivity` (optional) — soft, from published site ranges; seeds + ranks, does NOT
  prune. Literature-supported params are ALWAYS in the pool. Which parameters to calibrate is decided ONCE,
  by the agent from the KI (build step 8: the Morris screen and staged escalation are removed); nothing is
  silently excluded (literature is a PRIOR, not a pruning oracle).

## Provenance (mandatory, mirrors the validation-convention citation gate)
Each sensitive param and the holdout choice carries `cite` + `confidence`. Uncited → `low_confidence`
(stays screenable / flagged), never silently trusted.

## Holdout — the generalization AXIS, not just kind/fraction
`strategy.holdout` records: `protocol` (blocked_temporal | leave_region_out | blocked_space_time |
differential_split_sample …), `grouping_key`, `generalization_axis` (must match the promotion claim),
`fraction`, `rationale`, `cite`. Default per obs_shape: point_time_series → contiguous blocked temporal;
multi-site/panel claiming transfer → leave-region-out; gridded → spatial block; spatiotemporal → blocked
space-time on the claimed axis.
