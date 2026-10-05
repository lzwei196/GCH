# Preparing a calibration workflow

An agent's role in GCH is knowledge-guided calibration code formulation. It reads the scientific request and the model's knowledge infrastructure (KI), makes justified choices, and produces a reusable `calibration.yaml` and runner. The numerical engine then executes the fixed workflow without requiring the agent inside the search loop.

This guide is independent of an agent provider. This checkout supplies the engine and a runnable synthetic example; it does not supply a turnkey language-model launcher or the original KI host application.

## Inputs and decisions

Give the agent the scientific objective, model KI, accessible observations and a working model environment. The KI should establish model interfaces, parameter meanings and admissible domains, observation mappings, units, preprocessing, and any model-specific assessment standards. It constrains the agent's choices; it does not replace executable evidence.

The agent should record the following decisions and their sources:

| Decision | Required evidence |
| --- | --- |
| Parameters and domains | Actual model inputs, defaults, types, units, transformations and physical constraints |
| Observation mapping | Model variable, observation shape, spatial and temporal aggregation, units, masks, warm-up, calibration and holdout split |
| Objectives | Metrics appropriate to the observation shape and scientific question; aggregation or trade-off rationale |
| Optimizer | Compatibility with the objective count, parameter types, constraints and available software |
| Budget | User allowance or justified fixed cap, model runtime if known, seed count and stopping mode |
| Environment | Interpreter, model executable and library versions, input files and working-directory requirements |

Missing scientific inputs require clarification or explicit unresolved status. They are not evidence that a guessed configuration is correct.

## Contract and runner

Use the [schema](../calibration_kit/CALIBRATION_YAML_SCHEMA.md) and [standalone example](../examples/minimal/README.md). For `injection.mode: runner`, the runner must read `KDT_CALIB_PARAMS`, apply those values to the model, and return the metrics declared by the workflow. Parameter read-back must reflect the values actually used by the model, including any transformation or input-file rewrite.

Respect `KDT_CALIB_SPLIT`: score only the requested calibration or holdout observations. Return `__kdt__.applied_params`, the scored split and case identity. Supply raw-output `__kdt__.target_proofs` using `kdt-target-proof/1`, plus the watched panel and paired simulated/observed series when GCH requests them. GCH uses those records to check the runner reply, parameter consumption and assessment consistency. A parameter echo alone does not prove that a value affects model output.

Keep source KI and observations separate from run outputs. Declare parallel safety only if concurrent copies cannot overwrite shared inputs or outputs. Record required executables explicitly; the original host's path conventions are not portable defaults.

## Budget and convergence

The agent chooses an appropriate optimizer and budget policy before search. A fixed cap should have a stated purpose and rationale. For a wall-clock allowance, `strategy.budget.mode: measured` lets GCH estimate an affordable per-seed cap from its pilot, overhead and parallel capacity. Estimated affordability is not evidence of convergence.

Default `strategy.convergence.mode: keep_going` records diagnostics without ending search through the GCH convergence hook. Explicit `stop` mode enables the supported optimizer-native stopping checks at loop or generation boundaries. DDS retains its budget-dependent schedule; DREAM uses its sampling criterion. See [current rules and validation limits](../USAGE.md#budgets-and-convergence). Keep numerical convergence, scientific acceptance and execution completion distinct.

## Executable handoff

Before a full search, verify a default model run, requested split handling, parameter consumption and required metrics/series. GCH performs commissioning and pilot checks; a failure requires correcting the workflow or recording why it cannot proceed. A successful fit alone does not demonstrate correct formulation.

Archive the final contract, runner, KI identity, model/data/runtime versions, engine commit, seeds and budget rationale with the run report and evaluation history. Keep any later workflow revision identifiable as a new preparation. To reproduce the paper, follow the experiment-specific [companion package records](../paper/README.md); the current engine is not a substitute for an experiment's recorded framework version.

## Former host adapters

The removed `authoring/` scripts launched and reviewed agents inside a larger application. They imported missing orchestration services and contained host-specific paths. Their source remains in Git history at `5722f2b`. Optional detached execution and live agent repair elsewhere in the engine still require that external host; this guide does not claim to implement those services.
