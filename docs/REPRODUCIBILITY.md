# Versions and reproduction scope

GCH provides a current numerical engine and access to the retained evidence behind the GRL manuscript *Delegating Calibration Formulation to Language Model Agents*.

## Current engine

The code cleanup starts from GCH commit `5722f2b1e41c49f1a07a0284fdc60daadd1a4295` (4 October 2026). That snapshot incorporates convergence development from the Knowledge Dissection Toolkit at `a5bdb16`. The runtime version is recorded by `calibration_kit.__version__`; record the repository commit and whether it has local changes alongside every run.

`requirements.txt` records the direct dependency versions used to verify this repository on Python 3.12. It is not the historical paper environment and is not a complete transitive environment lock. The small synthetic example is a current-engine integration example, not a reproduction of an empirical paper result.

The current memory planner assumes Linux resource units. On macOS, the pilot overstates peak memory from `ru_maxrss` by a factor of 1024, which can suppress parallel lanes. Use the documented Linux environment for resource-planning comparisons; the synthetic example does not validate macOS parallel allocation. See [testing notes](TESTING.md).

## Original paper experiments

The archived record includes a frozen framework from toolkit commit `8acd3d7`, archived on 7 September 2026, together with experiment-specific contracts, runners, environments and settings. The retained six-model rerun substitutes this snapshot for the live toolkit used originally; it does not establish that every original experiment used this commit. Other retained framework snapshots document specific assessment and selection procedures. Use each experiment's recorded version and setup; these snapshots are not interchangeable engine upgrades.

`paper/reproduce.py` works with an explicitly supplied companion package root. It does not rewrite source records, replace the original engine with the current engine or launch the paper's costly physical-model and agent campaigns. File checks, saved-result scoring, figure reconstruction and fresh numerical reruns answer different questions.

`examples/paper_tests/` preserves selected subsequent rerun and convergence-validation scripts. These scripts still refer to their research server and require external model/data payloads. A recorded successful rerun or a compact check JSON is not the complete input and output bundle. The `validation_B` and `validation_C` folders are convergence studies; their letters do not identify the original GR4J range and repeated-formulation experiments.

## Companion package and access

The clean study package contains the selected reported experiments, KIs, environment manifests, source data and setup guides. It is not bundled into Git source history. Use the package's per-case instructions and access notes for model software, forcing and observations. No public archive URL is claimed until a versioned deposit exists.

Native executable payloads target Linux x86-64. Model-specific libraries and path staging remain necessary. Some observational data have access or redistribution restrictions. The metadata indexes identify files but do not grant access to their contents.

Known historical preservation limits include the exact external Python runtime required for sealed GR4J replay, some commissioning task assemblies, and the additional nine cross-provider authored contracts. Console evidence and reconstructed scripts are labelled as such. Existing source records and setup notes remain the authority for those limitations; a substitute runtime or newly generated contract is a new experiment.

## Preparing agent workflows

The numerical engine supports ordinary Python callables and subprocess runners without the original host application. The old host-specific `authoring/` adapters depended on an absent orchestrator, review service and provider launcher. Their source remains available in Git history, while [AUTHORING.md](AUTHORING.md) describes the transferable preparation procedure.

Host-dependent detached execution, live agent repair and optional external optimizer integrations require additional software. The standalone example and portable checks do not enable or pretend to supply those integrations.
