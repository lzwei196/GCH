# Testing GCH

Use Python 3.12 and install `requirements-dev.txt` in a virtual environment. From the repository root:

```bash
OMP_NUM_THREADS=1 python -m pytest -q
python examples/minimal/run.py --output ./outputs/minimal
```

The implementation suite uses synthetic objectives, temporary workspaces and stub runners. It checks numerical adapters, budget accounting, native convergence rules, seed handling, workflow provenance and saved-history replay. The synthetic example also exercises a real subprocess runner. Neither command launches the paper's physical models or live language-model campaigns.

## Optional host integration tests

Some retained tests exercise components of the original KI host application. They now report explicit skips when those components are unavailable:

- Host authoring, database and runner-repair tests require `calibrate`, `db_calibration` and `stage_calibrate` from that application.
- Shared-scorer comparison tests require the external `ki_tools_common` implementation.
- MADR backend tests require the optional MADR source tree.
- The original VIC convention test requires its server-side KI; synthetic convention tests still run.
- Seven Linux-specific lane, signal-mask and terminal checks require `/proc` or util-linux `script` and skip on other operating systems. The remaining portable lane checks still run.

A skip means the integration was not exercised. It is not a successful integration check. The original host adapters removed from `authoring/` remain available in Git history at `5722f2b`.

The machine-probe test retains a positive available-memory check on Linux. The current probe returns zero when `/proc/meminfo` is unavailable; the test checks that explicit fallback on other platforms. This cleanup does not change resource-allocation behavior.

The zero-call reporting test uses an available stub backend, so its coverage does not depend on an optional surrogate optimizer being installed. The lane-probe integration test controls its pilot-memory input while measuring real subprocess runtime and efficiency. The current pilot divides `ru_maxrss` by 1024 as on Linux, although macOS supplies bytes; on macOS this overstates peak memory by a factor of 1024 and can suppress parallel lanes. That production portability issue remains open; the controlled test does not establish macOS memory-planner correctness.

## Existing expected failures

`calibration_kit/tests/test_entry_points_checkpoint.py` already identifies two open behavior gaps:

1. A parameter whose response returns to its default at the tested endpoint may be labelled unreachable.
2. Small run-to-run noise may make an unused parameter appear responsive.

Their existing `known_gap` annotations now also register as strict pytest expected failures. They remain visible in the test report. An unexpected pass fails the suite so the annotation must be reviewed when a gap is fixed. This repository cleanup does not fix or broaden the scientific claims around those behaviors.

## Paper evidence checks

See [paper/README.md](../paper/README.md). Those commands use the companion study package and distinguish index validation, file integrity, saved numerical consistency and figure reconstruction. They do not rerun all scientific experiments or establish model validity from file presence.
