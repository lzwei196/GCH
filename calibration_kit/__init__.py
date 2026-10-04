"""calibration_kit — model-agnostic calibration for the self-improve KI engine.

A SIBLING module to the self-improve KI: the loop makes a model RUN CORRECTLY;
this kit tunes its PARAMETERS to fit the dag-defined obs. Data assimilation
(sequential state estimation) is a separate sibling, not part of this module.

Public entry: calib.calibrate(ki_path, workdir, run_model, obs_shape_by_var, ...).
See CALIBRATION_YAML_SCHEMA.md for the per-model contract and README.md for design.
"""
# ---------------------------------------------------------------------------
# VERSION (added 2026-09-02). The kit is a live working tree that several projects
# calibrate against and that is edited while in use, so a result must be able to name
# the ENGINE that produced it, not just the model and contract. Bump this on any change
# to scoring, stopping, holdout or objective behaviour, and add a CHANGELOG.md line.
# Consumers should record `calibration_kit.__version__` (plus the git commit and dirty
# flag) in their run reports — two numbers are only comparable if these match.
# ---------------------------------------------------------------------------
__version__ = "0.4.0-dev"
__version_note__ = (
    "0.4.0-dev (branch convergence-panel-2026-09): convergence rebuilt to design "
    "HANDOFF_CONVERGENCE_2026-09-27_v2.md — see CHANGELOG. Stopping behaviour changed. "
    "0.3.0 added the System 1 features: declared per-year holdout criteria (fail-closed when the "
    "runner returns no per-year payload), convention-band floors resolved from the KI's "
    "validation convention, explicit family weights for a composite objective, and the MADR "
    "leader/worker backend. Scoring modules calib/holdout/objectives/evaluator unchanged since "
    "2026-08-29, which is what makes the W4 structure comparison valid."
)

from . import objectives, applicator, calib  # noqa: F401
