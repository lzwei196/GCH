"""Runner-facing panel helper — import this from a case's calib_run.py (design §1.4, gap 2n).

    from calibration_kit.metrics_panel import panel_block, merge_panel_block
    merge_panel_block(metrics, panel_block({"Q": (sim, obs, dates)}, {}))

`panel_block` writes the kit-owned block `__kdt__.panel.<var>` with the kit's formulas — all eight
keys (r, alpha, beta, pbias, nse, kge, lnnse, nrmse) wherever they are defined, plus the
mean-near-zero rule — and the kit keeps what each variable's kind needs. `merge_panel_block` adds
it without replacing the runner's other __kdt__ keys (applied_params, case_id, split, series).

The helper never writes the plain keys: those stay the KI's own metrics (several KIs define
`nrmse` / `pbias` differently), which the standards check reads as the KI defines them.
Kept as its own tiny module so a runner does not need the rest of the kit.
"""
from .panel import panel_block, merge_panel_block, load_series      # noqa: F401
