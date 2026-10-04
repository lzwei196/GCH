Gauge "{subbasin_id}" ({station_code}) | Phase: {phase} | Best NSE so far: {best_nse}
Model: {{MODEL_NAME}}. One catchment, one gauge; every parameter is tunable each round.
Metrics are MEANS over the declared calibration years (NSE, signed PBIAS %, peak-timing days).

DIAGNOSIS:
{diagnosis_block}

FULL HISTORY (all rounds):
{history_table}
{explore_block}{leader_block}

Bounds: {bounds}
{frozen_note}
Propose parameter changes as JSON.
