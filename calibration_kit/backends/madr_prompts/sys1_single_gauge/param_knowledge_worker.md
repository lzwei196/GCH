## Reading the diagnosis
- "volume": PBIAS beyond ±10% — a storage / evaporation problem (capacity, ET shape, nonlinearity).
- "timing": the simulated flood peak is days away from the observed one — a lag / recession-speed problem.
- "shape": volume and timing acceptable but NSE low — peaks too sharp or too flat, recession wrong.
- "baseflow": dry-season flow off — slow-store drainage.
- Trend "stagnant" for several rounds with small steps = you are at a local optimum: take a large step on a DIFFERENT parameter or reverse an earlier decision.

## Sign conventions
- PBIAS = 100 * (sum(sim) - sum(obs)) / sum(obs). POSITIVE = simulated volume too HIGH.
- Peak-timing error is unsigned (days); the history rows carry it as `timing`.
