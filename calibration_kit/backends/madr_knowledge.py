"""Knowledge blocks the MADR rung renders into its single-gauge prompts (2026-08-30).

One block per model: what each parameter does (in words a hydrologist would use), whether it
is log-scaled, and a symptom -> fix table. These are the PRIORS the Worker LLM reasons with;
the numbers it proposes are clamped to the contract ranges by the backend, never by the prompt.
Sources: the KIs' calibration contracts (parameter descriptions + cites), dag.yaml notes, and
the model papers named there. Keep them short; the LLM also sees the full round history.
"""
from __future__ import annotations

KNOWLEDGE = {
    "HBV": {
        "model_name": "HBV-bmi (lumped 9-parameter bucket model: interception -> soil moisture -> fast + slow reservoirs -> triangular lag)",
        "params": {
            "Imax":  ("mm", "log", "maximum interception storage; more Imax = more canopy evaporation, slightly LESS runoff volume"),
            "Ce":    ("-", "lin", "evaporation shape: Ea = Ep*Su/(Sumax*Ce); LOWER Ce = MORE actual evaporation = less volume (a near-pure volume lever)"),
            "Sumax": ("mm", "log", "soil-moisture capacity (like field capacity); the main volume lever — LARGER Sumax = more storage/ET, LESS runoff, damped response"),
            "Beta":  ("-", "lin", "runoff-coefficient nonlinearity cr=(Su/Sumax)^Beta; HIGHER Beta = less runoff when dry, sharper wet-season response (wet/dry yield contrast)"),
            "Pmax":  ("mm/d", "log", "maximum percolation from soil to the slow reservoir; HIGHER = more baseflow, smaller fast peaks"),
            "Tlag":  ("d", "lin", "triangular unit-hydrograph lag (days); shifts and smooths the peak — timing lever"),
            "Kf":    ("1/d", "log", "fast-reservoir recession coefficient; HIGHER = quicker, higher peaks and faster recession"),
            "Ks":    ("1/d", "log", "slow-reservoir recession; HIGHER = faster baseflow drainage, lower dry-season flow"),
            "FM":    ("mm/°C/d", "lin", "degree-day snowmelt factor; only matters when temperature is below 0 (little snow at a humid basin)"),
        },
        "rules": [
            # Ea = Ep*Su/(Sumax*Ce): LOWER Ce = MORE evaporation = LESS runoff (kimi M4 caught the reversed rows)
            ("volume too HIGH in every year (PBIAS > +10%)", "raise Sumax and/or LOWER Ce (more ET); raise Beta if the excess is in dry years"),
            ("volume too LOW in every year (PBIAS < -10%)", "lower Sumax, RAISE Ce (less ET); lower Beta if dry years are the shortfall"),
            ("wet years right, dry years too WET (yield does not fall in dry years)", "raise Beta (stronger nonlinearity); raise Sumax moderately"),
            ("peaks too high, recession too fast", "lower Kf; raise Pmax (send more water to the slow store)"),
            ("peaks too low / too smooth", "raise Kf; lower Pmax; lower Tlag"),
            ("peak a few days LATE", "lower Tlag; raise Kf"),
            ("peak EARLY", "raise Tlag"),
            ("baseflow / dry-season flow too low", "lower Ks; raise Pmax"),
            ("baseflow too high", "raise Ks; lower Pmax"),
            ("NSE stuck while volume is right", "work on Kf/Tlag (shape), not Sumax/Ce (volume)"),
        ],
    },
    "GR4J": {
        "model_name": "GR4J (MARRMoT m_07_gr4j_4p_2s: production store, groundwater exchange, routing store, two unit hydrographs)",
        "params": {
            "x1": ("mm", "log", "production-store capacity; LARGER x1 = more storage and ET, LESS runoff, damped response — the main volume lever"),
            "x2": ("mm/d", "lin", "groundwater exchange; NEGATIVE = water lost from the catchment (lower volume, especially low flows); POSITIVE = gained"),
            "x3": ("mm", "log", "routing-store capacity; LARGER x3 = slower, smoother recession and more sustained baseflow"),
            "x4": ("d", "lin", "unit-hydrograph time base; LARGER x4 = later, flatter peaks — the timing/shape lever"),
        },
        "rules": [
            ("volume too HIGH (PBIAS > +10%)", "raise x1; make x2 more negative"),
            ("volume too LOW (PBIAS < -10%)", "lower x1; make x2 less negative / positive"),
            ("dry years too wet relative to wet years", "raise x1 (more nonlinearity through the production store)"),
            ("peaks too high / too sharp", "raise x4; raise x3"),
            ("peaks too low / too flat", "lower x4; lower x3"),
            ("peak LATE", "lower x4"),
            ("peak EARLY", "raise x4"),
            ("baseflow too low", "raise x3; raise x2"),
            ("baseflow too high", "lower x3; lower x2"),
        ],
    },
    "VIC": {
        "model_name": "VIC classic (distributed water balance, per-cell) + Lohmann routing to the gauge",
        # names = the System 1 VIC contract (models/vic/calibration_huaibin.yaml), kimi M4
        "params": {
            "binfilt":    ("-", "log", "variable infiltration curve exponent; HIGHER = more surface runoff (higher peaks, more volume)"),
            "expt_scale": ("-", "lin", "multiplier on the Brooks-Corey drainage exponent (all layers); HIGHER = slower drainage, wetter soil, more runoff later"),
            "Ds":         ("-", "log", "fraction of Dsmax where nonlinear baseflow starts; HIGHER = more baseflow at low soil moisture"),
            "Dsmax":      ("mm/d", "log", "maximum baseflow velocity; HIGHER = more baseflow, lower storm peaks"),
            "Ws":         ("-", "lin", "soil-moisture fraction where nonlinear baseflow starts; HIGHER = baseflow kicks in later"),
            "depth1":     ("m", "lin", "thin surface layer thickness; controls fast runoff and bare-soil evaporation"),
            "depth2":     ("m", "lin", "second soil layer thickness; DEEPER = more storage/ET, less runoff, slower response — the main volume lever"),
            "depth3":     ("m", "lin", "bottom layer thickness; DEEPER = more baseflow storage, smoother recession"),
            "lai_scale":  ("-", "lin", "multiplier on monthly LAI; HIGHER = more transpiration + interception = LESS runoff volume"),
            "rmin_scale": ("-", "lin", "multiplier on minimum stomatal resistance; HIGHER = LESS transpiration = MORE runoff volume"),
            "rout_velocity":    ("m/s", "lin", "Lohmann wave celerity; HIGHER = earlier, sharper peaks at the gauge (on the flat Huai plain an effective residence time)"),
            "rout_diffusivity": ("m2/s", "lin", "Lohmann diffusivity; HIGHER = broader response — it also ADVANCES the peak, so it is not a substitute for lowering velocity"),
        },
        "rules": [
            ("volume too HIGH", "lower binfilt; deepen depth2; raise lai_scale or lower rmin_scale (more ET); lower Dsmax"),
            ("volume too LOW", "raise binfilt; thin depth2; lower lai_scale or raise rmin_scale (less ET); raise Dsmax"),
            ("peaks too high", "lower binfilt; raise rout_diffusivity; deepen depth2"),
            ("peaks too low", "raise binfilt; lower rout_diffusivity"),
            ("peak LATE", "raise rout_velocity"),
            ("peak EARLY", "lower rout_velocity (raising diffusivity would advance it further)"),
            ("baseflow too low", "raise Dsmax, Ds; deepen depth3"),
            ("baseflow too high", "lower Dsmax; raise Ws"),
        ],
    },
}


def block_for(model_key: str, names: list[str], ranges: dict) -> dict:
    """Render-ready block restricted to the parameters actually in the search (names), with the
    contract ranges. Unknown parameters get a neutral description so nothing is invented."""
    k = KNOWLEDGE.get(model_key) or {"model_name": model_key, "params": {}, "rules": []}
    rows = []
    for n in names:
        unit, scale, desc = k["params"].get(n, ("-", "lin", "calibration parameter (no prior description on file)"))
        lo, hi = ranges[n]
        rows.append(f"| `{n}` | {unit} | {'LOG' if scale == 'log' else 'linear'} | {lo:.6g} to {hi:.6g} | {desc} |")
    table = "| Name | Unit | Scale | Propose value in | Role |\n|---|---|---|---|---|\n" + "\n".join(rows)
    rules = "| Symptom | Fix |\n|---|---|\n" + "\n".join(f"| {s} | {f} |" for s, f in k["rules"])
    return {"MODEL_NAME": k["model_name"], "PARAM_TABLE": table, "RULES_TABLE": rules,
            "LOG_PARAMS": ", ".join(f"`{n}`" for n in names if k["params"].get(n, ("", "lin", ""))[1] == "log") or "(none)"}
