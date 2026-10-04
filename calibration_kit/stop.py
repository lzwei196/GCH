"""The KI's validation-convention floor — `convention_floor` (kept from the old stopping layer).

`StopRule` and the `strategy.stop` block were REMOVED in build step 8 (design §5 item 8, gap 2w): the
convergence rule (rule.py, verdict.py) is the one test of "has the search settled", and it never looks
at how GOOD the fit is. An old contract's `strategy.stop` block is translated at load (rule.resolve_mode:
mode stop; its floor band becomes the band of the standards check, design §2.9). `kit_report["stop"]` is
no longer produced; readers use the report's `convergence` block.

What stays here is `convention_floor`: it reads the KI's OWN validation convention
(docs/validation_convention.yaml, cited bands — Moriasi 2007/2015 for discharge), never a number invented
here. The standards check (§2.9) and holdout.py's per-year band read the bands through it.
"""
from __future__ import annotations
import json
import math
from pathlib import Path


def convention_floor(ki_path: str, main_stat: str, band: str = "satisfactory",
                     dag_variable: str | None = None) -> dict:
    """Read the floor for `main_stat` from the KI's docs/validation_convention.yaml.
    Returns {"value", "band", "cites", "pbias_max", "entry"}; raises if the convention does
    not name that band — the rule refuses to make one up."""
    import yaml
    p = Path(ki_path) / "docs" / "validation_convention.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no validation convention at {p} — cannot set a floor")
    conv = yaml.safe_load(p.read_text())
    # schema_version 1 (VALIDATION_CONVENTION_SCHEMA.md): top-level `validation:` is the list of
    # per-obs-shape entries, each with headline_metrics[] and dag_variable
    entries = conv if isinstance(conv, list) else (conv.get("validation") or conv.get("conventions")
                                                     or conv.get("entries") or [])
    chosen = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        if dag_variable and str(e.get("dag_variable", "")).lower() != dag_variable.lower():
            continue
        heads = e.get("headline_metrics") or []
        if any(str(h.get("metric", "")).lower() == main_stat.lower() for h in heads):
            chosen = e
            break
    if chosen is None:
        raise LookupError(f"convention at {p} has no headline {main_stat!r}"
                          f"{' for ' + dag_variable if dag_variable else ''}")
    head = next(h for h in chosen["headline_metrics"] if str(h.get("metric", "")).lower() == main_stat.lower())
    bands = head.get("bands") or {}
    if band not in bands:
        raise LookupError(f"convention names bands {sorted(bands)} for {main_stat}; {band!r} is not one of them")
    pb = next((h for h in chosen["headline_metrics"] if str(h.get("metric", "")).lower() == "pbias"), None)
    pb_max = None
    if pb and (pb.get("bands") or {}):
        pb_bands = pb["bands"]
        pb_max = pb_bands.get(band, pb_bands.get(pb.get("pass_band", "satisfactory")))
    return {"value": float(bands[band]), "band": band, "direction": head.get("direction", "maximize"),
            "cites": head.get("cites", []), "pbias_max": (float(pb_max) if pb_max is not None else None),
            "dag_variable": chosen.get("dag_variable"), "source": str(p)}
