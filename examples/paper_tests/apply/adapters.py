"""Record format -> the standard call list of apply_rule.apply()."""
import json


def recorder_problem_rows(path, step_key):
    """Frozen-kit recorder records where the problem's evaluate() was called directly (HYMOD, E2, validation):
    kind 'problem', phase 'search'; the panel was handed in through problem.last_record (extra.panel)."""
    out = []
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "problem" or r.get("phase") != "search":
            continue
        out.append({"losses": r.get("losses"), "scalar": None, "panel": (r.get("extra") or {}).get("panel"),
                    "step": r.get(step_key)})
    return out
