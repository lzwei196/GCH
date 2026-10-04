"""MADR backend test with a STUB LLM (no CLI call): the stub reads the bounds and the last
metrics from the prompt and proposes a move of each parameter toward the optimum by a fraction,
so the loop, clamping, history, diagnosis, patience and the kit stop hook are all exercised."""
from __future__ import annotations
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.backends.base import Problem                     # noqa: E402
from calibration_kit.backends.madr_backend import MadrBackend         # noqa: E402

OPT = {"Sumax": 200.0, "Beta": 2.0, "Ce": 0.4}


class _Resp:
    def __init__(self, d): self.content = json.dumps(d); self.parsed = d; self.prompt_tokens = 10; self.completion_tokens = 5; self.total_tokens = 15


class StubLLM:
    """Moves halfway to the optimum from the best-so-far center the prompt names (parsed from Bounds/history)."""
    calls = 0
    def chat(self, messages, system="", **kw):
        StubLLM.calls += 1
        txt = messages[-1]["content"]
        rows = [l for l in txt.splitlines() if l.startswith("| ") and not l.startswith("| round")]
        # last row of the history table = latest tried point; use the best NSE row instead
        best = None
        for l in rows:
            cells = [c.strip() for c in l.strip("|").split("|")]
            try:
                nse = float(cells[4]); vals = [float(v) for v in cells[1:4]]
            except (ValueError, IndexError):
                continue
            if best is None or nse > best[0]:
                best = (nse, vals)
        cur = dict(zip(OPT, best[1])) if best else {"Sumax": 100.0, "Beta": 1.0, "Ce": 0.8}
        prop = {k: cur[k] + 0.5 * (OPT[k] - cur[k]) for k in OPT}
        return _Resp({"vic": prop, "reasoning": "halfway to the optimum (stub)"})


def _problem():
    metrics = {}
    def evaluate(x):
        d = dict(zip(OPT, x))
        nse = 0.95 - ((d["Sumax"] - 200) / 400) ** 2 - (d["Beta"] - 2) ** 2 / 4 - (d["Ce"] - 0.4) ** 2
        pb = 20 * (d["Sumax"] - 200) / 400
        metrics[len(metrics)] = {"nse": nse, "pbias": abs(pb), "pbias_signed_mean": pb, "day_bias": abs(d["Beta"] - 2)}
        return [1.0 - nse + 0.5 * abs(pb) / 100 + 0.02 * abs(d["Beta"] - 2)]
    prob = Problem(names=list(OPT), lower=[10.0, 0.5, 0.1], upper=[800.0, 5.0, 1.0], objective_names=["J"], evaluate=evaluate)
    return prob, metrics


def test_loop_and_patience():
    prob, metrics = _problem()
    with tempfile.TemporaryDirectory() as tmp:
        be = MadrBackend(model_key="HBV", mc_candidates=1, patience_rounds=4, llm_client=StubLLM(), workdir=tmp)
        specs = [{"name": "Sumax", "range": [10, 800], "default": 100.0}, {"name": "Beta", "range": [0.5, 5], "default": 1.0},
                 {"name": "Ce", "range": [0.1, 1.0], "default": 0.8}]
        res = be.optimize(prob, budget=40, seed=0, metrics_of=lambda i: metrics.get(i), param_specs=specs)
        assert res.n_evaluations >= 5 and res.best_loss[0] < 0.2, res.notes
        assert "patience" in res.notes or res.n_evaluations == 40, res.notes
        assert Path(tmp, "madr", "madr_summary.json").exists() and Path(tmp, "madr", "gauge_ledger.jsonl").exists()
        assert Path(tmp, "madr", "madr_prompts", "sys1_single_gauge", "worker_system.md").read_text().count("Sumax") >= 2
        print(f"T1 loop: {res.n_evaluations} evals, best loss {res.best_loss[0]:.4f}, best native {dict((k, round(v,3)) for k,v in min(res.history, key=lambda h: h['loss'])['native'].items())}; {res.notes[-60:]}: PASS")


def test_kit_stop_hook_wins():
    prob, metrics = _problem()
    with tempfile.TemporaryDirectory() as tmp:
        hook = lambda history: len(history) >= 10          # the kit's per-call hook asks MADR to stop
        be = MadrBackend(model_key="HBV", mc_candidates=1, patience_rounds=50, llm_client=StubLLM(), workdir=tmp)
        res = be.optimize(prob, budget=60, seed=0, on_eval=hook, metrics_of=lambda i: metrics.get(i),
                          param_specs=[{"name": n, "range": [lo, hi]} for n, lo, hi in zip(prob.names, prob.lower, prob.upper)])
        assert res.stopped_early and "kit stop rule" in res.notes, res.notes
        print(f"T2 kit stop rule ended MADR at {res.n_evaluations} evals: PASS")


if __name__ == "__main__":
    test_loop_and_patience(); test_kit_stop_hook_wins(); print("ALL MADR-BACKEND TESTS PASS")
