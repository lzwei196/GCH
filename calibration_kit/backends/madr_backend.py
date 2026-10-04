"""MADR backend — the agentic escalation rung of the calibration kit (System 1 W4, 2026-08-30).

MADR (Li et al., github.com/jovequli/MADR) is a leader/worker LLM calibration loop. Here it is
used the way the plan asks: **MADR proposes, the kit's evaluator scores, our stop rule and gates
judge.** We do NOT run MADR's LangGraph orchestration (multi-basin, needs langgraph + a SQLite
checkpointer that are not installed); we import its reasoning parts as libraries —
`WorkerAgent.propose_params` (one LLM call per proposal, JSON parsed/clamped), the code-only
`diagnose_round`, `WorkerLedger`/`WorkerMemory` (history + stagnation), and
`phase_config.check_phase_transition` (its patience rule, verbatim) — and re-implement the
~100-line round loop around `problem.evaluate`.

Interface (calibration_kit/backends/base.py): optimize(problem, budget, seed, on_eval=,
metrics_of=, param_specs=, transforms=). `param_specs` + `transforms` let the Worker reason in
NATIVE units (mm, 1/d) while the kit searches in its transformed box (log for wide ranges);
`metrics_of(i)` supplies the full metrics dict of evaluation i (NSE, PBIAS, day_bias...), which
the Worker's diagnosis needs — without it MADR degrades to an LLM random walk.

LLM: MADR's `claude_cli` / `codex_cli` providers (no API key; the CLIs installed on this server).
Prompts: `madr_prompts/sys1_single_gauge/*.md` rendered with `madr_knowledge.KNOWLEDGE[model]`
into <workdir>/madr_prompts and handed to MADR through MADR_PROMPTS_ROOT.

Stop: MADR's own patience (rounds without best-NSE improvement) is honoured AND reported as
evidence; the kit's stop rule (`on_eval`) can end the search earlier. Budget = cap on evaluations.
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from pathlib import Path

from .base import Backend, Problem, CalibResult, EarlyStop

MADR_ROOT = Path(os.environ.get("MADR_ROOT", "/mnt/disk1/Hydrocraft_server/models/MADR/source/repo/MADR"))
COMBO = "sys1_single_gauge"
STRATEGIES = {  # multi-candidate nudges (MADR graph_nodes._run_worker_multi, condensed)
    "conservative": "Strategy for THIS candidate: CONSERVATIVE — small steps (<10% of range) on the single most promising parameter.",
    "moderate": "Strategy for THIS candidate: MODERATE — medium steps on 1-2 parameters following the diagnosis.",
    "bold": "Strategy for THIS candidate: BOLD — large steps (30-50% of range) on 2-3 parameters; explore the far side of the range.",
    "reverse": "Strategy for THIS candidate: REVERSE — undo the direction of the last change that hurt NSE and go the other way.",
}


class MadrBackend(Backend):
    name = "madr"

    def __init__(self, provider: str = "claude_cli", model: str = "sonnet", *, model_key: str = "HBV",
                 mc_candidates: int = 3, mc_every: int = 3, patience_rounds: int = 10, max_rounds: int | None = None,
                 timeout_sec: float = 600.0, llm_client=None, workdir: str | None = None):
        self.provider, self.model, self.model_key = provider, model, model_key
        self.mc_candidates, self.mc_every, self.patience_rounds, self.max_rounds = mc_candidates, mc_every, patience_rounds, max_rounds
        self.timeout_sec, self._llm, self.workdir = timeout_sec, llm_client, workdir

    @staticmethod
    def available() -> bool:
        return (MADR_ROOT / "scripts" / "worker_agent.py").is_file()

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _import_madr():
        if str(MADR_ROOT) not in sys.path:
            sys.path.insert(0, str(MADR_ROOT))
        from scripts.worker_agent import WorkerAgent, WorkerMemory          # noqa
        from scripts.worker_ledger import WorkerLedger                      # noqa
        from scripts.worker_diagnostics import diagnose_round               # noqa
        from scripts import param_validator as PV                           # noqa
        from scripts.phase_config import check_phase_transition             # noqa
        return WorkerAgent, WorkerMemory, WorkerLedger, diagnose_round, PV, check_phase_transition

    def _render_prompts(self, wd: Path, names: list[str], native_ranges: dict) -> None:
        from .madr_knowledge import block_for
        blk = block_for(self.model_key, names, native_ranges)
        src = Path(__file__).parent / "madr_prompts" / COMBO
        dst = wd / "madr_prompts" / COMBO; dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.md"):
            t = f.read_text()
            for k, v in blk.items():
                t = t.replace("{{" + k + "}}", v)
            (dst / f.name).write_text(t)
        os.environ["MADR_PROMPTS_ROOT"] = str(wd / "madr_prompts")

    def _llm_client(self):
        if self._llm is not None:
            return self._llm
        from scripts.llm_client import create_client
        return create_client(model=self.model, provider=self.provider, temperature=0.3, max_tokens=4096,
                             json_mode=True, timeout_sec=self.timeout_sec)

    @staticmethod
    def _madr_metrics(m: dict | None) -> dict:
        """kit/driver metrics -> the keys MADR's diagnosis and prompts read."""
        if not m:
            return {"NSE": float("nan")}
        g = lambda *ks: next((float(m[k]) for k in ks if m.get(k) is not None and math.isfinite(float(m[k]))), None)
        out = {"NSE": g("nse", "NSE")}
        pb = g("pbias_signed_mean", "pbias", "PBIAS")
        if pb is not None: out["PBIAS"] = pb
        for src, dst in (("kge", "KGE"), ("rmse", "RMSE"), ("day_bias", "peak_timing_bias_days")):
            v = g(src)
            if v is not None: out[dst] = v
        out["timestep"] = "daily"
        return out

    # ---- the search ----------------------------------------------------------
    def optimize(self, problem: Problem, budget: int, seed: int = 0, **kw) -> CalibResult:
        WorkerAgent, WorkerMemory, WorkerLedger, diagnose_round, PV, check_phase_transition = self._import_madr()
        on_eval = kw.get("on_eval"); metrics_of = kw.get("metrics_of")
        specs = kw.get("param_specs") or [{"name": n, "range": [lo, hi]} for n, lo, hi in zip(problem.names, problem.lower, problem.upper)]
        fwd, inv = kw.get("transforms") or ({}, {})
        names = list(problem.names)
        native_ranges = {p["name"]: (float(p["range"][0]), float(p["range"][1])) for p in specs}
        to_native = lambda x: {n: (inv[n](xi) if n in inv else xi) for n, xi in zip(names, x)}
        to_search = lambda d: [(fwd[n](float(d[n])) if n in fwd else float(d[n])) for n in names]
        wd = Path(self.workdir or kw.get("workdir") or ".").resolve() / "madr"; wd.mkdir(parents=True, exist_ok=True)
        self._render_prompts(wd, names, native_ranges)
        for n in names:                                   # MADR rounds to 3 decimals by default — not for 1e-5 recession rates
            PV.PARAM_DECIMALS[n] = 8
            # kimi M1: MADR's MIN_NONZERO floors (velocity 0.5, diffusivity 200, ...) could silently lift a
            # proposal above the DECLARED lower bound before the kit sees it. The declared box wins.
            if n in PV.MIN_NONZERO:
                PV.MIN_NONZERO[n] = native_ranges[n][0]

        worker = WorkerAgent(agent_id="worker_gauge", subbasin_id="gauge", station_code="gauge",
                             llm_client=self._llm_client(), is_outlet=True, neighbors_downstream=[],
                             param_ranges={n: list(native_ranges[n]) for n in names}, model_combination=COMBO,
                             topology_note="One catchment, one gauge. No upstream inputs.", n_basins=1, outlet_id="gauge")
        worker.output_dir = str(wd); worker.ledger = WorkerLedger(output_dir=str(wd), subbasin_id="gauge")
        # generic history table: every parameter, native units (MADR's tables are combination-specific)
        def _table(phase="JOINT", model_combination=COMBO, _mem=worker.memory):
            if not _mem.history:
                return "(no history yet)"
            rows = ["| round | " + " | ".join(names) + " | NSE | PBIAS | timing |", "|" + "---|" * (len(names) + 4)]
            def _num(d, k):  # a present-but-None metric (failed eval) must read as nan, not crash the table
                v = d.get(k)
                return float(v) if isinstance(v, (int, float)) else float("nan")
            for e in _mem.history:
                p, m = e["params"], e["metrics"]
                rows.append(f"| {e['round_id']} | " + " | ".join(f"{_num(p, n):.4g}" for n in names) +
                            f" | {_num(m, 'NSE'):.3f} | {_num(m, 'PBIAS'):+.1f} | {_num(m, 'peak_timing_bias_days'):.1f} |")
            return "\n".join(rows)
        worker.memory.format_full_history_table = _table

        history: list[dict] = []; stopped_early = False; stop_note = ""
        max_rounds = self.max_rounds or max(2, int(budget))
        pipeline = [{"name": "JOINT", "max_rounds": max_rounds, "mc_rounds": 0, "mc_candidates": self.mc_candidates,
                     "patience_rounds": self.patience_rounds}]
        best = {"loss": float("inf"), "x": None, "native": None, "metrics": None, "nse": -float("inf")}
        prev_m = None; rounds_since_improve = 0; rounds_in_phase = 0; round_id = 0; llm_tokens = 0

        def _score(native: dict, label: str):
            nonlocal stopped_early
            x = to_search(native)
            x = [min(max(v, lo), hi) for v, lo, hi in zip(x, problem.lower, problem.upper)]
            losses = problem.evaluate(x); loss = float(losses[0]) if losses else float("inf")
            m = self._madr_metrics(metrics_of(len(history)) if metrics_of else None)
            history.append({"x": x, "loss": loss, "round": round_id, "label": label, "native": {n: float(native[n]) for n in names}, "metrics": m})
            if on_eval is not None and on_eval(history):
                raise EarlyStop()
            return x, loss, m

        try:
            # round 1 = the defaults (MADR does the same): a diagnosed baseline, no LLM call
            x0 = kw.get("x0")
            center = to_native(x0) if x0 is not None else {p["name"]: float(p.get("default", (native_ranges[p["name"]][0] + native_ranges[p["name"]][1]) / 2)) for p in specs}
            center = {n: min(max(center[n], native_ranges[n][0]), native_ranges[n][1]) for n in names}
            round_id = 1
            x, loss, m = _score(center, "baseline")
            diag = diagnose_round(m, None, {"model_combination": COMBO})
            worker._last_diagnosis = diag
            worker.memory.add_round(round_id, dict(center), m, -loss, None, "baseline (defaults)")
            worker.ledger.append(round_id, "JOINT", dict(center), m, diagnosis_summary=diag.reasoning, reasoning="baseline")
            best.update(loss=loss, x=x, native=dict(center), metrics=m, nse=m.get("NSE", -float("inf")))
            prev_m = m
            while len(history) < budget:
                round_id += 1; rounds_in_phase += 1
                multi = self.mc_candidates > 1 and (rounds_in_phase % self.mc_every == 0)
                tags = list(STRATEGIES)[: self.mc_candidates] if multi else [""]
                round_best = None
                for tag in tags:
                    if len(history) >= budget:
                        break
                    cands, reasoning, tok = worker.propose_params(round_id, best["native"], best["metrics"], "JOINT",
                                                                  leader_context=STRATEGIES.get(tag, ""))
                    llm_tokens += int((tok or {}).get("total", 0) or 0)
                    cand = {n: min(max(float(cands[0].get(n, best["native"][n])), native_ranges[n][0]), native_ranges[n][1]) for n in names}
                    if any(h["native"] == cand for h in history):        # never re-run an identical vector
                        cand = {n: (v + 0.02 * (native_ranges[n][1] - native_ranges[n][0]) * (1 if v < native_ranges[n][1] else -1)) if i == 0 else v
                                for i, (n, v) in enumerate(cand.items())}
                        cand = {n: min(max(v, native_ranges[n][0]), native_ranges[n][1]) for n, v in cand.items()}   # codex M1: re-clamp after the nudge
                    x, loss, m = _score(cand, tag or "single")
                    if round_best is None or loss < round_best[1]:
                        round_best = (cand, loss, m, x, reasoning)
                cand, loss, m, x, reasoning = round_best
                diag = diagnose_round(m, prev_m, {"model_combination": COMBO}); worker._last_diagnosis = diag
                worker.memory.add_round(round_id, dict(cand), m, -loss, None, reasoning[:200])
                worker.ledger.append(round_id, "JOINT", dict(cand), m, diagnosis_summary=diag.reasoning, reasoning=reasoning[:300])
                improved = loss < best["loss"] - 1e-9
                if improved:
                    best.update(loss=loss, x=x, native=dict(cand), metrics=m, nse=m.get("NSE", -float("inf")))
                    rounds_since_improve = 0
                elif not multi:                                     # MADR: multi-candidate rounds do not count
                    rounds_since_improve += 1
                prev_m = m
                done, _, reason = check_phase_transition(pipeline, 0, rounds_in_phase, rounds_since_improve)
                if done:
                    # codex M3: `stopped_early` means THE KIT'S stop rule fired (base.py); MADR's own patience
                    # is recorded separately. MADR's text says "best-NSE"; improvement here is judged on the
                    # composite loss the kit minimises.
                    stop_note = f"MADR patience/cap: {reason} (improvement judged on the composite loss)"
                    break
        except EarlyStop:
            stopped_early = True; stop_note = "kit stop rule (on_eval) ended the search"

        finite = [h for h in history if math.isfinite(h["loss"]) and h["loss"] < 1e29]
        if not finite:
            return CalibResult(best_x=[], best_loss=[float("inf")], backend=f"madr:{self.provider}:{self.model}",
                               n_evaluations=len(history), history=history, notes="no finite evaluation")
        b = min(finite, key=lambda h: h["loss"])
        (wd / "madr_summary.json").write_text(json.dumps({
            "rounds": round_id, "evaluations": len(history), "best_round": b["round"], "best_native": b["native"],
            "best_metrics": b["metrics"], "llm_tokens": llm_tokens, "stop": stop_note, "provider": self.provider, "model": self.model,
            "patience_rounds": self.patience_rounds, "mc_candidates": self.mc_candidates}, indent=1, default=str))
        return CalibResult(best_x=b["x"], best_loss=[b["loss"]], n_evaluations=len(history), history=history,
                           backend=f"madr:{self.provider}:{self.model}", stopped_early=stopped_early,
                           notes=f"{len(finite)}/{len(history)} finite; {round_id} rounds; best loss {b['loss']:.5g}; "
                                 f"llm tokens {llm_tokens}; {stop_note}")
