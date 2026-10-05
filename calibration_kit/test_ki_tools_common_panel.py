"""The shared scorer's new numbers (ki_tools_common.metrics: alpha, beta, lnnse, nrmse and
``all_metrics(..., extended=True)``) must equal the kit's own panel formulas (calibration_kit/panel.py).

Workflows report these so the kit's convergence check can watch every side of the fit (2026-09-30). The scorer
and the kit each hold the formulas; this test is what keeps them equal. It reads the installed ki_tools_common
when it already has the new functions, else the staged copy in this worktree (ki_runner_panel_staging/)."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_kit import panel as P                                                    # noqa: E402

STAGED = ROOT / "ki_runner_panel_staging" / "ki_tools_common"


def _load(path, name):
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scorer():
    installed = importlib.util.find_spec("ki_tools_common") is not None
    if installed:
        from ki_tools_common import metrics as M
        if hasattr(M, "lnnse"):
            return M
    if not (STAGED / "metrics.py").is_file():
        if installed:
            raise ImportError("Installed ki_tools_common lacks the required panel metrics and no staged scorer is available")
        pytest.skip("Requires the external ki_tools_common scorer", allow_module_level=True)
    return _load(STAGED / "metrics.py", "metrics_staged")


M = _scorer()
RNG = np.random.default_rng(11)


def _cases():
    n = 400
    base = 20 + 10 * np.sin(np.linspace(0, 20, n)) + RNG.normal(0, 1, n)
    sim = 0.9 * base + 1.5 + RNG.normal(0, 2, n)
    yield "ordinary", sim, base
    s2 = sim.copy(); s2[::17] = np.nan
    o2 = base.copy(); o2[::23] = np.nan
    yield "missing_values", s2, o2
    yield "negative_sim_no_log", sim - 30, base
    yield "zero_flows_allowed", np.clip(sim - 15, 0, None), np.clip(base - 15, 0, None)
    yield "constant_obs", sim, np.full(n, 5.0)
    yield "zero_mean_obs", sim, base - base.mean()
    yield "three_pairs", np.array([1.0, 2.0, 2.5]), np.array([1.5, 2.5, 2.0])
    yield "one_pair_snapshot", np.array([7001.14]), np.array([6461.90])


def _same(a, b):
    a = None if a is None or (isinstance(a, float) and not math.isfinite(a)) else a
    b = None if b is None or (isinstance(b, float) and not math.isfinite(b)) else b
    if a is None or b is None:
        return a is None and b is None
    return a == pytest.approx(b, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name, sim, obs", list(_cases()), ids=[c[0] for c in _cases()])
def test_the_scorers_numbers_equal_the_kits(name, sim, obs):
    s, o = P._finite_pairs(sim, obs)
    kit = P._metrics_from_pairs(s, o, "flow") if len(s) else {}
    got = M.all_metrics(obs, sim, extended=True)
    for kit_key, sc_key in (("alpha", "alpha"), ("beta", "beta"), ("lnnse", "lnNSE"), ("nrmse", "NRMSE")):
        assert _same(got[sc_key], kit.get(kit_key)), (name, kit_key, got[sc_key], kit.get(kit_key))
    # the older keys agree too wherever both define them (the kit needs n >= 3 for r)
    for kit_key, sc_key in (("nse", "NSE"), ("pbias", "PBIAS")):
        if kit.get(kit_key) is not None:
            assert _same(got[sc_key], kit[kit_key]), (name, kit_key)
    if len(s) >= 3 and kit.get("r") is not None:
        assert _same(got["r"], kit["r"]) and _same(got["KGE"], kit["kge"])


def test_the_default_output_is_unchanged():
    """Without extended=True, all_metrics returns exactly the five keys and values it always did (the golden
    contract the self-improve loop and the evidence records rely on)."""
    orig = STAGED / "metrics.py.orig"
    if not orig.exists():
        pytest.skip("no pre-change copy to compare with")
    old = _load(orig, "metrics_orig")
    for name, sim, obs in _cases():
        a, b = old.all_metrics(obs, sim), M.all_metrics(obs, sim)
        assert list(a) == list(b) == ["NSE", "KGE", "PBIAS", "RMSE", "r"], name
        assert all(_same(a[k], b[k]) for k in a), name


def test_a_workflow_that_lowercases_the_scorers_keys_gives_the_kit_what_it_reads():
    """The usual workflow pattern `{k.lower(): v ...}` yields alpha, beta, lnnse: the plain keys the kit's panel
    reads for a variable (calibration_kit.panel.PLAIN_KEYS)."""
    obs = 20 + RNG.normal(0, 3, 200)
    sim = obs * 1.05 + RNG.normal(0, 1, 200)
    out = {k.lower(): v for k, v in M.all_metrics(obs, sim, extended=True).items()}
    pan = P.Panel(["Q"], kinds={"Q": "flow"})
    got = pan.extract({"Q": out})["Q"]
    assert {"r", "alpha", "beta", "lnnse"} <= set(got)                  # every metric flow requires
    assert pan.missing_required("Q", got) == []


def test_two_pairs_the_scorer_gives_r_and_kge_the_kit_does_not():
    """codex 2026-10-01: with exactly 2 pairs the shared scorer's (unchanged, golden-locked) r and KGE are defined,
    while the kit needs 3 pairs for r. The new extended scores still equal the kit's; a workflow must not report r
    or KGE below 3 pairs (the contract prompt says so; HBV refuses to score below 3 pairs)."""
    obs, sim = np.array([1.0, 2.0]), np.array([1.1, 1.9])
    got = M.all_metrics(obs, sim, extended=True)
    s, o = P._finite_pairs(sim, obs)
    kit = P._metrics_from_pairs(s, o, "flow")
    assert kit["r"] is None and kit["kge"] is None
    assert math.isfinite(got["r"]) and math.isfinite(got["KGE"])           # the scorer's documented difference
    for kit_key, sc_key in (("alpha", "alpha"), ("beta", "beta"), ("lnnse", "lnNSE"), ("nrmse", "NRMSE")):
        assert _same(got[sc_key], kit.get(kit_key)), kit_key


def test_hbv_refuses_to_score_fewer_than_three_pairs():
    hbv = Path("/mnt/disk1/Hydrocraft_server/models/HBV/knowledge_infrastructure/tools/calib_run.py")
    if not hbv.exists():
        pytest.skip("HBV package not on this machine")
    src = hbv.read_text()
    assert "if len(paired) < 3:" in src and '"panel": {"Q":' in src
