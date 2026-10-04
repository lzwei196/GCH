"""Front-selection exhaustiveness + degenerate-objective tests (codex 2026-08-20).

Guards the fix that made _select_front_member EXHAUSTIVE: the committing member may be a
LOW-minimax-rank extreme of the Pareto front, and a cap must never hide it. Also checks that a
near-degenerate (zero-spread) objective is excluded from the minimax ranking rather than
dominating it via float-noise normalization.
"""
import types
import calibration_kit.holdout as _holdout
from calibration_kit.calib import _select_front_member


def _fake_result(px, pf):
    return types.SimpleNamespace(pareto_x=px, pareto_f=pf, best_x=None, best_loss=None, notes="")


def test_exhaustive_reaches_low_ranked_extreme():
    # 20-point convex front over 2 objectives. The most-balanced (minimax-min) sits near the
    # middle; index 0 is an EXTREME (obj1=0, obj2=1) -> worst normalized = 1.0 -> LAST in minimax
    # order (rank ~19). Only that extreme passes holdout, so a cap<19 would report none-passed.
    n = 20
    px = [[float(i)] for i in range(n)]
    pf = [[i / (n - 1), 1 - i / (n - 1)] for i in range(n)]
    winner = px[0]

    def fake_validate(ev, objs, best_x, spec, probe_x=None, band_ceilings=None, baseline_x=None):
        return {"passed": best_x == winner, "per_objective": []}

    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = fake_validate
    try:
        r = _fake_result(px, pf)
        idx, hd = _select_front_member(None, None, r, {}, {}, None, None)  # cap=None -> exhaustive
    finally:
        _holdout.validate_holdout = _orig
    assert idx == 0, f"expected the extreme member 0 to be committed, got {idx}"
    assert r.best_x == winner, r.best_x
    assert hd["passed"] is True
    assert "PASSED holdout" in r.notes and "exhaustive=True" in r.notes
    # and a CAP that stops before rank 19 must NOT silently claim success
    _holdout.validate_holdout = fake_validate
    try:
        r2 = _fake_result(px, pf)
        idx2, _ = _select_front_member(None, None, r2, {}, {}, None, None, cap=5)
    finally:
        _holdout.validate_holdout = _orig
    assert idx2 != 0, "capped run should miss the rank-19 extreme"
    assert "TRUNCATED" in r2.notes
    print("OK exhaustive_reaches_low_ranked_extreme")


def test_degenerate_objective_excluded():
    # obj2 is CONSTANT (zero spread) -> must be excluded from minimax; ranking driven by obj1 only.
    # Member with the smallest obj1 is the minimax-min; make ONLY it pass and confirm rank 0.
    n = 8
    px = [[float(i)] for i in range(n)]
    pf = [[i / (n - 1), 5.0] for i in range(n)]  # obj2 constant
    winner = px[0]  # smallest obj1

    def fake_validate(ev, objs, best_x, spec, probe_x=None, band_ceilings=None, baseline_x=None):
        return {"passed": best_x == winner, "per_objective": []}

    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = fake_validate
    try:
        r = _fake_result(px, pf)
        idx, hd = _select_front_member(None, None, r, {}, {}, None, None)
    finally:
        _holdout.validate_holdout = _orig
    assert idx == 0, f"degenerate obj2 must not perturb obj1-driven ranking; got rank for idx {idx}"
    assert hd["passed"] is True
    print("OK degenerate_objective_excluded")


def _objs(*names):
    return [types.SimpleNamespace(name=n) for n in names]


def _patch_validate(per_obj_by_idx, px):
    """fake validate_holdout: no member passes the full gate; per-objective holdout dicts are
    looked up by which member's params were passed as best_x (px[idx] == [float(idx)])."""
    def fake_validate(ev, objs, best_x, spec, probe_x=None, band_ceilings=None, baseline_x=None):
        idx = int(best_x[0])
        return {"passed": False, "per_objective": per_obj_by_idx[idx]}
    return fake_validate


def test_protect_tier_C_commits_bounded_degradation_over_knee():
    # 2 objectives: P (protected, baseline unbeatable) + S (free). NO member beats baseline on P,
    # so tiers A/B are infeasible; the raw minimax knee would sacrifice P badly to win S. Tier C must
    # instead commit the member that keeps P within tolerance of baseline AND is best on the free S.
    px = [[float(i)] for i in range(3)]
    pf = [[0.6, 0.3], [0.6, 0.9], [1.5, 0.05]]   # member 2 = P-wrecking, S-perfect knee
    base_P, base_S = 0.5, 1.0                      # baseline losses (uncalibrated best on P)
    per = {
        0: [{"objective": "P", "holdout_loss": 0.60, "baseline_holdout_loss": base_P,
             "ok": False, "beats_baseline": False},
            {"objective": "S", "holdout_loss": 0.30, "baseline_holdout_loss": base_S,
             "ok": True, "beats_baseline": True}],
        1: [{"objective": "P", "holdout_loss": 0.60, "baseline_holdout_loss": base_P,
             "ok": False, "beats_baseline": False},
            {"objective": "S", "holdout_loss": 0.90, "baseline_holdout_loss": base_S,
             "ok": True, "beats_baseline": True}],
        2: [{"objective": "P", "holdout_loss": 1.50, "baseline_holdout_loss": base_P,   # >25% over base
             "ok": False, "beats_baseline": False},
            {"objective": "S", "holdout_loss": 0.05, "baseline_holdout_loss": base_S,
             "ok": True, "beats_baseline": True}],
    }
    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = _patch_validate(per, px)
    try:
        r = _fake_result(px, pf)
        idx, hd = _select_front_member(None, _objs("P", "S"), r, {}, {}, None, None,
                                       protect={"P"})
    finally:
        _holdout.validate_holdout = _orig
    assert idx == 0, f"tier C should commit member 0 (P within tol, best free S), got {idx}"
    assert "tier C" in r.notes and "protecting ['P']" in r.notes, r.notes
    print("OK protect_tier_C_commits_bounded_degradation_over_knee")


def test_protect_tier_A_precedence_over_C():
    # Member 1 keeps P admissible with ok=True (tier A); member 0 is only within-tol (tier C) and has
    # a better free S. Tier A must win regardless of the free objective.
    px = [[float(i)] for i in range(2)]
    pf = [[0.6, 0.3], [0.55, 0.9]]
    per = {
        0: [{"objective": "P", "holdout_loss": 0.60, "baseline_holdout_loss": 0.5,
             "ok": False, "beats_baseline": False},
            {"objective": "S", "holdout_loss": 0.30, "baseline_holdout_loss": 1.0,
             "ok": True, "beats_baseline": True}],
        1: [{"objective": "P", "holdout_loss": 0.40, "baseline_holdout_loss": 0.5,
             "ok": True, "beats_baseline": True},
            {"objective": "S", "holdout_loss": 0.90, "baseline_holdout_loss": 1.0,
             "ok": True, "beats_baseline": True}],
    }
    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = _patch_validate(per, px)
    try:
        r = _fake_result(px, pf)
        idx, _ = _select_front_member(None, _objs("P", "S"), r, {}, {}, None, None, protect={"P"})
    finally:
        _holdout.validate_holdout = _orig
    assert idx == 1, f"tier A member must be preferred over a tier-C member, got {idx}"
    assert "tier A" in r.notes, r.notes
    print("OK protect_tier_A_precedence_over_C")


def test_protect_infeasible_keeps_knee():
    # No member keeps P even within tolerance of baseline -> INFEASIBLE; keep the minimax knee
    # (order[0]) and say so, never a silent protected commit.
    px = [[float(i)] for i in range(3)]
    pf = [[0.2, 0.9], [0.5, 0.5], [0.9, 0.2]]     # minimax knee = member 1 (balanced)
    per = {i: [{"objective": "P", "holdout_loss": 2.0, "baseline_holdout_loss": 0.3,
                "ok": False, "beats_baseline": False},
               {"objective": "S", "holdout_loss": 0.1, "baseline_holdout_loss": 1.0,
                "ok": True, "beats_baseline": True}] for i in range(3)}
    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = _patch_validate(per, px)
    try:
        r = _fake_result(px, pf)
        idx, _ = _select_front_member(None, _objs("P", "S"), r, {}, {}, None, None, protect={"P"})
    finally:
        _holdout.validate_holdout = _orig
    assert "INFEASIBLE" in r.notes, r.notes
    assert idx == 1, f"infeasible protection must keep the minimax knee (member 1), got {idx}"
    print("OK protect_infeasible_keeps_knee")


def test_protect_zero_free_loss_not_treated_as_missing():
    # codex 2026-08-25 #1: a PERFECT free loss (0.0) is falsy — must NOT be read as missing/inf.
    # Member 0 has free S loss exactly 0.0 (best possible); it must win over member 1 (S=0.5).
    px = [[float(i)] for i in range(2)]
    pf = [[0.6, 0.0], [0.6, 0.5]]
    per = {
        0: [{"objective": "P", "holdout_loss": 0.60, "baseline_holdout_loss": 0.5,
             "ok": False, "beats_baseline": False},
            {"objective": "S", "holdout_loss": 0.0, "baseline_holdout_loss": 1.0,
             "ok": True, "beats_baseline": True}],
        1: [{"objective": "P", "holdout_loss": 0.60, "baseline_holdout_loss": 0.5,
             "ok": False, "beats_baseline": False},
            {"objective": "S", "holdout_loss": 0.50, "baseline_holdout_loss": 1.0,
             "ok": True, "beats_baseline": True}],
    }
    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = _patch_validate(per, px)
    try:
        r = _fake_result(px, pf)
        idx, _ = _select_front_member(None, _objs("P", "S"), r, {}, {}, None, None, protect={"P"})
    finally:
        _holdout.validate_holdout = _orig
    assert idx == 0, f"member with 0.0 free loss must win, got {idx}"
    print("OK protect_zero_free_loss_not_treated_as_missing")


def test_protect_missing_free_loss_never_wins():
    # codex 2026-08-25 #2: a member with a MISSING free holdout_loss must not win via inf->nan.
    # Member 0 lacks the free S entry; member 1 has a finite (worse-than-perfect) S. The rankable
    # member 1 must be committed, never the un-rankable member 0.
    px = [[float(i)] for i in range(2)]
    pf = [[0.6, 0.9], [0.6, 0.3]]
    per = {
        0: [{"objective": "P", "holdout_loss": 0.60, "baseline_holdout_loss": 0.5,
             "ok": False, "beats_baseline": False}],   # no "S" entry -> free loss missing
        1: [{"objective": "P", "holdout_loss": 0.60, "baseline_holdout_loss": 0.5,
             "ok": False, "beats_baseline": False},
            {"objective": "S", "holdout_loss": 0.30, "baseline_holdout_loss": 1.0,
             "ok": True, "beats_baseline": True}],
    }
    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = _patch_validate(per, px)
    try:
        r = _fake_result(px, pf)
        idx, _ = _select_front_member(None, _objs("P", "S"), r, {}, {}, None, None, protect={"P"})
    finally:
        _holdout.validate_holdout = _orig
    assert idx == 1, f"a member with a rankable free loss must win over one with a missing loss, got {idx}"
    print("OK protect_missing_free_loss_never_wins")


def test_protect_unknown_names_ignored_no_reranking():
    # codex 2026-08-25 #3: protect names that match NO objective must NOT re-rank the front. The
    # committed member must equal the plain minimax knee (no protection applied), with a note saying
    # the names were ignored.
    px = [[float(i)] for i in range(3)]
    pf = [[0.2, 0.9], [0.5, 0.5], [0.9, 0.2]]      # minimax knee = member 1
    per = {i: [{"objective": "P", "holdout_loss": 0.5, "baseline_holdout_loss": 0.5,
                "ok": False, "beats_baseline": False},
               {"objective": "S", "holdout_loss": 0.5, "baseline_holdout_loss": 0.5,
                "ok": False, "beats_baseline": False}] for i in range(3)}
    _orig = _holdout.validate_holdout
    _holdout.validate_holdout = _patch_validate(per, px)
    try:
        r = _fake_result(px, pf)
        idx, _ = _select_front_member(None, _objs("P", "S"), r, {}, {}, None, None,
                                      protect={"typo_not_an_objective"})
    finally:
        _holdout.validate_holdout = _orig
    assert idx == 1, f"unknown protect names must fall through to the minimax knee (member 1), got {idx}"
    assert "ignored unknown protected" in r.notes, r.notes
    print("OK protect_unknown_names_ignored_no_reranking")


if __name__ == "__main__":
    test_exhaustive_reaches_low_ranked_extreme()
    test_degenerate_objective_excluded()
    test_protect_tier_C_commits_bounded_degradation_over_knee()
    test_protect_tier_A_precedence_over_C()
    test_protect_infeasible_keeps_knee()
    test_protect_zero_free_loss_not_treated_as_missing()
    test_protect_missing_free_loss_never_wins()
    test_protect_unknown_names_ignored_no_reranking()
    print("ALL PASS")
