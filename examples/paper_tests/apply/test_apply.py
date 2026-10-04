"""Tests of the part-D apply tool (apply_rule.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rule import apply

TOL = {"Q": {"r": 0.01, "alpha": 0.01, "beta": 0.01}}
PAN = {"Q": {"r": 0.9, "alpha": 1.0, "beta": 1.0}}


def test_weighted_single_objective_settle_without_scalar():
    calls = [{"losses": [1.0, 1.0], "panel": PAN}, {"losses": [0.9, 0.9], "panel": PAN}]
    out = apply(calls, "dds", [("Q:a", "Q"), ("Q:b", "Q")], {"Q": "series"}, TOL, weights=[1, 1])
    assert out["settle"]["settled_by_run"] == 2 and out["wording"] == "settled by run 2 of 2"


def test_weights_used():
    calls = [{"losses": [1.0, 0.0], "panel": PAN}, {"losses": [0.0, 1.2], "panel": PAN}]
    # weights 3:1 -> 0.75 then 0.3 (a change); equal weights -> 0.5 then 0.6 (not a new best)
    assert apply(calls, "dds", [("Q:a", "Q"), ("Q:b", "Q")], {"Q": "series"}, TOL, weights=[3, 1])["settle"]["settled_by_run"] == 2
    assert apply(calls, "dds", [("Q:a", "Q"), ("Q:b", "Q")], {"Q": "series"}, TOL)["settle"]["settled_by_run"] == 1


def test_partial_losses_not_a_best():
    calls = [{"losses": [1.0, 1.0], "panel": PAN}, {"losses": [0.1, None], "panel": PAN}]
    assert apply(calls, "dds", [("Q:a", "Q"), ("Q:b", "Q")], {"Q": "series"}, TOL)["settle"]["settled_by_run"] == 1


def test_partial_losses_with_stored_scalar_not_a_best():
    calls = [{"losses": [1.0, 1.0], "scalar": 1.0, "panel": PAN}, {"losses": [0.1, None], "scalar": 0.1, "panel": PAN}]
    assert apply(calls, "dds", [("Q:a", "Q"), ("Q:b", "Q")], {"Q": "series"}, TOL)["settle"]["settled_by_run"] == 1


def test_wrong_length_losses_refused():
    import pytest
    calls = [{"losses": [1.0, 1.0], "scalar": 1.0, "panel": PAN}, {"losses": [0.1], "scalar": 0.1, "panel": PAN}]
    with pytest.raises(ValueError):
        apply(calls, "dds", [("Q:a", "Q"), ("Q:b", "Q")], {"Q": "series"}, TOL)


def test_bad_step_tag_refused_not_rounded():
    import pytest
    calls = [{"losses": [1.0], "panel": PAN, "step": 0}, {"losses": [0.9], "panel": PAN, "step": 1.2}]
    with pytest.raises(ValueError):
        apply(calls, "sceua", [("Q:a", "Q")], {"Q": "series"}, TOL)


def test_bool_losses_are_not_finite():
    calls = [{"losses": [True], "panel": PAN}, {"losses": [False], "panel": PAN}]
    assert apply(calls, "dds", [("Q:a", "Q")], {"Q": "series"}, TOL)["settle"]["settled_by_run"] is None


def test_uses_the_validation_snapshot():
    import apply_rule
    assert "frozen_rule" in apply_rule.StepRule.__module__ or "kdt_rule_frozen" in apply_rule.StepRule.__module__
