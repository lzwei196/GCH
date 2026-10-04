"""resolve_reference_run (2026-08-29): a SITE contract's own reference wins; a reference for another
case is refused explicitly; the KI-level fallback and 'no reference' (skip) behaviours are unchanged."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_kit.calib import resolve_reference_run  # noqa: E402


def _ki(tmp, case=None):
    ki = Path(tmp) / "ki"; (ki / "calib").mkdir(parents=True)
    if case:
        (ki / "calib" / "reference_run.json").write_text(json.dumps({"case_id": case, "validated_run_id": "ki-ref", "headline_value": 0.5}))
    return ki


def test_ki_reference_for_another_case_is_refused():
    with tempfile.TemporaryDirectory() as t:
        ki = _ki(t, case="SITE:tangnaihai")
        ref, refuse = resolve_reference_run({"runner": {}}, ki, expected_case_id="SITE:huaibin")
        assert ref is None and refuse["status"] == "runner_uncertified"
        assert "SITE:tangnaihai" in refuse["reason"] and "SITE:huaibin" in refuse["reason"]
        assert refuse["validated_run_id"] == "ki-ref"


def test_site_reference_wins_over_ki_reference():
    with tempfile.TemporaryDirectory() as t:
        ki = _ki(t, case="SITE:tangnaihai")
        site = Path(t) / "site_ref.json"; site.write_text(json.dumps({"case_id": "SITE:huaibin", "validated_run_id": "site-ref"}))
        ref, refuse = resolve_reference_run({"runner": {"reference_run": str(site)}}, ki, expected_case_id="SITE:huaibin")
        assert refuse is None and ref["validated_run_id"] == "site-ref"


def test_site_reference_for_wrong_case_is_refused_too():
    with tempfile.TemporaryDirectory() as t:
        ki = _ki(t)
        site = Path(t) / "site_ref.json"; site.write_text(json.dumps({"case_id": "SITE:elsewhere"}))
        ref, refuse = resolve_reference_run({"runner": {"reference_run": str(site)}}, ki, expected_case_id="SITE:huaibin")
        assert ref is None and "SITE:elsewhere" in refuse["reason"]


def test_unreadable_site_reference_is_refused():
    with tempfile.TemporaryDirectory() as t:
        ki = _ki(t)
        ref, refuse = resolve_reference_run({"runner": {"reference_run": str(Path(t) / "missing.json")}}, ki, expected_case_id="SITE:huaibin")
        assert ref is None and "unreadable" in refuse["reason"]


def test_reference_without_case_id_is_refused_when_case_expected():
    with tempfile.TemporaryDirectory() as t:
        ki = _ki(t)
        (ki / "calib" / "reference_run.json").write_text(json.dumps({"validated_run_id": "legacy", "headline_value": 0.5}))
        ref, refuse = resolve_reference_run({"runner": {}}, ki, expected_case_id="SITE:huaibin")
        assert ref is None and "names no case_id" in refuse["reason"]
        # without an expected case (legacy callers) the case-less reference still passes through unchanged
        ref, refuse = resolve_reference_run({"runner": {}}, ki, expected_case_id=None)
        assert refuse is None and ref["validated_run_id"] == "legacy"


def test_no_reference_means_skip_unchanged():
    with tempfile.TemporaryDirectory() as t:
        ki = _ki(t)
        ref, refuse = resolve_reference_run({"runner": {}}, ki, expected_case_id="SITE:huaibin")
        assert ref is None and refuse is None


def test_ki_reference_same_case_passes_through():
    with tempfile.TemporaryDirectory() as t:
        ki = _ki(t, case="SITE:huaibin")
        ref, refuse = resolve_reference_run({"runner": {}}, ki, expected_case_id="SITE:huaibin")
        assert refuse is None and ref["case_id"] == "SITE:huaibin"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL reference-case TESTS PASS")
