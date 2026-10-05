"""Failure-boundary tests for the paper interface; no study data or models used."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper/checks"))
from common import inside, external_output
from validate_indexes import validate
from check_saved_results import complete_six_model_reports
from rebuild_figure_s2 import rebuild, CASES


class PaperChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "package"
        self.metadata = Path(self.tmp.name) / "metadata"
        self.root.mkdir()
        self.metadata.mkdir()
        (self.root / "04_research_archive").mkdir()

    def put(self, relative, content="fixture"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def fixture(self, mutate=None):
        test = {"test_id": "T", "paper_items": ["Figure 1"], **{key: [] for key in
                ("protocol", "results", "script", "contract", "ki", "environment")}}
        test["results"] = [{"package": "payload/result.json"}]
        docs = {
            "TEST_INDEX.json": {"tests": [test]},
            "PAPER_EVIDENCE_MAP.json": {"paper_items": [{"paper_item": "Figure 1", "test_ids": ["T"]}],
                "test_records": [{"test_id": "T", "files": [{"package": "payload/result.json"}]}]},
            "KI_INDEX.json": {"rows": [{"case_id": "K", "input_package_path": "payload/result.json"}]},
            "ENVIRONMENT_INDEX.json": {"cases": [{"case": "E", "package_directory": "env",
                "instructions": "env/RERUN.md", "manifest": "env/MANIFEST.sha256"}]},
        }
        self.put("payload/result.json", "{}")
        self.put("env/RERUN.md")
        self.put("env/MANIFEST.sha256")
        if mutate:
            mutate(docs)
        entries = []
        for name, doc in docs.items():
            text = json.dumps(doc)
            digest = hashlib.sha256(text.encode()).hexdigest()
            (self.metadata / name).write_text(text)
            path = "04_research_archive/" + name
            self.put(path, text)
            entries.append({"file": name, "package_path": path, "source_sha256": digest, "metadata_sha256": digest})
        (self.metadata / "SOURCE_MANIFEST.json").write_text(json.dumps({"indexes": entries}))

    def six_reports(self):
        root = self.root / "04_research_archive/record"
        data = {"six_family": {}}
        for index, case in enumerate(CASES):
            count = 3 if index == 5 else 4
            components = [{"objective": f"o{i}", "holdout_loss": 0.1, "baseline_holdout_loss": 0.2,
                           "ok": True, "beats_baseline": True} for i in range(count)]
            report = {"objectives": [x["objective"] for x in components], "holdout": {"per_objective": components}}
            self.put(f"04_research_archive/record/experiments/1_six_model/six_model_families/{case}/rerun_conv_report.json", json.dumps(report))
            data["six_family"][case] = {"components": copy.deepcopy(components), "accepted_components": count,
                                       "total_components": count, "source": "historical source"}
        return root, data

    def test_complete_index_resolves(self):
        self.fixture()
        self.assertTrue(validate(self.root, self.metadata)["passed"])

    def test_cli_missing_archive_fails_clearly(self):
        result = subprocess.run([sys.executable, "-B", str(REPO / "paper/reproduce.py"), "validate", "--paper-root",
                                 str(self.root / "absent")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not exist", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_path_traversal_and_absolute_paths_rejected(self):
        for locator in ("../outside", "/etc/passwd", "payload/../../outside"):
            with self.subTest(locator=locator), self.assertRaises(ValueError):
                inside(self.root, locator, must_exist=False)

    def test_symlink_escape_rejected(self):
        outside = Path(self.tmp.name) / "outside"
        outside.write_text("not an archive file")
        (self.root / "link").symlink_to(outside)
        with self.assertRaises(ValueError):
            inside(self.root, "link")

    def test_missing_file_only_in_evidence_map_fails(self):
        def modify(docs):
            docs["PAPER_EVIDENCE_MAP.json"]["test_records"][0]["files"].append({"package": "missing/evidence.json"})
        self.fixture(modify)
        result = validate(self.root, self.metadata)
        self.assertFalse(result["passed"])
        self.assertTrue(any(x.get("kind") == "evidence_file" for x in result["issues"]))

    def test_unknown_paper_test_link_fails(self):
        def modify(docs):
            docs["PAPER_EVIDENCE_MAP.json"]["paper_items"][0]["test_ids"].append("UNKNOWN")
        self.fixture(modify)
        self.assertFalse(validate(self.root, self.metadata)["passed"])

    def test_missing_test_record_fails(self):
        def modify(docs):
            docs["TEST_INDEX.json"]["tests"].append({**copy.deepcopy(docs["TEST_INDEX.json"]["tests"][0]), "test_id": "T2"})
        self.fixture(modify)
        result = validate(self.root, self.metadata)
        self.assertTrue(any(x.get("missing_records") == ["T2"] for x in result["issues"]))

    def test_archive_index_version_change_fails(self):
        self.fixture()
        self.put("04_research_archive/TEST_INDEX.json", "{}")
        self.assertFalse(validate(self.root, self.metadata)["passed"])

    def test_outputs_cannot_modify_archive(self):
        with self.assertRaises(ValueError):
            external_output(self.root, self.root / "new.json")

    def test_empty_report_is_not_a_success(self):
        record, _ = self.six_reports()
        self.assertEqual(complete_six_model_reports(record)["components"], 23)
        path = record / f"experiments/1_six_model/six_model_families/{CASES[0]}/rerun_conv_report.json"
        path.write_text('{"objectives": [], "holdout": {"per_objective": []}}')
        with self.assertRaises(ValueError):
            complete_six_model_reports(record)

    def test_figure_preflight_does_not_execute_builder_or_write_outputs(self):
        _, data = self.six_reports()
        builder = "03_code/Supporting_Figure_Builds/figures/S2/recovered_original/"
        self.put(builder + "scripts/build_quantitative_figures.py", "raise RuntimeError('must not execute')\n")
        self.put(builder + "source_data/figure_source_data.json", json.dumps(data))
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        result = rebuild(SimpleNamespace(package_root=self.root, output_dir=None, preflight=True))
        self.assertTrue(result["passed"])
        self.assertFalse(result["builder_imported"])
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
