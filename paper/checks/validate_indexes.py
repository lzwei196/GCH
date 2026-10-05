#!/usr/bin/env python3
"""Validate paper evidence navigation and companion-file existence; no models run."""
import argparse
import sys
from pathlib import Path

from common import METADATA, emit, inside, paper_root, read_json, sha256


def validate(root, metadata=METADATA):
    root = paper_root(root)
    metadata = Path(metadata)
    issues, refs = [], []
    documents = {}
    manifest = read_json(metadata / "SOURCE_MANIFEST.json")
    expected_names = {"TEST_INDEX.json", "PAPER_EVIDENCE_MAP.json", "KI_INDEX.json", "ENVIRONMENT_INDEX.json"}
    entries = manifest["indexes"]
    if len(entries) != 4 or {x["file"] for x in entries} != expected_names:
        raise ValueError("Source manifest must identify all four evidence indexes exactly once")
    for entry in entries:
        path = inside(metadata, entry["file"])
        if sha256(path) != entry["metadata_sha256"]:
            raise ValueError(f"Bundled metadata identity mismatch: {entry['file']}")
        documents[entry["file"]] = read_json(path)
        try:
            original = inside(root, entry["package_path"])
            if sha256(original) != entry["source_sha256"]:
                issues.append({"index": entry["file"], "problem": "companion index differs from the recorded version"})
        except (OSError, ValueError) as error:
            issues.append({"index": entry["file"], "problem": str(error)})

    tests = documents["TEST_INDEX.json"]["tests"]
    evidence = documents["PAPER_EVIDENCE_MAP.json"]
    ki = documents["KI_INDEX.json"]["rows"]
    environments = documents["ENVIRONMENT_INDEX.json"]["cases"]
    if not tests or not evidence["test_records"] or not evidence["paper_items"] or not ki or not environments:
        raise ValueError("Evidence indexes must contain tests, file records, paper items, KIs and environments")

    def unique(rows, key, label):
        ids = [row[key] for row in rows]
        if len(ids) != len(set(ids)):
            issues.append({"problem": f"duplicate {label}"})
        return set(ids)

    test_ids = unique(tests, "test_id", "test IDs")
    record_ids = unique(evidence["test_records"], "test_id", "evidence-map test IDs")
    if test_ids != record_ids:
        issues.append({"problem": "test index and evidence map disagree", "missing_records": sorted(test_ids - record_ids),
                       "unknown_records": sorted(record_ids - test_ids)})
    unique(ki, "case_id", "KI IDs")
    unique(environments, "case", "environment IDs")
    unique(evidence["paper_items"], "paper_item", "paper items")
    item_links = {}
    for item in evidence["paper_items"]:
        ids = item["test_ids"]
        # Schematics, inventories and literature tables explicitly have no numerical test.
        if (not ids and not item.get("source_resolution")) or len(ids) != len(set(ids)) or not set(ids) <= test_ids:
            issues.append({"paper_item": item["paper_item"], "problem": "unexplained empty, duplicate or unknown test links"})
        item_links[item["paper_item"]] = set(ids)
    for row in tests:
        if not row.get("paper_items") and not (row.get("inclusion") == "supporting_dependency" and row.get("paper_anchors")):
            issues.append({"test_id": row["test_id"], "problem": "missing paper-item links"})
        for item in row.get("paper_items", []):
            if row["test_id"] not in item_links.get(item, set()):
                issues.append({"test_id": row["test_id"], "paper_item": item, "problem": "missing reverse paper-item link"})
        for item, ids in item_links.items():
            if row["test_id"] in ids and item not in row.get("paper_items", []):
                issues.append({"test_id": row["test_id"], "paper_item": item, "problem": "paper-item link absent from test index"})
        for kind in ("protocol", "results", "script", "contract", "ki", "environment"):
            for loc in row[kind]:
                refs.append((row["test_id"], kind, loc["package"]))
    for row in evidence["test_records"]:
        for loc in row["files"]:
            refs.append((row["test_id"], "evidence_file", loc["package"]))
    for row in ki:
        for key, value in row.items():
            if key.endswith(("_package_path", "_package_paths")):
                for loc in value if isinstance(value, list) else [value]:
                    if loc:
                        refs.append((row["case_id"], key, loc))
    for row in environments:
        for key in ("package_directory", "instructions", "manifest"):
            refs.append((row["case"], key, row[key]))
    checked = {}
    for case, kind, locator in refs:
        if locator not in checked:
            try:
                inside(root, locator)
                checked[locator] = None
            except (OSError, ValueError) as error:
                checked[locator] = str(error)
        if checked[locator]:
            issues.append({"case": case, "kind": kind, "path": locator, "problem": checked[locator]})
    return {"passed": not issues, "paper_root": str(root), "counts": {
                "test_navigation_entries": len(tests), "ki_condition_entries": len(ki), "environment_entries": len(environments)},
            "references_checked": len(refs), "unique_paths": len(checked), "issues": issues,
            "scope": "Pinned index identities, relationships and locator existence only; no result recomputation or runtime/scientific-validity certification.",
            "evidence_limitations": [
                "Cross-provider authoring retains console evidence and reconstructed scripts; the additional nine contracts are unavailable.",
                "The exact sealed GR4J Python runtime is unrecovered; its identity manifest does not supply that runtime."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", "--package-root", dest="paper_root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.paper_root)
        emit(report, args.output, args.paper_root)
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Index check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
