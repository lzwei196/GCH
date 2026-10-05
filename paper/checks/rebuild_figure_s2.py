#!/usr/bin/env python3
"""Rebuild current Figure S2 with the unchanged recovered archived figure3().

Use through paper/reproduce.py with --paper-root. Requires NumPy and Matplotlib
for rendering. No model is run; the six saved reports are read and checked.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from common import paper_root, inside, external_output, RECORD


CASES = (
    "01_WOFOST", "02_HBV", "03_SUMMA", "04_VIC", "05_MODFLOW6", "06_CRHM"
)
ARCHIVED_STEM = "figure3_six_family_assessment"
OUTPUT_NAMES = [f"Figure_S2.{ext}" for ext in ("pdf", "svg", "png")]
MANIFEST_NAME = "rebuild_manifest.json"


def file_record(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root", "--package-root", dest="package_root", type=Path, required=True,
        help="External companion package root.",
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="Verify all inputs without importing the archived builder or rendering.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Writable output directory outside the companion package; required for rendering.",
    )
    return parser.parse_args()


def rebuild(args: argparse.Namespace) -> dict:
    package = paper_root(args.package_root)
    record = inside(package, RECORD)
    builder = inside(package, "03_code/Supporting_Figure_Builds/figures/S2/recovered_original")
    output = external_output(package, args.output_dir) if args.output_dir else None
    if not args.preflight and output is None:
        raise ValueError("Rendering requires --output-dir outside the companion package")
    # Keep all archived code, data, and canonical submission artwork immutable.
    protected = [record, builder, package / "01_journal_submission", package / "02_source_data"]
    if output and any(output == p or p in output.parents for p in protected):
        raise ValueError("Choose an output directory outside the record, builder, and submission/source-data folders.")
    for name in OUTPUT_NAMES + [MANIFEST_NAME]:
        if output and (output / name).exists():
            raise FileExistsError(f"Refusing to replace existing output: {output / name}")

    archived_script = inside(builder, "scripts/build_quantitative_figures.py")
    source_json = inside(builder, "source_data/figure_source_data.json")
    script_before = file_record(archived_script)
    source_before = file_record(source_json)
    data = json.loads(source_json.read_text(encoding="utf-8"))
    if set(data.get("six_family", {})) != set(CASES):
        raise ValueError("Source JSON must contain exactly the six expected model cases.")

    report_records = []
    mapping = []
    count = 0
    for case in CASES:
        report_path = inside(record, f"experiments/1_six_model/six_model_families/{case}/rerun_conv_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        family = data["six_family"][case]
        components = family["components"]
        if [c["objective"] for c in components] != report["objectives"]:
            raise ValueError(f"Objective ordering differs from the retained report: {case}")
        if components != report["holdout"]["per_objective"]:
            raise ValueError(f"Plotted component records differ from the retained report: {case}")
        if family["accepted_components"] != sum(bool(c["ok"]) for c in components):
            raise ValueError(f"Accepted-component count differs from the source components: {case}")
        if family["total_components"] != len(components):
            raise ValueError(f"Total-component count differs from the source components: {case}")
        mapping.append({"case": case, "original_source": family["source"], "resolved_source": str(report_path)})
        family["source"] = str(report_path)
        report_records.append(file_record(report_path))
        count += len(components)
    if count != 23:
        raise ValueError(f"Expected 23 components; found {count}.")

    if args.preflight:
        return {"passed": True, "preflight_only": True, "model_cases": 6, "components": count,
                "builder_imported": False, "model_execution_performed": False, "outputs": [],
                "original_builder": script_before, "original_source_data": source_before,
                "report_inputs": report_records,
                "scope": "Plotted component records equal the archived reports; no independent scientific acceptance recalculation or figure rendering."}

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".figure-s2-build-", dir=output) as temp_name:
        stage = Path(temp_name)
        (stage / "scripts").mkdir()
        (stage / "source_data").mkdir()
        staged_script = stage / "scripts" / archived_script.name
        shutil.copyfile(archived_script, staged_script)
        staged_json = stage / "source_data" / source_json.name
        staged_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        staged_json_hash = file_record(staged_json)["sha256"]
        cache_env = {
            "MPLCONFIGDIR": str(stage / "matplotlib_config"),
            "XDG_CACHE_HOME": str(stage / "cache"),
        }
        old_cache_env = {key: os.environ.get(key) for key in cache_env}
        os.environ.update(cache_env)
        try:
            spec = importlib.util.spec_from_file_location("archived_figure_s2_builder", staged_script)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot import the staged archived figure builder.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # The archived __main__ also builds superseded Figures 2 and 4.
            # Calling only this function preserves the current S2 build boundary.
            module.figure3()
        finally:
            for key, value in old_cache_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        rendered = [stage / "figures" / f"{ARCHIVED_STEM}.{ext}" for ext in ("pdf", "svg", "png")]
        if any(not p.is_file() or p.stat().st_size == 0 for p in rendered):
            raise RuntimeError("Archived figure3() did not produce all three nonempty exports.")
        # Verify preservation before publishing any exports.
        if file_record(archived_script) != script_before or file_record(source_json) != source_before:
            raise RuntimeError("An original builder input changed during rendering.")
        if any(file_record(Path(before["path"])) != before for before in report_records):
            raise RuntimeError("A retained model report changed during rendering.")
        for source, name in zip(rendered, OUTPUT_NAMES):
            shutil.copyfile(source, output / name)

        figure_description = dict(module.MANIFEST["figures"][ARCHIVED_STEM])
        figure_description["exports"] = list(OUTPUT_NAMES)
        manifest = {
            "purpose": "Current Figure S2 rebuilt from unchanged archived figure3() and saved reports",
            "no_model_runs": True,
            "archived_entrypoint": "figure3",
            "archived_figure_stem": ARCHIVED_STEM,
            "wrapper": file_record(Path(__file__).resolve()),
            "original_builder": script_before,
            "original_source_data": source_before,
            "staged_source_sha256": staged_json_hash,
            "source_path_mapping": mapping,
            "report_inputs": report_records,
            "validation": {
                "model_cases": 6,
                "components": count,
                "component_values_and_objective_order_match_reports": True,
                "original_builder_data_and_reports_unchanged": True,
            },
            "software": {
                "python_executable": sys.executable,
                "python": sys.version,
                "numpy": module.np.__version__,
                "matplotlib": module.matplotlib.__version__,
            },
            "figure": figure_description,
            "outputs": [file_record(output / name) for name in OUTPUT_NAMES],
        }
        reference = package / "01_journal_submission/03_Figures/Supporting_Figures/Figure_S2.png"
        if reference.is_file():
            manifest["historical_png"] = file_record(reference)
            manifest["historical_png"]["new_render_byte_identical"] = (
                file_record(reference)["sha256"] == file_record(output / "Figure_S2.png")["sha256"]
            )
        (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    try:
        manifest = rebuild(arguments())
    except (OSError, ValueError, KeyError, AssertionError, ImportError, RuntimeError) as exc:
        print(f"Figure S2 rebuild failed: {exc}", file=sys.stderr)
        return 1
    if manifest.get("preflight_only"):
        print(json.dumps(manifest, indent=2))
        return 0
    print("Rendered Figure S2 only; 6 model reports and 23 components verified.")
    for item in manifest["outputs"]:
        print(item["path"])
    print(str(Path(manifest["outputs"][0]["path"]).parent / MANIFEST_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
