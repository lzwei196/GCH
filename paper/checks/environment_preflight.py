#!/usr/bin/env python3
"""Read-only checks of archived case files; never imports/runs a model or installs software.

The only write is an explicitly requested JSON report. Use through paper/reproduce.py
with --paper-root. Exit 0 means selected archive
files passed integrity checks, not that execution has been demonstrated on this host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from common import paper_root, inside, external_output


RECORD_PREFIX = PurePosixPath("04_research_archive/record")
DEFAULT_INDEX = "03_code/Recovered_Dependencies/Environment_Setup"
PYTHON_DEPENDENCIES = {
    "01_WOFOST": {"pcse": "6.0.12", "numpy": None, "pandas": None},
    "02_HBV": {"HBV": "1.5.2", "numpy": None, "pandas": None, "xarray": None, "h5netcdf": None},
    "03_SUMMA": {"numpy": None, "pandas": None, "netCDF4": None, "xarray": None},
    "04_VIC": {"numpy": None, "pandas": None},
    "05_MODFLOW6": {"numpy": None, "pandas": None, "flopy": "3.10.0"},
    "06_CRHM": {"numpy": None, "pandas": None},
    "A_VIC_Tangnaihai": {"numpy": None, "pandas": None},
    "C_GR4J_L0123001": {},  # Exact SITE_MANIFEST supersedes a version-only check.
    "E1_airGR": {"numpy": None, "pandas": None},
    "E2_SACSMA_CAMELS": {"numpy": None, "pandas": None},
    "EXP6_SACSMA_Duan_HYDAT": {"spotpy": "1.6.7", "numpy": None},
    "FSM2": {"numpy": None},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_path(value: str, package_root: Path, record_root: Path) -> Path:
    """Resolve package-relative index paths without opening old absolute SSD paths."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Index path is not a safe package-relative path: {value}")
    try:
        relative = path.relative_to(RECORD_PREFIX)
    except ValueError:
        return inside(package_root, path, must_exist=False)
    return inside(record_root, relative, must_exist=False)


def manifest_path(name: str, case_dir: Path) -> tuple[Path, str]:
    """Retain hidden files and rebase only recognized historical case/payload paths.

    The shipped manifests use case-relative paths (including './' for Ibex).
    Absolute study-runtime paths can be checked against their archived files/
    counterpart, never against the live host path.  Historical absolute paths to
    this same case can also be rebased after an environments/10_model_packages
    component.  Parent traversal and unknown absolute paths are refused.
    """
    source = PurePosixPath(name)
    if ".." in source.parts:
        raise ValueError("Parent traversal in manifest entry")
    if not source.is_absolute():
        return case_dir.joinpath(*source.parts), "case_relative"
    parts = source.parts
    for marker in ("environments", "10_model_packages"):
        for i, part in enumerate(parts[:-1]):
            if part == marker and parts[i + 1] == case_dir.name:
                return case_dir.joinpath(*parts[i + 2:]), "historical_case_rebased"
    if len(parts) > 1 and parts[1] in {"mnt", "home", "tmp", "usr", "opt"}:
        return case_dir.joinpath("files", *parts[1:]), "historical_runtime_rebased"
    raise ValueError("Unrecognized absolute manifest path; no host path was opened")


def verify_manifest(case: dict, case_dir: Path, manifest: Path) -> dict:
    result = {
        "path": str(manifest), "exists": manifest.is_file(), "entries": 0,
        "matched": 0, "missing": [], "mismatched": [], "errors": [],
        "rebased_entries": [], "manifest_identity_matches_index": False,
    }
    if not manifest.is_file():
        result["integrity_ok"] = False
        return result
    try:
        actual_manifest_hash = sha256(manifest)
        result["actual_sha256"] = actual_manifest_hash
        result["expected_sha256"] = case.get("manifest_sha256")
        result["manifest_identity_matches_index"] = actual_manifest_hash == case.get("manifest_sha256")
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        result["errors"].append(str(error))
        result["integrity_ok"] = False
        return result
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?(.*)", line)
        if not match:
            result["errors"].append({"line": line_number, "reason": "Malformed/unsupported SHA-256 entry"})
            continue
        expected, name = match.groups()
        result["entries"] += 1
        try:
            target, mode = manifest_path(name, case_dir)
            if mode != "case_relative":
                result["rebased_entries"].append({"entry": name, "resolved": str(target), "mode": mode})
            # Preserve internal symlinks, but refuse a link to an external live host.
            if not target.resolve().is_relative_to(case_dir.resolve()):
                raise ValueError("Manifest entry resolves outside this case directory")
            if not target.is_file():
                result["missing"].append({"entry": name, "resolved": str(target)})
            else:
                actual = sha256(target)
                if actual == expected.lower():
                    result["matched"] += 1
                else:
                    result["mismatched"].append({"entry": name, "expected": expected.lower(), "actual": actual})
        except (OSError, ValueError) as error:
            result["errors"].append({"line": line_number, "entry": name, "reason": str(error)})
    result["expected_entry_count"] = case.get("manifest_entries_verified_2026_09_24")
    result["entry_count_matches_index"] = result["entries"] == result["expected_entry_count"]
    result["integrity_ok"] = bool(
        result["manifest_identity_matches_index"] and result["entry_count_matches_index"]
        and result["matched"] == result["entries"] and not result["errors"]
    )
    return result


def check_binary(entry: dict, package_root: Path, record_root: Path) -> dict:
    path = package_path(entry["package_path"], package_root, record_root)
    result = dict(entry, resolved_path=str(path), exists=path.is_file(), integrity_ok=False)
    if path.is_file():
        try:
            with path.open("rb") as stream:
                header = stream.read(20)
            result["is_elf"] = header[:4] == b"\x7fELF"
            byte_order = "little" if len(header) > 5 and header[5] == 1 else "big"
            result["elf_machine"] = int.from_bytes(header[18:20], byte_order) if result["is_elf"] else None
            result["actual_sha256"] = sha256(path)
            result["integrity_ok"] = bool(
                result["actual_sha256"] == entry["sha256"] and result["is_elf"]
                and result["elf_machine"] == 62  # ELF EM_X86_64.
            )
        except OSError as error:
            result["error"] = str(error)
    original = entry.get("study_path")
    result["original_runtime_path_present_on_host"] = Path(original).is_file() if original else None
    return result


def check_sealed_runtime(case_dir: Path) -> dict:
    candidates = list((case_dir / "files").rglob("SITE_MANIFEST.json"))
    if len(candidates) != 1:
        return {"status": "unresolved", "reason": f"Expected one SITE_MANIFEST, found {len(candidates)}"}
    manifest = candidates[0]
    inside(case_dir, manifest.relative_to(case_dir))
    data = json.loads(manifest.read_text())
    # The companion archive contains the identity manifest, not this external tree.
    # Never follow an arbitrary host directory supplied by an archive manifest.
    return {"identity_manifest": str(manifest), "expected_files": len(data["files"]),
            "status": "external_runtime_not_verified",
            "reason": "The exact sealed runtime is unrecovered in this package. No external host tree was read."}


def current_runtime(case: dict, case_dir: Path, binaries: list[dict], host: dict) -> dict:
    name = case["case"]
    checks = {"scope": "Current preflight Python interpreter and visible host paths only; no imports of model/dependency code.",
              "dependencies_recorded": case["runtime_dependencies"],
              "setup_conditions_recorded": case["setup_conditions"], "python_packages": [],
              "detected_conditions": [], "shared_libraries_checked": False,
              "fresh_model_execution_performed": False}
    if sys.version_info[:2] != (3, 12):
        checks["detected_conditions"].append("Preflight interpreter is not the recorded Python 3.12 runtime.")
    for distribution, expected in PYTHON_DEPENDENCIES.get(name, {}).items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        row = {"distribution": distribution, "recorded_version": expected, "installed_version": actual}
        row["status"] = "missing" if actual is None else ("version_differs" if expected and actual != expected else "present")
        checks["python_packages"].append(row)
        if row["status"] != "present":
            checks["detected_conditions"].append(f"{distribution}: {row['status']} in the preflight interpreter.")
    if binaries and not host["linux_x86_64"]:
        checks["detected_conditions"].append("Shipped Linux x86-64 ELF binaries are incompatible with this native host.")
    for entry in binaries:
        if entry["original_runtime_path_present_on_host"] is False:
            checks["detected_conditions"].append(f"Original binary path is not staged: {entry['study_path']}")
    aliases = []
    for alias in case.get("additional_aliases", []):
        expected, target = Path(alias["expected"]), Path(alias["target"])
        row = dict(alias, expected_exists=expected.exists(), target_exists=target.exists(), resolves_to_target=False)
        if expected.exists() and target.exists():
            row["resolves_to_target"] = expected.resolve() == target.resolve()
        aliases.append(row)
        if not row["resolves_to_target"]:
            checks["detected_conditions"].append(f"Original output alias not restored: {expected} -> {target}")
    checks["aliases"] = aliases
    if name in {"A_VIC_Tangnaihai", "04_VIC", "FSM2"} and not Path("/dev/shm").is_dir():
        checks["detected_conditions"].append("Linux /dev/shm directory is absent.")
    if name in {"C_GR4J_L0123001", "E1_airGR"}:
        checks["Rscript_on_path"] = shutil.which("Rscript")
        checks["recorded_Rscript_path_present"] = Path("/usr/bin/Rscript").is_file()
        checks["airGR_identity_checked"] = False
        if not checks["recorded_Rscript_path_present"]:
            checks["detected_conditions"].append("Recorded /usr/bin/Rscript is absent; airGR identity has not been established.")
    if name == "C_GR4J_L0123001":
        try:
            checks["sealed_runtime"] = check_sealed_runtime(case_dir)
        except (OSError, ValueError, KeyError) as error:
            checks["sealed_runtime"] = {"status": "unresolved", "reason": str(error)}
        if checks["sealed_runtime"]["status"] != "identity_matches":
            checks["detected_conditions"].append("Exact sealed Python runtime identity is not satisfied.")
    checks["execution_readiness"] = (
        "blocked_by_detected_conditions" if checks["detected_conditions"] else
        "not_established_by_static_preflight"
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-root", "--package-root", dest="package_root", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases", help="Check one case; repeat to select several. Default: all 12.")
    parser.add_argument("--json", "--output-json", dest="json_output", metavar="PATH", help="Write a JSON report; '-' prints JSON to stdout.")
    args = parser.parse_args(argv)
    try:
        package_root = paper_root(args.package_root)
        record_root = inside(package_root, RECORD_PREFIX)
        index_dir = inside(package_root, DEFAULT_INDEX)
        index = json.loads(inside(index_dir, "ENVIRONMENT_INDEX.json").read_text())
        binary_index = json.loads(inside(index_dir, "BINARY_PATHS.json").read_text())
        cases_by_name = {entry["case"]: entry for entry in index["cases"]}
        selected = args.cases or list(cases_by_name)
        unknown = set(selected) - set(cases_by_name)
        if unknown:
            parser.error("Unknown case(s): " + ", ".join(sorted(unknown)))
        selected = list(dict.fromkeys(selected))
        host = {"system": platform.system(), "machine": platform.machine(), "platform": platform.platform(),
                "python": platform.python_version(), "python_executable": sys.executable,
                "linux_x86_64": platform.system() == "Linux" and platform.machine().lower() in {"x86_64", "amd64"}}
        results = []
        for name in selected:
            case = cases_by_name[name]
            case_dir = package_path(case["package_directory"], package_root, record_root)
            manifest = package_path(case["manifest"], package_root, record_root)
            binaries = [check_binary(entry, package_root, record_root) for entry in binary_index if entry["case"] == name]
            instruction = package_path(case["instructions"], package_root, record_root)
            verification = verify_manifest(case, case_dir, manifest)
            results.append({"case": name, "study": case["study"], "case_directory": str(case_dir),
                            "instructions": str(instruction), "instructions_present": instruction.is_file(),
                            "manifest": verification, "binaries": binaries,
                            "archive_integrity_ok": verification["integrity_ok"] and instruction.is_file() and all(b["integrity_ok"] for b in binaries),
                            "runtime": current_runtime(case, case_dir, binaries, host)})
        findings_path = inside(index_dir, "RECOVERY_FINDINGS.json", must_exist=False)
        findings = json.loads(findings_path.read_text()) if findings_path.is_file() else {}
        report = {"schema": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "package_root": str(package_root), "record_root": str(record_root), "index_directory": str(index_dir),
                  "host": host, "selected_cases": selected, "case_count": len(results),
                  "manifest_entries": sum(r["manifest"]["entries"] for r in results),
                  "manifest_entries_matched": sum(r["manifest"]["matched"] for r in results),
                  "binary_files_checked": sum(len(r["binaries"]) for r in results),
                  "archive_integrity_ok": all(r["archive_integrity_ok"] for r in results),
                  "execution_readiness": "not_established; see per-case detected conditions",
                  "model_execution_performed": False, "dependency_installation_performed": False,
                  "native_source_and_external_dependency_conditions": findings.get("unresolved", []),
                  "source_condition_scope": "Recorded recovery audit conditions, not a new full-disk source search. Missing native source does not itself invalidate an included replay binary.",
                  "cases": results}
        encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.json_output and args.json_output != "-":
            destination = external_output(package_root, args.json_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded)
        if args.json_output == "-":
            print(encoded, end="")
        else:
            print(f"Archive integrity: {'PASS' if report['archive_integrity_ok'] else 'FAIL'}; {len(results)} case(s), "
                  f"{report['manifest_entries_matched']}/{report['manifest_entries']} manifest entries, "
                  f"{report['binary_files_checked']} binary/library hashes.")
            print(f"Host: {host['system']} {host['machine']}, Python {host['python']}. No models run or dependencies installed.")
            for result in results:
                print(f"  {result['case']}: archive={'PASS' if result['archive_integrity_ok'] else 'FAIL'}; "
                      f"runtime={result['runtime']['execution_readiness']}")
            print("Execution readiness is not certified by file integrity. See JSON and per-case setup instructions.")
            if args.json_output:
                print(f"JSON report: {args.json_output}")
        return 0 if report["archive_integrity_ok"] else 2
    except (OSError, ValueError, KeyError) as error:
        print(f"Preflight configuration/read error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
