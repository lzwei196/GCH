#!/usr/bin/env python3
"""Inspect paper coverage or check an external companion package. No model campaigns."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "checks"))
from common import METADATA, paper_root, read_json, external_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List reported tests and evidence status; no archive needed")
    listing.add_argument("--test", help="Exact test ID to inspect, including archive-relative locators")
    listing.add_argument("--json", action="store_true")
    for name in ("validate", "check-saved", "environments", "figure-s2"):
        child = commands.add_parser(name)
        child.add_argument("--paper-root", "--package-root", dest="paper_root", type=Path, required=True,
                           help="External GRL companion package root")
        if name == "figure-s2":
            child.add_argument("--preflight", action="store_true", help="Verify saved inputs without importing the builder or rendering")
            child.add_argument("--output-dir", type=Path)
        else:
            child.add_argument("--output", type=Path, help="JSON report outside the companion package")
        if name == "check-saved":
            child.add_argument("--check", choices=("all", "six_model", "gr4j_current", "selector_unit"), default="all")
        if name == "environments":
            child.add_argument("--case", action="append")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            rows = read_json(METADATA / "TEST_INDEX.json")["tests"]
            if args.test:
                rows = [row for row in rows if row["test_id"] == args.test]
                if not rows:
                    parser.error(f"Unknown test ID: {args.test}")
            if args.json or args.test:
                print(json.dumps(rows, indent=2, ensure_ascii=False))
            else:
                print("Paper evidence navigation (entries include subtests and dependencies; not an experiment count):")
                for row in rows:
                    print(f"{row['test_id']} | {row['paper']} | {row['evidence_status']}")
            return 0
        root = paper_root(args.paper_root)
        scripts = {"validate": "validate_indexes.py", "check-saved": "check_saved_results.py",
                   "environments": "environment_preflight.py", "figure-s2": "rebuild_figure_s2.py"}
        command = [sys.executable, "-B", str(HERE / "checks" / scripts[args.command]), "--paper-root", str(root)]
        if args.command == "figure-s2":
            if args.preflight:
                command.append("--preflight")
            elif not args.output_dir:
                parser.error("figure-s2 requires --output-dir unless --preflight is selected")
            if args.output_dir:
                command.extend(["--output-dir", str(external_output(root, args.output_dir))])
        else:
            if args.output:
                flag = "--json" if args.command == "environments" else "--output"
                command.extend([flag, str(external_output(root, args.output))])
            if args.command == "check-saved":
                command.extend(["--check", args.check])
            if args.command == "environments":
                for case in args.case or []:
                    command.extend(["--case", case])
        return subprocess.run(command, check=False).returncode
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Paper check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
