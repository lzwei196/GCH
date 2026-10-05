#!/usr/bin/env python3
"""Read-only checks of archived results; no numerical models or agent calls."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import math

from common import paper_root, inside, external_output, RECORD

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k not in ('file', 'contract_dir')}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    return value


def gr4j(record):
    base = record / 'experiments/3_gr4j_ranges/B_ki_vs_memory_gr4j/full'
    code = base / 'compare_contracts_v3.py'
    manifest = base / 'CAMPAIGN_MANIFEST.json'
    for path in (code, manifest, base / 'CAMPAIGN_MANIFEST.sha256', base.parent / 'comparison_current_v3.json'):
        inside(record, path.relative_to(record))
    man = json.loads(manifest.read_text())
    if sha(manifest) != (base / 'CAMPAIGN_MANIFEST.sha256').read_text().split()[0]:
        raise ValueError('Campaign manifest hash mismatch')
    if sha(code) != man['scorer']['sha256']:
        raise ValueError('Archived scorer hash mismatch')
    e1 = record / 'experiments/7_reference_comparisons/E1_E2_reproductions/E1_airGR/agent_authored.yaml'
    inside(record, e1.relative_to(record))
    if sha(e1) != man['side_inputs']['E1_agent_authored']['sha256']:
        raise ValueError('E1 side-input hash mismatch')
    spec = importlib.util.spec_from_file_location('archived_gr4j_current_scorer', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.E1 = e1
    inside(record, (base / 'contracts').relative_to(record))
    for path in (base / 'contracts').glob('*_calibration.yaml'):
        inside(record, path.relative_to(record))
    contracts = module.load_set(base / 'contracts')
    if len(contracts) != 10 or not all('params' in v for v in contracts.values()):
        raise ValueError('Missing or invalid contract')
    ok, per, pairs, groups = module.score(contracts)
    generated = dict(contracts=ok, per_contract_vs_canonical=per, pairwise=pairs, group_agreement=groups, missing=[])
    saved = json.loads((base.parent / 'comparison_current_v3.json').read_text())['v3']
    if normalized(generated) != normalized(saved):
        raise ValueError('Recomputed current-cohort fields differ from saved v3 results')
    inputs = [code, manifest, e1, base.parent / 'comparison_current_v3.json', *sorted(p for p in (base / 'contracts').glob('*_calibration.yaml') if not p.name.startswith('._'))]
    return {
        'check': 'gr4j_current_cohort', 'passed': True, 'authored_contracts': 9,
        'reference_contracts': 1, 'comparison': 'All normalized parsed fields and numerical scores equal saved v3 results',
        'input_sha256': {str(p.relative_to(record)): sha(p) for p in inputs},
        'scope': 'Current cohort only. Historical v1/v2 contracts are not recovered or rescored. Current contract hashes record this check; the scorer and E1 hashes are verified against the historical manifest.'
    }


def complete_six_model_reports(record):
    """Fail on incomplete/empty reports before the historical consistency printer."""
    cases = ('01_WOFOST', '02_HBV', '03_SUMMA', '04_VIC', '05_MODFLOW6', '06_CRHM')
    count = 0
    for case in cases:
        path = inside(record, f'experiments/1_six_model/six_model_families/{case}/rerun_conv_report.json')
        report = json.loads(path.read_text())
        objectives = report.get('objectives')
        components = report.get('holdout', {}).get('per_objective')
        if not objectives or not isinstance(components, list) or not components:
            raise ValueError(f'{case}: missing objectives or holdout components')
        if not isinstance(objectives, list) or not all(isinstance(x, str) and x for x in objectives) or len(set(objectives)) != len(objectives):
            raise ValueError(f'{case}: invalid or duplicate objective names')
        if [c.get('objective') for c in components] != objectives:
            raise ValueError(f'{case}: objective order/count differs from holdout components')
        for component in components:
            for key in ('holdout_loss', 'baseline_holdout_loss'):
                value = component.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{case}: missing/nonfinite {key}')
            for key in ('ok', 'beats_baseline'):
                if not isinstance(component.get(key), bool):
                    raise ValueError(f'{case}: missing/nonboolean {key}')
        count += len(components)
    if count != 23:
        raise ValueError(f'Expected 23 components across six reports; found {count}')
    return {'model_cases': 6, 'components': count}


def six_model(record):
    completeness = complete_six_model_reports(record)
    script = inside(record, 'framework/reproduce/verify_six_model_table.py')
    inside(record, 'framework/reproduce/verify_table1.py')
    result = run_script('six_model_saved_report_consistency', script)
    result['completeness'] = completeness
    return result


def run_script(name, script, pythonpath=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if pythonpath:
        env['PYTHONPATH'] = str(pythonpath)
    with tempfile.TemporaryDirectory(prefix='grl-saved-check-') as tmp:
        proc = subprocess.run([sys.executable, '-B', str(script)], cwd=tmp, env=env,
                              capture_output=True, text=True, timeout=120)
    extra = {}
    if name == 'selector_unit':
        extra['tests_passed'] = sum(line.startswith('OK ') for line in proc.stdout.splitlines())
    return dict(check=name, passed=proc.returncode == 0, exit_code=proc.returncode,
                stdout=proc.stdout, stderr=proc.stderr, **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root', '--package-root', dest='package_root', type=Path, required=True)
    parser.add_argument('--check', choices=['all', 'six_model', 'gr4j_current', 'selector_unit'], default='all')
    parser.add_argument('--output', type=Path, help='Optional JSON result outside the frozen record')
    args = parser.parse_args()
    try:
        package = paper_root(args.package_root)
        record = inside(package, RECORD)
        if args.output:
            args.output = external_output(package, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    snapshot = record / 'framework/kdt_repo_snapshot'
    checks = {
        'six_model': lambda: six_model(record),
        'gr4j_current': lambda: gr4j(record),
        'selector_unit': lambda: run_script('selector_unit', inside(record, 'framework/kdt_repo_snapshot/calibration_kit/test_front_select.py'), inside(record, 'framework/kdt_repo_snapshot')),
    }
    results = []
    for name in checks if args.check == 'all' else [args.check]:
        try:
            results.append(checks[name]())
        except Exception as exc:
            results.append(dict(check=name, passed=False, error=f'{type(exc).__name__}: {exc}'))
    report = dict(scope='Saved-report consistency, current GR4J contract scoring, and eight selector implementation tests. No model or agent campaign rerun; six-model band flags are archived flags, not an independent acceptance audit.',
                  python=sys.version, platform=platform.platform(), record_root=str(record),
                  passed=all(r['passed'] for r in results), results=results)
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
