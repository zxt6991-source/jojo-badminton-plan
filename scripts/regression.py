#!/usr/bin/env python3
"""Run all labeled clips with current pipeline, then compare to a saved report."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).resolve().parents[1] / 'worker/main.py'
    for index, case in enumerate(manifest['cases']):
        output = destination / f'case-{index:03d}'
        command = [args.python, str(worker), '--input', str((manifest_path.parent/case['video']).resolve()),
                   '--output-dir', str(output), '--analysis-id', case['id'], '--video-id', case['id'],
                   '--swing-type', case.get('swingType', 'smash')]
        if 'targetPoint' in case:
            command += ['--target-point', ','.join(str(n) for n in case['targetPoint'])]
        if 'captureFps' in case:
            command += ['--capture-fps', str(case['captureFps'])]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        case['result'] = str(output/'result.json')
    run_manifest = destination/'manifest.json'
    run_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    report = evaluate(run_manifest)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        report['baselinePipelineVersions'] = baseline.get('pipelineVersions', [])
        report['metricDeltas'] = {key: value-baseline['metrics'][key] for key, value in report['metrics'].items()
                                   if value is not None and baseline['metrics'].get(key) is not None}
        report['regressed'] = any((delta > 0 if key.endswith('Ms') else delta < 0) for key, delta in report['metricDeltas'].items())
    (destination/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get('regressed') or report['acceptance'] != 'pass':
        sys.exit(1)


if __name__ == '__main__':
    main()
