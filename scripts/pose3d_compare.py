#!/usr/bin/env python3
"""Compare normalized A/B predictions on identical authorized frames, not accuracy."""
import argparse
from datetime import date
import math
from pathlib import Path
import sys
from pose3d_evaluate import read_json, require, load_clip, load_run, point, digest, write_report, ensure_new_output, output_error


def compare(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    require(manifest.get('schemaVersion') == 'pose3d-evaluation-0.1.0' and isinstance(manifest.get('videos'), list), 'invalid_manifest')
    reports = []
    for clip in manifest['videos']:
        require(isinstance(clip, dict), 'clip_object_required')
        item = {'id': clip.get('id')}
        reports.append(item)
        try:
            times = load_clip(path.parent, clip, date.today())
            runs = [load_run(path.parent, r, clip['frameCount'], times, clip['videoSha256']) for r in clip['runs']]
            a = [r for r in runs if r[0]['pose3d']['runtime'] == 'rtmlib-onnx']
            b = [r for r in runs if r[0]['pose3d']['runtime'] == 'mmpose-pytorch']
            require(len(a) == len(b) == 1, 'requires_exactly_one_A_and_one_B_run')
            differences, z_matches, z_pairs = [], 0, 0
            for frame in range(clip['frameCount']):
                for joint in range(65):
                    pa, pb = point(a[0][1], frame, joint), point(b[0][1], frame, joint)
                    if pa[3] > 0 and pb[3] > 0:
                        differences.append(math.dist(pa[:3], pb[:3]))
                for left, right in ((5, 6), (7, 8), (9, 10)):
                    pts = [point(data, frame, j) for data in (a[0][1], b[0][1]) for j in (left, right)]
                    if all(p[3] > 0 for p in pts):
                        da = (pts[0][2] - pts[1][2]) * a[0][0]['provenance']['depthNearerSign']
                        db = (pts[2][2] - pts[3][2]) * b[0][0]['provenance']['depthNearerSign']
                        if da != 0 and db != 0:
                            z_pairs += 1
                            z_matches += int((da > 0) == (db > 0))
            require(bool(differences), 'no_common_valid_points')
            item.update(status='compared', commonPoints=len(differences),
                        meanModelRelativeDifference=sum(differences) / len(differences),
                        pairDepthSignAgreement=z_matches / z_pairs if z_pairs else None,
                        pairDepthCount=z_pairs,
                        weightsSha256=[run[0]['pose3d']['weightsSha256'] for run in (a[0], b[0])])
        except (KeyError, ValueError, TypeError, OSError) as error:
            item.update(status='invalid_evidence', error=str(error))
    return {'status': 'evidence_collected' if any(c['status'] == 'compared' for c in reports) else 'blocked',
            'manifestSha256': digest(path), 'clips': reports, 'gateDecision': 'not_assessed', 'releaseReady': False,
            'limitations': ['differences_are_not_ground_truth_error', 'model_relative_scales_must_be_reviewed',
                           'left_right_correctness_requires_manual_annotations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output', required=True, help='新的报告路径；已有文件不会覆盖')
    args = parser.parse_args()
    try:
        output = ensure_new_output(args.output)
    except OSError as error:
        return output_error(error)
    try:
        report = compare(args.manifest)
    except (KeyError, ValueError, TypeError, OSError) as error:
        report = {'status': 'invalid_evidence', 'error': str(error), 'releaseReady': False}
    try:
        write_report(output, report)
    except (OSError, ValueError, TypeError) as error:
        return output_error(error)
    print(f"{report['status']}: {args.output}")
    return 0 if report['status'] == 'evidence_collected' else 2


if __name__ == '__main__':
    sys.exit(main())
