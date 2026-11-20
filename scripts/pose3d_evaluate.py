#!/usr/bin/env python3
"""Evaluate normalized research artifacts; never infer or certify a release gate.

All paths are relative to the JSON that contains them. Video consent is required.
Synthetic unit tests validate arithmetic, never model performance.
"""
from __future__ import annotations
import argparse
from array import array
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

VIEWS = ('side', 'oblique', 'front', 'rear', 'unknown')
BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError(f'non_finite_json: {value}')
    result = json.loads(Path(path).read_text(), parse_constant=invalid)
    require(isinstance(result, dict), 'json_object_required')
    return result


def resolve(base, value):
    require(isinstance(value, str) and bool(value), 'missing_file_path')
    return (base / value).resolve()


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def floats(path, count):
    require(path.stat().st_size == count * 4, 'invalid_binary_length')
    result = array('f')
    result.frombytes(path.read_bytes())
    if sys.byteorder != 'little':
        result.byteswap()
    return result


def frame_number(value, count):
    require(type(value) is int and 0 <= value < count, 'invalid_frame')
    return value


def load_clip(base, clip, today):
    require(isinstance(clip, dict), 'clip_object_required')
    require(isinstance(clip.get('id'), str) and clip['id'], 'missing_clip_id')
    require(clip.get('synthetic') is not True, 'synthetic_is_not_model_evidence')
    require(clip.get('viewClass') in VIEWS, 'missing_manual_view')
    require(clip.get('split') in ('calibration', 'validation', 'test'), 'missing_dataset_split')
    evidence = resolve(base, clip.get('authorizationEvidence'))
    require(evidence.is_file() and evidence.stat().st_size > 0, 'missing_authorization_evidence')
    require(date.fromisoformat(clip['retentionUntil']) >= today, 'authorization_expired')
    video = resolve(base, clip.get('videoPath'))
    require(digest(video) == clip.get('videoSha256'), 'video_hash_mismatch')
    count = clip.get('frameCount')
    require(type(count) is int and 0 < count <= 20000, 'invalid_frame_count')
    times = floats(resolve(base, clip.get('timestampsPath')), count)
    require(all(math.isfinite(t) and t >= 0 for t in times)
            and all(a < b for a, b in zip(times, times[1:])), 'invalid_timestamps')
    require(isinstance(clip.get('runs'), list), 'missing_runs')
    return times


def load_run(base, run_ref, count, times, input_video_sha256):
    run_path = resolve(base, run_ref['path'])
    run = read_json(run_path)
    pose = run['pose3d']
    require(pose['status'] == 'succeeded', 'run_not_succeeded')
    require(pose['source'] == 'ai-monocular-predicted', 'not_ai_prediction')
    require(pose['frameCount'] == count and pose['pointSpec'] == 'cwb-65', 'run_frame_spec_mismatch')
    require(pose['coordinateSpace'] == 'camera-root-relative'
            and pose['axisConvention'] == 'x-right-y-up-z-forward'
            and pose['unit'] == 'model-relative' and pose['rootJoint'] == 'mid-hip', 'unverified_coordinate_contract')
    require(re.fullmatch('[a-fA-F0-9]{64}', pose['weightsSha256']) is not None, 'invalid_weights_hash')
    require(pose.get('runtime') in ('rtmlib-onnx', 'mmpose-pytorch', 'motionbert-pytorch'), 'invalid_runtime')
    require(all(isinstance(pose.get(k), str) and pose[k] for k in ('modelId', 'modelVersion', 'pipelineVersion', 'viewGatePolicyVersion')), 'missing_model_provenance')
    provenance = run['provenance']
    require(all(provenance.get(k) for k in ('sourceRevision', 'dependencyVersions', 'device', 'configHash', 'coordinateEvidence')), 'missing_provenance')
    source_hash = provenance.get('inputVideoSha256')
    require(isinstance(source_hash, str) and re.fullmatch('[a-fA-F0-9]{64}', source_hash), 'invalid_input_video_hash')
    require(isinstance(input_video_sha256, str) and re.fullmatch('[a-fA-F0-9]{64}', input_video_sha256), 'invalid_manifest_video_hash')
    require(source_hash.lower() == input_video_sha256.lower(), 'run_video_hash_mismatch')
    require(resolve(run_path.parent, provenance['coordinateEvidence']).is_file(), 'missing_coordinate_evidence')
    require(type(provenance.get('depthNearerSign')) is int and provenance['depthNearerSign'] in (-1, 1), 'missing_verified_depth_sign')
    run_times = floats(resolve(run_path.parent, run['timestampsPath']), count)
    require(run_times == times, 'timestamp_mismatch')
    data = floats(resolve(run_path.parent, run['keypointsPath']), count * 65 * 4)
    allowed = set()
    previous_end = -1
    for window in pose['windows']:
        start, end = (frame_number(window[k], count) for k in ('startFrame', 'endFrame'))
        require(start <= end and start > previous_end, 'invalid_windows')
        require(window['viewClass'] in VIEWS and window['viewGate'] in ('pass', 'limited', 'blocked'), 'invalid_view_gate')
        previous_end = end
        if window['viewGate'] != 'blocked':
            allowed.update(range(start, end + 1))
    valid_points = 0
    for f in range(count):
        for joint in range(65):
            x, y, z, confidence = point(data, f, joint)
            missing = confidence == 0 and all(math.isnan(v) for v in (x, y, z))
            valid = all(math.isfinite(v) for v in (x, y, z, confidence)) and 0 < confidence <= 1
            require(missing or valid, 'invalid_point_encoding')
            require(f in allowed or missing, 'data_outside_windows')
            valid_points += int(valid)
        if any(point(data, f, j)[3] > 0 for j in range(65)):
            hips = [point(data, f, j) for j in (11, 12)]
            require(all(p[3] > 0 for p in hips), 'missing_root')
            require(all(abs((hips[0][i] + hips[1][i]) / 2) <= 1e-4 for i in range(3)), 'nonzero_root')
    require(valid_points > 0, 'successful_run_has_no_valid_points')
    return run, data


def point(data, frame, joint):
    offset = (frame * 65 + joint) * 4
    return data[offset:offset + 4]


def ratio(successes, total):
    return {'count': total, 'value': successes / total if total else None,
            'status': 'measured' if total else 'not_measurable'}


def ordinal_metrics(clip, data, depth_nearer_sign):
    require(depth_nearer_sign in (-1, 1), 'missing_verified_depth_sign')
    buckets = {view: {'correct': 0, 'usable': 0, 'annotated': 0} for view in VIEWS}
    seen = set()
    for label in clip.get('ordinalDepth', []):
        frame = frame_number(label['frame'], clip['frameCount'])
        a, b = label['a'], label['b']
        require(type(a) is int and type(b) is int and 0 <= a < 65 and 0 <= b < 65 and a != b, 'invalid_joint_pair')
        require(label['closer'] in ('a', 'b'), 'ordinal_label_must_choose_a_or_b')
        require((frame, min(a, b), max(a, b)) not in seen, 'duplicate_depth_annotation')
        seen.add((frame, min(a, b), max(a, b)))
        view = label.get('viewClass', clip['viewClass'])
        require(view in VIEWS, 'invalid_manual_view')
        bucket = buckets[view]
        bucket['annotated'] += 1
        pa, pb = point(data, frame, a), point(data, frame, b)
        if pa[3] > 0 and pb[3] > 0:
            # The POC evidence must state which sign means closer to camera;
            # do not infer this from an ambiguous "z-forward" label.
            bucket['usable'] += 1
            delta = (pa[2] - pb[2]) * depth_nearer_sign
            bucket['correct'] += int(delta > 0 if label['closer'] == 'a' else delta < 0)
    return {view: {'accuracy': ratio(v['correct'], v['usable']),
                   'coverage': ratio(v['usable'], v['annotated'])} for view, v in buckets.items()}


def drift(data, count, windows):
    values = []
    for window in windows:
        for f in range(window['startFrame'] + 1, window['endFrame'] + 1):
            for a, b in BONES:
                pts = [point(data, t, j) for t in (f - 1, f) for j in (a, b)]
                if all(p[3] > 0 for p in pts):
                    before = math.dist(pts[0][:3], pts[1][:3])
                    after = math.dist(pts[2][:3], pts[3][:3])
                    if before > 1e-8:
                        values.append(abs(after - before) / before)
    return {'count': len(values), 'meanRelativeChange': sum(values) / len(values) if values else None,
            'status': 'measured' if values else 'not_measurable'}


def evaluate_run(clip, run, data):
    count, pose = clip['frameCount'], run['pose3d']
    reviews = run.get('manualReview', [])
    reviewed_frames = set()
    by_view = {v: [] for v in VIEWS}
    for review in reviews:
        frame = frame_number(review['frame'], count)
        require(frame not in reviewed_frames, 'duplicate_review_frame')
        reviewed_frames.add(frame)
        require(all(type(review.get(k)) is bool for k in ('leftRightSwapped', 'identitySwitch', 'severeError')), 'incomplete_manual_review')
        view = review.get('viewClass', clip['viewClass'])
        require(view in VIEWS, 'invalid_review_view')
        by_view[view].append(review)
    covered = {f for w in pose['windows'] if w['viewGate'] != 'blocked'
               for f in range(w['startFrame'], w['endFrame'] + 1)}
    hits = clip.get('manualHitFrames', [])
    require(len(set(hits)) == len(hits), 'duplicate_manual_hit')
    for f in hits:
        frame_number(f, count)
    measurements = run.get('measurements')
    if measurements is not None:
        for key in ('elapsedSec', 'peakRamBytes'):
            require(type(measurements.get(key)) in (int, float) and math.isfinite(measurements[key]) and measurements[key] >= 0, 'invalid_measurement')
        require(type(measurements.get('coldStart')) is bool, 'missing_cold_start')
        require(isinstance(measurements.get('windowSeconds'), list) and len(measurements['windowSeconds']) == len(pose['windows']), 'missing_window_measurements')
        require(all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in measurements['windowSeconds']), 'invalid_window_measurement')
    return {'runtime': pose['runtime'], 'modelId': pose['modelId'], 'weightsSha256': pose['weightsSha256'],
            'status': 'evaluated', 'ordinalDepthByView': ordinal_metrics(clip, data, run['provenance']['depthNearerSign']),
            'manualReviewByView': {v: {'leftRightSwapRate': ratio(sum(r['leftRightSwapped'] for r in rows), len(rows)),
                'identitySwitches': sum(r['identitySwitch'] for r in rows) if rows else None,
                'severeErrors': sum(r['severeError'] for r in rows) if rows else None} for v, rows in by_view.items()},
            'windowHitCoverage': ratio(sum(f in covered for f in hits), len(hits)),
            'boneLengthDrift': drift(data, count, pose['windows']),
            'performance': measurements if measurements is not None else {'status': 'not_measurable'},
            'mpjpe': {'status': 'not_measurable', 'reason': 'no_validated_3d_ground_truth_protocol'},
            'limitations': ['manual_review_counts_only_annotated_frames', 'internal_consistency_does_not_validate_depth']}


def evaluate(manifest_path, today=None):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    require(manifest.get('schemaVersion') == 'pose3d-evaluation-0.1.0' and isinstance(manifest.get('videos'), list), 'invalid_manifest')
    clips = []
    seen = set()
    for clip in manifest['videos']:
        require(isinstance(clip, dict), 'clip_object_required')
        item = {'id': clip.get('id'), 'runs': []}
        clips.append(item)
        try:
            require(clip.get('id') not in seen, 'duplicate_clip_id')
            seen.add(clip.get('id'))
            times = load_clip(path.parent, clip, today or date.today())
            item['viewClass'] = clip['viewClass']
            item['status'] = 'validated_input'
            for run_ref in clip['runs']:
                try:
                    run, data = load_run(path.parent, run_ref, clip['frameCount'], times, clip['videoSha256'])
                    item['runs'].append(evaluate_run(clip, run, data))
                except (ValueError, TypeError, KeyError, OSError) as error:
                    item['runs'].append({'status': 'invalid_evidence', 'error': str(error)})
        except (ValueError, TypeError, KeyError, OSError) as error:
            item.update(status='invalid_evidence', error=str(error))
    valid_runs = sum(run['status'] == 'evaluated' for clip in clips for run in clip['runs'])
    return {'reportVersion': 'pose3d-evaluation-0.1.0', 'status': 'evidence_collected' if valid_runs else 'blocked',
            'manifestSha256': digest(path), 'frozenDataset': manifest.get('frozen') is True,
            'clips': clips, 'evaluatedRuns': valid_runs,
            'gateDecision': 'not_assessed',
            'releaseReady': False,
            'remaining': ['review_runtime_choice_and_failure_examples', 'approve_licenses_and_costs',
                          'freeze_view_policy_on_held_out_data', 'complete_user_value_study']}


def ensure_new_output(path):
    # Do not resolve away a dangling symlink: it is still an existing target.
    output = Path(path).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'output_already_exists: {output}; choose a new report path')
    return output


def output_error(error):
    print(json.dumps({'status': 'output_rejected',
                      'errorCode': 'output_already_exists' if isinstance(error, FileExistsError) else 'output_unavailable',
                      'error': str(error), 'releaseReady': False}, ensure_ascii=False), file=sys.stderr)
    return 2


def write_report(path, report):
    """Publish a complete new report atomically, never replace existing evidence."""
    output = ensure_new_output(path)
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix=f'.{output.name}.',
                                         suffix='.tmp', dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # link() is an atomic exclusive publish on the same filesystem. A second
        # writer creating the target after preflight still cannot be overwritten.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        report = evaluate(args.manifest)
    except (ValueError, TypeError, KeyError, OSError) as error:
        report = {'status': 'invalid_evidence', 'error': str(error), 'releaseReady': False}
    try:
        write_report(output, report)
    except (OSError, ValueError, TypeError) as error:
        return output_error(error)
    print(json.dumps({'status': report['status'], 'output': args.output, 'releaseReady': False}))
    return 0 if report['status'] == 'evidence_collected' else 2


if __name__ == '__main__':
    sys.exit(main())
