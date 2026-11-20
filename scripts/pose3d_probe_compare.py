#!/usr/bin/env python3
"""Compare BGR/RGB preprocessing of identical local RAW probe frames.

This is a preprocessing diagnostic, not a model A/B experiment or depth accuracy
evaluation. Scores retain their upstream meaning and are not depth confidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import zipfile
import numpy as np

from pose3d_evaluate import ensure_new_output, write_report, output_error

from pose3d_probe import (ProbeError, require, digest, existing_local_file, authorize,
                          load_timestamps, load_keypoints, validate_outputs,
                          OUTPUT_NAMES, MAX_PROBE_FRAMES, BBOX_CONFIDENCE)


def read_json(path):
    def reject_constant(value):
        raise ProbeError('nonfinite_json', value)
    value = json.loads(Path(path).read_text(), parse_constant=reject_constant)
    require(isinstance(value, dict), 'invalid_report')
    return value


def finite_nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'invalid_measurement', name)
    return float(value)


def verified_file(value, expected, label):
    require(isinstance(expected, str) and re.fullmatch('[a-fA-F0-9]{64}', expected) is not None,
            'invalid_hash', label)
    path = existing_local_file(value, f'{label}_unavailable')
    require(digest(path) == expected, 'evidence_hash_mismatch', label)
    return path


def video_size(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        require(capture.isOpened(), 'video_decode_failed')
        ok, first = capture.read()
        require(ok and first is not None, 'video_decode_failed')
        return first.shape[1], first.shape[0]
    finally:
        capture.release()


def load_raw(directory, item, timestamp):
    frame = item['frame']
    require(item['rawPath'] == f'frame-{frame:06d}.raw.npz', 'invalid_raw_path')
    path = directory / item['rawPath']
    require(not path.is_symlink() and path.resolve().parent == directory.resolve(), 'invalid_raw_path')
    verified_file(path, item['rawSha256'], 'raw_npz')
    require(path.stat().st_size <= 2 * 1024 * 1024, 'raw_npz_too_large')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) <= 16 and sum(entry.file_size for entry in entries) <= 2 * 1024 * 1024,
                'raw_npz_too_large')
    with np.load(path, allow_pickle=False) as archive:
        arrays = validate_outputs(tuple(archive[name] for name in OUTPUT_NAMES))
        raw_frame, raw_time, raw_bbox = archive['frame_index'], archive['timestamp_sec'], archive['bbox_xyxy']
        require(raw_frame.shape == () and np.issubdtype(raw_frame.dtype, np.integer)
                and int(raw_frame) == frame, 'raw_frame_mismatch')
        require(raw_time.shape == () and float(raw_time) == timestamp == item['captureTimeSec'], 'raw_timestamp_mismatch')
        bbox = np.asarray(item['bboxXYXY'], dtype=float)
        require(bbox.shape == (4,) and np.isfinite(bbox).all() and raw_bbox.shape == (4,)
                and np.array_equal(raw_bbox, bbox), 'raw_bbox_mismatch')
        require(0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3], 'invalid_bbox')
        return arrays


def load_report(path, expected_color):
    path = existing_local_file(path, 'report_unavailable')
    report = read_json(path)
    require(report.get('schemaVersion') == 'pose3d-raw-probe-0.1.0'
            and report.get('status') == 'raw_evidence_collected'
            and report.get('evaluationMode') is True and report.get('releaseReady') is False
            and report.get('productCompatible') is False
            and report.get('coordinateStatus') == 'unverified-raw-mixed-scales', 'invalid_report_status')
    require(report['preprocess']['decodedImageColor'] == 'bgr'
            and report['preprocess']['passedToLibraryColor'] == expected_color, 'unexpected_color_order')
    expected_conversion = 'none' if expected_color == 'bgr' else 'BGR-to-RGB before RTMPose3d'
    require(report['preprocess']['colorConversion'] == expected_conversion, 'unexpected_color_conversion')
    source, model, auth = report['input'], report['model'], report['authorization']
    require(source['kind'] == 'capture-proxy' and source['clockSource'] == 'existing-2d-capture-timestamps', 'invalid_clock')
    verified_file(model['path'], model['weightsSha256'], 'model')
    verified_file(source['path'], source['sha256'], 'proxy')
    verified_file(source['timestampsPath'], source['timestampsSha256'], 'timestamps')
    verified_file(source['keypoints2dPath'], source['keypoints2dSha256'], 'keypoints2d')
    verified_file(auth['evidencePath'], auth['evidenceSha256'], 'authorization')
    authorization = authorize(auth['evidencePath'], source['sha256'])
    require(authorization == auth, 'authorization_mismatch')
    _, times = load_timestamps(source['timestampsPath'])
    require(type(source['frameCount']) is int and source['frameCount'] == len(times), 'frame_count_mismatch')
    _, baseline = load_keypoints(source['keypoints2dPath'], len(times))
    frames = report['frames']
    require(isinstance(frames, list) and 1 <= len(frames) <= MAX_PROBE_FRAMES, 'invalid_frames')
    indexed = {}
    for item in frames:
        frame = item['frame']
        require(type(frame) is int and 0 <= frame < len(times) and frame not in indexed, 'invalid_frames')
        finite_nonnegative(item['inferenceSec'], 'inferenceSec')
        require(item['inferenceTimingIncludes'] == ['preprocess', 'onnx-session', 'SimCC-postprocess'], 'timing_scope_mismatch')
        indexed[frame] = (item, load_raw(path.parent, item, float(times[frame])))
    for name in ('elapsedSec', 'modelLoadSec', 'decodeSec', 'peakProcessRamBytes'):
        finite_nonnegative(report[name], name)
    return path, report, times, baseline, indexed


def mean(values):
    return float(np.mean(values)) if len(values) else None


def compare_reports(bgr_path, rgb_path, size_provider=video_size):
    bgr = load_report(bgr_path, 'bgr')
    rgb = load_report(rgb_path, 'rgb')
    left, right = bgr[1], rgb[1]
    for name in ('weightsSha256', 'implementation', 'sourceFileSha256', 'apiVersion', 'modelInputSize',
                 'backend', 'requestedDevice', 'executionProviders', 'zRange', 'upstreamDecode', 'onnxInputs', 'onnxOutputs'):
        require(name in left['model'] and left['model'][name] == right['model'].get(name), 'model_or_runtime_mismatch', name)
    for name in ('sha256', 'timestampsSha256', 'keypoints2dSha256', 'frameCount'):
        require(left['input'][name] == right['input'][name], 'input_evidence_mismatch', name)
    require(left['dependencyVersions'] == right['dependencyVersions'] and bool(left['dependencyVersions']), 'dependency_mismatch')
    require(all(left['authorization'][key] == right['authorization'][key]
                for key in ('matchingVideoId', 'originalSha256')), 'authorization_mismatch')
    require(left['bboxPolicy'] == right['bboxPolicy'], 'bbox_policy_mismatch')
    require(set(bgr[4]) == set(rgb[4]), 'frame_set_mismatch')
    require(np.array_equal(bgr[2], rgb[2]), 'timestamp_mismatch')
    width, height = size_provider(Path(left['input']['path']))
    require(type(width) is int and type(height) is int and width > 0 and height > 0, 'invalid_video_size')
    collected = {color: {'distances': [], 'pairedDistances': [], 'scores': [], 'eligible': 0, 'common': 0,
                         'inferenceSec': []} for color in ('bgr', 'rgb')}
    all_z_delta, frame_results = [], []
    for frame in sorted(bgr[4]):
        items = {'bgr': bgr[4][frame], 'rgb': rgb[4][frame]}
        a, b = items['bgr'][0], items['rgb'][0]
        require(a['captureTimeSec'] == b['captureTimeSec'] and a['bboxXYXY'] == b['bboxXYXY'], 'frame_crop_mismatch')
        require(a['bboxXYXY'][2] <= width and a['bboxXYXY'][3] <= height, 'invalid_bbox')
        target = bgr[3][frame, :17]
        eligible = np.isfinite(target).all(axis=1) & (target[:, 2] >= BBOX_CONFIDENCE)
        selected = {color: eligible & (arrays['scores'][0, :17] >= BBOX_CONFIDENCE)
                    for color, (_item, arrays) in items.items()}
        paired = selected['bgr'] & selected['rgb']
        per_frame = {'frame': frame, 'captureTimeSec': a['captureTimeSec'], 'bboxXYXY': a['bboxXYXY'],
                     'eligibleExisting2DBodyJoints': int(eligible.sum()), 'pairedCommonBodyJoints': int(paired.sum()),
                     'runs': {}}
        for color, (item, arrays) in items.items():
            differences = np.linalg.norm(arrays['keypoints_2d'][0, :17] - target[:, :2] * [width, height], axis=1)
            distances, paired_distances = differences[selected[color]], differences[paired]
            scores = arrays['scores'][0]
            accumulator = collected[color]
            accumulator['distances'].extend(distances.tolist())
            accumulator['pairedDistances'].extend(paired_distances.tolist())
            accumulator['scores'].extend(scores.tolist())
            accumulator['eligible'] += int(eligible.sum())
            accumulator['common'] += int(selected[color].sum())
            accumulator['inferenceSec'].append(item['inferenceSec'])
            per_frame['runs'][color] = {'commonBodyJoints': int(selected[color].sum()),
                'coverageOfEligible2DBodyJoints': float(selected[color].sum() / eligible.sum()) if eligible.any() else None,
                'meanImage2DDistancePx': mean(distances), 'meanImage2DDistanceOnPairedJointsPx': mean(paired_distances),
                'meanRawScore133': mean(scores), 'inferenceSec': item['inferenceSec']}
        delta = items['rgb'][1]['keypoints_simcc'][0, :, 2] - items['bgr'][1]['keypoints_simcc'][0, :, 2]
        all_z_delta.extend(delta.tolist())
        per_frame['simccZChangeRgbMinusBgr'] = {'jointCount': len(delta), 'meanSignedDelta': mean(delta),
            'meanAbsoluteDelta': mean(np.abs(delta)), 'maxAbsoluteDelta': float(np.abs(delta).max())}
        frame_results.append(per_frame)
    summary = {}
    for color, run in (('bgr', left), ('rgb', right)):
        acc = collected[color]
        summary[color] = {'eligibleExisting2DBodyJoints': acc['eligible'], 'commonBodyJoints': acc['common'],
            'coverageOfEligible2DBodyJoints': acc['common'] / acc['eligible'] if acc['eligible'] else None,
            'pairedCommonBodyJoints': len(acc['pairedDistances']), 'meanImage2DDistancePx': mean(acc['distances']),
            'meanImage2DDistanceOnPairedJointsPx': mean(acc['pairedDistances']), 'meanRawScore133': mean(acc['scores']),
            'timing': {name: run[name] for name in ('elapsedSec', 'modelLoadSec', 'decodeSec')},
            'inferenceTotalSec': sum(acc['inferenceSec']), 'inferenceMeanSec': mean(acc['inferenceSec']),
            'peakProcessRamBytes': run['peakProcessRamBytes']}
    return {'schemaVersion': 'pose3d-preprocessing-comparison-0.1.0', 'status': 'preprocessing_compared',
        'comparisonKind': 'same-model-bgr-rgb-preprocessing', 'evaluationMode': True, 'releaseReady': False,
        'rawCoordinatesValidated': False, 'winner': None,
        'reports': {color: {'path': str(run[0]), 'sha256': digest(run[0])} for color, run in (('bgr', bgr), ('rgb', rgb))},
        'weightsSha256': left['model']['weightsSha256'], 'proxySha256': left['input']['sha256'],
        'timestampsSha256': left['input']['timestampsSha256'], 'keypoints2dSha256': left['input']['keypoints2dSha256'],
        'modelImplementationSha256': left['model']['sourceFileSha256'], 'imageSize': [width, height],
        'matchingVideoId': left['authorization']['matchingVideoId'], 'frameCountCompared': len(frame_results),
        'confidenceThresholdForImage2DComparison': BBOX_CONFIDENCE, 'bodyJointIndices': [0, 16],
        'summary': summary, 'frames': frame_results,
        'simccZChangeRgbMinusBgr': {'jointCount': len(all_z_delta), 'meanSignedDelta': mean(all_z_delta),
            'meanAbsoluteDelta': mean(np.abs(all_z_delta)), 'maxAbsoluteDelta': float(np.abs(all_z_delta).max()),
            'unit': 'upstream-argmax-bin-divided-by-2', 'isAccuracy': False},
        'limitations': ['preprocessing_comparison_not_model_A_B', 'existing_2d_is_not_ground_truth',
            'lower_image_2d_distance_does_not_establish_depth_accuracy', 'raw_scores_are_not_depth_confidence',
            'simcc_z_difference_is_not_depth_accuracy', 'single_runs_are_not_a_performance_benchmark',
            'no_winner_selected_no_release_or_value_gate_assessed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bgr_report')
    parser.add_argument('rgb_report')
    parser.add_argument('--output', required=True, help='新的报告路径；已有文件不会覆盖')
    args = parser.parse_args()
    try:
        output = ensure_new_output(args.output)
    except OSError as error:
        return output_error(error)
    try:
        report = compare_reports(args.bgr_report, args.rgb_report)
    except Exception as error:
        report = {'status': 'invalid_evidence', 'errorCode': getattr(error, 'code', 'invalid_evidence'),
                  'error': str(error), 'releaseReady': False, 'winner': None}
    try:
        write_report(output, report)
    except (OSError, ValueError, TypeError) as error:
        return output_error(error)
    print(json.dumps({'status': report['status'], 'output': str(output), 'releaseReady': False}))
    return 0 if report['status'] == 'preprocessing_compared' else 2


if __name__ == '__main__':
    sys.exit(main())
