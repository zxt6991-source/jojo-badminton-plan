#!/usr/bin/env python3
"""Compare bounded local research windows, retaining failed-frame denominators."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

from pose3d_camera_research import read_config, recover_camera, pose_metrics, BONES, ELBOWS
from pose3d_decode_audit import verify_authorization, verify_decoder, read_frame, decode_frame, OFFICIAL_Z_RANGE
from pose3d_evaluate import ensure_new_output, write_report, output_error
from pose3d_probe import ProbeError, require, digest, existing_local_file, validate_proxy_clock, validate_bbox
from pose3d_probe_compare import read_json, verified_file
from pose3d_runtime_probe_compare import verify_b_decoder, read_b_frame, decode_b, stats
from pose3d_window_spec import load_window_spec, validate_window_report_selection

VERSION = 'pose3d-window-evaluation-0.1.0'
TERMINAL_STATUSES = {'raw_evidence_collected', 'raw_evidence_collected_with_failures', 'failed'}


def nonnegative(value, code):
    require(type(value) in (int, float) and np.isfinite(value) and value >= 0, code)
    return float(value)


def read_window_report(path, runtime):
    require(path.stat().st_size <= 5_000_000, 'report_size_limit')
    report = read_json(path)
    require(report.get('schemaVersion') == 'pose3d-window-probe-0.1.0'
            and report.get('runtime') == runtime and report.get('status') in TERMINAL_STATUSES,
            'unsupported_window_report')
    require(report.get('evaluationMode') is True and report.get('productCompatible') is False
            and report.get('releaseReady') is False and report.get('rawCoordinatesValidated') is False,
            'unsafe_coordinate_claim')
    return report


def verify_sources(paths, reports, clock_validator):
    timestamps = [verify_authorization(report) for report in reports]
    a, b = reports
    for key in ('sha256', 'timestampsSha256', 'keypoints2dSha256', 'frameCount'):
        require(a['input'].get(key) == b['input'].get(key), 'input_evidence_mismatch', key)
    for key in ('evidenceSha256', 'matchingVideoId', 'originalSha256', 'proxySha256'):
        require(a['authorization'][key] == b['authorization'][key], 'authorization_mismatch', key)
    require(np.array_equal(*timestamps), 'timestamp_mismatch')
    clock = (clock_validator or validate_proxy_clock)(Path(a['input']['path']), timestamps[0])
    require(all(report['input'].get('proxyClockValidation') == clock for report in reports),
            'proxy_clock_metadata_mismatch')
    selections = [validate_window_report_selection(report, timestamps[0]) for report in reports]
    require(selections[0] == selections[1], 'window_selection_mismatch')
    for key in ('origin', 'confidenceThreshold', 'extentPaddingRatio', 'detectorUsed', 'identityValidation'):
        require(key in a.get('bboxPolicy', {}) and a['bboxPolicy'][key] == b.get('bboxPolicy', {}).get(key),
                'bbox_policy_mismatch', key)
    for path, report in zip(paths, reports):
        verified_file(report['model'].get('path'), report['model'].get('weightsSha256'), 'model')
        source = report['input']
        if source.get('keypoints2dSha256') is not None:
            verified_file(source.get('keypoints2dPath'), source['keypoints2dSha256'], 'keypoints2d')
        for window in report['windows']:
            directory = path.parent / window['id']
            require(directory.resolve().parent == path.parent.resolve(), 'window_path_escape')
        spec_path = report.get('windowSpecPath')
        spec_sha = report.get('windowSpecSha256')
        require(spec_path is not None and spec_sha is not None, 'window_spec_evidence_missing')
        verified_file(spec_path, spec_sha, 'window_spec')
        require(load_window_spec(spec_path, timestamps[0]) == selections[0], 'window_spec_selection_mismatch')
    # A completed zero-success run still contributes its requested/failed frames.
    # There are no decoder outputs to validate for that run.
    for index, report in enumerate(reports):
        successful = any(frame['status'] == 'succeeded' for window in report['windows'] for frame in window['frames'])
        if not successful:
            continue
        if index == 0:
            require(verify_decoder(report) == 144, 'unsupported_decoder')
            require(report.get('preprocess', {}).get('passedToLibraryColor') == 'rgb'
                    and report['preprocess'].get('decodedImageColor') == 'bgr'
                    and report['preprocess'].get('colorConversion') == 'BGR-to-RGB before RTMPose3d',
                    'unsupported_a_preprocess')
        else:
            verify_b_decoder(report['model'])
    return timestamps[0], clock, selections[0]


def frame_evidence(directory, item, runtime, times, clock, camera):
    frame = item['frame']
    require(type(item.get('captureTimeSec')) in (int, float)
            and item['captureTimeSec'] == float(times[frame]), 'capture_timestamp_mismatch')
    bbox = item.get('bboxXYXY')
    if bbox is not None:
        validate_bbox(bbox, clock['width'], clock['height'])
    attempted = item.get('inferenceAttempted')
    elapsed = item.get('inferenceSec')
    require(type(attempted) is bool, 'inference_attempt_missing')
    if attempted:
        nonnegative(elapsed, 'invalid_inference_timing')
    else:
        require(elapsed is None, 'unexpected_inference_timing')
    result = {'status': item['status'], 'errorCode': item.get('errorCode'),
              'inferenceAttempted': attempted, 'inferenceSec': elapsed, 'cameraRecovery': 'not_available',
              'cameraErrorCode': None, 'decodedBodyJoints': 0,
              'metrics': {'elbows': {side: {'angleDeg': None, 'reason': 'raw_frame_failed'} for side in ELBOWS},
                          'boneLengthsModelRelative': {bone: None for bone in BONES}}}
    if item['status'] == 'failed':
        require(all(item.get(key) is None for key in ('rawPath', 'rawSha256', 'outputShapes')),
                'failed_frame_has_raw_evidence')
        return result, None
    require(attempted and bbox is not None, 'successful_frame_missing_input')
    loader = read_frame if runtime == 'rtmlib-onnx' else read_b_frame
    arrays = loader(directory, item, len(times))
    require(np.array_equal(arrays['bbox_xyxy'], bbox), 'raw_bbox_mismatch')
    if runtime == 'rtmlib-onnx':
        decoded = decode_frame(arrays, 144)
        z, mask = decoded['official_codec_relative_z'][0], decoded['decoded_point_mask'][0].copy()
    else:
        decode_b(arrays)  # independently check all 133 decoded values/formulas
        z = (arrays['keypoints_simcc'][0, :, 2].astype(float) / 144 - 1) * OFFICIAL_Z_RANGE
        mask = arrays['scores'][0] > 0
    mask[17:] = False
    data = {'xy': arrays['keypoints_2d'][0], 'z': z, 'valid': mask, 'recovered': None, 'metrics': None}
    result.update(rawSha256=item['rawSha256'], decodedBodyJoints=int(mask[:17].sum()))
    try:
        recovered, error = recover_camera(data['xy'], z, mask, camera)
        data['recovered'] = recovered['camera_root_relative_xyz']
        data['metrics'] = pose_metrics(data['recovered'], mask)
        result.update(cameraRecovery='recovered_conditional_on_parameters',
                      projectionRoundtripMaxPixelError=error, metrics=data['metrics'])
    except ProbeError as error:
        # Missing hips/nonpositive assumed depth are missing research geometry,
        # not model execution failures and never fabricated as zero coordinates.
        result.update(cameraRecovery='not_measurable', cameraErrorCode=error.code)
        for elbow in result['metrics']['elbows'].values():
            elbow['reason'] = error.code
    return result, data


def bone_variation(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    mean = float(values.mean()) if values.size else None
    usable = values.size and mean > 1e-12
    return {'sampleCount': len(values), 'meanModelRelativeLength': mean,
        'coefficientOfVariation': float(values.std() / mean) if usable else None,
        'spanOverMean': float((values.max() - values.min()) / mean) if usable else None,
        'maxAbsoluteChangeFromFirstOverFirst': float(np.abs(values - values[0]).max() / values[0])
            if values.size and values[0] > 1e-12 else None}


def temporal_pair(previous, current, previous_index, index, dt):
    output = {'fromFrame': previous_index, 'toFrame': index, 'dtProxySec': dt, 'available': False,
              'reason': 'raw_frame_failed', 'commonBodyJoints': 0}
    require(index == previous_index + 1 and np.isfinite(dt) and dt > 0, 'nonadjacent_or_invalid_dt')
    if previous is None or current is None:
        return output, [], [], {side: [] for side in ELBOWS}, {side: [] for side in ELBOWS}
    if previous['recovered'] is None or current['recovered'] is None:
        output['reason'] = 'camera_geometry_unavailable'
        return output, [], [], {side: [] for side in ELBOWS}, {side: [] for side in ELBOWS}
    common = (previous['valid'] & current['valid'])[:17]
    displacement = np.linalg.norm(current['recovered'][:17][common] - previous['recovered'][:17][common], axis=1)
    rates = displacement / dt
    output.update(available=True, reason=None, commonBodyJoints=int(common.sum()),
        rootRelativeCoordinateDisplacement=stats(displacement),
        rootRelativeCoordinateChangePerProxySec=stats(rates), elbows={})
    angle_changes, angle_rates = {}, {}
    for side in ELBOWS:
        before = previous['metrics']['elbows'][side]['angleDeg']
        after = current['metrics']['elbows'][side]['angleDeg']
        change = abs(after - before) if before is not None and after is not None else None
        angle_changes[side] = [] if change is None else [change]
        angle_rates[side] = [] if change is None else [change / dt]
        output['elbows'][side] = {'absoluteAngleChangeDeg': change,
                                 'absoluteAngleChangeDegPerProxySec': change / dt if change is not None else None}
    return output, displacement.tolist(), rates.tolist(), angle_changes, angle_rates


def run_window(selection, windows, paths, reports, times, clock, camera):
    series, rows = {'a': [], 'b': []}, []
    all_xy, all_z = [], []
    for item_a, item_b in zip(windows[0]['frames'], windows[1]['frames']):
        # An unprocessed frame has no crop; keep it in the denominator when the
        # other runtime reached this frame. Every attempted crop must still match.
        for item in (item_a, item_b):
            require(item.get('bboxXYXY') is not None or
                    (item['status'] == 'failed' and item.get('inferenceAttempted') is False),
                    'attempted_frame_missing_bbox')
        if item_a['bboxXYXY'] is not None and item_b['bboxXYXY'] is not None:
            require(item_a['bboxXYXY'] == item_b['bboxXYXY'], 'frame_crop_mismatch')
        frame = item_a['frame']
        require(frame == item_b['frame'], 'frame_pair_mismatch')
        row = {'frame': frame, 'captureTimeSec': float(times[frame]),
               'bboxXYXY': item_a['bboxXYXY'] or item_b['bboxXYXY'], 'runs': {}}
        for label, item, path, report in zip(('a', 'b'), (item_a, item_b), paths, reports):
            state, data = frame_evidence(path.parent / selection['id'], item, report['runtime'], times, clock, camera)
            series[label].append(data)
            row['runs'][label] = state
        a, b = series['a'][-1], series['b'][-1]
        xy, z = [], []
        if a is not None and b is not None:
            common = (a['valid'] & b['valid'])[:17]
            xy = np.linalg.norm(a['xy'][:17][common].astype(float) - b['xy'][:17][common], axis=1).tolist()
            if common[11] and common[12]:
                za = a['z'][:17] - a['z'][[11, 12]].mean()
                zb = b['z'][:17] - b['z'][[11, 12]].mean()
                z = (zb[common] - za[common]).tolist()
        all_xy.extend(xy)
        all_z.extend(z)
        row.update(imageXYDistancePx=stats(xy), rootCenteredRelativeZDifference=stats(z, True))
        rows.append(row)
    runs = {}
    for label in ('a', 'b'):
        states = [row['runs'][label] for row in rows]
        failed = [state for state in states if state['status'] == 'failed']
        inferred = [state['inferenceSec'] for state in states if state['inferenceAttempted']]
        recoveries = [data for data in series[label] if data is not None and data['recovered'] is not None]
        pairs, displacement, rates = [], [], []
        angle_changes, angle_rates = {side: [] for side in ELBOWS}, {side: [] for side in ELBOWS}
        for pos in range(1, len(rows)):
            previous, current = rows[pos - 1], rows[pos]
            pair, changes, per_sec, angles, angles_per_sec = temporal_pair(
                series[label][pos - 1], series[label][pos], previous['frame'], current['frame'],
                current['captureTimeSec'] - previous['captureTimeSec'])
            pairs.append(pair)
            displacement.extend(changes)
            rates.extend(per_sec)
            for side in ELBOWS:
                angle_changes[side].extend(angles[side])
                angle_rates[side].extend(angles_per_sec[side])
        runs[label] = {'requestedFrames': len(rows), 'successfulFrames': len(rows) - len(failed),
            'failedFrames': len(failed), 'successCoverage': (len(rows) - len(failed)) / len(rows),
            'failureReasons': dict(Counter(state['errorCode'] for state in failed)),
            'cameraRecoveredFrames': len(recoveries), 'cameraRecoveryCoverage': len(recoveries) / len(rows),
            'cameraFailureReasons': dict(Counter(state['cameraErrorCode'] for state in states if state['cameraErrorCode'])),
            'inference': {'attemptedFrames': len(inferred), 'unattemptedFrames': len(rows) - len(inferred),
                'totalSec': sum(inferred), 'meanSecPerAttempt': float(np.mean(inferred)) if inferred else None,
                'includesFailedAttempts': True, 'scope': 'recorded_model_call_wall_time'},
            'boneLengthVariation': {bone: bone_variation([data['metrics']['boneLengthsModelRelative'][bone]
                                                         for data in recoveries]) for bone in BONES},
            'temporal': {'candidateAdjacentPairs': max(0, len(rows) - 1),
                'availableAdjacentPairs': sum(pair['available'] for pair in pairs),
                'rootRelativeCoordinateDisplacement': stats(displacement),
                'rootRelativeCoordinateChangePerProxySec': stats(rates),
                'elbows': {side: {'absoluteAngleChangeDeg': stats(angle_changes[side]),
                                 'absoluteAngleChangeDegPerProxySec': stats(angle_rates[side])} for side in ELBOWS},
                'pairs': pairs}}
    return {**selection, 'runs': runs, 'frames': rows,
        'commonSuccessfulFrames': sum(row['runs']['a']['status'] == row['runs']['b']['status'] == 'succeeded' for row in rows),
        'imageXYDistancePx': stats(all_xy), 'rootCenteredRelativeZDifference': stats(all_z, True)}


def evaluate_windows(a_path, b_path, config_path, scenario_id, clock_validator=None):
    paths = [existing_local_file(path, 'report_unavailable') for path in (a_path, b_path)]
    reports = [read_window_report(path, runtime) for path, runtime in zip(paths, ('rtmlib-onnx', 'mmpose'))]
    times, clock, selections = verify_sources(paths, reports, clock_validator)
    config_path = existing_local_file(config_path, 'camera_config_required')
    cameras = read_config(config_path)['scenarios']
    matches = [camera for camera in cameras if camera['id'] == scenario_id]
    require(len(matches) == 1, 'camera_scenario_missing')
    camera = matches[0]
    require([camera['imageWidth'], camera['imageHeight']] == [clock['width'], clock['height']],
            'camera_image_dimensions_mismatch')
    windows = [run_window(selection, pair, paths, reports, times, clock, camera)
               for selection, pair in zip(selections, zip(reports[0]['windows'], reports[1]['windows']))]
    requested = sum(len(selection['frameIndices']) for selection in selections)
    summary = {'windowCount': len(windows), 'requestedFramesPerRun': requested, 'runs': {}}
    for label, report in zip(('a', 'b'), reports):
        success = sum(window['runs'][label]['successfulFrames'] for window in windows)
        errors = Counter()
        for window in windows:
            errors.update(window['runs'][label]['failureReasons'])
        summary['runs'][label] = {'successfulFrames': success, 'failedFrames': requested - success,
            'successCoverage': success / requested, 'failureReasons': dict(errors),
            'inferenceAttemptedFrames': sum(window['runs'][label]['inference']['attemptedFrames'] for window in windows),
            'inferenceTotalSec': sum(window['runs'][label]['inference']['totalSec'] for window in windows),
            'recordedRunTiming': {key: nonnegative(report[key], 'invalid_run_timing')
                                  for key in ('elapsedSec', 'modelLoadSec', 'decodeSec') if key in report},
            'decoderValidated': success > 0}
    return {'schemaVersion': VERSION, 'status': 'research_windows_evaluated', 'purpose': 'local-research',
        'evaluationMode': True, 'productCompatible': False, 'releaseReady': False, 'rawCoordinatesValidated': False,
        'accuracy': {'status': 'not_measurable', 'reason': 'no_independent_3d_ground_truth'},
        'physicalMotionMeasured': False, 'cameraCoordinateUnit': 'model-relative; not measured meters',
        'camera': camera, 'cameraConfigPath': str(config_path), 'cameraConfigSha256': digest(config_path),
        'reports': {label: {'path': str(path), 'sha256': digest(path), 'runtime': report['runtime'],
                           'status': report['status'], 'model': report['model'],
                           'windowSpecPath': report['windowSpecPath'], 'windowSpecSha256': report['windowSpecSha256']}
                    for label, path, report in zip(('a', 'b'), paths, reports)},
        'authorization': reports[0]['authorization'], 'input': reports[0]['input'],
        'clockEvidence': {'independentProxyValidation': clock, 'bothRecordedValidationsMatch': True,
            'originalCaptureTimingVerified': False, 'sourceEditingHistoryVerified': False},
        'measurementPolicy': {'bodyJointIndices': list(range(17)), 'decodedAvailability': 'raw score > 0',
            'zDecodeDenominator': 144, 'rootIndices': [11, 12], 'imageXYUnit': 'proxy pixels',
            'cameraRecovery': 'same explicit camera parameters for both runs; default B camera XYZ unused',
            'timeUnit': 'actual proxy PTS seconds', 'adjacency': 'consecutive frame indices within one window only',
            'gaps': 'do not bridge failed frames, unavailable camera geometry, or window boundaries',
            'boneVariation': 'within-window available samples; relative to mean and first available sample',
            'rates': 'model-relative coordinate or angle changes per proxy second, not physical velocity'},
        'summary': summary, 'windows': windows,
        'limitations': ['camera_parameters_are_explicit_assumptions_not_independently_verified',
            'model_agreement_and_temporal_smoothness_are_not_accuracy', 'no_measured_world_scale_or_physical_speed',
            'different_models_and_runtimes_do_not_isolate_a_runtime_effect',
            'single_run_timing_is_not_a_benchmark', 'scores_are_not_calibrated_depth_confidence',
            'view_identity_and_depth_have_no_independent_annotations',
            'no_winner_selection_no_release_or_value_gate_assessed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rtmlib_report')
    parser.add_argument('mmpose_report')
    parser.add_argument('--camera-config', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--output', required=True, help='New report path; existing evidence is never overwritten')
    args = parser.parse_args()
    try:
        output = ensure_new_output(args.output)
    except OSError as error:
        return output_error(error)
    try:
        report = evaluate_windows(args.rtmlib_report, args.mmpose_report, args.camera_config, args.scenario)
    except Exception as error:
        report = {'schemaVersion': VERSION, 'status': 'invalid_evidence',
                  'errorCode': getattr(error, 'code', 'invalid_evidence'), 'error': str(error),
                  'evaluationMode': True, 'productCompatible': False, 'releaseReady': False}
    try:
        write_report(output, report)
    except (OSError, ValueError, TypeError) as error:
        return output_error(error)
    print(json.dumps({'status': report['status'], 'errorCode': report.get('errorCode'),
                      'output': str(output), 'releaseReady': False}))
    return 0 if report['status'] == 'research_windows_evaluated' else 2


if __name__ == '__main__':
    sys.exit(main())
