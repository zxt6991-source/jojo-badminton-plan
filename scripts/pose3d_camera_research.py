#!/usr/bin/env python3
"""Explicit-camera research backprojection. Never publishes product pose3d.f32."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

from pose3d_decode_audit import (read_report, verify_authorization, verify_decoder,
                                read_frame, decode_frame, finite_number)
from pose3d_probe import ProbeError, require, digest, MAX_PROBE_FRAMES, validate_proxy_clock

CONFIG_SCHEMA = 'pose3d-camera-research-config-0.1.0'
DISPLAY_MATRIX = np.diag([1., -1., -1.])
BONES = {'leftUpperArm': (5, 7), 'leftForearm': (7, 9),
         'rightUpperArm': (6, 8), 'rightForearm': (8, 10),
         'leftThigh': (11, 13), 'leftShin': (13, 15),
         'rightThigh': (12, 14), 'rightShin': (14, 16),
         'shoulders': (5, 6), 'hips': (11, 12)}
ELBOWS = {'left': (5, 7, 9), 'right': (6, 8, 10)}
SCENARIO_FIELDS = {'id', 'source', 'evidence', 'imageWidth', 'imageHeight',
                   'fx', 'fy', 'cx', 'cy', 'rootDepth'}


def read_config(path):
    require(path.is_file() and path.stat().st_size <= 1_000_000, 'camera_config_required')
    try:
        config = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise ProbeError('invalid_camera_config', str(error)) from error
    require(isinstance(config, dict) and config.get('schemaVersion') == CONFIG_SCHEMA,
            'unsupported_camera_config')
    scenarios = config.get('scenarios')
    require(isinstance(scenarios, list) and 1 <= len(scenarios) <= 12, 'invalid_camera_scenarios')
    identifiers = set()
    for scenario in scenarios:
        require(isinstance(scenario, dict) and set(scenario) == SCENARIO_FIELDS,
                'camera_parameters_required', '每个情景须显式提供全部参数，不接受隐式参数或未支持字段')
        name = scenario['id']
        require(isinstance(name, str) and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}', name)
                and name not in identifiers, 'invalid_scenario_id')
        identifiers.add(name)
        require(scenario['source'] in ('assumed-research', 'calibrated')
                and isinstance(scenario['evidence'], str) and scenario['evidence'].strip(),
                'camera_evidence_required')
        require(all(type(scenario[k]) is int and 1 <= scenario[k] <= 8192
                    for k in ('imageWidth', 'imageHeight'))
                and scenario['imageWidth'] * scenario['imageHeight'] <= 3840 * 2160,
                'invalid_image_dimensions')
        require(all(finite_number(scenario[k]) for k in ('fx', 'fy', 'cx', 'cy', 'rootDepth'))
                and scenario['fx'] > 0 and scenario['fy'] > 0 and scenario['rootDepth'] > 0,
                'invalid_camera_parameters')
    return config


def video_dimensions(path):
    """Read the hash-verified capture proxy, without rerunning model inference."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        require(capture.isOpened(), 'proxy_decode_failed')
        success, image = capture.read()
        require(success and isinstance(image, np.ndarray) and image.ndim == 3, 'proxy_decode_failed')
        return image.shape[1], image.shape[0]
    finally:
        capture.release()


def project(camera_xyz, scenario):
    """Pinhole forward projection, for mathematical roundtrip checks only."""
    points = np.asarray(camera_xyz, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            and (points[:, 2] > 0).all(), 'invalid_projection_points')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        image = points[:, :2] / points[:, 2:] * [scenario['fx'], scenario['fy']] + [scenario['cx'], scenario['cy']]
    require(np.isfinite(image).all(), 'nonfinite_projection')
    return image


def recover_camera(image_xy, relative_z, valid, scenario):
    """Same-unit XYZ conditional on explicitly supplied intrinsics/root depth."""
    image_xy, relative_z = np.asarray(image_xy, dtype=np.float64), np.asarray(relative_z, dtype=np.float64)
    valid = np.asarray(valid)
    require(image_xy.shape == (133, 2) and relative_z.shape == (133,)
            and valid.shape == (133,) and valid.dtype == np.bool_, 'invalid_recovery_arrays')
    require(valid[11] and valid[12], 'root_hips_missing')
    require(np.isfinite(image_xy[valid]).all() and np.isfinite(relative_z[valid]).all(), 'nonfinite_recovery_input')
    depth = relative_z[valid] + scenario['rootDepth']
    require(np.isfinite(depth).all() and (depth > 0).all(), 'nonpositive_camera_depth',
            '显式根深度加预测相对 Z 后必须为正；不自动平移或夹紧深度')
    absolute = np.full((133, 3), np.nan, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        absolute[valid, :2] = ((image_xy[valid] - [scenario['cx'], scenario['cy']])
                               / [scenario['fx'], scenario['fy']] * depth[:, None])
    absolute[valid, 2] = depth
    require(np.isfinite(absolute[valid]).all(), 'nonfinite_camera_reconstruction')
    with np.errstate(over='ignore', invalid='ignore'):
        root = absolute[[11, 12]].mean(axis=0)
        relative = absolute - root
        display = relative @ DISPLAY_MATRIX.T
    require(np.isfinite(relative[valid]).all(), 'nonfinite_camera_reconstruction')
    roundtrip = project(absolute[valid], scenario)
    error = float(np.abs(roundtrip - image_xy[valid]).max())
    require(np.isfinite(error) and error <= 1e-7, 'projection_roundtrip_failed')
    return {'camera_xyz': absolute, 'camera_root_xyz': root,
            'camera_root_relative_xyz': relative, 'display_root_relative_xyz': display,
            'decoded_point_mask': valid.copy()}, error


def angle_degrees(points):
    a, b, c = points
    u, v = a - b, c - b
    product = np.linalg.norm(u) * np.linalg.norm(v)
    if not np.isfinite(product) or product <= 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / product, -1, 1))))


def pose_metrics(points, valid):
    lengths, elbows = {}, {}
    for label, indices in BONES.items():
        with np.errstate(over='ignore', invalid='ignore'):
            length = float(np.linalg.norm(points[indices[0]] - points[indices[1]])) if valid[list(indices)].all() else None
        require(length is None or np.isfinite(length), 'nonfinite_research_metric')
        lengths[label] = length
    for label, indices in ELBOWS.items():
        available = bool(valid[list(indices)].all())
        angle = angle_degrees(points[list(indices)]) if available else None
        elbows[label] = {'angleDeg': angle, 'reason': None if angle is not None else
                         ('degenerate_segments' if available else 'joint_missing')}
    return {'elbows': elbows, 'boneLengthsModelRelative': lengths}


def spread(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'sampleCount': 0, 'min': None, 'max': None, 'span': None, 'spanOverMean': None}
    low, high, mean = min(values), max(values), float(np.mean(values))
    return {'sampleCount': len(values), 'min': low, 'max': high, 'span': high - low,
            'spanOverMean': (high - low) / mean if mean > 0 else None}


def run_research(report_path, config_path, output_dir, image_size_reader=None, clock_validator=None):
    path, config_path = Path(report_path).expanduser().resolve(), Path(config_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'output_not_empty')
    output.mkdir(parents=True, exist_ok=True)
    # Two concurrent invocations may both observe an empty folder. Reserve it
    # exclusively before writing any arrays; preserve the marker as evidence.
    try:
        with (output / '.camera-research-reserved').open('x') as reservation:
            reservation.write('This research output directory has been reserved.\n')
    except FileExistsError as error:
        raise ProbeError('output_not_empty', '研究输出目录已被另一个运行占用') from error
    result = {'schemaVersion': 'pose3d-camera-research-0.1.0', 'status': 'invalid_evidence',
              'evaluationMode': True, 'releaseReady': False, 'productCompatible': False,
              'rawCoordinatesValidated': False, 'cameraCoordinatesValidated': False,
              'coordinateStatus': 'research-camera-xyz-conditional-on-explicit-parameters',
              'pointSpec': 'coco-wholebody-133', 'unit': 'model-relative', 'rootJoint': 'mid-hip',
              'cameraAxes': 'x-right-y-down-z-away-from-camera',
              'displayAxes': 'x-right-y-up-z-toward-camera', 'displayMatrix': DISPLAY_MATRIX.tolist(),
              'scenarios': [], 'parameterSensitivity': [],
              'limitations': ['not_validated_motion_capture', 'no_measured_world_scale',
                              'camera_config_evidence_not_independently_verified',
                              'depth_and_identity_not_independently_validated',
                              'positive_raw_scores_only_not_calibrated_3d_confidence',
                              'projection_roundtrip_checks_arithmetic_not_prediction_accuracy',
                              'sparse_sample_variation_not_motion_quality_or_injury_risk',
                              'proxy_pts_agreement_does_not_verify_original_capture_fps_or_editing_history']}
    try:
        config = read_config(config_path)
        report = read_report(path)
        timestamps = verify_authorization(report)
        denominator = verify_decoder(report)
        clock = (clock_validator or validate_proxy_clock)(Path(report['input']['path']), timestamps)
        recorded_clock = report['input'].get('proxyClockValidation')
        if recorded_clock is not None:
            require(recorded_clock == clock, 'proxy_clock_metadata_mismatch')
        dimensions = (image_size_reader or video_dimensions)(report['input']['path'])
        require(isinstance(dimensions, (tuple, list)) and len(dimensions) == 2
                and all(type(v) is int and v > 0 for v in dimensions), 'invalid_proxy_dimensions')
        require(tuple(dimensions) == (clock['width'], clock['height']), 'proxy_clock_dimensions_mismatch')
        require(all((s['imageWidth'], s['imageHeight']) == tuple(dimensions) for s in config['scenarios']),
                'camera_image_dimensions_mismatch', '相机参数必须对应已授权代理视频的实际像素尺寸')
        items = report.get('frames')
        require(isinstance(items, list) and 1 <= len(items) <= MAX_PROBE_FRAMES, 'invalid_probe_frames')
        evidence, indices = [], []
        for item in items:
            arrays = read_frame(path.parent, item, report['input']['frameCount'])
            require(item['captureTimeSec'] == float(timestamps[item['frame']]), 'capture_timestamp_mismatch')
            x1, y1, x2, y2 = arrays['bbox_xyxy']
            require(0 <= x1 < x2 <= dimensions[0] and 0 <= y1 < y2 <= dimensions[1], 'bbox_outside_camera_image')
            evidence.append((item, arrays, decode_frame(arrays, denominator)))
            indices.append(item['frame'])
        require(indices == sorted(set(indices)), 'duplicate_or_unsorted_frames')
        prepared = []
        for scenario in config['scenarios']:
            frames = []
            for item, arrays, decoded in evidence:
                recovered, roundtrip_error = recover_camera(arrays['keypoints_2d'][0],
                    decoded['official_codec_relative_z'][0], decoded['decoded_point_mask'][0], scenario)
                metrics = pose_metrics(recovered['camera_root_relative_xyz'], recovered['decoded_point_mask'])
                frames.append((item, arrays, decoded, recovered, metrics, roundtrip_error))
            prepared.append((scenario, frames))
        # Validate every scenario/frame before publishing any research arrays.
        result.update(probeReportSha256=digest(path), cameraConfigSha256=digest(config_path),
                      authorization=report['authorization'], weightsSha256=report['model']['weightsSha256'],
                      inputSha256=report['input']['sha256'], imageDimensions=list(dimensions),
                      decoder={'simccSplitRatio': 2, 'officialZDenominator': denominator},
                      sourceEvidence={'probeReportPath': str(path), 'probeReportSha256': digest(path),
                          'cameraConfigPath': str(config_path), 'cameraConfigSha256': digest(config_path),
                          'proxyPath': report['input']['path'], 'proxySha256': report['input']['sha256'],
                          'timestampsPath': report['input']['timestampsPath'],
                          'timestampsSha256': report['input']['timestampsSha256'],
                          'frameCount': report['input']['frameCount']},
                      clockEvidence={'independentProxyValidation': clock,
                          'rawProbeClockValidationRecorded': recorded_clock is not None,
                          'originalCaptureTimingVerified': False, 'sourceEditingHistoryVerified': False,
                          'scope': 'decoded_proxy_pts_against_existing_2d_timestamps_only'},
                      formulaEvidence='https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/rtmpose3d/pose_estimator.py#L46-L74')
        for scenario, frames in prepared:
            scenario_result = {'camera': scenario, 'frames': [], 'sampledBoneLengthVariation': {}}
            for item, arrays, decoded, recovered, metrics, error in frames:
                name = f"{scenario['id']}.frame-{item['frame']:06d}.camera-research.npz"
                np.savez_compressed(output / name, **recovered, image_xy=arrays['keypoints_2d'][0],
                    scores=arrays['scores'][0], official_relative_z=decoded['official_codec_relative_z'][0],
                    frame_index=arrays['frame_index'], timestamp_sec=arrays['timestamp_sec'])
                scenario_result['frames'].append({'frame': item['frame'], 'captureTimeSec': item['captureTimeSec'],
                    'rawSha256': item['rawSha256'], 'researchPath': name, 'researchSha256': digest(output / name),
                    'decodedPointCount': int(recovered['decoded_point_mask'].sum()),
                    'projectionRoundtripMaxPixelError': error, 'metrics': metrics})
            for bone in BONES:
                scenario_result['sampledBoneLengthVariation'][bone] = spread([
                    f['metrics']['boneLengthsModelRelative'][bone] for f in scenario_result['frames']])
            result['scenarios'].append(scenario_result)
        for position, frame in enumerate(indices):
            metrics = [s['frames'][position]['metrics'] for s in result['scenarios']]
            result['parameterSensitivity'].append({'frame': frame,
                'elbowAngleDegrees': {side: spread([m['elbows'][side]['angleDeg'] for m in metrics]) for side in ELBOWS},
                'boneLengthsModelRelative': {bone: spread([m['boneLengthsModelRelative'][bone] for m in metrics]) for bone in BONES}})
        result['status'] = 'camera_research_recovered'
    except (ProbeError, OSError, ValueError, TypeError, KeyError) as error:
        result.update(errorCode=getattr(error, 'code', 'invalid_evidence'), error=str(error))
    temporary = output / '.camera-research.json.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(output / 'camera-research.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', help='已授权本地 pose3d_probe report.json')
    parser.add_argument('--camera-config', required=True, help='显式相机参数与来源 JSON；无默认参数')
    parser.add_argument('--output-dir', required=True, help='新的空研究目录')
    args = parser.parse_args()
    try:
        result = run_research(args.report, args.camera_config, args.output_dir)
    except (ProbeError, OSError) as error:
        result = {'status': 'invalid_evidence', 'errorCode': getattr(error, 'code', 'io_error'), 'error': str(error)}
    print(json.dumps({'status': result['status'], 'errorCode': result.get('errorCode'),
                      'report': str(Path(args.output_dir).resolve() / 'camera-research.json'), 'releaseReady': False}))
    return 0 if result['status'] == 'camera_research_recovered' else 2


if __name__ == '__main__':
    sys.exit(main())
