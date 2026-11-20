#!/usr/bin/env python3
"""Compare local rtmlib/MMPose raw evidence in image XY and relative Z only.

Different weights and runtimes are allowed and recorded. These differences are
neither accuracy measurements nor an isolated runtime benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

from pose3d_decode_audit import (read_report as read_a_report, verify_authorization,
    verify_decoder, read_frame, decode_frame, sha, OFFICIAL_Z_RANGE, PAIRS)
from pose3d_evaluate import ensure_new_output, write_report, output_error
from pose3d_mmpose_probe import RAW_SHAPES
from pose3d_probe import (ProbeError, require, digest, existing_local_file,
    load_keypoints, validate_proxy_clock, validate_bbox, MAX_PROBE_FRAMES, BBOX_CONFIDENCE)
from pose3d_probe_compare import read_json, verified_file

BODY_COUNT = 17
SIGN_TOLERANCE = 1e-6
SOURCE_FILES = ('rtmpose3d/pose_estimator.py', 'rtmpose3d/rtmw3d_head.py',
                'rtmpose3d/simcc_3d_label.py')


def read_b_report(path):
    require(path.stat().st_size <= 1_000_000, 'invalid_probe_report')
    report = read_json(path)
    require(report.get('schemaVersion') == 'pose3d-mmpose-raw-probe-0.1.0'
            and report.get('status') == 'raw_evidence_collected', 'unsupported_probe_schema_or_status')
    require(report.get('evaluationMode') is True and report.get('releaseReady') is False
            and report.get('productCompatible') is False and report.get('rawCoordinatesValidated') is False
            and report.get('coordinateStatus') == 'official-decoder-and-assumed-camera-unvalidated',
            'unsafe_coordinate_claim')
    return report


def verify_b_decoder(model):
    require(model.get('runtime') == 'mmpose-pytorch' and model.get('modelId') == 'rtmw3d-mmpose-l'
            and model.get('sourceRef') == 'v1.3.2'
            and model.get('dependencyVersions', {}).get('mmpose') == '1.3.2', 'unsupported_b_runtime')
    geometry = model.get('sourceGeometry', {})
    require(geometry.get('codecInputSize') == [288, 384, 288]
            and geometry.get('simccSplitRatio') == 2 and geometry.get('decodeDepthDenominator') == 144
            and geometry.get('zRange') == OFFICIAL_Z_RANGE and geometry.get('useDark') is False
            and geometry.get('rootIndices') == [11, 12]
            and geometry.get('isotropicXYZValidated') is False, 'unsupported_b_decoder')
    require(model.get('preprocess') == {'inputColor': 'bgr', 'modelBgrToRgb': True, 'inputSize': [288, 384]},
            'unsupported_b_preprocess')
    require(model.get('decoderObservation') ==
            'copy official head.decode outputs before camera postprocess; no numerical replacement',
            'unsupported_b_decoder_observation')
    require(model.get('cameraAssumptions', {}).get('calibrated') is False, 'unsupported_camera_claim')
    require(model.get('configOverrides') == {'model.backbone.init_cfg': None, 'model.test_cfg.mode': '3d'},
            'unsupported_b_config_overrides')
    sha(model.get('effectiveConfigSha256'), 'effective_config_hash_missing')
    config = verified_file(model.get('configPath'), model.get('configSha256'), 'config')
    source_root = Path(model.get('sourceRoot', '')).expanduser().resolve() / 'projects/rtmpose3d'
    require(config == source_root / 'configs/rtmw3d-l_8xb64_cocktail14-384x288.py', 'unsupported_b_config')
    hashes = model.get('projectSourceSha256')
    require(isinstance(hashes, dict) and set(hashes) == set(SOURCE_FILES), 'source_hashes_missing')
    for name in SOURCE_FILES:
        verified_file(source_root / name, hashes[name], 'source')


def read_b_frame(directory, item, count):
    require(isinstance(item, dict), 'invalid_frame_metadata')
    frame = item.get('frame')
    require(type(frame) is int and 0 <= frame < count, 'invalid_frame_index')
    name = f'frame-{frame:06d}.mmpose-raw.npz'
    require(item.get('rawPath') == name, 'invalid_raw_path')
    path = directory / name
    require(not path.is_symlink() and path.resolve().parent == directory.resolve(), 'invalid_raw_path')
    verified_file(path, item.get('rawSha256'), 'raw_npz')
    require(path.stat().st_size <= 1_000_000, 'raw_size_limit')
    expected = set(RAW_SHAPES) | {'frame_index', 'timestamp_sec', 'bbox_xyxy'}
    require(item.get('outputShapes') == {name: list(shape) for name, shape in RAW_SHAPES.items()},
            'raw_shape_metadata_mismatch')
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            require(len(entries) == len(expected) and sum(entry.file_size for entry in entries) <= 1_000_000,
                    'raw_size_limit')
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == expected, 'raw_fields_mismatch')
            arrays = {name: archive[name] for name in archive.files}
    except (ValueError, zipfile.BadZipFile, OSError) as error:
        if isinstance(error, ProbeError):
            raise
        raise ProbeError('invalid_raw_archive', str(error)) from error
    for name, shape in RAW_SHAPES.items():
        value = arrays[name]
        require(value.shape == shape and np.issubdtype(value.dtype, np.number)
                and not np.iscomplexobj(value) and np.isfinite(value).all(), 'invalid_raw_array', name)
    index, timestamp, bbox = (arrays[key] for key in ('frame_index', 'timestamp_sec', 'bbox_xyxy'))
    require(index.shape == () and np.issubdtype(index.dtype, np.integer) and int(index) == frame,
            'raw_frame_mismatch')
    require(timestamp.shape == () and np.issubdtype(timestamp.dtype, np.number)
            and not np.iscomplexobj(timestamp) and np.isfinite(timestamp)
            and type(item.get('captureTimeSec')) in (int, float)
            and float(timestamp) == item['captureTimeSec'], 'raw_timestamp_mismatch')
    require(bbox.shape == (4,) and np.issubdtype(bbox.dtype, np.number) and not np.iscomplexobj(bbox)
            and np.isfinite(bbox).all() and isinstance(item.get('bboxXYXY'), list)
            and len(item['bboxXYXY']) == 4 and np.array_equal(bbox, item['bboxXYXY']), 'raw_bbox_mismatch')
    return arrays


def decode_b(arrays):
    q = arrays['keypoints_simcc'].astype(np.float64)
    present = arrays['scores'] > 0
    bins = q * 2
    valid = (bins == np.floor(bins)) & (bins >= 0) & (bins < [576, 768, 576])
    require(valid[present].all() and (q[~present] == -.5).all(), 'invalid_simcc_locations')
    z = (q[..., 2] / 144 - 1) * OFFICIAL_Z_RANGE
    require(np.array_equal(arrays['decoded_keypoints'][..., :2], q[..., :2]), 'b_xy_formula_mismatch')
    require(np.allclose(arrays['decoded_keypoints'][..., 2], z, atol=1e-6, rtol=1e-6),
            'b_z_formula_mismatch')
    return z[0, :BODY_COUNT], present[0, :BODY_COUNT]


def stats(values, signed=False):
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    result = {'count': len(values), 'meanAbsolute': float(absolute.mean()) if values.size else None,
              'medianAbsolute': float(np.median(absolute)) if values.size else None,
              'p95Absolute': float(np.percentile(absolute, 95)) if values.size else None,
              'maxAbsolute': float(absolute.max()) if values.size else None}
    if signed:
        result['meanSignedBMinusA'] = float(values.mean()) if values.size else None
    return result


def sign(value):
    return 0 if abs(value) <= SIGN_TOLERANCE else (1 if value > 0 else -1)


def sign_stats(values):
    comparable = [(a, b) for a, b in values if a != 0 and b != 0]
    agree = sum(a == b for a, b in comparable)
    return {'availableCount': len(values), 'nonTieCount': len(comparable),
            'tieInEitherCount': len(values) - len(comparable), 'agreeCount': agree,
            'disagreeCount': len(comparable) - agree,
            'agreementFraction': agree / len(comparable) if comparable else None}


def compare_reports(a_path, b_path, clock_validator=None):
    paths = [existing_local_file(path, 'report_unavailable') for path in (a_path, b_path)]
    reports = [read_a_report(paths[0]), read_b_report(paths[1])]
    times = [verify_authorization(report) for report in reports]
    a, b = reports
    for key in ('sha256', 'timestampsSha256', 'keypoints2dSha256', 'frameCount'):
        require(a['input'].get(key) == b['input'].get(key), 'input_evidence_mismatch', key)
    for key in ('evidenceSha256', 'matchingVideoId', 'originalSha256', 'proxySha256'):
        require(a['authorization'][key] == b['authorization'][key], 'authorization_mismatch', key)
    require(np.array_equal(*times), 'timestamp_mismatch')
    clock = (clock_validator or validate_proxy_clock)(Path(a['input']['path']), times[0])
    require(all(report['input'].get('proxyClockValidation') == clock for report in reports),
            'proxy_clock_metadata_mismatch')
    for report in reports:
        verified_file(report['model'].get('path'), report['model'].get('weightsSha256'), 'model')
    denominator = verify_decoder(a)
    require(denominator == 144, 'unsupported_decoder')
    verify_b_decoder(b['model'])
    require(a.get('preprocess', {}).get('passedToLibraryColor') == 'rgb'
            and a['preprocess'].get('decodedImageColor') == 'bgr'
            and a['preprocess'].get('colorConversion') == 'BGR-to-RGB before RTMPose3d',
            'unsupported_a_preprocess')
    for key in ('origin', 'confidenceThreshold', 'extentPaddingRatio', 'detectorUsed', 'identityValidation'):
        require(key in a.get('bboxPolicy', {}) and a['bboxPolicy'][key] == b.get('bboxPolicy', {}).get(key),
                'bbox_policy_mismatch', key)
    baseline = None
    for report in reports:
        source = report['input']
        if source.get('keypoints2dSha256') is not None:
            file = verified_file(source.get('keypoints2dPath'), source['keypoints2dSha256'], 'keypoints2d')
            _, baseline = load_keypoints(file, len(times[0]))
        else:
            require(source.get('keypoints2dPath') is None, 'keypoints2d_evidence_missing')
    indexed = []
    for path, report, loader in zip(paths, reports, (read_frame, read_b_frame)):
        frames = report.get('frames')
        require(isinstance(frames, list) and 1 <= len(frames) <= MAX_PROBE_FRAMES, 'invalid_probe_frames')
        current = {}
        for item in frames:
            arrays = loader(path.parent, item, len(times[0]))
            frame = item['frame']
            require(frame not in current, 'duplicate_frames')
            require(item['captureTimeSec'] == float(times[0][frame]), 'capture_timestamp_mismatch')
            validate_bbox(item['bboxXYXY'], clock['width'], clock['height'])
            require(np.array_equal(arrays['bbox_xyxy'], item['bboxXYXY']), 'raw_bbox_mismatch')
            current[frame] = (item, arrays)
        indexed.append(current)
    require(set(indexed[0]) == set(indexed[1]), 'frame_set_mismatch')

    image_values, depth_values, root_signs, pair_signs = [], [], [], []
    baseline_values = {'a': [], 'b': []}
    frame_results = []
    missing_root_frames = 0
    for frame in sorted(indexed[0]):
        (item_a, raw_a), (item_b, raw_b) = (run[frame] for run in indexed)
        require(item_a['bboxXYXY'] == item_b['bboxXYXY'], 'frame_crop_mismatch')
        decoded_a = decode_frame(raw_a, denominator)
        z_a, valid_a = (decoded_a[key][0, :BODY_COUNT] for key in ('official_codec_relative_z', 'decoded_point_mask'))
        z_b, valid_b = decode_b(raw_b)
        common = valid_a & valid_b
        root_available = bool(common[11] and common[12])
        root_a = float(z_a[[11, 12]].mean()) if root_available else None
        root_b = float(z_b[[11, 12]].mean()) if root_available else None
        missing_root_frames += not root_available
        xy_a, xy_b = raw_a['keypoints_2d'][0, :BODY_COUNT], raw_b['keypoints_2d'][0, :BODY_COUNT]
        xy_distances = np.linalg.norm(xy_b.astype(float) - xy_a, axis=1)
        image_values.extend(xy_distances[common].tolist())
        joint_rows, local_z, local_signs, local_pairs = [], [], [], []
        for joint in range(BODY_COUNT):
            available = bool(common[joint])
            row = {'joint': joint, 'commonDecoded': available,
                   'imageXYDistancePx': float(xy_distances[joint]) if available else None,
                   'rootCenteredRelativeZA': None, 'rootCenteredRelativeZB': None,
                   'relativeZBMinusA': None, 'rootRelativeSignA': None, 'rootRelativeSignB': None}
            if available and root_available:
                za, zb = float(z_a[joint] - root_a), float(z_b[joint] - root_b)
                local_z.append(zb - za)
                local_signs.append((sign(za), sign(zb)))
                row.update(rootCenteredRelativeZA=za, rootCenteredRelativeZB=zb,
                           relativeZBMinusA=zb - za, rootRelativeSignA=sign(za), rootRelativeSignB=sign(zb))
            joint_rows.append(row)
        pair_rows = {}
        for label, (left, right) in PAIRS.items():
            available = bool(common[left] and common[right])
            za = float(z_a[left] - z_a[right]) if available else None
            zb = float(z_b[left] - z_b[right]) if available else None
            if available:
                local_pairs.append((sign(za), sign(zb)))
            pair_rows[label] = {'indices': [left, right], 'commonDecoded': available,
                'leftMinusRightRelativeZA': za, 'leftMinusRightRelativeZB': zb,
                'signA': sign(za) if available else None, 'signB': sign(zb) if available else None}
        against_2d = None
        if baseline is not None:
            target = baseline[frame, :BODY_COUNT]
            eligible = np.isfinite(target).all(axis=1) & (target[:, 2] >= BBOX_CONFIDENCE)
            paired = common & eligible
            against_2d = {'pairedBodyJoints': int(paired.sum()), 'isGroundTruth': False}
            for label, xy in (('a', xy_a), ('b', xy_b)):
                distances = np.linalg.norm(xy[paired] - target[paired, :2] * [clock['width'], clock['height']], axis=1)
                baseline_values[label].extend(distances.tolist())
                against_2d[label] = stats(distances)
        depth_values.extend(local_z)
        root_signs.extend(local_signs)
        pair_signs.extend(local_pairs)
        frame_results.append({'frame': frame, 'captureTimeSec': item_a['captureTimeSec'],
            'bboxXYXY': item_a['bboxXYXY'], 'rawSha256': {'a': item_a['rawSha256'], 'b': item_b['rawSha256']},
            'commonDecodedBodyJoints': int(common.sum()), 'missingBodyJointsA': int((~valid_a).sum()),
            'missingBodyJointsB': int((~valid_b).sum()), 'rootAvailableInBoth': root_available,
            'imageXYDistancePx': stats(xy_distances[common]), 'rootCenteredRelativeZDifference': stats(local_z, True),
            'rootRelativeSignAgreement': sign_stats(local_signs), 'leftRightSignAgreement': sign_stats(local_pairs),
            'leftRightPairs': pair_rows, 'existing2DDeviationPx': against_2d, 'joints': joint_rows})
    return {'schemaVersion': 'pose3d-runtime-probe-comparison-0.1.0', 'status': 'raw_outputs_compared',
        'comparisonKind': 'cross-model-cross-runtime-raw-output-difference', 'evaluationMode': True,
        'releaseReady': False, 'productCompatible': False, 'rawCoordinatesValidated': False,
        'accuracyMeasured': False, 'metricDepthMeasured': False, 'cameraXYZCompared': False,
        'reports': {label: {'path': str(path), 'sha256': digest(path), 'model': report['model']}
                    for label, path, report in zip(('a', 'b'), paths, reports)},
        'authorization': a['authorization'], 'input': a['input'],
        'clockEvidence': {'independentProxyValidation': clock, 'bothRecordedValidationsMatch': True,
                          'originalCaptureTimingVerified': False, 'sourceEditingHistoryVerified': False},
        'decoder': {'simccSplitRatio': 2, 'axisBins': [576, 768, 576], 'zRange': OFFICIAL_Z_RANGE,
            'aRecordedZDenominator': 192, 'comparisonZDenominator': 144,
            'formula': '(keypoints_simcc_z / 144 - 1) * zRange; then subtract mean of hips 11 and 12',
            'relativeZUnit': 'model-relative-depth-unit; not measured meters',
            'signConvention': '-1 nearer / +1 farther in model camera direction; 0 tie',
            'signTieTolerance': SIGN_TOLERANCE, 'isAccuracy': False},
        'selection': {'bodyJointIndices': list(range(BODY_COUNT)), 'decodedAvailability': 'raw score > 0',
            'scoresAreCalibratedAcrossModels': False, 'rootIndices': [11, 12],
            'existing2DThreshold': BBOX_CONFIDENCE, 'existing2DComparisonUsesSamePairedJoints': True},
        'summary': {'frameCount': len(frame_results), 'framesWithoutCommonHipRoot': missing_root_frames,
            'imageXYDistancePx': stats(image_values), 'rootCenteredRelativeZDifference': stats(depth_values, True),
            'rootRelativeSignAgreement': sign_stats(root_signs), 'leftRightSignAgreement': sign_stats(pair_signs),
            'existing2DDeviationPx': {key: stats(value) for key, value in baseline_values.items()} if baseline is not None else None},
        'frames': frame_results,
        'limitations': ['different_model_sizes_weights_and_runtimes_do_not_isolate_one_cause',
            'model_agreement_is_not_accuracy', 'existing_2d_is_not_ground_truth',
            'relative_z_is_not_measured_metric_depth', 'camera_xyz_and_mixed_scale_xyz_are_not_compared',
            'raw_scores_indicate_availability_not_calibrated_depth_confidence',
            'joint_identity_and_near_far_order_have_no_independent_annotations',
            'no_winner_selection_no_release_or_value_gate_assessed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rtmlib_report', help='RGB raw report with independently checked proxy PTS')
    parser.add_argument('mmpose_report', help='MMPose raw report including observed decoder attributes')
    parser.add_argument('--output', required=True, help='New report path; existing evidence is never overwritten')
    args = parser.parse_args()
    try:
        output = ensure_new_output(args.output)
    except OSError as error:
        return output_error(error)
    try:
        report = compare_reports(args.rtmlib_report, args.mmpose_report)
    except Exception as error:
        report = {'schemaVersion': 'pose3d-runtime-probe-comparison-0.1.0', 'status': 'invalid_evidence',
                  'errorCode': getattr(error, 'code', 'invalid_evidence'), 'error': str(error), 'releaseReady': False}
    try:
        write_report(output, report)
    except (OSError, ValueError, TypeError) as error:
        return output_error(error)
    print(json.dumps({'status': report['status'], 'errorCode': report.get('errorCode'),
                      'output': str(output), 'releaseReady': False}))
    return 0 if report['status'] == 'raw_outputs_compared' else 2


if __name__ == '__main__':
    sys.exit(main())
