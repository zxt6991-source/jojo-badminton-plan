#!/usr/bin/env python3
"""Audit local raw probe evidence and independently decode relative Z, never XYZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np

from pose3d_probe import (OUTPUT_NAMES, OUTPUT_SHAPES, MAX_PROBE_FRAMES,
                         MAX_CLIP_FRAMES, ProbeError, require, digest, authorize, load_timestamps)

ONNX_SHAPES = [[1, 133, 576], [1, 133, 768], [1, 133, 576]]
OFFICIAL_Z_RANGE = 2.1744869
PAIRS = {'shoulders': (5, 6), 'elbows': (7, 8), 'wrists': (9, 10),
         'hips': (11, 12), 'knees': (13, 14), 'ankles': (15, 16)}


def sha(value, code):
    require(isinstance(value, str) and re.fullmatch(r'[a-fA-F0-9]{64}', value) is not None, code)
    return value.lower()


def finite_number(value):
    return type(value) in (int, float) and np.isfinite(value)


def read_report(path):
    require(path.is_file() and path.stat().st_size <= 1_000_000, 'invalid_probe_report')
    try:
        report = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise ProbeError('invalid_probe_report', str(error)) from error
    require(isinstance(report, dict) and report.get('schemaVersion') == 'pose3d-raw-probe-0.1.0'
            and report.get('status') == 'raw_evidence_collected', 'unsupported_probe_schema_or_status')
    require(report.get('evaluationMode') is True and report.get('releaseReady') is False
            and report.get('productCompatible') is False and report.get('rawCoordinatesValidated') is False
            and report.get('coordinateStatus') == 'unverified-raw-mixed-scales', 'unsafe_coordinate_claim')
    return report


def verify_authorization(report):
    evidence, source = report.get('authorization'), report.get('input')
    require(isinstance(evidence, dict) and isinstance(source, dict), 'authorization_missing')
    for key in ('evidenceSha256', 'originalSha256', 'proxySha256'):
        sha(evidence.get(key), 'authorization_hash_missing')
    proxy_sha = sha(source.get('sha256'), 'input_hash_missing')
    require(source.get('kind') == 'capture-proxy' and evidence['proxySha256'].lower() == proxy_sha,
            'authorization_proxy_mismatch')
    path = Path(evidence.get('evidencePath', ''))
    require(path.is_file() and digest(path) == evidence['evidenceSha256'].lower(), 'authorization_hash_mismatch')
    current = authorize(path, source['sha256'])
    require(all(current.get(key) == evidence.get(key) for key in current), 'authorization_metadata_mismatch')
    proxy = Path(source.get('path', ''))
    require(proxy.is_file() and digest(proxy) == proxy_sha, 'input_hash_mismatch')
    require(source.get('clockSource') == 'existing-2d-capture-timestamps', 'capture_clock_required')
    sha(source.get('timestampsSha256'), 'timestamps_hash_missing')
    count = source.get('frameCount')
    require(type(count) is int and 1 <= count <= MAX_CLIP_FRAMES, 'invalid_frame_count')
    timestamps_path, times = load_timestamps(source.get('timestampsPath'))
    require(digest(timestamps_path) == source['timestampsSha256'].lower(), 'timestamps_hash_mismatch')
    require(len(times) == count, 'timestamp_count_mismatch')
    return times


def verify_decoder(report):
    model = report.get('model')
    require(isinstance(model, dict), 'model_metadata_missing')
    for key in ('weightsSha256', 'sourceFileSha256'):
        sha(model.get(key), 'model_hash_missing')
    require(model.get('apiVersion') == '0.0.16'
            and model.get('implementation') == 'rtmlib.tools.pose_estimation.rtmpose3d.RTMPose3d'
            and model.get('modelInputSize') == [288, 384], 'unsupported_decoder')
    outputs = model.get('onnxOutputs')
    require(isinstance(outputs, list) and len(outputs) == 3
            and all(isinstance(o, dict) for o in outputs)
            and [o.get('shape') for o in outputs] == ONNX_SHAPES, 'incompatible_onnx_outputs',
            '需按 X/Y/Z 顺序提供 [1,133,576]/[1,133,768]/[1,133,576] 输出')
    decoder = model.get('upstreamDecode')
    require(isinstance(decoder, dict) and decoder.get('simccSplitRatio') == 2
            and decoder.get('zDenominator') == 192 and decoder.get('isotropicXYZVerified') is False,
            'unsupported_decoder')
    require(finite_number(model.get('zRange')) and abs(model['zRange'] - OFFICIAL_Z_RANGE) < 1e-9,
            'unsupported_z_range')
    # Derive depth input size from actual ONNX metadata and the recorded bin ratio.
    return outputs[2]['shape'][-1] / decoder['simccSplitRatio'] / 2


def read_frame(directory, item, frame_count):
    require(isinstance(item, dict), 'invalid_frame_metadata')
    frame = item.get('frame')
    require(type(frame) is int and 0 <= frame < frame_count, 'invalid_frame_index')
    expected_name = f'frame-{frame:06d}.raw.npz'
    require(item.get('rawPath') == expected_name, 'invalid_raw_path')
    path = directory / expected_name
    require(path.is_file() and path.resolve().parent == directory.resolve(), 'raw_path_escape_or_missing')
    require(path.stat().st_size <= 1_000_000, 'raw_size_limit')
    require(digest(path) == sha(item.get('rawSha256'), 'raw_hash_missing'), 'raw_hash_mismatch')
    require(item.get('outputShapes') == {name: list(shape) for name, shape in zip(OUTPUT_NAMES, OUTPUT_SHAPES)},
            'raw_shape_metadata_mismatch')
    try:
        with zipfile.ZipFile(path) as archive:
            require(len(archive.infolist()) == 7 and sum(i.file_size for i in archive.infolist()) <= 1_000_000,
                    'raw_size_limit')
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(OUTPUT_NAMES) | {'frame_index', 'timestamp_sec', 'bbox_xyxy'}, 'raw_fields_mismatch')
            arrays = {name: archive[name] for name in archive.files}
    except (ValueError, zipfile.BadZipFile, OSError) as error:
        if isinstance(error, ProbeError):
            raise
        raise ProbeError('invalid_raw_archive', str(error)) from error
    for name, shape in zip(OUTPUT_NAMES, OUTPUT_SHAPES):
        array = arrays[name]
        require(array.shape == shape and np.issubdtype(array.dtype, np.number)
                and not np.iscomplexobj(array), 'invalid_raw_array', name)
        require(np.isfinite(array).all(), 'nonfinite_raw_array', name)
    require(arrays['frame_index'].shape == () and np.issubdtype(arrays['frame_index'].dtype, np.integer)
            and int(arrays['frame_index']) == frame, 'raw_frame_mismatch')
    require(arrays['timestamp_sec'].shape == () and np.issubdtype(arrays['timestamp_sec'].dtype, np.number)
            and not np.iscomplexobj(arrays['timestamp_sec']) and finite_number(item.get('captureTimeSec'))
            and item['captureTimeSec'] >= 0
            and float(arrays['timestamp_sec']) == item['captureTimeSec'], 'raw_timestamp_mismatch')
    bbox = arrays['bbox_xyxy']
    require(bbox.shape == (4,) and np.issubdtype(bbox.dtype, np.number) and np.isfinite(bbox).all()
            and isinstance(item.get('bboxXYXY'), list) and len(item['bboxXYXY']) == 4
            and all(finite_number(v) for v in item['bboxXYXY'])
            and np.allclose(bbox, item['bboxXYXY'], rtol=0, atol=1e-6), 'raw_bbox_mismatch')
    return arrays


def decode_frame(arrays, denominator):
    q = arrays['keypoints_simcc'].astype(np.float64)
    scores = arrays['scores'].astype(np.float64)
    present = scores > 0
    bins = q * 2
    valid_bins = (bins == np.floor(bins)) & (bins >= 0) & (bins < np.array([576, 768, 576]))
    require(valid_bins[present].all() and (q[~present] == -.5).all(), 'invalid_simcc_locations')
    upstream = arrays['keypoints'][..., 2].astype(np.float64)
    reconstructed = (q[..., 2] / 192 - 1) * OFFICIAL_Z_RANGE
    require(np.allclose(upstream, reconstructed, atol=1e-6, rtol=1e-6), 'upstream_z_formula_mismatch')
    require(np.array_equal(arrays['keypoints'][..., :2], q[..., :2]), 'upstream_xy_formula_mismatch')
    independent = (q[..., 2] / denominator - 1) * OFFICIAL_Z_RANGE
    return {'upstream_relative_z': upstream, 'official_codec_relative_z': independent,
            'relative_z_difference': independent - upstream, 'scores': scores,
            'decoded_point_mask': present, 'frame_index': arrays['frame_index'],
            'timestamp_sec': arrays['timestamp_sec']}


def stats(values, mask):
    selected = values[mask]
    return {'count': int(selected.size), 'min': float(selected.min()) if selected.size else None,
            'max': float(selected.max()) if selected.size else None,
            'meanAbsolute': float(np.abs(selected).mean()) if selected.size else None}


def run_audit(report_path, output_dir):
    path, output = Path(report_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'output_not_empty')
    output.mkdir(parents=True, exist_ok=True)
    result = {'schemaVersion': 'pose3d-decode-audit-0.1.0', 'status': 'invalid_evidence',
              'evaluationMode': True, 'releaseReady': False, 'productCompatible': False,
              'rawCoordinatesValidated': False, 'cameraXYZRecovered': False, 'frames': [],
              'limitations': ['relative_z_only_not_camera_xyz', 'decoder_difference_is_not_accuracy',
                              'upstream_scores_do_not_measure_depth_confidence',
                              'left_right_depth_order_not_independently_validated']}
    try:
        report = read_report(path)
        timestamps = verify_authorization(report)
        denominator = verify_decoder(report)
        items = report.get('frames')
        require(isinstance(items, list) and 1 <= len(items) <= MAX_PROBE_FRAMES, 'invalid_probe_frames')
        evidence, indices = [], []
        for item in items:
            arrays = read_frame(path.parent, item, report['input']['frameCount'])
            require(item['captureTimeSec'] == float(timestamps[item['frame']]), 'capture_timestamp_mismatch')
            indices.append(item['frame'])
            evidence.append((item, arrays, decode_frame(arrays, denominator)))
        require(indices == sorted(set(indices)), 'duplicate_or_unsorted_frames')
        # All evidence passes before writing any research arrays.
        result.update(probeReportSha256=digest(path), authorization=report['authorization'],
                      weightsSha256=report['model']['weightsSha256'],
                      decoder={'simccSplitRatio': 2, 'actualZBins': report['model']['onnxOutputs'][2]['shape'][-1],
                               'upstreamDenominator': 192, 'officialDepthInputSize': denominator * 2,
                               'officialDenominator': denominator, 'zRange': OFFICIAL_Z_RANGE})
        for item, arrays, decoded in evidence:
            name = f"frame-{item['frame']:06d}.relative-z-audit.npz"
            np.savez_compressed(output / name, **decoded)
            mask = decoded['decoded_point_mask']
            pairs = {}
            for label, (left, right) in PAIRS.items():
                available = bool(mask[0, left] and mask[0, right])
                pairs[label] = {'indices': [left, right], 'bothDecoded': available,
                    'leftMinusRightRelativeZ': float(decoded['official_codec_relative_z'][0, left] - decoded['official_codec_relative_z'][0, right]) if available else None}
            result['frames'].append({'frame': item['frame'], 'captureTimeSec': item['captureTimeSec'],
                'rawSha256': item['rawSha256'], 'researchPath': name, 'researchSha256': digest(output / name),
                'finiteRatios': {key: float(np.isfinite(arrays[key]).mean()) for key in OUTPUT_NAMES},
                'upstreamRelativeZ': stats(decoded['upstream_relative_z'], mask),
                'officialCodecRelativeZ': stats(decoded['official_codec_relative_z'], mask),
                'decodeDifference': stats(decoded['relative_z_difference'], mask), 'pairDepthDifferences': pairs})
        result['status'] = 'decode_audited'
    except (ProbeError, OSError, ValueError, TypeError, KeyError) as error:
        result.update(errorCode=getattr(error, 'code', 'invalid_evidence'), error=str(error))
    temporary = output / '.audit.json.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(output / 'audit.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', help='pose3d_probe.py 的本地 report.json')
    parser.add_argument('--output-dir', required=True, help='新的空研究目录')
    args = parser.parse_args()
    try:
        report = run_audit(args.report, args.output_dir)
    except (ProbeError, OSError) as error:
        report = {'status': 'invalid_evidence', 'errorCode': getattr(error, 'code', 'io_error'), 'error': str(error)}
    print(json.dumps({'status': report['status'], 'errorCode': report.get('errorCode'),
                      'audit': str(Path(args.output_dir).resolve() / 'audit.json'), 'releaseReady': False}))
    return 0 if report['status'] == 'decode_audited' else 2


if __name__ == '__main__':
    sys.exit(main())
