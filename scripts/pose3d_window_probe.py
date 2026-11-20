#!/usr/bin/env python3
"""Stream authorized capture-proxy windows into local RAW research evidence.

Run each runtime in its own existing environment. This runner never downloads a
model, writes product pose3d.f32, or interprets RAW scores as depth confidence.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

from pose3d_probe import (ProbeError, require, existing_local_file, authorize,
    digest, load_timestamps, load_keypoints, validate_proxy_clock, bbox_from_2d,
    load_rtmlib, validate_outputs, dependency_versions, write_report,
    BBOX_CONFIDENCE, BBOX_PADDING)
from pose3d_mmpose_probe import load_mmpose, validate_raw_arrays
from pose3d_window_spec import load_window_spec


VERSION = 'pose3d-window-probe-0.1.0'
RUNTIMES = ('rtmlib-onnx', 'mmpose')


def claim_output(value):
    """mkdir is the exclusive claim; even an existing empty directory is refused."""
    path = Path(value).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ProbeError('output_exists', '研究输出目录必须不存在，禁止覆盖或并发复用') from error
    return path.resolve()


def frame_record(index, timestamps):
    return {'frame': index, 'captureTimeSec': float(timestamps[index]),
            'status': 'failed', 'errorCode': 'frame_not_processed',
            'bboxXYXY': None, 'inferenceAttempted': False, 'inferenceSec': None,
            'inferenceCpuSec': None, 'decodeSec': None,
            'rawPath': None, 'rawSha256': None, 'outputShapes': None}


def count_frames(report):
    success = failure = 0
    for window in report['windows']:
        window['succeededFrames'] = sum(item['status'] == 'succeeded' for item in window['frames'])
        window['failedFrames'] = len(window['frames']) - window['succeededFrames']
        success += window['succeededFrames']
        failure += window['failedFrames']
    report['succeededFrames'], report['failedFrames'] = success, failure
    report['selectedFrames'] = success + failure


def save_arrays(directory, index, timestamp, bbox, arrays, runtime):
    suffix = '.raw.npz' if runtime == 'rtmlib-onnx' else '.mmpose-raw.npz'
    name = f'frame-{index:06d}{suffix}'
    path = directory / name
    # A new directory is already exclusively claimed; xb also detects an
    # unexpected collision instead of replacing any previously written evidence.
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays, frame_index=np.array(index, dtype=np.int64),
            timestamp_sec=np.array(timestamp, dtype=np.float64), bbox_xyxy=np.asarray(bbox))
    return {'rawPath': name, 'rawSha256': digest(path),
            'outputShapes': {name: list(value.shape) for name, value in arrays.items()},
            'rawScoreRange': [float(arrays['scores'].min()), float(arrays['scores'].max())]}


def process_frame(image, record, baseline, infer, runtime, color_order, directory):
    """A missing target or failed model call is explicit and never reuses points."""
    try:
        record['bboxXYXY'] = bbox_from_2d(baseline, image.shape[1], image.shape[0])
        wall, cpu = time.perf_counter(), time.process_time()
        record['inferenceAttempted'] = True
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if runtime == 'rtmlib-onnx':
                    model_image = image if color_order == 'bgr' else image[:, :, ::-1].copy()
                    arrays = validate_outputs(infer(model_image, bboxes=[record['bboxXYXY']]))
                else:
                    arrays = validate_raw_arrays(infer(image, record['bboxXYXY']))
        finally:
            record['inferenceSec'] = time.perf_counter() - wall
            record['inferenceCpuSec'] = time.process_time() - cpu
        write_started = time.perf_counter()
        record.update(save_arrays(directory, record['frame'], record['captureTimeSec'],
                                  record['bboxXYXY'], arrays, runtime))
        record['rawPersistenceSec'] = time.perf_counter() - write_started
        record['status'] = 'succeeded'
        record.pop('errorCode', None)
    except Exception as error:
        record.update(status='failed', errorCode=getattr(error, 'code', 'frame_inference_failed'),
                      error=str(error))
    return record


def stream_frames(source, times, baseline, clock, report, output, infer, cv, color_order):
    selected = {item['frame']: (window, item) for window in report['windows'] for item in window['frames']}
    window_starts = {}
    capture = None
    count = 0
    try:
        start = time.perf_counter()
        try:
            capture = cv.VideoCapture(str(source))
            require(capture.isOpened(), 'video_decode_failed')
        finally:
            report['decodeSec'] += time.perf_counter() - start
        while True:
            read_started = time.perf_counter()
            try:
                ok, image = capture.read()
            finally:
                read_elapsed = time.perf_counter() - read_started
                report['decodeSec'] += read_elapsed
            if not ok:
                break
            require(count < len(times), 'frame_count_mismatch', '解码帧数超过已核验采集时间轴')
            require(isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3,
                    'video_decode_failed')
            require(image.shape[0] * image.shape[1] <= 3840 * 2160, 'resolution_limit_exceeded')
            require(image.shape[1] == clock['width'] and image.shape[0] == clock['height'],
                    'proxy_dimensions_mismatch')
            if count in selected:
                window, record = selected[count]
                window_starts.setdefault(window['id'], read_started)
                record['decodeSec'] = read_elapsed
                process_frame(image, record, baseline[count], infer, report['runtime'],
                              color_order, output / window['id'])
                window['decodeSec'] += read_elapsed
                window['inferenceSec'] += record['inferenceSec'] or 0
                window['inferenceCpuSec'] += record['inferenceCpuSec'] or 0
                window['rawPersistenceSec'] += record.get('rawPersistenceSec', 0)
                report['inferenceSec'] += record['inferenceSec'] or 0
                report['inferenceCpuSec'] += record['inferenceCpuSec'] or 0
                report['rawPersistenceSec'] += record.get('rawPersistenceSec', 0)
                count_frames(report)
                persist = time.perf_counter()
                write_report(output, report)
                report['reportPersistenceSec'] += time.perf_counter() - persist
                window['elapsedSec'] = time.perf_counter() - window_starts[window['id']]
            count += 1
            report['decodedFrames'] = count
            # No selected images are retained; the next read replaces this frame.
            del image
        require(count == len(times), 'frame_count_mismatch', '代理实际解码帧数与采集时间轴不同')
        report['decodeCompleted'] = True
    finally:
        if capture is not None:
            start = time.perf_counter()
            capture.release()
            report['decodeSec'] += time.perf_counter() - start


def run_probe(args, rtmlib_loader=None, mmpose_loader=None, cv_module=None, clock_validator=None):
    started, cpu_started = time.perf_counter(), time.process_time()
    output = claim_output(args.output_dir)
    report = {'schemaVersion': VERSION, 'status': 'running', 'runtime': args.runtime,
        'evaluationMode': True, 'releaseReady': False, 'productCompatible': False,
        'rawCoordinatesValidated': False, 'createdAt': datetime.now(timezone.utc).isoformat(),
        'selectionMode': 'research-sampling-not-production-window-policy', 'windows': [],
        'decodeSec': 0.0, 'inferenceSec': 0.0, 'inferenceCpuSec': 0.0,
        'rawPersistenceSec': 0.0, 'reportPersistenceSec': 0.0,
        'decodedFrames': 0, 'decodeCompleted': False,
        'limitations': ['research_windows_are_not_production_pre_post_settings',
            'raw_xyz_and_assumed_camera_coordinates_are_unvalidated',
            'raw_scores_are_not_depth_confidence', 'no_3d_ground_truth_or_release_gate_assessed'],
        'timingBoundaries': {
            'elapsedSec': 'entry through final bookkeeping; includes authorization, hashing, clock validation, model load, full sequential decode, inference and intermediate evidence writes; excludes final report write',
            'modelLoadSec': 'runtime loader including runtime imports, session or checkpoint initialization and loader provenance',
            'decodeSec': 'VideoCapture construction, every read including unselected frames/EOF, and release; excludes inference, frame validation, and ffprobe',
            'windowElapsedSec': 'first selected frame read through last processed frame report write; excludes model loading and decode between windows',
            'windowDecodeSec': 'reads of selected frames only',
            'inferenceSec': 'preprocess including optional BGR-to-RGB, model call, upstream postprocess and raw output shape validation; failed attempts included',
            'rawPersistenceSec': 'successful NPZ compression, file write and hash only',
            'cpuSec': 'process CPU time; process-lifetime RSS is separate and includes runtime imports'}}
    try:
        require(args.runtime in RUNTIMES, 'invalid_runtime')
        require(getattr(args, 'input_kind', 'capture-proxy') == 'capture-proxy', 'capture_proxy_required')
        source = existing_local_file(args.input, 'video_unavailable')
        source_hash = digest(source)
        report['authorization'] = authorize(args.authorization_evidence, source_hash)
        timestamps_path, times = load_timestamps(args.timestamps)
        spec_path = existing_local_file(args.window_spec, 'window_spec_unavailable')
        windows = load_window_spec(spec_path, times)
        specification = json.loads(spec_path.read_text())
        report.update(purpose=specification['purpose'], windowSpecPath=str(spec_path),
                      windowSpecSha256=digest(spec_path))
        for selection in windows:
            window = dict(selection, frames=[frame_record(index, times) for index in selection['frameIndices']],
                          elapsedSec=None, decodeSec=0.0, inferenceSec=0.0,
                          inferenceCpuSec=0.0, rawPersistenceSec=0.0)
            report['windows'].append(window)
            (output / selection['id']).mkdir()
        count_frames(report)
        pose_path, baseline = load_keypoints(args.keypoints2d, len(times))
        clock_started = time.perf_counter()
        clock = (clock_validator or validate_proxy_clock)(source, times)
        report['clockValidationSec'] = time.perf_counter() - clock_started
        report['input'] = {'kind': 'capture-proxy', 'path': str(source), 'sha256': source_hash,
            'frameCount': len(times), 'timestampsPath': str(timestamps_path),
            'timestampsSha256': digest(timestamps_path), 'clockSource': 'existing-2d-capture-timestamps',
            'proxyClockValidation': clock, 'keypoints2dPath': str(pose_path), 'keypoints2dSha256': digest(pose_path)}
        report['bboxPolicy'] = {'origin': 'same-frame-existing-2d', 'bodyIndices': [0, 16],
            'confidenceThreshold': BBOX_CONFIDENCE, 'minBodyPoints': 5,
            'extentPaddingRatio': BBOX_PADDING, 'detectorUsed': False,
            'identityValidation': 'not_independently_validated'}
        color_order = getattr(args, 'color_order', 'bgr')
        require(color_order in ('bgr', 'rgb'), 'invalid_color_order')
        if args.runtime == 'rtmlib-onnx':
            model_path = existing_local_file(args.pose_model, 'model_unavailable')
            require(model_path.suffix.lower() == '.onnx', 'invalid_model')
            report['model'] = {'path': str(model_path), 'weightsSha256': digest(model_path)}
            report['dependencyVersions'] = dependency_versions()
            report['preprocess'] = {'decodedImageColor': 'bgr', 'passedToLibraryColor': color_order,
                'colorConversion': 'none' if color_order == 'bgr' else 'BGR-to-RGB before RTMPose3d',
                'purpose': 'controlled-preprocessing-probe; not a validated model configuration'}
            loader = lambda: (rtmlib_loader or load_rtmlib)(model_path, args.device)
        else:
            require(color_order == 'bgr', 'mmpose_color_override_forbidden',
                    'MMPose 使用官方内部 bgr_to_rgb；不要再转换输入')
            checkpoint = existing_local_file(args.checkpoint, 'model_unavailable')
            config = existing_local_file(args.config, 'config_unavailable')
            report['model'] = {'path': str(checkpoint), 'weightsSha256': digest(checkpoint),
                              'configPath': str(config), 'configSha256': digest(config)}
            loader = lambda: (mmpose_loader or load_mmpose)(config, checkpoint, args.source_root, args.threads)
        if cv_module is None:
            import cv2 as cv_module
        load_wall, load_cpu = time.perf_counter(), time.process_time()
        try:
            infer, metadata = loader()
        finally:
            report['modelLoadSec'] = time.perf_counter() - load_wall
            report['modelLoadCpuSec'] = time.process_time() - load_cpu
        report['model'].update(metadata)
        report['setupSec'] = time.perf_counter() - started - report['modelLoadSec']
        stream_frames(source, times, baseline, clock, report, output, infer, cv_module, color_order)
        count_frames(report)
        if not report['succeededFrames']:
            report.update(status='failed', errorCode='no_successful_frames')
        else:
            report['status'] = ('raw_evidence_collected_with_failures' if report['failedFrames']
                                else 'raw_evidence_collected')
    except Exception as error:
        report.update(status='failed', errorCode=getattr(error, 'code', 'window_probe_failed'), error=str(error))
        (output / 'failure-traceback.txt').write_text(traceback.format_exc())
    finally:
        count_frames(report)
        report['elapsedSec'] = time.perf_counter() - started
        report['cpuSec'] = time.process_time() - cpu_started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peakProcessRamBytes'] = int(peak if sys.platform == 'darwin' else peak * 1024)
        report['memoryMeasurement'] = 'process-lifetime-peak-rss; not model-only allocation'
        write_report(output, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--runtime', choices=RUNTIMES, required=True)
    for name in ('input', 'timestamps', 'keypoints2d', 'authorization-evidence', 'window-spec', 'output-dir'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--input-kind', choices=['capture-proxy'], default='capture-proxy')
    result.add_argument('--pose-model', help='RTMLib: existing local .onnx')
    result.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    result.add_argument('--color-order', choices=['bgr', 'rgb'], default='bgr',
                        help='RTMLib only; MMPose retains official internal bgr_to_rgb')
    result.add_argument('--checkpoint', help='MMPose: existing local .pth')
    result.add_argument('--config', help='MMPose: verified official local config')
    result.add_argument('--source-root', help='MMPose: verified local v1.3.2 source root')
    result.add_argument('--threads', type=int, default=4)
    return result


def main():
    args = parser().parse_args()
    try:
        report = run_probe(args)
    except (OSError, ProbeError) as error:
        report = {'status': 'failed', 'errorCode': getattr(error, 'code', 'io_error'), 'error': str(error)}
    print(json.dumps({'status': report['status'], 'errorCode': report.get('errorCode'),
        'report': str(Path(args.output_dir).absolute() / 'report.json'), 'releaseReady': False}, ensure_ascii=False))
    return 0 if report['status'] == 'raw_evidence_collected' else 2


if __name__ == '__main__':
    sys.exit(main())
