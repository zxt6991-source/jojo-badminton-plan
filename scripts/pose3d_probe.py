#!/usr/bin/env python3
"""Local-only RTMW3D raw-output probe; never emits product pose3d.f32.

Frame numbers refer exclusively to an existing capture-clock proxy and its
timestamps.f32. Raw XYZ remains in the upstream decoder's mixed coordinate
conventions and must not be interpreted as a validated camera-relative pose.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone, date
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
import numpy as np

from pose3d_prepare import probe_proxy, TIMESTAMP_TOLERANCE_SEC

MAX_PROBE_FRAMES = 10
MAX_CLIP_FRAMES = 20000
BBOX_CONFIDENCE = .4
BBOX_PADDING = .15
SUPPORTED_RTMLIB_VERSION = '0.0.16'
OUTPUT_NAMES = ('keypoints', 'scores', 'keypoints_simcc', 'keypoints_2d')
OUTPUT_SHAPES = ((1, 133, 3), (1, 133), (1, 133, 3), (1, 133, 2))
BODY_BONES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
              (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


class ProbeError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message=None):
    if not condition:
        raise ProbeError(code, message or code)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def parse_frames(value):
    try:
        parts = value.split(',')
        require(all(re.fullmatch(r'\d+', part) for part in parts), 'invalid_frames')
        frames = [int(part) for part in parts]
        require(1 <= len(frames) <= MAX_PROBE_FRAMES and len(set(frames)) == len(frames)
                and all(frame < MAX_CLIP_FRAMES for frame in frames), 'invalid_frames')
        return sorted(frames)
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f'frames 必须为 1–{MAX_PROBE_FRAMES} 个不重复的非负整数') from error


def parse_bbox(value):
    try:
        values = tuple(float(part) for part in value.split(','))
        require(len(values) == 4 and all(math.isfinite(v) and v >= 0 for v in values)
                and values[0] < values[2] and values[1] < values[3], 'invalid_bbox')
        return values
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError('bbox 必须为像素坐标 x1,y1,x2,y2，且面积为正') from error


def existing_local_file(value, code):
    require(isinstance(value, (str, Path)) and '://' not in str(value), code, '只接受已存在的本地文件，禁止 URL 或自动下载')
    path = Path(value).expanduser().resolve()
    require(path.is_file() and path.stat().st_size > 0, code, f'本地文件不存在或为空：{path}')
    return path


def authorize(path, input_sha256):
    evidence = existing_local_file(path, 'authorization_required')
    try:
        document = json.loads(evidence.read_text())
        require(isinstance(document, dict), 'invalid_authorization')
        require(document.get('schemaVersion') == 'pose3d-local-authorization-0.1.0'
                and document.get('scope') == 'local-internal-3d-validation'
                and document.get('externalUploadAllowed') is False
                and isinstance(document.get('userStatement'), str) and document['userStatement'].strip(),
                'invalid_authorization')
        date.fromisoformat(document['authorizedAt'])
        retained = document.get('retentionUntil')
        require(retained is None or date.fromisoformat(retained) >= date.today(), 'authorization_expired')
        items = document['items']
        require(isinstance(items, list), 'invalid_authorization')
        matches = [item for item in items if isinstance(item, dict) and item.get('proxySha256') == input_sha256]
        require(len(matches) == 1, 'input_not_authorized', '代理视频 hash 未唯一匹配本机研究授权')
        item = matches[0]
        require(all(isinstance(item.get(k), str) and item[k] for k in ('videoId', 'originalPath', 'proxyPath'))
                and re.fullmatch('[a-fA-F0-9]{64}', item.get('originalSha256', '')) is not None,
                'invalid_authorization')
        return {'evidencePath': str(evidence), 'evidenceSha256': digest(evidence),
                'matchingVideoId': item['videoId'], 'authorizedAt': document['authorizedAt'],
                'scope': document['scope'], 'externalUploadAllowed': False, 'retentionUntil': retained,
                'originalSha256': item['originalSha256'], 'proxySha256': input_sha256}
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ProbeError):
            raise
        raise ProbeError('invalid_authorization', f'授权 JSON 不完整：{error}') from error


def load_timestamps(path):
    path = existing_local_file(path, 'timestamps_required')
    require(path.stat().st_size % 4 == 0, 'invalid_timestamps')
    count = path.stat().st_size // 4
    require(1 <= count <= MAX_CLIP_FRAMES, 'invalid_frame_count')
    times = np.fromfile(path, dtype='<f4')
    require(np.isfinite(times).all() and times[0] >= 0 and np.all(np.diff(times) > 0), 'invalid_timestamps')
    return path, times


def validate_proxy_clock(path, timestamps):
    """Independently match the actual proxy PTS to every capture timestamp.

    Production always requires ffprobe. Count agreement or a claimed clockSource
    string is insufficient; a same-length shifted/scaled time file is rejected.
    """
    try:
        decoded = probe_proxy(path)
    except (OSError, subprocess.SubprocessError) as error:
        raise ProbeError('proxy_clock_unavailable', f'无法独立核验代理时间轴（需要 ffprobe）：{error}') from error
    except (ValueError, TypeError, KeyError) as error:
        raise ProbeError('invalid_proxy_clock', f'代理时间轴无效：{error}') from error
    require(type(decoded.get('width')) is int and type(decoded.get('height')) is int
            and decoded['width'] > 0 and decoded['height'] > 0, 'proxy_dimensions_invalid')
    pts = np.asarray(decoded['timestamps'], dtype=np.float64)
    capture_times = np.asarray(timestamps, dtype=np.float64)
    require(pts.ndim == capture_times.ndim == 1 and len(pts) == len(capture_times)
            and 1 <= len(pts) <= MAX_CLIP_FRAMES, 'proxy_clock_frame_count_mismatch')
    require(np.isfinite(pts).all() and np.isfinite(capture_times).all()
            and np.all(np.diff(pts) > 0) and np.all(np.diff(capture_times) > 0)
            and abs(pts[0]) <= TIMESTAMP_TOLERANCE_SEC, 'invalid_proxy_clock')
    max_error = float(np.max(np.abs(pts - capture_times)))
    require(max_error <= TIMESTAMP_TOLERANCE_SEC, 'proxy_timestamp_mismatch',
            '代理真实逐帧时间戳与所提供的采集时间轴不一致')
    return {'source': 'ffprobe-best_effort_timestamp_time', 'frameCount': len(pts),
            'width': decoded['width'], 'height': decoded['height'],
            'maxTimestampErrorSec': max_error, 'toleranceSec': TIMESTAMP_TOLERANCE_SEC}


def load_keypoints(path, count):
    path = existing_local_file(path, 'keypoints2d_required')
    require(path.stat().st_size == count * 65 * 3 * 4, 'invalid_keypoints2d_length')
    pose = np.fromfile(path, dtype='<f4').reshape(count, 65, 3)
    xy, score = pose[..., :2], pose[..., 2]
    present = (np.isfinite(xy).all(axis=2) & (xy >= 0).all(axis=2) & (xy <= 1).all(axis=2)
               & np.isfinite(score) & (score >= 0) & (score <= 1))
    missing = np.isnan(xy).all(axis=2) & (score == 0)
    require((present | missing).all(), 'invalid_keypoints2d')
    return path, pose


def bbox_from_2d(pose, width, height):
    body = pose[:17]
    visible = np.isfinite(body).all(axis=1) & (body[:, 2] >= BBOX_CONFIDENCE)
    require(visible.sum() >= 5, 'target_bbox_unavailable', '同帧身体关键点少于 5 个，不自动选择其他球员')
    xy = body[visible, :2] * [width, height]
    minimum, maximum = xy.min(axis=0), xy.max(axis=0)
    extent = maximum - minimum
    require((extent > 1).all(), 'target_bbox_unavailable', '目标框退化')
    lower, upper = minimum - extent * BBOX_PADDING, maximum + extent * BBOX_PADDING
    return validate_bbox((max(0, lower[0]), max(0, lower[1]), min(width, upper[0]), min(height, upper[1])), width, height)


def validate_bbox(bbox, width, height):
    require(len(bbox) == 4 and all(math.isfinite(float(v)) for v in bbox), 'invalid_bbox')
    x1, y1, x2, y2 = map(float, bbox)
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, 'invalid_bbox', '目标框超出代理视频画幅')
    return [x1, y1, x2, y2]


def decode_frames(path, indices, expected_count, cv):
    capture = cv.VideoCapture(str(path))
    selected, seen = {}, 0
    try:
        require(capture.isOpened(), 'video_decode_failed')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            require(seen < expected_count and isinstance(frame, np.ndarray)
                    and frame.ndim == 3 and frame.shape[2] == 3, 'frame_count_mismatch')
            require(frame.shape[0] * frame.shape[1] <= 3840 * 2160, 'resolution_limit_exceeded')
            if seen in indices:
                selected[seen] = frame
            seen += 1
        require(seen == expected_count and len(selected) == len(indices), 'frame_count_mismatch',
                '代理视频实际解码帧数与既有采集时间戳不一致')
        return selected
    finally:
        capture.release()


def load_rtmlib(model_path, device):
    # BaseTool otherwise invokes download_checkpoint for a nonexistent path.
    # Validate twice and replace that function during construction to fail closed
    # even if a file disappears between the checks and upstream model loading.
    path = existing_local_file(model_path, 'model_unavailable')
    require(path.suffix.lower() == '.onnx', 'invalid_model', '只接受本地 .onnx 文件')
    require(importlib.metadata.version('rtmlib') == SUPPORTED_RTMLIB_VERSION,
            'unsupported_rtmlib_version', f'探针已核验 rtmlib {SUPPORTED_RTMLIB_VERSION} API，其他版本须重新核验')
    from rtmlib.tools.pose_estimation.rtmpose3d import RTMPose3d
    import rtmlib.tools.base as base
    import onnxruntime as ort
    ort.disable_telemetry_events()
    original_download = base.download_checkpoint

    def refuse_download(*_args, **_kwargs):
        raise ProbeError('model_download_forbidden', '研究探针不下载模型')

    base.download_checkpoint = refuse_download
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = RTMPose3d(onnx_model=str(path), model_input_size=(288, 384),
                              to_openpose=False, backend='onnxruntime', device=device)
    finally:
        base.download_checkpoint = original_download
    # Inspect ONNX metadata before a potentially expensive inference call.
    inputs = model.session.get_inputs()
    require(len(inputs) == 1 and inputs[0].shape[1:] == [3, 384, 288], 'incompatible_model_input')
    require(len(model.session.get_outputs()) == 3, 'incompatible_model_outputs')
    if device == 'cuda':
        require('CUDAExecutionProvider' in model.session.get_providers(), 'requested_device_unavailable')
    metadata = {
        'implementation': 'rtmlib.tools.pose_estimation.rtmpose3d.RTMPose3d',
        'sourceFileSha256': digest(inspect.getsourcefile(RTMPose3d)),
        'apiVersion': SUPPORTED_RTMLIB_VERSION, 'modelInputSize': [288, 384],
        'libraryPreprocessColorConversion': 'none in installed RTMPose3d.preprocess',
        'backend': 'onnxruntime', 'requestedDevice': device,
        'executionProviders': model.session.get_providers(), 'zRange': model.z_range,
        'onnxInputs': [{'name': value.name, 'shape': value.shape, 'type': value.type} for value in inputs],
        'onnxOutputs': [{'name': value.name, 'shape': value.shape, 'type': value.type} for value in model.session.get_outputs()],
        'upstreamDecode': {'simccSplitRatio': 2, 'zDenominator': 192,
                           'zFormula': '((z_bin / 2) / (model_input_height / 2) - 1) * z_range',
                           'keypointsSimccMeaning': 'argmax locations / 2 before z remapping; not raw logits',
                           'isotropicXYZVerified': False},
    }
    return model, metadata


def validate_outputs(outputs):
    require(isinstance(outputs, (tuple, list)) and len(outputs) == 4, 'invalid_model_output', 'RTMPose3d 必须返回 4 元组')
    result = {}
    for name, shape, value in zip(OUTPUT_NAMES, OUTPUT_SHAPES, outputs):
        require(isinstance(value, np.ndarray) and value.shape == shape
                and np.issubdtype(value.dtype, np.number) and not np.iscomplexobj(value)
                and np.isfinite(value).all(), 'invalid_model_output', f'{name} 必须为有限数值数组 {shape}')
        result[name] = value.copy()  # preserve raw scores, signs and scales exactly
    return result


def draw_overlay(image, arrays, bbox, baseline, frame_index, timestamp, cv):
    left, right = image.copy(), image.copy()
    height, width = image.shape[:2]
    rect = tuple(int(v) for v in bbox)
    for canvas in (left, right):
        cv.rectangle(canvas, rect[:2], rect[2:], (0, 200, 255), 2)

    def skeleton(canvas, xy, scores, color):
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(scores) & (scores >= BBOX_CONFIDENCE)
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        for a, b in BODY_BONES:
            if valid[a] and valid[b]:
                cv.line(canvas, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 2)
        for joint in np.flatnonzero(valid):
            point = tuple(xy[joint].astype(int))
            cv.circle(canvas, point, 3, color, -1)
            if joint < 17:
                cv.putText(canvas, str(joint), point, cv.FONT_HERSHEY_SIMPLEX, .35, color, 1)

    if baseline is not None:
        skeleton(left, baseline[:, :2] * [width, height], baseline[:, 2], (50, 220, 80))
    skeleton(right, arrays['keypoints_2d'][0], arrays['scores'][0], (230, 160, 50))
    for canvas, label in ((left, 'Existing 2D' if baseline is not None else 'Original proxy'),
                           (right, 'RAW RTMW3D image-2D; depth UNVERIFIED')):
        cv.putText(canvas, label, (10, 24), cv.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        cv.putText(canvas, f'Capture frame {frame_index} / {timestamp:.6f}s', (10, 48),
                   cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return np.concatenate([left, right], axis=1)


def dependency_versions():
    result = {'python': platform.python_version(), 'numpy': np.__version__}
    for name in ('rtmlib', 'onnxruntime', 'opencv-python', 'opencv-contrib-python'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not-installed'
    return result


def write_report(directory, report):
    path = directory / 'report.json'
    temporary = directory / '.report.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def run_probe(args, model_factory=None, cv_module=None, clock_validator=None):
    started = time.perf_counter()
    output = Path(args.output_dir).expanduser().resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            'output_not_empty', '使用空的研究输出目录，以保留既有证据')
    output.mkdir(parents=True, exist_ok=True)
    report = {'schemaVersion': 'pose3d-raw-probe-0.1.0', 'status': 'running', 'evaluationMode': True,
              'rawCoordinatesValidated': False,
              'releaseReady': False, 'productCompatible': False, 'coordinateStatus': 'unverified-raw-mixed-scales',
              'createdAt': datetime.now(timezone.utc).isoformat(), 'frames': [],
              'limitations': ['not_camera_root_relative_xyz', 'not_a_product_3d_result',
                             'image_xy_agreement_does_not_validate_depth', 'no_3d_accuracy_or_value_gate_assessed']}
    try:
        require(args.input_kind == 'capture-proxy', 'capture_proxy_required')
        input_path = existing_local_file(args.input, 'video_unavailable')
        input_hash = digest(input_path)
        report['authorization'] = authorize(args.authorization_evidence, input_hash)
        model_path = existing_local_file(args.pose_model, 'model_unavailable')
        require(model_path.suffix.lower() == '.onnx', 'invalid_model')
        timestamps_path, times = load_timestamps(args.timestamps)
        indices = args.frames
        require(isinstance(indices, list) and 1 <= len(indices) <= MAX_PROBE_FRAMES
                and all(type(f) is int and 0 <= f < len(times) for f in indices)
                and len(indices) == len(set(indices)), 'invalid_frames')
        clock_validation = (clock_validator or validate_proxy_clock)(input_path, times)
        baseline_path, baseline = load_keypoints(args.keypoints2d, len(times)) if args.keypoints2d else (None, None)
        require(args.bbox is not None or baseline is not None, 'target_bbox_required', '提供同帧二维骨架或明确的像素 bbox')
        report.update(input={'kind': 'capture-proxy', 'path': str(input_path), 'sha256': input_hash,
                             'frameCount': len(times), 'timestampsPath': str(timestamps_path),
                             'timestampsSha256': digest(timestamps_path), 'clockSource': 'existing-2d-capture-timestamps',
                             'proxyClockValidation': clock_validation,
                             'keypoints2dPath': str(baseline_path) if baseline_path else None,
                             'keypoints2dSha256': digest(baseline_path) if baseline_path else None},
                      model={'path': str(model_path), 'weightsSha256': digest(model_path)},
                      dependencyVersions=dependency_versions(),
                      bboxPolicy={'origin': 'explicit-pixel-bbox' if args.bbox is not None else 'same-frame-existing-2d',
                                  'bodyIndices': [0, 16], 'confidenceThreshold': BBOX_CONFIDENCE,
                                  'minBodyPoints': 5, 'extentPaddingRatio': BBOX_PADDING,
                                  'detectorUsed': False, 'identityValidation': 'not_independently_validated'})
        color_order = getattr(args, 'color_order', 'bgr')
        require(color_order in ('bgr', 'rgb'), 'invalid_color_order')
        report['preprocess'] = {'decodedImageColor': 'bgr', 'passedToLibraryColor': color_order,
                                'colorConversion': 'none' if color_order == 'bgr' else 'BGR-to-RGB before RTMPose3d',
                                'purpose': 'controlled-preprocessing-probe; not a validated model configuration'}
        if cv_module is None:
            import cv2 as cv_module
        decode_started = time.perf_counter()
        frames = decode_frames(input_path, set(indices), len(times), cv_module)
        report['decodeSec'] = time.perf_counter() - decode_started
        require(all(frame.shape[1] == clock_validation['width'] and frame.shape[0] == clock_validation['height']
                    for frame in frames.values()), 'proxy_dimensions_mismatch')
        bboxes = {index: (validate_bbox(args.bbox, frame.shape[1], frame.shape[0]) if args.bbox is not None
                         else bbox_from_2d(baseline[index], frame.shape[1], frame.shape[0]))
                  for index, frame in frames.items()}
        load_started = time.perf_counter()
        model, metadata = (model_factory or load_rtmlib)(model_path, args.device)
        report['modelLoadSec'] = time.perf_counter() - load_started
        report['model'].update(metadata)
        for index in sorted(indices):
            frame, bbox = frames.pop(index), bboxes[index]
            inference_started = time.perf_counter()
            model_image = frame if color_order == 'bgr' else frame[:, :, ::-1].copy()
            with contextlib.redirect_stdout(sys.stderr):
                arrays = validate_outputs(model(model_image, bboxes=[bbox]))
            elapsed = time.perf_counter() - inference_started
            raw_name, overlay_name = f'frame-{index:06d}.raw.npz', f'frame-{index:06d}.overlay.png'
            np.savez_compressed(output / raw_name, **arrays, frame_index=np.array(index, dtype=np.int64),
                                timestamp_sec=np.array(times[index], dtype=np.float64), bbox_xyxy=np.asarray(bbox))
            overlay = draw_overlay(frame, arrays, bbox, baseline[index] if baseline is not None else None,
                                   index, float(times[index]), cv_module)
            require(cv_module.imwrite(str(output / overlay_name), overlay), 'overlay_write_failed')
            item = {'frame': index, 'captureTimeSec': float(times[index]), 'bboxXYXY': bbox,
                    'inferenceSec': elapsed, 'inferenceTimingIncludes': ['preprocess', 'onnx-session', 'SimCC-postprocess'],
                    'rawPath': raw_name, 'rawSha256': digest(output / raw_name), 'overlayPath': overlay_name,
                    'outputShapes': {name: list(array.shape) for name, array in arrays.items()},
                    'rawXYZRange': {'min': arrays['keypoints'][0].min(axis=0).tolist(),
                                    'max': arrays['keypoints'][0].max(axis=0).tolist()},
                    'rawScoreRange': [float(arrays['scores'].min()), float(arrays['scores'].max())]}
            if baseline is not None:
                target = baseline[index, :17]
                candidate, confidence = arrays['keypoints_2d'][0, :17], arrays['scores'][0, :17]
                common = np.isfinite(target).all(axis=1) & (target[:, 2] >= BBOX_CONFIDENCE) & (confidence >= BBOX_CONFIDENCE)
                item['image2DComparison'] = {'commonBodyJoints': int(common.sum()),
                    'meanPixelDistance': float(np.linalg.norm(candidate[common] - target[common, :2] * [frame.shape[1], frame.shape[0]], axis=1).mean()) if common.any() else None,
                    'depthValidated': False}
            report['frames'].append(item)
            write_report(output, report)
        report['status'] = 'raw_evidence_collected'
    except Exception as error:
        report.update(status='failed', errorCode=getattr(error, 'code', 'probe_failed'), error=str(error))
    finally:
        report['elapsedSec'] = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peakProcessRamBytes'] = int(peak if sys.platform == 'darwin' else peak * 1024)
        report['memoryMeasurement'] = 'process-lifetime-peak-rss; not model-only allocation'
        write_report(output, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input', required=True)
    result.add_argument('--input-kind', required=True, choices=['capture-proxy'])
    result.add_argument('--timestamps', required=True)
    result.add_argument('--pose-model', required=True)
    result.add_argument('--bbox', type=parse_bbox, help='同一目标的固定像素 x1,y1,x2,y2；省略时使用每帧二维骨架')
    result.add_argument('--keypoints2d', help='原 cwb-65 [T,65,3] f32，使用同一采集帧')
    result.add_argument('--frames', required=True, type=parse_frames)
    result.add_argument('--output-dir', required=True)
    result.add_argument('--authorization-evidence', required=True)
    result.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    result.add_argument('--color-order', choices=['bgr', 'rgb'], default='bgr',
                        help='传入已安装RTMPose3d的通道顺序；默认保留库原路径BGR，可选RGB用于同帧诊断')
    return result


def main():
    args = parser().parse_args()
    try:
        report = run_probe(args)
    except (OSError, ProbeError) as error:
        report = {'status': 'failed', 'errorCode': getattr(error, 'code', 'io_error'), 'error': str(error)}
    print(json.dumps({'status': report['status'], 'errorCode': report.get('errorCode'),
                      'report': str(Path(args.output_dir).resolve() / 'report.json'), 'releaseReady': False}, ensure_ascii=False))
    return 0 if report['status'] == 'raw_evidence_collected' else 2


if __name__ == '__main__':
    sys.exit(main())
