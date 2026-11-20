#!/usr/bin/env python3
"""Isolated official MMPose RTMW3D-L CPU research probe; no product f32 output."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np

from pose3d_probe import (ProbeError, require, existing_local_file, authorize, digest,
    load_timestamps, load_keypoints, validate_proxy_clock, decode_frames,
    bbox_from_2d, validate_bbox, parse_bbox, parse_frames, draw_overlay,
    write_report, MAX_PROBE_FRAMES, BBOX_CONFIDENCE, BBOX_PADDING)

VERSION = 'pose3d-mmpose-raw-probe-0.1.0'
RAW_SHAPES = {'decoded_keypoints': (1, 133, 3), 'keypoints_simcc': (1, 133, 3),
              'scores': (1, 133), 'official_camera_keypoints': (1, 133, 3), 'keypoints_2d': (1, 133, 2)}


def array(value, shape, name):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    require(result.shape == shape and np.issubdtype(result.dtype, np.number)
            and not np.iscomplexobj(result) and np.isfinite(result).all(),
            'invalid_model_output', f'{name}: expected finite array {shape}')
    return result.copy()


def validate_prediction(decoded, samples):
    require(isinstance(decoded, list) and len(decoded) == 1
            and isinstance(samples, list) and len(samples) == 1,
            'invalid_model_output', 'One explicit bbox must produce exactly one prediction')
    original, final = decoded[0], samples[0].pred_instances
    return {
        'decoded_keypoints': array(original['keypoints'], (1, 133, 3), 'decoded_keypoints'),
        'keypoints_simcc': array(original['keypoints_simcc'], (1, 133, 3), 'keypoints_simcc'),
        'scores': array(original['keypoint_scores'], (1, 133), 'scores'),
        'official_camera_keypoints': array(final.keypoints, (1, 133, 3), 'official_camera_keypoints'),
        'keypoints_2d': array(final.transformed_keypoints, (1, 133, 2), 'keypoints_2d'),
    }


def validate_raw_arrays(values):
    require(isinstance(values, dict) and set(values) == set(RAW_SHAPES), 'invalid_model_output')
    return {name: array(values[name], shape, name) for name, shape in RAW_SHAPES.items()}


def load_mmpose(config_path, checkpoint_path, source_root, threads):
    config_path = existing_local_file(config_path, 'config_unavailable')
    checkpoint_path = existing_local_file(checkpoint_path, 'model_unavailable')
    source_root = Path(source_root).expanduser().resolve()
    project_root = source_root / 'projects/rtmpose3d'
    require((project_root / 'rtmpose3d/__init__.py').is_file(), 'source_unavailable')
    require(config_path.parent == project_root / 'configs'
            and config_path.name == 'rtmw3d-l_8xb64_cocktail14-384x288.py', 'unsupported_config')
    require(checkpoint_path.suffix == '.pth', 'invalid_checkpoint')
    require(type(threads) is int and 1 <= threads <= 16, 'invalid_threads')
    # Keep plotting caches inside the research environment and import the actual
    # custom project. No dummy mmcv modules or fake operators are installed.
    os.environ.setdefault('MPLCONFIGDIR', str(source_root.parent / 'matplotlib-cache'))
    sys.path.insert(0, str(project_root))
    import torch
    from mmengine.config import Config
    from mmpose.apis import init_model, inference_topdown
    import mmpose
    require(mmpose.__version__ == '1.3.2', 'unsupported_mmpose_version')
    torch.set_num_threads(threads)
    config = Config.fromfile(str(config_path))
    require(config.model.type == 'TopdownPoseEstimator3D'
            and config.model.head.type == 'RTMW3DHead'
            and config.model.data_preprocessor.bgr_to_rgb is True, 'unsupported_config')
    # init_model clears this by default; make the disabled remote backbone
    # initialization explicit as well. The full local checkpoint is mandatory.
    config.model.backbone.init_cfg = None
    config.model.test_cfg.mode = '3d'
    with contextlib.redirect_stdout(sys.stderr):
        model = init_model(config, str(checkpoint_path), device='cpu')
    require(model.dataset_meta.get('num_keypoints') == 133, 'invalid_dataset_metadata')
    captured = []
    original_decode = model.head.decode

    def observed_decode(outputs):
        predictions = original_decode(outputs)
        captured[:] = [{key: np.array(getattr(prediction, key), copy=True)
                       for key in ('keypoints', 'keypoints_simcc', 'keypoint_scores')}
                      for prediction in predictions]
        return predictions  # observation only: preserve original values/objects

    model.head.decode = observed_decode

    def infer(frame, bbox):
        captured.clear()
        with contextlib.redirect_stdout(sys.stderr):
            samples = inference_topdown(model, frame, np.array([bbox], dtype=np.float32), bbox_format='xyxy')
        return validate_prediction(captured, samples)

    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'mmengine', 'mmcv', 'mmcv-lite', 'mmdet', 'mmpose', 'opencv-python', 'setuptools'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    source_files = ('rtmpose3d/pose_estimator.py', 'rtmpose3d/rtmw3d_head.py', 'rtmpose3d/simcc_3d_label.py')
    decoder = model.head.decoder
    root_indices = decoder.root_index if isinstance(decoder.root_index, (tuple, list)) else [decoder.root_index]
    metadata = {'runtime': 'mmpose-pytorch', 'modelId': 'rtmw3d-mmpose-l', 'device': 'cpu',
        'torchThreads': threads, 'sourceRef': 'v1.3.2', 'sourceRoot': str(source_root),
        'projectSourceSha256': {name: digest(project_root / name) for name in source_files},
        'dependencyVersions': versions, 'configPath': str(config_path), 'configSha256': digest(config_path),
        'configOverrides': {'model.backbone.init_cfg': None, 'model.test_cfg.mode': '3d'},
        'effectiveConfigSha256': __import__('hashlib').sha256(config.pretty_text.encode()).hexdigest(),
        'decoderObservation': 'copy official head.decode outputs before camera postprocess; no numerical replacement',
        'cameraAssumptions': {'calibrated': False, 'source': 'official defaults; not measured for this video',
                             'focalXY': [1145.04940459, 1143.78109572],
                             'principalPoint': 'official np.array(ori_shape)/2 (height,width order)',
                             'rootDepth': 5.14388},
        'preprocess': {'inputColor': 'bgr', 'modelBgrToRgb': True, 'inputSize': [288, 384]},
        'sourceGeometry': {'codecInputSize': list(decoder.input_size),
            'simccSplitRatio': float(decoder.simcc_split_ratio), 'zRange': float(decoder.z_range),
            'useDark': bool(decoder.use_dark), 'rootIndices': list(root_indices),
            'decodeDepthDenominator': float(decoder.input_size[2] / 2), 'isotropicXYZValidated': False}}
    return infer, metadata


def run_probe(args, model_loader=None, cv_module=None, clock_validator=None):
    started = time.perf_counter()
    output = Path(args.output_dir).expanduser().resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'output_not_empty')
    output.mkdir(parents=True, exist_ok=True)
    report = {'schemaVersion': VERSION, 'status': 'running', 'evaluationMode': True,
        'releaseReady': False, 'productCompatible': False, 'rawCoordinatesValidated': False,
        'coordinateStatus': 'official-decoder-and-assumed-camera-unvalidated',
        'createdAt': datetime.now(timezone.utc).isoformat(), 'frames': [],
        'limitations': ['camera_intrinsics_and_root_depth_are_assumptions', 'decoded_xyz_mixes_crop_xy_and_depth',
                        'raw_score_is_not_depth_confidence', 'not_product_3d_or_accuracy_validation']}
    try:
        require(args.input_kind == 'capture-proxy', 'capture_proxy_required')
        source = existing_local_file(args.input, 'video_unavailable')
        report['authorization'] = authorize(args.authorization_evidence, digest(source))
        checkpoint = existing_local_file(args.checkpoint, 'model_unavailable')
        config = existing_local_file(args.config, 'config_unavailable')
        timestamps_path, times = load_timestamps(args.timestamps)
        require(isinstance(args.frames, list) and 1 <= len(args.frames) <= MAX_PROBE_FRAMES
                and len(set(args.frames)) == len(args.frames)
                and all(type(f) is int and 0 <= f < len(times) for f in args.frames), 'invalid_frames')
        clock = (clock_validator or validate_proxy_clock)(source, times)
        pose_path, baseline = load_keypoints(args.keypoints2d, len(times)) if args.keypoints2d else (None, None)
        require(args.bbox is not None or baseline is not None, 'target_bbox_required')
        report['input'] = {'kind': 'capture-proxy', 'path': str(source), 'sha256': digest(source),
            'frameCount': len(times), 'timestampsPath': str(timestamps_path),
            'timestampsSha256': digest(timestamps_path), 'clockSource': 'existing-2d-capture-timestamps',
            'proxyClockValidation': clock, 'keypoints2dPath': str(pose_path) if pose_path else None,
            'keypoints2dSha256': digest(pose_path) if pose_path else None}
        report['model'] = {'path': str(checkpoint), 'weightsSha256': digest(checkpoint),
                           'configPath': str(config), 'configSha256': digest(config)}
        if cv_module is None:
            import cv2 as cv_module
        decode_start = time.perf_counter()
        frames = decode_frames(source, set(args.frames), len(times), cv_module)
        report['decodeSec'] = time.perf_counter() - decode_start
        require(all(frame.shape[1] == clock['width'] and frame.shape[0] == clock['height'] for frame in frames.values()),
                'proxy_dimensions_mismatch')
        bboxes = {index: validate_bbox(args.bbox, frame.shape[1], frame.shape[0]) if args.bbox is not None
                  else bbox_from_2d(baseline[index], frame.shape[1], frame.shape[0]) for index, frame in frames.items()}
        report['bboxPolicy'] = {'origin': 'explicit-pixel-bbox' if args.bbox else 'same-frame-existing-2d',
            'confidenceThreshold': BBOX_CONFIDENCE, 'extentPaddingRatio': BBOX_PADDING,
            'detectorUsed': False, 'identityValidation': 'not_independently_validated'}
        load_start = time.perf_counter()
        infer, metadata = (model_loader or load_mmpose)(config, checkpoint, args.source_root, args.threads)
        report['modelLoadSec'] = time.perf_counter() - load_start
        report['model'].update(metadata)
        for index in sorted(args.frames):
            frame = frames.pop(index)
            infer_start = time.perf_counter()
            arrays = validate_raw_arrays(infer(frame, bboxes[index]))
            elapsed = time.perf_counter() - infer_start
            raw_name = f'frame-{index:06d}.mmpose-raw.npz'
            np.savez_compressed(output / raw_name, **arrays, frame_index=np.array(index, dtype=np.int64),
                timestamp_sec=np.array(times[index], dtype=np.float64), bbox_xyxy=np.asarray(bboxes[index]))
            overlay_name = f'frame-{index:06d}.overlay.png'
            overlay = draw_overlay(frame, arrays, bboxes[index], baseline[index] if baseline is not None else None,
                                   index, float(times[index]), cv_module)
            require(cv_module.imwrite(str(output / overlay_name), overlay), 'overlay_write_failed')
            item = {'frame': index, 'captureTimeSec': float(times[index]), 'bboxXYXY': bboxes[index],
                'rawPath': raw_name, 'rawSha256': digest(output / raw_name), 'overlayPath': overlay_name,
                'inferenceSec': elapsed, 'outputShapes': {name: list(value.shape) for name, value in arrays.items()},
                'rawScoreRange': [float(arrays['scores'].min()), float(arrays['scores'].max())]}
            if baseline is not None:
                target = baseline[index, :17]
                common = np.isfinite(target).all(axis=1) & (target[:, 2] >= BBOX_CONFIDENCE) & (arrays['scores'][0, :17] >= BBOX_CONFIDENCE)
                item['image2DComparison'] = {'commonBodyJoints': int(common.sum()), 'depthValidated': False,
                    'meanPixelDistance': float(np.linalg.norm(arrays['keypoints_2d'][0, :17][common] - target[common, :2]
                        * [frame.shape[1], frame.shape[0]], axis=1).mean()) if common.any() else None}
            report['frames'].append(item)
            write_report(output, report)
        report['status'] = 'raw_evidence_collected'
    except Exception as error:
        report.update(status='failed', errorCode=getattr(error, 'code', 'mmpose_probe_failed'), error=str(error))
        (output / 'failure-traceback.txt').write_text(traceback.format_exc())
    finally:
        report['elapsedSec'] = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peakProcessRamBytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
        write_report(output, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'timestamps', 'checkpoint', 'config', 'source-root', 'output-dir', 'authorization-evidence'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--input-kind', required=True, choices=['capture-proxy'])
    result.add_argument('--frames', type=parse_frames, required=True)
    result.add_argument('--keypoints2d')
    result.add_argument('--bbox', type=parse_bbox)
    result.add_argument('--threads', type=int, default=4)
    return result


def main():
    args = parser().parse_args()
    report = run_probe(args)
    print(json.dumps({'status': report['status'], 'errorCode': report.get('errorCode'),
                     'report': str(Path(args.output_dir).resolve() / 'report.json'), 'releaseReady': False}))
    return 0 if report['status'] == 'raw_evidence_collected' else 2


if __name__ == '__main__':
    sys.exit(main())
