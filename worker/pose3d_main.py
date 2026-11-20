#!/usr/bin/env python3
"""Independent 3D job CLI. P0 integration gates currently fail closed.

Input: --request JSON --output-dir DIR. Stdout's final line is a terminal
{pose3d, error?} object. No dependency or model download is performed here.
This does not change or overwrite any existing 2D result.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


def run(request, output_dir):
    from pose3d.backend import Pose3DError, create_provider
    from pose3d.windows import validate_timestamps
    import numpy as np

    result = copy.deepcopy(request['pose3d'])
    result.pop('keypointsKey', None)
    result.pop('errorCode', None)
    provider = None
    try:
        analysis, video, config = request['analysis'], request['video'], request['config']
        if analysis.get('status') != 'succeeded' or not analysis.get('skeleton'):
            raise Pose3DError('analysis_not_ready', '需要先完成二维分析。')
        count = result['frameCount']
        if type(count) is not int or not 0 < count <= config['limits']['maxFrames']:
            raise Pose3DError('invalid_frame_count', '无效的 3D 帧数。', True)
        if count != analysis['skeleton']['frameCount'] or count != video['frameCount']:
            raise Pose3DError('frame_mismatch', '3D 与原视频、二维姿态帧数不一致。', True)
        timestamps_path, keypoints_path = Path(request['timestampsPath']), Path(request['keypointsPath'])
        if timestamps_path.stat().st_size != count * 4 or keypoints_path.stat().st_size != count * 65 * 3 * 4:
            raise Pose3DError('invalid_binary', '二维输入文件长度不符。', True)
        validate_timestamps(np.fromfile(timestamps_path, dtype='<f4'), count)
        points = np.fromfile(keypoints_path, dtype='<f4').reshape(count, 65, 3)
        xy, scores = points[:, :, :2], points[:, :, 2]
        valid = (np.isfinite(xy).all(axis=2) & (xy >= 0).all(axis=2) & (xy <= 1).all(axis=2)
                 & np.isfinite(scores) & (scores >= 0) & (scores <= 1))
        missing = np.isnan(xy).all(axis=2) & (scores == 0)
        if not (valid | missing).all():
            raise Pose3DError('invalid_binary', '二维输入文件包含无效关键点。', True)
        if not Path(request['inputPath']).is_file():
            raise Pose3DError('invalid_video', '未找到与采集时钟对齐的代理视频。')
        windows = result['windows']
        if not windows:
            raise Pose3DError('no_swing_windows', '未识别到击球片段，可先手动标帧。', True)
        from pose3d.io import validate_pose
        from pose3d.mapping import missing_pose
        validate_pose(missing_pose(count), count, windows)
        if all(window['viewGate'] == 'blocked' for window in windows):
            raise Pose3DError('view_blocked', '视角未知或置信度不足，无法展示 3D。', True)
        if result['viewGatePolicyVersion'] != config['viewGatePolicyVersion']:
            raise Pose3DError('policy_mismatch', '视角门控策略版本不一致。')
        provider = create_provider(result['modelId'], config)
        provider.load()
        # Deliberate stop: a future adapter implementation still cannot bypass
        # P0 by merely changing load(). The inference runner must be reviewed
        # alongside actual per-frame coordinate/identity/provenance evidence.
        raise Pose3DError('model_unavailable', '3D 推理流水线尚未通过 P0 验证。')
    except Pose3DError as error:
        result.update(status='rejected_quality' if error.reject else 'failed', errorCode=error.code)
        return {'pose3d': result, 'error': str(error)}
    except (KeyError, TypeError, ValueError, OSError) as error:
        result.update(status='failed', errorCode='invalid_request')
        return {'pose3d': result, 'error': f'无效的 3D 输入：{error}'}
    finally:
        if provider is not None:
            provider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request).read_text())
        response = run(request, Path(args.output_dir))
    except Exception as error:
        # Includes missing numpy in an unconfigured environment. Never fabricate
        # provenance or an output binary when the input contract cannot be read.
        result = request.get('pose3d') if 'request' in locals() and isinstance(request, dict) else None
        response = {'error': f'3D Worker 不可用：{error}', 'errorCode': 'model_unavailable'}
        if isinstance(result, dict):
            result = {**result, 'status': 'failed', 'errorCode': 'model_unavailable'}
            result.pop('keypointsKey', None)
            response['pose3d'] = result
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return 0 if response.get('pose3d') else 1


if __name__ == '__main__':
    sys.exit(main())
