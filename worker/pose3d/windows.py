"""Seconds-based windows on the existing capture timestamps, inclusive frames."""
import math
import numpy as np
from .backend import Pose3DError


def validate_timestamps(times, frame_count=None):
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)
            or (frame_count is not None and len(times) != frame_count)):
        raise ValueError('invalid_timestamps')
    return times


def normalize_windows(windows, frame_count, origin):
    ranges = []
    for window in windows:
        start, end = window.get('startFrame'), window.get('endFrame')
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= end < frame_count):
            raise Pose3DError('invalid_windows', '3D 窗口帧号越界或顺序错误。', True)
        ranges.append((start, end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1]['endFrame'] + 1:
            merged[-1]['endFrame'] = max(end, merged[-1]['endFrame'])
        else:
            merged.append({'startFrame': start, 'endFrame': end, 'origin': origin})
    return merged


def select_windows(times, scope, config, events=(), manual=()):
    times = validate_timestamps(times)
    limits = config['limits']
    if len(times) > limits['maxFrames']:
        raise Pose3DError('scope_limit_exceeded', '3D 帧数超出上限。', True)
    if scope == 'full':
        duration = times[-1] + (float(np.median(np.diff(times))) if len(times) > 1 else 0)
        if len(times) > limits['maxFullFrames'] or duration > limits['maxFullDurationSec']:
            raise Pose3DError('scope_limit_exceeded', '整段 3D 处理超出时长或帧数上限。', True)
        return normalize_windows([{'startFrame': 0, 'endFrame': len(times) - 1}], len(times), 'full-clip')
    if scope == 'manual':
        if not manual or len(manual) > limits['maxWindows']:
            raise Pose3DError('invalid_windows', '手动处理需要有效且数量受限的窗口。', True)
        return normalize_windows(manual, len(times), 'manual-annotation')
    if scope != 'swing-windows':
        raise Pose3DError('invalid_scope', '未知的 3D 处理范围。', True)
    if not events:
        raise Pose3DError('no_swing_windows', '未识别到击球片段，可先手动标帧。', True)
    pre, post = config['windows']['preSec'], config['windows']['postSec']
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (pre, post)):
        raise Pose3DError('window_policy_unconfigured', '挥拍前后时长需在 P0 后明确配置。')
    windows = []
    for event in events:
        frame = event.get('impact', {}).get('frame')
        if type(frame) is not int or not 0 <= frame < len(times):
            raise Pose3DError('invalid_windows', '击球帧与采集时间轴不一致。', True)
        hit_time = times[frame]  # authoritative capture clock, not guessed FPS
        start = int(np.searchsorted(times, hit_time - pre, side='left'))
        end = int(np.searchsorted(times, hit_time + post, side='right') - 1)
        windows.append({'startFrame': start, 'endFrame': end})
    result = normalize_windows(windows, len(times), 'auto-swing-event')
    if len(result) > limits['maxWindows']:
        raise Pose3DError('scope_limit_exceeded', '3D 窗口数量超出上限。', True)
    return result


def margin_frames(window, times, margin_sec):
    times = validate_timestamps(times)
    if not math.isfinite(margin_sec) or margin_sec < 0:
        raise ValueError('invalid_margin')
    start, end = window['startFrame'], window['endFrame']
    return (int(np.searchsorted(times, times[start] - margin_sec, side='left')),
            int(np.searchsorted(times, times[end] + margin_sec, side='right') - 1))
