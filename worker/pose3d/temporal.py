"""Optional timestamp-aware smoothing, with no interpolation or stale poses.

Each independent window gets a fresh filter. Call with margin observations and
crop afterwards to warm up the filter without publishing extra frames. Disabled
by default pending raw-vs-filtered peak and timing measurements in P0.
"""
import math
import numpy as np
from .windows import validate_timestamps


def smooth_window(pose, times, policy):
    pose = np.array(pose, dtype='<f4', copy=True)
    times = validate_timestamps(times, len(pose))
    if pose.ndim != 3 or pose.shape[1:] != (65, 4):
        raise ValueError('invalid_pose3d_shape')
    valid = np.isfinite(pose).all(axis=2) & (pose[:, :, 3] > 0) & (pose[:, :, 3] <= 1)
    pose[~valid] = [np.nan, np.nan, np.nan, 0]
    if not policy['enabled']:
        return pose
    for name in ('maxGapSec', 'minCutoff', 'derivativeCutoff'):
        if not math.isfinite(policy[name]) or policy[name] <= 0:
            raise ValueError('invalid_temporal_policy')
    if (not math.isfinite(policy['beta']) or policy['beta'] < 0
            or not 0 <= policy['observationWeight'] <= 1):
        raise ValueError('invalid_temporal_policy')
    for joint in range(65):
        observations = pose[:, joint, :3].copy()

        def filtered(order):
            result = observations.copy()
            previous_raw = previous_filtered = previous_time = None
            derivative = np.zeros(3)
            for frame in order:
                if not valid[frame, joint]:
                    previous_raw = previous_filtered = previous_time = None
                    derivative = np.zeros(3)
                    continue
                value, t = observations[frame], times[frame]
                dt = abs(t - previous_time) if previous_time is not None else None
                if previous_time is not None and dt <= policy['maxGapSec']:
                    alpha_d = 1 / (1 + 1 / (2 * math.pi * policy['derivativeCutoff'] * dt))
                    derivative = alpha_d * (value - previous_raw) / dt + (1 - alpha_d) * derivative
                    cutoff = policy['minCutoff'] + policy['beta'] * np.linalg.norm(derivative)
                    alpha = 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
                    result[frame] = alpha * value + (1 - alpha) * previous_filtered
                else:
                    derivative = np.zeros(3)
                previous_raw, previous_filtered, previous_time = value, result[frame].copy(), t
            return result

        symmetric = (filtered(range(len(times))) + filtered(range(len(times) - 1, -1, -1))) / 2
        weight = policy['observationWeight']
        pose[:, joint, :3] = weight * observations + (1 - weight) * symmetric
    return pose
