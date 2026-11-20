"""Internal consistency signals; these never establish correctness of depth."""
import numpy as np

BODY_JOINTS = tuple(range(5, 17))
BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16))


def clean_window(pose, policy):
    result = np.array(pose, dtype='<f4', copy=True)
    if result.ndim != 3 or result.shape[1:] != (65, 4):
        raise ValueError('invalid_pose3d_shape')
    valid = (np.isfinite(result).all(axis=2) & (result[:, :, 3] >= policy['minConfidence'])
             & (result[:, :, 3] > 0) & (result[:, :, 3] <= 1))
    result[~valid] = [np.nan, np.nan, np.nan, 0]
    warnings = set()
    # Compare measured lengths within this one window only. Missing spans and
    # outliers stay missing; no last-frame carry forward is permitted.
    for parent, child in BONES:
        good = valid[:, parent] & valid[:, child]
        lengths = np.linalg.norm(result[:, parent, :3] - result[:, child, :3], axis=1)
        reference = np.median(lengths[good]) if good.any() else 0
        if reference <= 0:
            continue
        abnormal = good & (np.abs(lengths / reference - 1) > policy['maxBoneLengthDriftRatio'])
        if abnormal.any():
            result[abnormal, child] = [np.nan, np.nan, np.nan, 0]
            warnings.add('bone_length_drift')
    root_missing = (result[:, 11, 3] == 0) | (result[:, 12, 3] == 0)
    result[root_missing] = [np.nan, np.nan, np.nan, 0]
    if root_missing.any():
        warnings.add('root_missing')
    return result, sorted(warnings)


def summarize(pose, windows, identity_locked_frames=(), warnings=()):
    pose = np.asarray(pose)
    valid = np.isfinite(pose).all(axis=2) & (pose[:, :, 3] > 0) & (pose[:, :, 3] <= 1)
    usable = valid[:, list(BODY_JOINTS)].all(axis=1)
    covered = valid.any(axis=1)
    selected = np.zeros(len(pose), dtype=bool)
    result_windows = []
    warnings = set(warnings)
    for window in windows:
        start, end = window['startFrame'], window['endFrame'] + 1
        if window['viewGate'] != 'blocked':
            selected[start:end] = True
        if window['viewGate'] == 'limited':
            warnings.add('view_limited')
        result_windows.append({**window, 'usableFrameRatio': float(np.mean(usable[start:end]))})
    count = int(selected.sum())
    locked = set(identity_locked_frames)
    quality = {
        'coverageRatio': float(covered.mean()) if len(pose) else 0,
        'usableFrameRatio': float(usable[selected].mean()) if count else 0,
        'identityLockedRatio': sum(1 for frame in locked if 0 <= frame < len(pose) and selected[frame]) / count if count else 0,
        'meanJointConfidence': float(pose[:, :, 3][valid].mean()) if valid.any() else 0,
        'warnings': sorted(warnings),
    }
    return quality, result_windows
