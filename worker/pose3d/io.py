"""Strict Float32 LE contract and atomically published artifacts."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def validate_pose(pose, frame_count, windows):
    pose = np.asarray(pose)
    if pose.shape != (frame_count, 65, 4):
        raise ValueError('invalid_pose3d_shape')
    allowed = np.zeros(frame_count, dtype=bool)
    previous_end = -1
    for window in windows:
        start, end = window['startFrame'], window['endFrame']
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end < frame_count
                or start <= previous_end or window['viewGate'] not in ('pass', 'limited', 'blocked')):
            raise ValueError('invalid_windows')
        previous_end = end
        if window['viewGate'] != 'blocked':
            allowed[start:end + 1] = True
    xyz, scores = pose[:, :, :3], pose[:, :, 3]
    valid = np.isfinite(xyz).all(axis=2) & np.isfinite(scores) & (scores > 0) & (scores <= 1)
    missing = np.isnan(xyz).all(axis=2) & (scores == 0)
    if not (valid | missing).all():
        raise ValueError('invalid_pose3d_missing_pattern')
    if not missing[~allowed].all():
        raise ValueError('pose3d_outside_windows')
    # The format promises a hip-centered origin. A populated frame needs both
    # hip points; tolerate only floating-point rounding around zero.
    populated = valid.any(axis=1)
    if not valid[populated][:, [11, 12]].all():
        raise ValueError('pose3d_root_missing')
    if not np.allclose(xyz[populated][:, [11, 12]].mean(axis=1), 0, atol=1e-5):
        raise ValueError('pose3d_root_not_centered')


def read_pose(path, frame_count, windows):
    path = Path(path)
    if path.stat().st_size != frame_count * 65 * 4 * 4:
        raise ValueError('invalid_pose3d_length')
    pose = np.fromfile(path, dtype='<f4').reshape(frame_count, 65, 4)
    validate_pose(pose, frame_count, windows)
    return pose


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_pose(output_dir, pose, windows, name='pose3d.f32'):
    if name not in ('pose3d.f32', 'pose3d.raw.f32'):
        raise ValueError('invalid_pose3d_filename')
    pose = np.asarray(pose, dtype='<f4')
    validate_pose(pose, len(pose), windows)
    atomic_write(Path(output_dir) / name, pose.tobytes(order='C'))
    return name


def write_json(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode())
