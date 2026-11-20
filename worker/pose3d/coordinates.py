"""Explicit provider coordinate transform; no implicit sign or unit guesses."""
import numpy as np
from .mapping import missing_pose


def validate_transform(transform):
    if not isinstance(transform, dict):
        raise ValueError('coordinate_transform_required')
    if any(not isinstance(transform.get(key), str) or not transform[key].strip()
           for key in ('sourceConvention', 'evidence')):
        raise ValueError('coordinate_evidence_required')
    if transform.get('targetConvention') != 'x-right-y-up-z-forward':
        raise ValueError('invalid_target_convention')
    matrix = np.asarray(transform.get('matrix'), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('invalid_coordinate_matrix')
    # Coordinate conversion may change axis signs/order and uniform scale, but
    # must not stretch different body axes or skew angles.
    gram = matrix @ matrix.T
    scale_squared = float(np.trace(gram) / 3)
    if scale_squared <= 0 or not np.allclose(gram, np.eye(3) * scale_squared):
        raise ValueError('coordinate_matrix_must_be_uniform_orthogonal')
    return matrix


def normalize_root(pose, transform):
    matrix = validate_transform(transform)
    pose = np.asarray(pose)
    if pose.shape != (65, 4):
        raise ValueError('invalid_pose3d_shape')
    result = missing_pose()
    valid = np.isfinite(pose).all(axis=1) & (pose[:, 3] > 0) & (pose[:, 3] <= 1)
    if not valid[[11, 12]].all():
        return result  # no independently invented root when either hip is absent
    root = pose[[11, 12], :3].mean(axis=0)
    result[valid, :3] = (pose[valid, :3] - root) @ matrix.T
    result[valid, 3] = pose[valid, 3]
    return result
