"""COCO-WholeBody -> cwb-65, matching the stable 2D index contract."""
import numpy as np

CWB_INDICES = tuple(range(23)) + tuple(range(91, 133))


def missing_pose(frame_count=None):
    shape = (65, 4) if frame_count is None else (frame_count, 65, 4)
    result = np.full(shape, np.nan, dtype='<f4')
    result[..., 3] = 0
    return result


def to_cwb65(points, scores):
    points, scores = np.asarray(points), np.asarray(scores)
    if points.shape != (133, 3) or scores.shape != (133,):
        raise ValueError('pose3d_model_must_output_133_keypoints')
    result = missing_pose()
    xyz, confidence = points[list(CWB_INDICES)], scores[list(CWB_INDICES)]
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(confidence) & (confidence > 0) & (confidence <= 1)
    result[valid, :3], result[valid, 3] = xyz[valid], confidence[valid]
    return result
