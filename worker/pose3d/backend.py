"""Provider boundary: adapters must document coordinates before returning data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np


class Pose3DError(Exception):
    def __init__(self, code: str, message: str, reject: bool = False):
        super().__init__(message)
        self.code, self.reject = code, reject


@dataclass(frozen=True)
class Candidate:
    # Full COCO-WholeBody order, image-normalized xy for existing 2D association.
    points2d: np.ndarray  # (133, 2)
    points3d: np.ndarray  # (133, 3), provider's documented raw coordinates
    scores: np.ndarray    # (133,), confidence in [0, 1]


class Pose3DProvider(Protocol):
    def load(self) -> None: ...
    def infer_frame(self, frame: Any, target_bbox: tuple | None) -> list[Candidate]: ...
    def metadata(self) -> dict: ...
    def close(self) -> None: ...


def create_provider(model_id: str, config: dict) -> Pose3DProvider:
    if model_id == 'rtmw3d-rtmlib-x':
        from .rtmw3d_rtmlib import RTMW3DRtmlib
        return RTMW3DRtmlib(config)
    if model_id == 'rtmw3d-mmpose-l':
        from .rtmw3d_mmpose import RTMW3DMMPose
        return RTMW3DMMPose(config)
    if model_id == 'motionbert-experimental':
        raise Pose3DError('model_unavailable', 'MotionBERT 的 H36M 映射、权重和坐标尚未验证。')
    raise Pose3DError('unsupported_model', '不支持的 3D 模型。')


class UnverifiedProvider:
    """Fail closed until real P0 evidence establishes the adapter's contract.

    Merely supplying an ONNX path, or allowing downloads for the existing 2D
    worker, must never promote this placeholder into a working 3D provider.
    """
    model_id = ''
    runtime = ''

    def __init__(self, config: dict):
        self.config = config

    def load(self):
        raise Pose3DError('model_unavailable',
                          f'{self.model_id} 尚未通过 P0 真实视频、坐标及权重核验；未启用 3D 推理。')

    def infer_frame(self, frame, target_bbox):
        raise Pose3DError('model_unavailable', '未验证的模型不产生姿态数据。')

    def metadata(self):
        return {'modelId': self.model_id, 'runtime': self.runtime, 'validationStatus': 'unverified'}

    def close(self):
        pass
