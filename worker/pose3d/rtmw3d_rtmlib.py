"""Path A integration gate; no guessed library API or coordinate conversion.

Official source currently returns crop-space XY and separately scaled depth:
https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmpose3d.py
An axis sign matrix alone cannot turn those raw values into isotropic XYZ.
The future adapter must first recover/validate the provider's actual geometry.
"""
from .backend import UnverifiedProvider


class RTMW3DRtmlib(UnverifiedProvider):
    model_id = 'rtmw3d-rtmlib-x'
    runtime = 'rtmlib-onnx'
