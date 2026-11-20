"""Path B integration gate; isolate its future environment from the 2D worker."""
from .backend import UnverifiedProvider


class RTMW3DMMPose(UnverifiedProvider):
    model_id = 'rtmw3d-mmpose-l'
    runtime = 'mmpose-pytorch'
