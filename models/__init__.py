from .base import DeepfakeDetector, FrameDetector, SeriesManipulatorDetector, GradCamCaricatureModel
from .utils import Normalize
from .face_extraction import MTCNN, FaceModel
from .xception import Xception, xception
from .mxresnet import mxresnet18, mxresnet50, MXResNet
from .deep_motion_mag.network import MagNet, Manipulator

__all__ = [
    'xception',
    'Xception',
    'FrameDetector',
    'FaceModel',
    'DeepfakeDetector',
    'MTCNN',
    'Normalize',
    'ManipulatorDectector',
    'MagNet',
    'Manipulator',
    'GradCamCaricatureModel'
    'mxresnet18',
    'mxresnet50',
    'MXResNet',
]
