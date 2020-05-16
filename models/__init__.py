from .base import DeepfakeDetector, FrameModel, ManipulatorDetector, Normalize, CaricatureModel
from .face_extraction import MTCNN, FaceModel
from .xception import Xception, xception
from .deep_motion_mag.network import MagNet, Manipulator

__all__ = [
    'xception',
    'Xception',
    'FrameModel',
    'FaceModel',
    'DeepfakeDetector',
    'MTCNN',
    'Normalize',
    'ManipulatorDectector',
    'MagNet',
    'Manipulator',
    'CaricatureModel'
]
