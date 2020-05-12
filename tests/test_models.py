import os
import pytest
import torch

import cv2
from PIL import Image
import numpy as np
import mmcv

import pretorched
import pretorched.models as pmodels

from .. import config as cfg
from .. import models

dir_path = os.path.dirname(os.path.realpath(__file__))
BATCH_SIZE = 4
TEST_VIDEO_FILE = os.path.join(dir_path, 'data', 'aassnaulhq.mp4')
TEST_VIDEO_FILE = os.path.join(dir_path, 'data', 'aayfryxljh.mp4')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

video = mmcv.VideoReader(TEST_VIDEO_FILE)


def _frames_pil():
    return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]


def _frames_np():
    return np.stack([np.uint8(img) for img in _frames_pil()])


@pytest.fixture
def frames_pil():
    return _frames_pil()


@pytest.fixture
def frames_np():
    return _frames_np()


@pytest.fixture
def frames():
    return torch.as_tensor(_frames_np(), device=device).permute(0, 3, 1, 2).float()


@pytest.fixture
def frames3D():
    f = torch.as_tensor(_frames_np(), device=device).permute(0, 3, 1, 2).float()
    f = torch.stack([f[::12], f[::12]])
    # f = torch.stack([f, f])[0:1]
    # f = torch.stack([f, f])
    frames = f.permute(0, 2, 1, 3, 4)
    return frames


@pytest.fixture
def frames3D_small():
    return torch.randn(BATCH_SIZE, 3, 16, 224, 224, device=device)


@pytest.mark.skip
def test_facenet(frames):
    print(frames.shape)
    facenet = models.MTCNN(
        image_size=224, keep_all=True, device=device, post_process=False, margin=50
    )
    out = facenet(frames)
    for i, f in enumerate(out):
        print(i, f.shape)


def test_facenet3D_get_faces(frames3D):
    print(frames3D.shape)
    facenet = models.FaceModel(keep_all=True, select_largest=False, chunk_size=100)
    with torch.no_grad():
        out = facenet.get_faces(frames3D)


@pytest.mark.skip
def test_deepfake_detector(frames3D):
    facenet = models.FaceModel(device=device)
    detector = models.FrameModel(
        pretorched.resnet18(num_classes=2, pretrained=None), normalize=True
    )
    dfmodel = models.DeepfakeDetector(facenet, detector).to(device)
    out = dfmodel(frames3D)
    print(f'out: {out.shape}')


def test_manipulate_detector(frames3D_small):
    model = pmodels.resnet18()
    manipulator_detector = models.ManipulatorDetector(
        manipulator_model=models.MagNet(), detector_model=models.FrameModel(model)
    ).to(device)
    out = manipulator_detector(frames3D_small)
    print(out.shape)
