import os
import sys

import cv2
import mmcv
import numpy as np
import pytest
import torch
from PIL import Image

import pretorched
import pretorched.models as pmodels

from .. import models

try:
    sys.path.extend(['.', '..'])
    import core
except ImportError:
    SKIP_GET_TESTS = True
else:
    SKIP_GET_TESTS = False


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
    facenet = models.FaceModel(
        keep_all=True, select_largest=False, chunk_size=100, device=device
    )
    with torch.no_grad():
        out = facenet.get_faces(frames3D)
        print(out)


@pytest.mark.skip
def test_deepfake_detector(frames3D):
    facenet = models.FaceModel(device=device)
    detector = models.FrameDetector(
        pretorched.resnet18(num_classes=2, pretrained=None), normalize=True
    )
    dfmodel = models.DeepfakeDetector(facenet, detector).to(device)
    out = dfmodel(frames3D)
    print(f'out: {out.shape}')


def test_manipulate_detector(frames3D_small):
    model = pmodels.resnet18()
    manipulator_detector = models.SeriesManipulatorDetector(
        manipulator_model=models.MagNet(), detector_model=models.FrameDetector(model)
    ).to(device)
    out = manipulator_detector(frames3D_small)
    print(out.shape)


def test_manipulate_video(frames3D_small):
    model = models.MagNet().to(device)
    out = model.manipulate_video(frames3D_small)
    assert tuple(out.shape) == frames3D_small.shape


@pytest.mark.parametrize(
    'model_name, basemodel_name',
    [('FrameDetector', 'resnet18'), ('SeriesManipulatorDetector', 'resnet18')],
)
def test_get_model(model_name, basemodel_name, frames3D_small):
    model = core.get_model(model_name, basemodel_name)
    model = model.to(device)
    out = model(frames3D_small)
    assert tuple(out.size()) == (BATCH_SIZE, 2)


def test_manipulator_with_attn(frames3D_small):
    manipulator_model = models.MagNet().to(device)
    frames = frames3D_small[0].transpose(0, 1)
    attn_map = torch.randn(16, 8, 8).to(device)
    out = manipulator_model.manipulate(frames, attn_map=attn_map)
    assert tuple(out.shape) == (16, 3, 224, 224)


def test_caricature_model(frames3D_small):
    model = core.get_model('GradCamCaricatureModel').to(device)
    # frames = frames3D_small[0]
    out, attn_maps = model(frames3D_small[0:1])
    assert tuple(out.size()) == (1, 3, 16, 224, 224)


@pytest.mark.skip
def test_smooth_attn():
    from models.base import smooth_attn

    attn_maps = torch.randn(16, 8, 8)
    smoothed = smooth_attn(attn_maps)
    print(smoothed.shape)


def test_simple_self_attn_detector(frames3D_small):
    model = models.AttnFrameDetector(model=core.get_basemodel('ssamxresnet18'))
    model = model.to(device)
    out, attn_map = model(frames3D_small)
    assert tuple(out.shape) == (BATCH_SIZE, 2)
    assert tuple(attn_map.shape) == (BATCH_SIZE, frames3D_small.size(2), 56, 56)


def test_self_attn_detector(frames3D_small):
    model = models.AttnFrameDetector(model=core.get_basemodel('samxresnet18'))
    model = model.to(device)
    out, attn_map = model(frames3D_small)
    assert tuple(out.shape) == (BATCH_SIZE, 2)
    assert tuple(attn_map.shape) == (BATCH_SIZE, frames3D_small.size(2), 56, 56)


def test_ResManipulatorAttnDetector(frames3D_small):
    model = core.get_model('ResPretrainedManipulatorAttnDetector')
    model = model.to(device)
    out = model(frames3D_small)
    print(out.shape)
