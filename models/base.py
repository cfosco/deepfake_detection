import torch
import torch.nn as nn
from torch.nn import Parameter as P

from pretorched.utils import chunk
from pretorched.visualizers import grad_cam


class Normalize(nn.Module):
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        shape=(1, -1, 1, 1, 1),
        rescale=True,
    ):
        super().__init__()
        self.shape = shape
        self.mean = P(torch.tensor(mean).view(shape), requires_grad=False)
        self.std = P(torch.tensor(std).view(shape), requires_grad=False)
        self.rescale = rescale

    def forward(self, x, rescale=None):
        rescale = self.rescale if rescale is None else rescale
        x.div_(255.0) if rescale else None
        return (x - self.mean) / self.std


class FrameModel(torch.nn.Module):

    frame_dim = 2

    def __init__(self, model, consensus_func=torch.mean, normalize=False):
        super().__init__()
        self.model = model
        self.consensus_func = consensus_func
        self.norm = Normalize() if normalize else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(2, 0, 1, 3, 4)
        return self.consensus_func(torch.stack([self.model(f) for f in x]), dim=0)

    @property
    def input_size(self):
        return self.model.input_size


class DeepfakeDetector(torch.nn.Module):
    def __init__(self, face_model, fake_model):
        super().__init__()
        self.face_model = face_model
        self.fake_model = fake_model

    def forward(self, x):
        faces = self.face_model(x)
        return self.fake_model(faces)
        # min_faces = torch.min([f.shape[0] for f in faces])
        # batched_faces = [torch.stack([f[i] for f in faces]) for i in range(min_faces)]
        # return self.consensus_func(
        # torch.stack([self.detector(f) for f in batched_faces]), dim=0)


class ManipulatorDetector(torch.nn.Module):
    def __init__(self, manipulator_model, detector_model):
        super().__init__()
        self.manipulator_model = manipulator_model
        self.detector_model = detector_model

    def manipulate(self, x):
        return torch.stack(
            [self.manipulator_model.manipulate(f.transpose(0, 1)) for f in x]
        ).transpose(1, 2)

    def forward(self, x):
        # x: [bs, 3, D, H, W]
        o = self.manipulate(x)
        o = self.detector_model(o)
        return o

    @property
    def input_size(self):
        return self.detector_model.input_size


class CaricatureModel(torch.nn.Module):
    def __init__(
        self, face_model, fake_model, mag_model, norm=None, device='cuda', target_layer='layer4'
    ):
        super().__init__()
        self.face_model = face_model
        self.fake_model = fake_model
        self.mag_model = mag_model
        self.gcam_model = grad_cam.GradCAM(fake_model)
        self.target_layer = target_layer
        if norm is None:
            self.norm = Normalize().to(device)

    def face_forward(self, x):
        with torch.no_grad():
            faces = self.face_model(x)
            norm_faces = self.norm(faces)
        return faces, norm_faces

    def forward(self, x):
        faces, norm_faces = self.face_forward(x)
        self.gcam_model.zero_grad()
        probs, idx = self.gcam_model(norm_faces)
        self.gcam_model.backward(idx=idx[0])
        attn_maps = self.gcam_model.generate(target_layer=self.target_layer)
        # TODO: Reshape attn_maps for manipulator
