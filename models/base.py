import torch
import torch.nn as nn
from torch.nn import Parameter as P

from pretorched.utils import chunk


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 shape=(1, -1, 1, 1, 1), rescale=True):
        super().__init__()
        self.shape = shape
        self.mean = P(torch.tensor(mean).view(shape),
                      requires_grad=False)
        self.std = P(torch.tensor(std).view(shape),
                     requires_grad=False)
        self.rescale = rescale

    def forward(self, x, rescale=None):
        rescale = self.rescale if rescale is None else rescale
        x.div_(255.) if rescale else None
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
