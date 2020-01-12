import torch
import torch.nn as nn
from torch.nn import Parameter as P

from pretorched.models.detection.facenet import MTCNN
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


class FaceModel(torch.nn.Module):

    def __init__(self, size=224, device='cuda', margin=50, keep_all=False,
                 post_process=False, select_largest=True):
        super().__init__()
        self.model = MTCNN(image_size=size,
                           device=device,
                           margin=margin,
                           keep_all=keep_all,
                           post_process=post_process,
                           select_largest=select_largest)

    def forward(self, x):
        """
        Args:
            x: [bs, nc, d, h, w]
        NOTE: For now, assume keep_all=False for 1 face per frame.
        lists of lists or nested tensors should be used to handle variable
        number of faces per example in batch (avoid this for now).
        """
        bs, nc, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [bs, d, nc, h, w]
        # x = x.view(-1, *x.shape[2:])  # [bs * d, nc, h, w]
        x = x.reshape(-1, *x.shape[2:])  # [bs * d, nc, h, w]
        out = self.model(x)
        for i, o in enumerate(out):
            if o is None:
                try:
                    out[i] = out[i-1]
                except IndexError:
                    pass
        out = torch.stack(out)
        out = out.view(bs, d, nc, *out.shape[-2:])
        out = out.permute(0, 2, 1, 3, 4)  # [bs, nc, d, h, w]
        return out


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
