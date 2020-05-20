import torch
import torch.nn as nn

from pretorched.visualizers import grad_cam

from .utils import Normalize


class Detector(torch.nn.Module):
    def __init__(
        self, model, consensus_func=nn.Identity(), normalize=False, rescale=True
    ):
        super().__init__()
        self.model = model
        self.consensus_func = consensus_func
        self.norm = Normalize(rescale) if normalize else nn.Identity()

    @property
    def input_size(self):
        return self.model.input_size


class FrameDetector(Detector):

    frame_dim = 2

    def __init__(self, model, consensus_func=torch.mean, normalize=False, rescale=True):
        super().__init__(model, consensus_func, normalize, rescale)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(2, 0, 1, 3, 4)
        return self.consensus_func(torch.stack([self.model(f) for f in x]), dim=0)


class VideoDetector(Detector):
    def __init__(
        self, model, consensus_func=nn.Identity(), normalize=False, rescale=True
    ):
        super().__init__(model, consensus_func, normalize, rescale)

    def forward(self, x):
        x = self.norm(x)
        x = self.model(x)
        return self.consensus_func(x)


class Distorter(nn.Module):
    pass


class DeepfakeDetector(torch.nn.Module):
    def __init__(self, face_model, fake_model):
        super().__init__()
        self.face_model = face_model
        self.fake_model = fake_model

    def forward(self, x):
        faces = self.face_model(x)
        return self.fake_model(faces)


class SeriesManipulatorDetector(torch.nn.Module):
    def __init__(self, manipulator_model, detector_model):
        super().__init__()
        self.manipulator_model = manipulator_model
        self.detector_model = detector_model

    def manipulate(self, x, amp=None):
        return torch.stack(
            [self.manipulator_model.manipulate(f.transpose(0, 1), amp=amp) for f in x]
        ).transpose(1, 2)

    def forward(self, x):
        # x: [bs, 3, D, H, W]
        o = self.manipulate(x)
        o = self.detector_model(o)
        return o

    @property
    def input_size(self):
        return self.detector_model.input_size


class SharedEncoder(torch.nn.Module):
    def __init__(
        self, detector: Detector, distorter: Distorter,
    ):
        super().__init__()
        self.detector = detector
        self.distorter = distorter

    def forward(self, x):
        # TODO: FINISH
        return x


class GradCamCaricatureModel(torch.nn.Module):
    def __init__(
        self,
        face_model,
        fake_model,
        mag_model,
        norm=None,
        device='cuda',
        target_layer='layer4',
    ):
        super().__init__()
        self.face_model = face_model
        self.fake_model = fake_model
        self.mag_model = mag_model
        self.gcam_model = grad_cam.GradCAM(fake_model.model)
        self.target_layer = target_layer
        if norm is None:
            self.norm = Normalize().to(device)

    def face_forward(self, x):
        with torch.no_grad():
            faces = self.face_model(x)
            norm_faces = self.norm(faces)
        return faces, norm_faces

    def forward(self, x, extract_face=False):
        if extract_face:
            faces, norm_faces = self.face_forward(x)
        else:
            faces, norm_faces = x, self.norm(x)

        outputs = []
        attn_maps = []
        for face, norm_face in zip(faces, norm_faces):
            norm_face = norm_face.transpose(0, 1)
            self.gcam_model.model.zero_grad()
            probs, idx = self.gcam_model.forward(norm_face)
            self.gcam_model.backward(idx=idx[0])
            attn_map = self.gcam_model.generate(target_layer=self.target_layer)
            out = self.mag_model.manipulate(norm_face, attn_map=attn_map)
            out = out.transpose(0, 1)
            outputs.append(out)
            attn_maps.append(attn_map)
        # TODO: smooth and threshold attn maps
        attn_maps = torch.stack(attn_maps)
        outputs = torch.stack(outputs)
        return outputs, attn_maps


def smooth_attn(attn_maps, kernel_size=3):
    # TODO: Fix and finish this!
    bs, c, d, h, w = attn_maps.size()
    conv3d = torch.conv3d(1, 1, kernel_size, bias=False)
    conv3d.weight = torch.ones_like(conv3d.weight)
    smoothed = conv3d(attn_maps)
    return smoothed
