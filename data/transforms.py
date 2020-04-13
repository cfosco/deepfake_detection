import numpy as np
import torch
import torchvision
from PIL import Image

from pretorched.data import transforms

try:
    import accimage
except ImportError:
    accimage = None


def get_transform(name='DFDC', split='train', size=224, resolution=256,
                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                  normalize=True, degrees=10):
    norm = transforms.NormalizeVideo(mean=mean, std=std)
    cropping = {
        'train': torchvision.transforms.Compose([
            transforms.RandomResizedCropVideo(size),
            transforms.RandomHorizontalFlipVideo(),
            transforms.RandomRotationVideo(degrees)]),
        'val': torchvision.transforms.Compose([
            transforms.ResizeVideo(resolution),
            transforms.CenterCropVideo(size),
        ])
    }.get(split, 'val')
    transform = torchvision.transforms.Compose([
        cropping,
        transforms.CollectFrames(),
        transforms.PILVideoToTensor(),
        norm if normalize else transforms.IdentityTransform(),
    ])
    return transform


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def to_tensor(pic, rescale=False):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        rescale: Whether or not to rescale video from :math:`[0, 255]` to
            :math:`[0, 1]`. If ``False`` the tensor will be in range
            :math:`[0, 255]`.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            img = img.float()
            return img.div(255) if rescale else img
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        img = img.float()
        return img.div(255) if rescale else img
    else:
        return img


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) optionally rescaled to the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __init__(self, rescale=False):
        self.rescale = rescale

    def __call__(self, pic, rescale=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic, self.rescale if rescale is None else rescale)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class VideoToTensor:
    """Convert a list of PIL Images to a tensor :math:`(C, T, H, W)` or
    :math:`(T, C, H, W)`.
    """

    def __init__(self, rescale: bool = True, ordering: str = "CTHW"):
        """
        Args:
            rescale: Whether or not to rescale video from :math:`[0, 255]` to
                :math:`[0, 1]`. If ``False`` the tensor will be in range
                :math:`[0, 255]`.
            ordering: What channel ordering to convert the tensor to. Either `'CTHW'`
                or `'TCHW'`
        """
        self.rescale = rescale
        self.ordering = ordering.upper()
        self.to_tensor = ToTensor(rescale)
        acceptable_ordering = ["CTHW", "TCHW"]
        if self.ordering not in acceptable_ordering:
            raise ValueError(
                "Ordering must be one of {} but was {}".format(
                    acceptable_ordering, self.ordering))

    def __call__(self, frames):
        """
        PIL Images are in the format (H, W, C)
        F.to_tensor converts (H, W, C) to (C, H, W)
        Since we have a list of these tensors, when we stack them we get shape
        (T, C, H, W)
        """
        tensor = torch.stack(list(map(self.to_tensor, frames)))
        if self.ordering == "CTHW":
            tensor = tensor.transpose(0, 1)
        return tensor

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(rescale={rescale!r}, ordering={ordering!r})".format(
                rescale=self.rescale, ordering=self.ordering
            )
        )
