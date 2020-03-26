import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from PIL import Image
from torch.nn import Parameter as P
from torch.nn import functional as F
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import functional as TF

# from pretorched.data import transforms

# TEST_VIDEO_DIR = '/kaggle/input/deepfake-detection-challenge/test_videos/'
# SAMPLE_SUBMISSION_CSV = '/kaggle/input/deepfake-detection-challenge/sample_submission.csv'
# WEIGHT_DIR = '/kaggle/input/deepfake-data/data'

TEST_VIDEO_DIR = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'test_videos')
SAMPLE_SUBMISSION_CSV = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'sample_submission.csv')
TARGET_FILE = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'test_targets.json')
if not os.path.exists(TARGET_FILE):
    TARGET_FILE = None
dir_path = os.path.dirname(os.path.realpath(__file__))
WEIGHT_DIR = os.path.join(dir_path, 'data')
# WEIGHT_DIR = 'data'
NUM_WORKERS = 4
STEP = 20


def main(video_dir, margin=100, step=20, batch_size=1, chunk_size=300, num_workers=4):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = VideoFolder(video_dir, step=step, target_file=TARGET_FILE)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    fakenet = resnet18(num_classes=2, pretrained=None)
    fakenet.load_state_dict({k.replace('module.model.', ''): v
                             for k, v in torch.load(WEIGHT_DIR + '/'
                                                    'resnet18_dfdc_seg_count-24_init-imagenet-'
                                                    'ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs'
                                                    '-64_best.pth.tar')['state_dict'].items()})
    facenet = FaceModel(size=fakenet.input_size[-1],
                        device=device,
                        margin=margin,
                        min_face_size=50,
                        keep_all=True,
                        post_process=False,
                        select_largest=False,
                        chunk_size=chunk_size)
    fakenet.eval()
    detector = FrameModel(fakenet, normalize=True)
    model = DeepfakeDetector(facenet, detector)
    model.to(device)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda(device)
    sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub.label = 0.5
    sub = sub.set_index('filename', drop=False)

    preds, acc, loss = validate(dataloader, model, criterion, device=device)

    # batch_time = AverageMeter('Time', ':6.3f')
    # progress = ProgressMeter(
    #     len(dataset),
    #     [batch_time],
    #     prefix='Test: ')

    # # switch to evaluate mode
    # model.eval()
    # with torch.no_grad():
    #     end = time.time()
    #     for i, (filenames, images) in enumerate(dataloader):
    #         images = images.to(device, non_blocking=True)
    #         images.mul_(255)
    #         print(images.shape, images.min(), images.max())
    #         try:
    #             # compute output
    #             output = model(images)
    #             print(output.shape)

    #             probs = torch.softmax(output, 1)
    #             for fn, prob in zip(filenames, probs):
    #                 p = prob[1].item()
    #                 if np.isnan(p):
    #                     p = 0.5
    #                 elif p <= 0.0:
    #                     p = 0.0001
    #                 elif p >= 1.0:
    #                     p = 0.9999
    #                 sub.loc[fn, 'label'] = p
    #                 print(f'filename: {fn}; prob: {prob[1]:.3f}')
    #         except Exception:
    #             sub.loc[filenames[0], 'label'] = 0.5

    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         progress.display(i)

    sub.to_csv('submission.csv', index=False)


def validate(val_loader, model, criterion, device='cuda', display=True, print_freq=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    preds = {}

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (filenames, images, target) in enumerate(val_loader):
            if device is not None:
                images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            probs = torch.softmax(output, 1)
            for fn, prob in zip(filenames, probs):
                preds[fn] = prob[1].item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and display:
                progress.display(i)

        if display:
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return preds, top1.avg, losses.avg


def read_frames(video, fps=30, step=1):
    # Open video file
    video_capture = cv2.VideoCapture(video)
    video_capture.set(cv2.CAP_PROP_FPS, fps)

    count = 0
    while video_capture.isOpened():
        # Grab a single frame of video
        ret = video_capture.grab()

        # Bail out when the video file ends
        if not ret:
            break
        if count % step == 0:
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            ret, frame = video_capture.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        count += 1


# class VideoDataset(torch.utils.data.Dataset):
#     def __init__(self, root, step=12, transform=None):
#         self.root = root
#         self.step = step
#         self.videos_filenames = sorted([f for f in os.listdir(root) if f.endswith('.mp4')])
#         self.transform = transform

#     def __getitem__(self, index):
#         name = self.videos_filenames[index]
#         video_filename = os.path.join(self.root, name)
#         frames = read_frames(video_filename, step=self.step)
#         frames = torch.stack(list(map(TF.to_tensor, frames))).transpose(0, 1)
#         if self.transform is not None and callable(self.transform):
#             frames = self.transform(frames)
#         return name, frames

#     def __len__(self):
#         return len(self.videos_filenames)


class VideoFolder(torch.utils.data.Dataset):
    def __init__(self, root, step=2, transform=None, target_file=None):
        self.root = root
        self.step = step
        self.videos_filenames = sorted([f for f in os.listdir(root) if f.endswith('.mp4')])
        if transform is None:
            transform = VideoToTensor(rescale=False)
        self.transform = transform

        if target_file is not None:
            with open(target_file) as f:
                self.targets = json.load(f)
        else:
            self.targets = {}

    def __getitem__(self, index):
        name = self.videos_filenames[index]
        video_filename = os.path.join(self.root, name)
        frames = read_frames(video_filename, step=self.step)
        if self.transform is not None:
            frames = self.transform(frames)
        target = int(self.targets.get(name, 0))
        return name, frames, target

    def __len__(self):
        return len(self.videos_filenames)


try:
    import accimage
except ImportError:
    accimage = None


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


def modify_resnets(model):
    # Modify attributs
    model.last_linear = model.fc
    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    setattr(model.__class__, 'features', features)
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    setattr(model.__class__, 'input_size', (3, 224, 224))
    setattr(model.__class__, 'mean', [0.485, 0.456, 0.406])
    setattr(model.__class__, 'std', [0.229, 0.224, 0.225])
    # model.features = types.MethodType(features, model)
    # model.logits = types.MethodType(logits, model)
    # model.forward = types.MethodType(forward, model)
    # model.features = types.MethodType(features, model)
    # model.logits = types.MethodType(logits, model)
    # model.forward = types.MethodType(forward, model)
    return model


def resnet18(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet50'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-101 model.
    """
    model = models.resnet101(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet101'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


class FaceModel(torch.nn.Module):

    def __init__(self, size=224, device='cuda', margin=50, keep_all=False,
                 post_process=False, select_largest=True, min_face_size=50,
                 chunk_size=None):
        super().__init__()
        self.model = MTCNN(image_size=size,
                           device=device,
                           margin=margin,
                           keep_all=keep_all,
                           min_face_size=min_face_size,
                           post_process=post_process,
                           select_largest=select_largest,
                           chunk_size=chunk_size)

    def input_transform(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [bs, d, nc, h, w]
        return x
        # return x.view(-1, *x.shape[2:])            # [bs * d, nc, h, w]

    def get_faces(self, x):
        bs, nc, d, h, w = x.shape
        batched_face_images = []
        for x in self.input_transform(x):
            faces_out = []
            out = self.model(x, smooth=True, smooth_len=d)
            out = torch.stack(out).cpu()  # [num_frames, num_faces, 3, 360, 360]
            faces_out.append(out)

            min_face = min([f.shape[1] for f in faces_out])
            faces_out = torch.cat([f[:, :min_face] for f in faces_out])
            face_images = {i: [Image.fromarray(ff.permute(1, 2, 0).numpy().astype(np.uint8)) for ff in f]
                           for i, f in enumerate(faces_out.permute(1, 0, 2, 3, 4))}
            batched_face_images.append(face_images)
        return batched_face_images

    def forward(self, x):
        """
        Args:
            x: [bs, nc, d, h, w]
        NOTE: For now, assume keep_all=False for 1 face per frame.
        lists of lists or nested tensors should be used to handle variable
        number of faces per example in batch (avoid this for now).
        """
        bs, nc, d, h, w = x.shape
        x = self.input_transform(x)
        x = x.view(-1, *x.shape[2:])
        out = self.model(x, smooth=True)
        out = torch.stack(out)
        out = out.view(bs, -1, nc, *out.shape[-2:])
        out = out.permute(0, 2, 1, 3, 4)  # [bs, nc, d, h, w]

        return out


class MTCNN(nn.Module):
    """MTCNN face detection module.

    This class loads pretrained P-, R-, and O-nets and, given raw input images as PIL images,
    returns images cropped to include the face only. Cropped faces can optionally be saved to file
    also.

    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether or not to post process images tensors before returning. (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned. (default: {True})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
        self, image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7, 0.98], factor=0.709, post_process=True,
        select_largest=True, keep_all=False, device=None, chunk_size=None,
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.chunk_size = chunk_size

        self.pnet = PNet(pretrained=os.path.join(WEIGHT_DIR, 'pnet.pth'))
        self.rnet = RNet(pretrained=os.path.join(WEIGHT_DIR, 'rnet.pth'))
        self.onet = ONet(pretrained=os.path.join(WEIGHT_DIR, 'onet.pth'))

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, img, save_path=None, return_prob=False, smooth=False, amount=0.4, smooth_len=None):
        """Run MTCNN face detection on a PIL image. This method performs both detection and
        extraction of faces, returning tensors representing detected faces rather than the bounding
        boxes. To access bounding boxes, see the MTCNN.detect() method below.

        Arguments:
            img {PIL.Image or list} -- A PIL image or a list of PIL images.

        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved face
                image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})

        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra
                dimension (batch) as the first dimension.

        Example:
        >>> from facenet_pytorch import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """

        # Detect faces
        with torch.no_grad():
            cs = len(img) if self.chunk_size is None else self.chunk_size
            batch_boxes, batch_probs = [], []
            for im in chunk(img, cs):
                bb, bp = self.detect(im)
                batch_boxes.extend(bb)
                batch_probs.extend(bp)

        if smooth:
            bboxes = []
            sl = len(batch_boxes) if smooth_len is None else smooth_len
            for bb in chunk(batch_boxes, sl):
                bboxes.extend(smooth_boxes(bb, amount=amount))
            batch_boxes = bboxes

        # Determine if a batch or single image was passed
        batch_mode = True
        if not isinstance(img, Iterable):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_probs = [batch_probs]
            batch_mode = False

        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]

        # Process all bounding boxes and probabilities
        faces, probs = [], []
        for im, box_im, prob_im, path_im in zip(img, batch_boxes, batch_probs, save_path):
            if box_im is None:
                faces.append(None)
                probs.append([None] if self.keep_all else None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]
                prob_im = prob_im[0]

            faces.append(faces_im)
            probs.append(prob_im)

        if not batch_mode:
            faces = faces[0]
            probs = probs[0]

        if return_prob:
            return faces, probs
        else:
            return faces

    def detect(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.

        Arguments:
            img {PIL.Image or list} -- A PIL image or a list of PIL images.

        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})

        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )
        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                inds = box[:, 4] >= self.thresholds[-1]
                inds[0] = True  # Keep at least one face
                # boxes.append(box[:, :4])
                # probs.append(box[:, 4])
                boxes.append(box[inds, :4])
                probs.append(box[inds, 4])
                points.append(point)
        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)

        if not isinstance(img, Iterable):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        if landmarks:
            return boxes, probs, points

        return boxes, probs


class PNet(nn.Module):
    """MTCNN PNet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained='data/pnet.pth'):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained is not None:
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained='data/rnet.pth'):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained is not None:
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained='data/onet.pth'):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained is not None:
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict)
            # self.load_state_dict(model_zoo.load_url(model_urls['onet']))

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


def interp_nans(arr):
    arr = np.array(arr).ravel()
    arr = arr.astype(float)
    missing = np.isnan(arr)
    n = len(arr)
    missing = np.isnan(arr.astype(float))
    if sum(missing) == 0 or all(missing):
        return arr
    inds = np.arange(n)[missing]
    x = np.arange(n)[~missing]
    out = np.interp(inds, x, arr[x])
    arr[inds] = out
    return arr


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0 / (float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y


def smooth_data(data, amount=1.0):
    if not amount > 0.0:
        return data
    data_len = len(data)
    ksize = max(1, int(amount * (data_len // 2)))
    kernel = np.ones(ksize) / ksize
    return np.convolve(data, kernel, mode='same')


def smooth(x, amount=0.2, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    data_len = len(x)
    window_len = max(1, int(amount * (data_len // 2)))

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len // 2):-(window_len // 2)]


def smooth_boxes(batch_boxes, amount=0.5):
    known_coords = None
    boxes = defaultdict(list)
    for i, bb in enumerate(batch_boxes):
        if bb is None:
            for face_num in boxes:
                boxes[face_num].append(4 * [None])
        else:
            if known_coords is None:
                known_coords = bb
            added = []
            for face_num, b in enumerate(bb):
                diff = np.abs(np.array(b) - np.array(known_coords)).sum(1)
                face_idx = int(np.argmin(diff))
                if face_idx not in added:
                    known_coords[face_idx] = b
                    boxes[face_idx].append(b)
                    added.append(face_idx)
    for face_num, coords in boxes.items():
        out = [np.array(x) for x in zip(*[smooth(interp_nans(dim), amount=amount) for dim in zip(*coords)])]
        boxes[face_num] = out
    return list(map(np.stack, zip(*boxes.values())))


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    if not isinstance(imgs, Iterable):
        imgs = [imgs]
    if any(img.shape != imgs[0].shape for img in imgs):
        raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
    if not isinstance(imgs, torch.Tensor):
        imgs_np = np.stack([np.uint8(img) for img in imgs])
        imgs = torch.as_tensor(imgs_np, device=device).permute(0, 3, 1, 2)

    imgs = imgs.to(device)
    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage
    boxes = []
    image_inds = []
    all_inds = []
    all_i = 0
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)

        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)
        all_inds.append(all_i + image_inds_scale)
        all_i += batch_size

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0).cpu()
    all_inds = torch.cat(all_inds, dim=0)

    # NMS within each scale + image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], all_inds, 0.5)
    boxes, image_inds = boxes[pick], image_inds[pick]

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    y, ey, x, ex = pad(boxes, w, h)

    # Second stage
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        out = rnet(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        out = onet(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)

        # NMS within each image using "Min" strategy
        # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds])
        batch_points.append(points[b_i_inds])

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)

    return batch_boxes, batch_points


def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].float().flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method == "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[:counter]
    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(boxes, w, h):
    boxes = boxes.trunc().int()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y.cpu().tolist(), ey.cpu().tolist(), x.cpu().tolist(), ex.cpu().tolist()


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bboxA


def imresample(img, sz):
    im_data = F.interpolate(img, size=sz, mode="area")
    return im_data


def crop_tensor(tensor, box):
    """
    Args:
        tensor: tensor to be cropped.
        box: coords with format: (left, top, right, bottom)
    """
    left, top, right, bottom = box
    return tensor[..., top:bottom, left:right]


def crop_resize(tensor, box, size):
    return F.interpolate(
        crop_tensor(tensor, box).unsqueeze(0),
        size=size, mode='area')[0]


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    if isinstance(img, torch.Tensor):
        height, width = img.shape[-2:]
    else:
        width, height = img.size

    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, width)),
        int(min(box[3] + margin[1] / 2, height)),
    ]

    if isinstance(img, torch.Tensor):
        face = crop_resize(img, box, (image_size, image_size))
    else:
        face = img.crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        if isinstance(img, torch.Tensor):
            face_im = Image.fromarray(face.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_args = {"compress_level": 0} if ".png" in save_path else {}
        face_im.save(save_path, **save_args)

    return face


def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, verbose=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg) if verbose else None
        return msg

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def reset(self):
        for meter in self.meters:
            meter.reset()


def accuracy(output, target, topk=(1, 5)):
    """Compute the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.reshape(-1).size(0)

        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.eq(target.unsqueeze(1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:, :k, ...].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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


# class FaceModel(torch.nn.Module):

#     def __init__(self, size=224, device='cuda', margin=50, keep_all=False,
#                  post_process=False, select_largest=True):
#         super().__init__()
#         self.model = MTCNN(image_size=size,
#                            device=device,
#                            margin=margin,
#                            keep_all=keep_all,
#                            post_process=post_process,
#                            select_largest=select_largest)

#     def forward(self, x):
#         """
#         Args:
#             x: [bs, nc, d, h, w]
#         NOTE: For now, assume keep_all=False for 1 face per frame.
#         lists of lists or nested tensors should be used to handle variable
#         number of faces per example in batch (avoid this for now).
#         """
#         bs, nc, d, h, w = x.shape
#         x = x.permute(0, 2, 1, 3, 4)  # [bs, d, nc, h, w]
#         # x = x.view(-1, *x.shape[2:])  # [bs * d, nc, h, w]
#         x = x.reshape(-1, *x.shape[2:])  # [bs * d, nc, h, w]
#         out = self.model(x)
#         for i, o in enumerate(out):
#             if o is None:
#                 try:
#                     out[i] = out[i - 1]
#                 except IndexError:
#                     pass
#         out = torch.stack(out)
#         out = out.view(bs, d, nc, *out.shape[-2:])
#         out = out.permute(0, 2, 1, 3, 4)  # [bs, nc, d, h, w]
#         return out


class DeepfakeDetector(torch.nn.Module):

    def __init__(self, face_model, fake_model):
        super().__init__()
        self.face_model = face_model
        self.fake_model = fake_model

    def forward(self, x):
        try:
            faces = self.face_model(x)
        except TypeError:
            print('Error finding faces')
            return 0.5 * torch.ones(x.size(0), 2)
        return self.fake_model(faces)
        # min_faces = torch.min([f.shape[0] for f in faces])
        # batched_faces = [torch.stack([f[i] for f in faces]) for i in range(min_faces)]
        # return self.consensus_func(
        # torch.stack([self.detector(f) for f in batched_faces]), dim=0)


if __name__ == '__main__':
    main(TEST_VIDEO_DIR)
