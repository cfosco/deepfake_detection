# import functools
import io
import json
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.utils.data as data
from numpy.random import randint
from PIL import Image
from torch._six import int_classes as _int_classes

from torchvideo.datasets import VideoDataset
from torchvideo.internal.readers import _get_videofile_frame_count
from torchvideo.samplers import FrameSampler, _default_sampler, frame_idx_to_list
from torchvideo.transforms import PILVideoToTensor

from . import transforms


class DeepfakeRecord:
    """Represents a video record.
    A video record has the following properties:
        path (str): path to directory containing frames.
        num_frames (int): number of frames in path dir.
        label (int): primary label associated with video.
        labels (list[int]): all labels associated with video.
    """

    label_mapping = {'FAKE': 1, 'REAL': 0}

    def __init__(self, part, name, data):
        self.part = part
        self.name = name
        self.data = data

    @property
    def path(self):
        return self.data['path']

    @property
    def filename(self):
        return self.data['filename']

    @property
    def num_frames(self):
        return int(self.data['num_frames'])

    @property
    def label_name(self):
        return self.data['label']

    @property
    def label(self):
        return int(self.label_mapping[self.label_name])

    def todict(self):
        return self.data

    def __hash__(self):
        return hash(self.data.values())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.path == other.path
        else:
            return False


class DeepfakeFaceRecord(DeepfakeRecord):
    def __init__(self, part, name, data, face_data_key='facenet_videos'):
        self.part = part
        self.name = name
        self.data = data
        self.face_data_key = face_data_key

    @property
    def num_faces(self):
        return self.face_data['num_faces']

    @property
    def face_names(self):
        return self.face_data['face_names']

    @property
    def face_nums(self):
        return self.face_data['face_nums']

    @property
    def num_face_frames(self):
        return self.face_data['num_frames']

    @property
    def face_locations(self):
        return self.face_data['face_coords']

    @property
    def size(self):
        return self.face_data['size']

    @property
    def has_face_data(self):
        return self.face_data_key in self.data

    @property
    def face_data(self):
        return self.data[self.face_data_key]

    @property
    def face_path(self):
        return self.face_data.get('path', self.path)


class DeepfakeSet:
    def __init__(self, metafile, blacklist_file=None, record_func=DeepfakeRecord):
        self.records = []
        self.metafile = metafile
        self.record_func = record_func
        with open(metafile) as f:
            self.data = json.load(f)

        self.blacklist = []
        if blacklist_file is not None:
            with open(blacklist_file) as f:
                self.blacklist.extend(json.load(f))

        self.records = []
        self.records_dict = defaultdict(list)
        self.blacklist_records = []
        for part, part_data in self.data.items():
            for name, rec in part_data.items():
                record = self.record_func(part, name, rec)
                if issubclass(self.record_func, DeepfakeFaceRecord):
                    if not record.has_face_data:
                        continue

                if (
                    name not in self.blacklist
                    and os.path.basename(name) not in self.blacklist
                ):
                    self.records.append(record)
                    self.records_dict[part].append(record)
                else:
                    self.blacklist_records.append(record)

        # for b in self.blacklist_records:
        # assert b not in self.records

    def __getitem__(self, idx: int) -> Union[DeepfakeRecord, DeepfakeFaceRecord]:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


class DeepfakeFaceSet(DeepfakeSet):
    def __init__(self, metafile, blacklist_file=None):
        super().__init__(
            metafile, blacklist_file=blacklist_file, record_func=DeepfakeFaceRecord
        )

    def __getitem__(self, idx: int) -> DeepfakeFaceRecord:
        return self.records[idx]


class DeepfakeFrame(data.Dataset):

    offset = 1

    def __init__(
        self,
        root: Union[str, Path],
        record_set: Union[DeepfakeSet, DeepfakeFaceSet],
        sampler: FrameSampler = _default_sampler(),
        image_tmpl: str = '{:06d}.jpg',
        transform=None,
        target_transform=None,
    ):

        self.root = root
        self.sampler = sampler
        self.record_set = record_set
        self.image_tmpl = image_tmpl

        if transform is None:
            transform = PILVideoToTensor()
        self.transform = transform

        if target_transform is None:
            target_transform = int
        self.target_transform = target_transform

    def _load_image(self, directory, idx):
        idx += self.offset
        filename_tmpl = os.path.join(self.root, directory, self.image_tmpl)
        try:
            return Image.open(filename_tmpl.format(idx)).convert('RGB')
        except Exception:
            print('Error loading image:', filename_tmpl.format(idx))
            return Image.open(filename_tmpl.format(1)).convert('RGB')

    def _load_frames(self, frame_dir, frame_inds):
        return (self._load_image(frame_dir, idx) for idx in frame_inds)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        record = self.record_set[index]
        frame_path = os.path.join(self.root, record.path)
        frame_inds = self.sampler.sample(record.num_frames)
        frames = list(self._load_frames(frame_path, frame_inds))
        label = record.label

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label

    def __len__(self):
        return len(self.record_set)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '  Number of datapoints: {}\n'.format(len(self.record_set))
        tmp = '  Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        return fmt_str


class DeepfakeFaceFrame(DeepfakeFrame):
    def __init__(
        self,
        root: Union[str, Path],
        record_set: DeepfakeFaceSet,
        sampler: FrameSampler = _default_sampler(),
        image_tmpl: str = '{:06d}.jpg',
        crop_faces: bool = False,
        margin=100,
        transform=None,
        target_transform=None,
    ):

        self.root = root
        self.sampler = sampler
        self.record_set = record_set
        self.image_tmpl = image_tmpl
        self.crop_faces = crop_faces
        self.margin = margin

        if transform is None:
            transform = PILVideoToTensor()
        self.transform = transform

        if target_transform is None:
            target_transform = int
        self.target_transform = target_transform

    def _get_face_frames(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        # TODO
        record = self.record_set[index]
        frame_path = os.path.join(self.root, record.path)
        frame_inds = self.sampler.sample(record.num_frames)
        frames = self._load_frames(frame_path, frame_inds)
        label = record.label

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label

    def _crop_frame(self, frame, coords, size=360):
        top, right, bottom, left = coords
        mtop = max(top - self.margin, 0)
        mbottom = min(bottom + self.margin, frame.size[0])
        mleft = max(left - self.margin, 0)
        mright = min(right + self.margin, frame.size[1])
        return frame.crop((mleft, mtop, mright, mbottom)).resize((size, size))

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        record = self.record_set[index]
        frame_path = os.path.join(self.root, record.path)
        face_num = np.random.choice(record.face_nums)
        num_face_frames = record.num_face_frames[face_num]
        frame_inds = self.sampler.sample(num_face_frames)

        if not self.crop_faces:
            frame_path = os.path.join(frame_path, record.face_names[face_num])
        frames = list(self._load_frames(frame_path, frame_inds))
        if self.crop_faces:
            face_coords = [
                record.face_locations[str(face_num)][idx] for idx in frame_inds
            ]
            frames = [
                self._crop_frame(f, c, record.size) for f, c in zip(frames, face_coords)
            ]
        label = record.label

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label


class DeepfakeFaceCropFrame(DeepfakeFaceFrame):
    def __init__(
        self,
        root,
        record_set,
        sampler=_default_sampler(),
        image_tmpl='{:06d}.jpg',
        margin=100,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root,
            record_set,
            sampler=sampler,
            image_tmpl=image_tmpl,
            crop_faces=True,
            margin=margin,
            transform=transform,
            target_transform=target_transform,
        )


class DeepfakeVideo(VideoDataset):
    def __init__(
        self,
        root: Union[str, Path],
        record_set: DeepfakeSet,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:

        self.root = root
        self.record_set = record_set
        self.sampler = sampler

        if frame_counter is None:
            frame_counter = _get_videofile_frame_count
        self.frame_counter = frame_counter

        if transform is None:
            transform = PILVideoToTensor()
        self.transform = transform

        if target_transform is None:
            target_transform = int
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        record = self.record_set[index]
        video_path = os.path.join(self.root, record.path)
        # video_length = self.frame_counter(video_path)
        video_length = record.num_frames
        frame_inds = self.sampler.sample(video_length)
        frames = self._load_frames(video_path, frame_inds)
        label = record.label

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label

    def __len__(self):
        return len(self.record_set)


class DeepfakeFaceVideo(DeepfakeVideo):
    def __getitem__(self, index: int):
        record = self.record_set[index]
        video_dir = os.path.join(self.root, record.face_path)
        face_num = np.random.choice(record.face_nums)
        num_face_frames = record.num_face_frames[face_num]
        video_path = os.path.join(video_dir, record.face_names[face_num] + '.mp4')
        # video_length = self.frame_counter(video_path)
        # frame_inds = self.sampler.sample(video_length)
        frame_inds = self.sampler.sample(num_face_frames)
        # print(num_face_frames, frame_inds)
        frames = list(self._load_frames(video_path, frame_inds))
        label = record.label

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label


class DeepfakeFaceHeatvolVideo(DeepfakeFaceVideo):

    max_heatvol_len: int = 90
    default_heatvol_size: int = 360

    def __init__(
        self,
        root: Union[str, Path],
        record_set: DeepfakeSet,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
        heatvols_root: Optional[Union[str, Path]] = None,
        pt_ext: str = '.pt',
        heatvol_freq: int = 10,
    ) -> None:
        super().__init__(
            root, record_set, sampler, transform, target_transform, frame_counter
        )
        self.heatvol_freq = heatvol_freq
        self.heatvol_inds: List[int] = []
        if heatvols_root is None:
            # Assume default location
            basedir = os.path.dirname(root)
            heatvols_root = os.path.join(basedir, 'heatvols')
            if not os.path.exists(heatvols_root):
                return
            heatvol_files = [f for f in os.listdir(heatvols_root) if f.endswith(pt_ext)]
            heatvol_names = [os.path.splitext(f)[0] for f in heatvol_files]

            for i, record in enumerate(self.record_set):
                name = os.path.splitext(record.filename)[0]
                if name in heatvol_names:
                    record.heatvol_path = os.path.join(heatvols_root, name + pt_ext)
                    self.heatvol_inds.append(i)

    # def _find_heatvols(self):

    def __getitem__(self, index: int):
        valid_heatvol = index in self.heatvol_inds
        record = self.record_set[index]
        video_dir = os.path.join(self.root, record.face_path)
        face_num = np.random.choice(record.face_nums)
        video_path = os.path.join(video_dir, record.face_names[face_num] + '.mp4')
        num_face_frames = (
            self.max_heatvol_len if valid_heatvol else record.num_face_frames[face_num]
        )
        frame_inds = self.sampler.sample(num_face_frames)
        frames = list(self._load_frames(video_path, frame_inds))
        label = record.label
        if valid_heatvol:
            heatvol = torch.load(record.heatvol_path).index_select(
                0, torch.tensor(frame_idx_to_list(frame_inds))
            )
        else:
            heatvol = torch.zeros(
                len(frames), self.default_heatvol_size, self.default_heatvol_size
            )

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label, heatvol, int(valid_heatvol)


class HeatvolBatchSampler(torch.utils.data.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            with ``__len__`` implemented.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        heatvol_inds,
        heatvols_per_batch=1,
        random_heatvol=True,
    ):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.heatvol_inds = heatvol_inds
        self.heatvols_per_batch = heatvols_per_batch
        self.random_heatvol = random_heatvol
        self.num_heatvols = len(heatvol_inds)
        self.curr_heatvol_idx = 0

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == (self.batch_size - self.heatvols_per_batch):
                batch.extend(self._get_heatvol_inds())
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def _get_heatvol_inds(self):
        if self.random_heatvol:
            heatvol_inds = [np.random.choice(self.heatvol_inds) for i in range(self.heatvols_per_batch)]
        else:
            heatvol_inds = []
            for _ in range(self.heatvols_per_batch):
                heatvol_inds.append(self.curr_heatvol_idx)
                self.curr_heatvol_idx += 1
                if self.curr_heatvol_idx == self.num_heatvols:
                    self.curr_heatvol_idx = 0
        return heatvol_inds

    def __len__(self):
        if self.drop_last:
            return (len(self.sampler) // (self.batch_size - self.heatvols_per_batch))
        else:
            return ((len(self.sampler) + self.batch_size - 1) // (self.batch_size - self.heatvols_per_batch))


class DeepfakeZipVideo(DeepfakeVideo):

    zip_ext = '.zip'

    def __getitem__(self, index):
        record = self.record_set[index]
        part, video_path, label = record.part, record.path, record.label
        with zipfile.ZipFile(os.path.join(self.root, part + self.zip_ext)) as z:
            video_path = io.BytesIO(z.read(video_path))
        video_length = record.num_frames or self.frame_counter(video_path)
        frame_inds = self.sampler.sample(video_length)
        frames = self._load_frames(video_path, frame_inds)

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label


class DeepfakeZipFaceVideo(DeepfakeFaceVideo):

    zip_ext = '.zip'

    def __getitem__(self, index: int):
        record = self.record_set[index]
        part, video_dir, label = record.part, record.face_path, record.label
        face_num = np.random.choice(record.face_nums)
        num_face_frames = record.num_face_frames[face_num]
        frame_inds = self.sampler.sample(num_face_frames)

        video_path = os.path.join(video_dir, record.face_names[face_num] + '.mp4')
        with zipfile.ZipFile(os.path.join(self.root, part + self.zip_ext)) as z:
            video_path = io.BytesIO(z.read(video_path))

        frames = list(self._load_frames(video_path, frame_inds))

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label


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


class VideoFolder(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        step=2,
        transform=None,
        target_file=None,
        default_target=0,
        load_as='pil',
    ):
        self.root = root
        self.step = step
        self.load_as = load_as
        self.videos_filenames = sorted(
            [f for f in os.listdir(root) if f.endswith('.mp4')]
        )
        if transform is None:
            transform = transforms.VideoToTensor(rescale=False)
        self.transform = transform

        if target_file is not None:
            with open(target_file) as f:
                self.targets = json.load(f)
        else:
            self.targets = {}

        self.default_target = default_target

    def __getitem__(self, index) -> Tuple[str, torch.Tensor, int]:
        name = self.videos_filenames[index]
        basename = os.path.basename(name)
        video_filename = os.path.join(self.root, name)
        frames = read_frames(video_filename, step=self.step)
        if self.load_as == 'pil':
            frames = map(Image.fromarray, frames)
        if self.transform is not None:
            frames = self.transform(frames)
        target = int(self.targets.get(basename, self.default_target))
        return name, frames, target

    def __len__(self):
        return len(self.videos_filenames)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '  Root dir: {}\n'.format(self.root)
        fmt_str += '  Number of datapoints: {}\n'.format(len(self.videos_filenames))
        tmp = '  Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        return fmt_str


class VideoZipFile(VideoFolder):
    def __init__(
        self, filename, step=2, transform=None, target_file=None, default_target=0
    ):
        self.filename = filename
        self.step = step
        with zipfile.ZipFile(filename) as z:
            self.videos_filenames = sorted(
                [f for f in z.namelist() if f.endswith('.mp4')]
            )

        if transform is None:
            transform = transforms.VideoToTensor(rescale=False)
        self.transform = transform

        if target_file is not None:
            with open(target_file) as f:
                self.targets = json.load(f)
        else:
            self.targets = {}

        self.default_target = default_target

    def __getitem__(self, index):
        name = self.videos_filenames[index]
        basename = os.path.basename(name)
        with tempfile.NamedTemporaryFile() as temp:
            with zipfile.ZipFile(self.filename) as z:
                temp.write(z.read(name))
            frames = read_frames(temp.name, step=self.step)
            if self.transform is not None:
                frames = self.transform(frames)
        target = int(self.targets.get(basename, self.default_target))
        return name, frames, target


def video_collate_fn(batch):
    names, frames, targets = zip(*batch)
    nc, _, h, w = frames[0].shape
    num_frames = [f.size(1) for f in frames]
    max_len = max(num_frames)
    frames = torch.stack(
        [
            torch.cat([f, f[:, -1:].expand(nc, max_len - nf, h, w)], 1)
            for f, nf in zip(frames, num_frames)
        ]
    )
    return names, frames, targets


class Record(object):
    """Represents a record.

    A record has the following properties:
        path (str): path to file.
        label (int): primary label associated with video.
        labels (list[int]): all labels associated with video.
    """

    def __init__(self, path, label):
        self._path = path
        self._label = label

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return int(self._label)

    def todict(self):
        return {'label': self.label, 'path': self.path}

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.path == other.path
        else:
            return False


class VideoRecord(object):
    """Represents a video record.
    A video record has the following properties:
        path (str): path to directory containing frames.
        num_frames (int): number of frames in path dir.
        label (int): primary label associated with video.
        labels (list[int]): all labels associated with video.
    """

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def labels(self):
        return [int(label) for label in self._data[2:]]

    def todict(self):
        return {
            'labels': self.labels,
            'num_frames': self.num_frames,
            'path': self.path,
            'label': self.label,
        }

    def __hash__(self):
        return hash((self.path, self.num_frames))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.path == other.path
        else:
            return False


class VideoDataset(data.Dataset):
    def __init__(
        self,
        root,
        metadata_file,
        num_frames=1,
        image_tmpl='img_{:06d}.jpg',
        sampler=None,
        transform=None,
        target_transform=None,
    ):
        self.root = root
        self.metadata_file = metadata_file
        self.num_frames = num_frames
        self.sampler = sampler
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.target_transform = target_transform

        self._parse_list()

    def _load_image(self, directory, idx):
        filename_tmpl = os.path.join(self.root, directory, self.image_tmpl)
        try:
            return Image.open(filename_tmpl.format(idx)).convert('RGB')
        except Exception:
            print('Error loading image:', filename_tmpl.format(idx))
            return Image.open(filename_tmpl.format(1)).convert('RGB')

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.metadata_file)]
        tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('Video number: {}'.format(len(self.video_list)))

    def _sample_indices(self, record):
        """Sample frame indices.
        :param record: VideoRecord
        :return: list
        Args:
            record (TYPE): Description.
        Returns:
            TYPE: Description.
        """
        average_duration = (
            (record.num_frames - self.new_length + 1) * 1.0 / self.num_frames
        )
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_frames)), average_duration
            ) + np.random.uniform(0, average_duration, size=self.num_frames)
            offsets = np.floor(offsets)
        elif record.num_frames > self.num_frames:
            offsets = np.sort(
                randint(record.num_frames - self.new_length + 1, size=self.num_frames)
            )
        else:
            offsets = np.zeros((self.num_frames,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_frames + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) * 1.0 / self.num_frames
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_frames)]
            )
        else:
            offsets = np.zeros((self.num_frames,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) * 1.0 / self.num_frames
        offsets = np.array(
            [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
        )
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # Check this is a legit video folder
        test_filename = os.path.join(self.root, record.path, self.image_tmpl.format(1))
        while not os.path.exists(test_filename):
            print('Could not find: {}'.format(test_filename))
            index = np.random.randint(len(self.video_list))
            # Try another video.
            record = self.video_list[index]

        if self.sampler is None:
            frame_indices = self._sample_indices(record)

        return self.get(record, frame_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        label = record.label

        if self.transform is not None:
            images = self.transform(images)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return images, label

    def __len__(self):
        return len(self.video_list)
