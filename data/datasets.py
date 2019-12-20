import os
import os.path

import numpy as np
import torch.utils.data as data
from numpy.random import randint
from PIL import Image


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
        return {'labels': self.labels, 'num_frames': self.num_frames,
                'path': self.path, 'label': self.label}

    def __hash__(self):
        return hash((self.path, self.num_frames))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.path == other.path
        else:
            return False


class VideoDataset(data.Dataset):
    def __init__(self, root, list_file, num_frames=1, image_tmpl='img_{:06d}.jpg',
                 sampler=None, transform=None, target_transform=None):
        self.root = root
        self.list_file = list_file
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
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
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
        average_duration = (record.num_frames - self.new_length + 1) * 1.0 / self.num_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_frames)), average_duration) + \
                np.random.uniform(0, average_duration, size=self.num_frames)
            offsets = np.floor(offsets)
        elif record.num_frames > self.num_frames:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_frames))
        else:
            offsets = np.zeros((self.num_frames,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_frames + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) * 1.0 / self.num_frames
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_frames)])
        else:
            offsets = np.zeros((self.num_frames,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) * 1.0 / self.num_frames
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
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


if __name__ == '__main__':

    # TODO: Finish implementing this!
    root = ''
    list_file = ''
    transform = None
    dataset = VideoDataset(root, list_file, transform=transform)

    for i, (frames, label) in enumerate(dataset):
        print(i, frames.shape, label)
