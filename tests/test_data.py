import os
import pytest
import torch

from pretorched.data import samplers

from .. import config as cfg
from .. import data


DATA_ROOT = cfg.DATA_ROOT
MAX_ITERS = 10
BATCH_SIZE = 16

SKIP_FRAME = True
SKIP_VIDEO = True
SKIP_VIDEO_ZIP = False


@pytest.mark.skipif(SKIP_VIDEO, reason='Data not present')
@pytest.mark.parametrize('name, split, num_frames, size, dataset_type, record_set_type', [
    ('DFDC', 'train', 16, 224, 'DeepfakeVideo', 'DeepfakeSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeVideo', 'DeepfakeSet'),
    ('DFDC', 'train', 16, 224, 'DeepfakeFaceVideo', 'DeepfakeFaceSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFaceVideo', 'DeepfakeFaceSet'),
])
def test_get_dataset_video(name, split, num_frames, size, dataset_type, record_set_type):
    metadata = cfg.get_metadata(name, split=split, dataset_type=dataset_type,
                                data_root=DATA_ROOT)
    print(metadata)
    root = metadata['root']
    metafile = metadata['metafile']
    blacklist_file = metadata['blacklist_file']
    record_set = getattr(data, record_set_type)(metafile, blacklist_file=blacklist_file)
    sampler = samplers.TSNFrameSampler(num_frames)
    Dataset = getattr(data, dataset_type, 'ImageFolder')
    transform = data.get_transform(split=split, size=size)
    dataset = Dataset(root, record_set, sampler, transform=transform)
    for i, (frames, label) in enumerate(dataset):
        if not (i < MAX_ITERS):
            break
        print(i, frames.shape, label)
        assert label < cfg.num_classes_dict[name]
        assert frames.shape == torch.Size((3, num_frames, size, size))


@pytest.mark.skipif(SKIP_VIDEO_ZIP, reason='Data not present')
@pytest.mark.parametrize('name, split, num_frames, size, dataset_type, record_set_type', [
    ('DFDC', 'train', 16, 224, 'DeepfakeZipVideo', 'DeepfakeSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeZipVideo', 'DeepfakeSet'),
    ('DFDC', 'train', 16, 224, 'DeepfakeZipFaceVideo', 'DeepfakeFaceSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeZipFaceVideo', 'DeepfakeFaceSet'),
])
def test_get_dataset_video_zip(name, split, num_frames, size, dataset_type, record_set_type):
    metadata = cfg.get_metadata(name, split=split, dataset_type=dataset_type,
                                data_root=DATA_ROOT)
    print(metadata)
    root = metadata['root']
    metafile = metadata['metafile']
    blacklist_file = metadata['blacklist_file']
    record_set = getattr(data, record_set_type)(metafile, blacklist_file=blacklist_file)
    sampler = samplers.TSNFrameSampler(num_frames)
    Dataset = getattr(data, dataset_type, 'ImageFolder')
    transform = data.get_transform(split=split, size=size)
    dataset = Dataset(root, record_set, sampler, transform=transform)
    for i, (frames, label) in enumerate(dataset):
        if not (i < MAX_ITERS):
            break
        print(i, frames.shape, label)
        assert label < cfg.num_classes_dict[name]
        assert frames.shape == torch.Size((3, num_frames, size, size))


@pytest.mark.skipif(SKIP_FRAME, reason='Data not present')
@pytest.mark.parametrize('name, split, num_frames, size, dataset_type, record_set_type', [
    ('DFDC', 'train', 16, 224, 'DeepfakeFrame', 'DeepfakeSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFrame', 'DeepfakeSet'),
    ('DFDC', 'train', 16, 224, 'DeepfakeFaceFrame', 'DeepfakeFaceSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFaceFrame', 'DeepfakeFaceSet'),
    ('DFDC', 'train', 16, 224, 'DeepfakeFaceCropFrame', 'DeepfakeFaceSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFaceCropFrame', 'DeepfakeFaceSet'),
])
def test_get_dataset_frame(name, split, num_frames, size, dataset_type, record_set_type):
    metadata = cfg.get_metadata(name, split=split, dataset_type=dataset_type,
                                data_root=DATA_ROOT)
    print(metadata)
    root = metadata['root']
    metafile = metadata['metafile']
    blacklist_file = metadata['blacklist_file']
    record_set = getattr(data, record_set_type)(metafile, blacklist_file=blacklist_file)
    sampler = samplers.TSNFrameSampler(num_frames)
    Dataset = getattr(data, dataset_type, 'ImageFolder')
    transform = data.get_transform(split=split, size=size)
    dataset = Dataset(root, record_set, sampler, transform=transform)
    for i, (frames, label) in enumerate(dataset):
        if not (i < MAX_ITERS):
            break
        print(i, frames.shape, label)
        assert label < cfg.num_classes_dict[name]
        assert frames.shape == torch.Size((3, num_frames, size, size))


@pytest.mark.parametrize('video_dir, step', [
    (os.path.join(cfg.DATA_ROOT, 'DeepfakeDetection', 'videos', 'dfdc_train_part_0'), 2),
])
def test_videofolder_dataset(video_dir, step):
    dataset = data.VideoFolder(video_dir, step=step)
    for i, (name, frames, target) in enumerate(dataset):
        if not (i < MAX_ITERS):
            break
        print(i, frames.shape, target)


@pytest.mark.parametrize('filename, step', [
    (os.path.join(cfg.DATA_ROOT, 'DeepfakeDetection', 'videos', 'dfdc_train_part_00.zip'), 2),
])
def test_videozipfile(filename, step):
    dataset = data.VideoZipFile(filename, step=step)
    for i, (name, frames, target) in enumerate(dataset):
        if not (i < MAX_ITERS):
            break
        print(i, frames.shape, target)