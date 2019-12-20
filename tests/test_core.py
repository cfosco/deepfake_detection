import pytest
import torch

from pretorched.data import samplers

from .. import config as cfg
from .. import core, data

DATA_ROOT = cfg.DATA_ROOT
MAX_ITERS = 10
BATCH_SIZE = 16


@pytest.mark.parametrize('name, split, num_frames, size, dataset_type', [
    ('DFDC', 'train', 16, 224, 'DeepfakeFrame'),
])
def test_get_dataset(name, split, num_frames, size, dataset_type):
    metadata = cfg.get_metadata(name, dataset_type=dataset_type,
                                data_root=DATA_ROOT)

    root = metadata['root']
    metafile = metadata['metafile']
    record_set = data.DeepfakeSet(metafile)
    sampler = samplers.TSNFrameSampler(num_frames)
    Dataset = getattr(data, dataset_type, 'ImageFolder')
    transform = data.get_transform(split=split, size=size)
    dataset = Dataset(root, record_set, sampler, transform=transform)
    # dataset = core.get_dataset(name=name, root=root,
    #    split=split, size=224,
    #    dataset_type=dataset_type)
    frames, label = dataset[0]
    assert label < cfg.num_classes_dict[name]
    assert frames.shape == torch.Size((3, num_frames, size, size))


@pytest.mark.parametrize('name, split, num_frames, size, dataset_type', [
    ('DFDC', 'train', 16, 224, 'DeepfakeFrame'),
])
def test_get_dataloader(name, split, num_frames, size, dataset_type):

    loader = core.get_dataloader(
        name, data_root=DATA_ROOT, split=split, size=size, dataset_type=dataset_type,
        batch_size=BATCH_SIZE, num_workers=16, shuffle=True, load_in_mem=False, pin_memory=True,
        drop_last=True, distributed=False)

    for i, (x, y) in enumerate(loader):
        if i >= MAX_ITERS:
            break

        assert y.shape == torch.Size((BATCH_SIZE,))
        assert x.shape == torch.Size((BATCH_SIZE, 3, num_frames, size, size))
