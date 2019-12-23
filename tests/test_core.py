import pytest
import torch

from pretorched.data import samplers
# import sys
# sys.path.append('../')
# import config as cfg
from .. import config as cfg
from .. import data
# from .. import core, data
# import config as cfg
# import core, data

DATA_ROOT = cfg.DATA_ROOT
MAX_ITERS = 10
BATCH_SIZE = 16


@pytest.mark.parametrize('name, split, num_frames, size, dataset_type, record_set_type', [
    ('DFDC', 'train', 16, 224, 'DeepfakeFrame', 'DeepfakeSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFrame', 'DeepfakeSet'),
    ('DFDC', 'train', 16, 224, 'DeepfakeFaceFrame', 'DeepfakeFaceSet'),
])
def test_get_dataset(name, split, num_frames, size, dataset_type, record_set_type):
    metadata = cfg.get_metadata(name, dataset_type=dataset_type,
                                data_root=DATA_ROOT)

    root = metadata['root']
    metafile = metadata['metafile']
    record_set = getattr(data, record_set_type)(metafile)
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
