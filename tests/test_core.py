import pytest
import torch

from pretorched.data import samplers

from .. import config as cfg
from .. import data


DATA_ROOT = cfg.DATA_ROOT
MAX_ITERS = 10
BATCH_SIZE = 16


@pytest.mark.parametrize('name, split, num_frames, size, dataset_type, record_set_type', [
    ('DFDC', 'train', 16, 224, 'DeepfakeFrame', 'DeepfakeSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFrame', 'DeepfakeSet'),
    ('DFDC', 'train', 16, 224, 'DeepfakeFaceFrame', 'DeepfakeFaceSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFaceFrame', 'DeepfakeFaceSet'),
    ('DFDC', 'train', 16, 224, 'DeepfakeFaceCropFrame', 'DeepfakeFaceSet'),
    ('DFDC', 'val', 16, 224, 'DeepfakeFaceCropFrame', 'DeepfakeFaceSet'),
])
def test_get_dataset(name, split, num_frames, size, dataset_type, record_set_type):
    metadata = cfg.get_metadata(name, split=split, dataset_type=dataset_type,
                                data_root=DATA_ROOT)

    root = metadata['root']
    metafile = metadata['metafile']
    record_set = getattr(data, record_set_type)(metafile)
    sampler = samplers.TSNFrameSampler(num_frames)
    Dataset = getattr(data, dataset_type, 'ImageFolder')
    transform = data.get_transform(split=split, size=size)
    dataset = Dataset(root, record_set, sampler, transform=transform)
    print(len(dataset))
    frames, label = dataset[0]
    assert label < cfg.num_classes_dict[name]
    assert frames.shape == torch.Size((3, num_frames, size, size))
