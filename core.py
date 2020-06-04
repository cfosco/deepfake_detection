import os
import functools
import time
from operator import add
from typing import Union, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.models as torchvision_models
from torch.nn import init
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import config as cfg
import data
import models as deepfake_models
from pretorched import models, optim, utils
from pretorched.data import samplers, transforms
from pretorched.metrics import accuracy
from pretorched.models import utils as mutils
from pretorched.runners.utils import AverageMeter, ProgressMeter

torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)
torchvision_model_names.extend(['xception'])

torchvision_model_names.extend(['xception', 'mxresnet18', 'mxresnet50'])

dir_path = os.path.dirname(os.path.realpath(__file__))


class CorrCoefLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.reduction_func = self._get_reduction_func(reduction)

    def _get_reduction_func(self, reduction):
        return {'mean': torch.mean}.get(reduction, 'mean')

    def __call__(self, output, target):
        return self.reduction_func(
            torch.cat([corrcoef_loss(x, y) for x, y in zip(output, target)])
        )


def corrcoef_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    )
    return cost


def get_optimizer(model, optimizer_name='SGD', lr=0.001, **kwargs):
    optim_func = getattr(optim, optimizer_name)
    func_kwargs, _ = utils.split_kwargs_by_func(optim_func, kwargs)
    optim_kwargs = {**cfg.optimizer_defaults.get(optimizer_name, {}), **func_kwargs}
    optimizer = optim_func(model.parameters(), lr=lr, **optim_kwargs)
    return optimizer


def get_scheduler(optimizer, scheduler_name='CosineAnnealingLR', **kwargs):
    sched_func = getattr(torch.optim.lr_scheduler, scheduler_name)
    func_kwargs, _ = utils.split_kwargs_by_func(sched_func, kwargs)
    sched_kwargs = {**cfg.scheduler_defaults.get(scheduler_name, {}), **func_kwargs}
    scheduler = sched_func(optimizer, **sched_kwargs)
    return scheduler


def init_weights(model, init_name='ortho'):
    def _init_weights(m, init_func):
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
            init_func(m.weight)
        for l in m.children():
            _init_weights(l, init_func)

    init_func = {
        'ortho': init.orthogonal_,
        'N02': functools.partial(init.normal_, mean=0, std=0.02),
        'glorot': init.xavier_normal_,
        'xavier': init.xavier_normal_,
        'kaiming': init.kaiming_normal_,
    }.get(init_name, 'kaiming')

    _init_weights(model, init_func)
    return model


def get_model(
    model_name: str,
    basemodel_name: str = 'resnet18',
    pretrained: str = 'imagenet',
    init_name: Optional[str] = None,
    num_classes: int = 2,
    normalize=False,
    rescale=True,
) -> Union[
    deepfake_models.Detector,
    deepfake_models.ResManipulatorDetector,
    deepfake_models.ResManipulatorAttnDetector,
    deepfake_models.SeriesManipulatorDetector,
    deepfake_models.SeriesManipulatorAttnDetector,
    deepfake_models.GradCamCaricatureModel,
]:
    if model_name == 'FrameDetector':
        basemodel = get_basemodel(
            basemodel_name,
            pretrained=pretrained,
            num_classes=num_classes,
            init_name=init_name,
        )
        return deepfake_models.FrameDetector(
            basemodel, normalize=normalize, rescale=rescale
        )

    elif model_name == 'AttnFrameDetector':
        basemodel = get_basemodel(
            basemodel_name,
            pretrained=pretrained,
            num_classes=num_classes,
            init_name=init_name,
        )
        return deepfake_models.AttnFrameDetector(
            basemodel,
            normalize=normalize,
            rescale=rescale,
            basemodel_name=basemodel_name,
        )

    elif model_name == 'VideoDetector':
        basemodel = get_basemodel(
            basemodel_name,
            pretrained=pretrained,
            num_classes=num_classes,
            init_name=init_name,
        )
        return deepfake_models.VideoDetector(basemodel)

    elif model_name == 'SeriesManipulatorDetector':
        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=deepfake_models.MagNet(),
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
            ),
        )

    elif model_name == 'SeriesPretrainedManipulatorDetector':
        magnet = deepfake_models.MagNet()
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_e11.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'SeriesPretrainedFrozenManipulatorDetector':
        magnet = deepfake_models.MagNet()
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_e11.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        for p in magnet.parameters():
            p.requires_grad = False

        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'SeriesPretrainedSmallManipulatorDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=3
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_3_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'SeriesPretrainedFrozenSmallManipulatorDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=3
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_3_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        for p in magnet.parameters():
            p.requires_grad = False
        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'SeriesPretrainedFrozenSmallManipulatorAttnDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=3
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_3_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        for p in magnet.parameters():
            p.requires_grad = False

        if basemodel_name not in [
            'samxresnet18',
            'samxresnet34',
            'samxresnet50',
            'ssamxresnet18',
            'ssamxresnet34',
            'ssamxresnet50',
        ]:
            print(f'Warning: {basemodel_name} does not support attention')
            print(f'Switching basemodel to samxresnet18...')
            basemodel_name = 'samxresnet18'
        return deepfake_models.SeriesManipulatorAttnDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'AttnFrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'SeriesPretrainedMediumManipulatorDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=6
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_6_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'SeriesPretrainedFrozenMediumManipulatorDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=6
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_6_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        for p in magnet.parameters():
            p.requires_grad = False
        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'ResPretrainedManipulatorDetector':
        magnet = deepfake_models.MagNet()
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_e11.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        return deepfake_models.ResManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'ResPretrainedSmallManipulatorDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=3
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_3_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        return deepfake_models.ResManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'ResPretrainedFrozenSmallManipulatorDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=3
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_3_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        for p in magnet.parameters():
            p.requires_grad = False

        return deepfake_models.ResManipulatorDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'FrameDetector',  # TODO: add option for VideoDetector
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'ResPretrainedManipulatorAttnDetector':
        magnet = deepfake_models.MagNet()
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_e11.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        if basemodel_name not in [
            'samxresnet18',
            'samxresnet34',
            'samxresnet50',
            'ssamxresnet18',
            'ssamxresnet34',
            'ssamxresnet50',
        ]:
            print(f'Warning: {basemodel_name} does not support attention')
            print(f'Switching basemodel to samxresnet18...')
            basemodel_name = 'samxresnet18'
        return deepfake_models.ResManipulatorAttnDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'AttnFrameDetector',
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'ResPretrainedFrozenSmallManipulatorAttnDetector':
        magnet = deepfake_models.MagNet(
            num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=3
        )
        magnet_ckpt_file = os.path.join(
            dir_path, 'models/deep_motion_mag/ckpt/ckpt_3_1_3_22.pth.tar'
        )
        magnet_ckpt = torch.load(magnet_ckpt_file, map_location='cpu')
        magnet.load_state_dict(mutils.remove_prefix(magnet_ckpt['state_dict']))
        for p in magnet.parameters():
            p.requires_grad = False

        if basemodel_name not in [
            'ssamxresnet18',
            'ssamxresnet50',
            'samxresnet18',
            'samxresnet50',
        ]:
            print(f'Warning: {basemodel_name} does not support attention')
            print(f'Switching basemodel to samxresnet18...')
            basemodel_name = 'samxresnet18'
        return deepfake_models.ResManipulatorAttnDetector(
            manipulator_model=magnet,
            detector_model=get_model(
                'AttnFrameDetector',
                basemodel_name,
                pretrained=pretrained,
                init_name=init_name,
                normalize=True,
                rescale=True,
            ),
        )

    elif model_name == 'GradCamCaricatureModel':
        return deepfake_models.GradCamCaricatureModel(
            face_model=deepfake_models.FaceModel(),
            fake_model=get_model(
                'FrameDetector', basemodel_name, pretrained, init_name=init_name
            ),
            mag_model=deepfake_models.MagNet(),
        )
    else:
        raise ValueError(f'Unreconized model type {model_name}')


def do_normalize(model):
    return not isinstance(
        model,
        (
            deepfake_models.SeriesManipulatorDetector,
            deepfake_models.ResManipulatorDetector,
            deepfake_models.SeriesManipulatorAttnDetector,
            deepfake_models.ResManipulatorAttnDetector,
        ),
    )


def do_rescale(model):
    return not isinstance(
        model,
        (
            deepfake_models.SeriesManipulatorDetector,
            deepfake_models.SeriesManipulatorAttnDetector,
            deepfake_models.ResManipulatorDetector,
            deepfake_models.ResManipulatorAttnDetector,
        ),
    )


def get_basemodel(
    model_name: str,
    num_classes: int = 2,
    pretrained: Optional[str] = 'imagenet',
    init_name: Optional[str] = None,
    **kwargs,
):
    if model_name in ['mxresnet18', 'mxresnet50', 'samxresnet18', 'samxresnet50']:
        model_func = getattr(deepfake_models, model_name)
    else:
        model_func = getattr(models, model_name)
    pretrained = None if pretrained == 'None' else pretrained
    if pretrained is not None:
        # TODO Update THIS!
        nc = {k.lower(): v for k, v in cfg.num_classes_dict.items()}.get(pretrained)
        model = model_func(num_classes=nc, pretrained=pretrained, **kwargs)
        if nc != num_classes:
            in_feat = model.last_linear.in_features
            last_linear = nn.Linear(in_feat, num_classes)
            if init_name is not None:
                print(f'Re-initializing last_linear of {model_name} with {init_name}.')
                last_linear = init_weights(last_linear, init_name)
            model.last_linear = last_linear
    else:
        model = model_func(num_classes=num_classes, pretrained=pretrained, **kwargs)
        if init_name is not None:
            print(f'Initializing {model_name} with {init_name}.')
            model = init_weights(model, init_name)
    return model


def get_transform(
    name='DFDC',
    split='train',
    size=224,
    resolution=256,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    normalize=True,
    degrees=25,
):
    norm = transforms.NormalizeVideo(mean=mean, std=std)
    cropping = {
        'train': torchvision.transforms.Compose(
            [
                transforms.RandomResizedCropVideo(size),
                transforms.RandomHorizontalFlipVideo(),
                transforms.RandomRotationVideo(degrees),
            ]
        ),
        'val': torchvision.transforms.Compose(
            [transforms.ResizeVideo(resolution), transforms.CenterCropVideo(size)]
        ),
    }.get(split, 'val')
    transform = torchvision.transforms.Compose(
        [
            cropping,
            transforms.CollectFrames(),
            transforms.PILVideoToTensor(),
            norm if normalize else transforms.IdentityTransform(),
        ]
    )
    return transform


def get_dataset(
    name,
    data_root,
    split='train',
    size=224,
    resolution=256,
    num_frames=16,
    dataset_type='DeepfakeFrame',
    sampler_type='TemporalSegmentSampler',
    record_set_type='DeepfakeSet',
    segment_count=None,
    normalize=True,
    rescale=True,
    **kwargs,
):

    if isinstance(name, (list, tuple)):
        return torch.utils.data.ConcatDataset(
            [
                get_dataset(
                    n,
                    data_root,
                    split=split,
                    size=size,
                    resolution=resolution,
                    num_frames=num_frames,
                    dataset_type=dataset_type,
                    sampler_type=sampler_type,
                    record_set_type=record_set_type,
                    segment_count=segment_count,
                    normalize=normalize,
                    rescale=rescale,
                    **kwargs,
                )
                for n in name
            ]
        )
    elif name.lower() in ['all', 'full']:
        names = cfg.ALL_DATASETS
        return get_dataset(
            names,
            data_root,
            split=split,
            size=size,
            resolution=resolution,
            num_frames=num_frames,
            dataset_type=dataset_type,
            sampler_type=sampler_type,
            record_set_type=record_set_type,
            segment_count=segment_count,
            normalize=normalize,
            rescale=rescale,
            **kwargs,
        )

    segment_count = num_frames if segment_count is None else segment_count

    metadata = cfg.get_metadata(
        name,
        split=split,
        dataset_type=dataset_type,
        record_set_type=record_set_type,
        data_root=data_root,
    )
    kwargs = {**metadata, **kwargs, 'segment_count': segment_count}

    Dataset = getattr(data, dataset_type, 'DeepfakeFrame')
    RecSet = getattr(data, record_set_type, 'DeepfakeSet')
    Sampler = getattr(samplers, sampler_type, 'TSNFrameSampler')
    r_kwargs, _ = utils.split_kwargs_by_func(RecSet, kwargs)
    s_kwargs, _ = utils.split_kwargs_by_func(Sampler, kwargs)

    record_set = RecSet(**r_kwargs)
    sampler = Sampler(**s_kwargs)
    full_kwargs = {
        'record_set': record_set,
        'sampler': sampler,
        'transform': data.get_transform(
            split=split, size=size, normalize=normalize, rescale=rescale
        ),
        **kwargs,
    }
    dataset_kwargs, _ = utils.split_kwargs_by_func(Dataset, full_kwargs)
    return Dataset(**dataset_kwargs)


def get_hybrid_dataset(
    name,
    root,
    split='train',
    size=224,
    resolution=256,
    dataset_type='ImageFolder',
    load_in_mem=False,
):
    if name != 'Hybrid1365':
        raise ValueError(f'Hybrid Dataset: {name} not implemented')
    imagenet_root = cfg.get_root_dirs(
        'ImageNet', dataset_type=dataset_type, resolution=resolution, data_root=root
    )
    places365_root = cfg.get_root_dirs(
        'Places365', dataset_type=dataset_type, resolution=resolution, data_root=root
    )
    imagenet_dataset = get_dataset(
        'ImageNet',
        resolution=resolution,
        size=size,
        dataset_type=dataset_type,
        load_in_mem=load_in_mem,
        split=split,
        root=imagenet_root,
    )
    placess365_dataset = get_dataset(
        'Places365',
        resolution=resolution,
        size=size,
        dataset_type=dataset_type,
        load_in_mem=load_in_mem,
        target_transform=functools.partial(add, 1000),
        split=split,
        root=places365_root,
    )
    return torch.utils.data.ConcatDataset((imagenet_dataset, placess365_dataset))


def get_dataloader(
    name,
    data_root=None,
    split='train',
    num_frames=16,
    size=224,
    resolution=256,
    dataset_type='DeepfakeFrame',
    sampler_type='TSNFrameSampler',
    record_set_type='DeepfakeSet',
    batch_sampler=None,
    batch_size=64,
    num_workers=8,
    shuffle=True,
    load_in_mem=False,
    pin_memory=True,
    drop_last=False,
    distributed=False,
    segment_count=None,
    normalize=True,
    rescale=True,
    **kwargs,
):

    dataset = get_dataset(
        name,
        data_root,
        split=split,
        size=size,
        resolution=resolution,
        num_frames=num_frames,
        segment_count=segment_count,
        dataset_type=dataset_type,
        sampler_type=sampler_type,
        record_set_type=record_set_type,
        normalize=normalize,
        rescale=rescale,
        **kwargs,
    )
    loader_sampler = (
        DistributedSampler(dataset) if (distributed and split == 'train') else None
    )
    if batch_sampler is not None:
        batch_size = 1
        shuffle = False
        loader_sampler = None
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=loader_sampler,
        batch_sampler=batch_sampler,
        shuffle=(loader_sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_dataloaders(
    name,
    root,
    dataset_type='ImageFolder',
    size=224,
    resolution=256,
    batch_size=32,
    num_workers=12,
    shuffle=[True, False],
    batch_sampler=None,
    distributed=False,
    load_in_mem=False,
    pin_memory=True,
    drop_last=False,
    splits=['train', 'val'],
    normalize=True,
    rescale=True,
    **kwargs,
):
    if not isinstance(shuffle, list):
        shuffle = len(splits) * [shuffle]
    dataloaders = {
        split: get_dataloader(
            name,
            data_root=root,
            split=split,
            size=size,
            resolution=resolution,
            dataset_type=dataset_type,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            shuffle=shuffle[i],
            load_in_mem=load_in_mem,
            pin_memory=pin_memory,
            drop_last=drop_last,
            distributed=distributed,
            normalize=normalize,
            rescale=rescale,
            **kwargs,
        )
        for i, split in enumerate(splits)
    }
    return dataloaders


def get_heatvol_dataloader(
    name,
    data_root=None,
    split='train',
    num_frames=16,
    size=224,
    resolution=256,
    dataset_type='DeepfakeFrame',
    sampler_type='TSNFrameSampler',
    record_set_type='DeepfakeSet',
    batch_size=64,
    num_workers=8,
    shuffle=True,
    load_in_mem=False,
    pin_memory=True,
    drop_last=False,
    distributed=False,
    segment_count=None,
    normalize=True,
    rescale=True,
    **kwargs,
):
    dataset = get_dataset(
        name,
        data_root,
        split=split,
        num_frames=num_frames,
        size=size,
        dataset_type=dataset_type,
        record_set_type=record_set_type,
    )
    if split == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler = data.HeatvolBatchSampler(
        sampler, batch_size, drop_last, dataset.heatvol_inds,
    )
    loader_sampler = (
        DistributedSampler(dataset) if (distributed and split == 'train') else None
    )
    if batch_sampler is not None:
        batch_size = 1
        shuffle = False
        loader_sampler = None
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=loader_sampler,
        batch_sampler=batch_sampler,
        shuffle=(loader_sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_heatvol_dataloaders(
    name,
    root,
    dataset_type='ImageFolder',
    size=224,
    resolution=256,
    batch_size=32,
    num_workers=12,
    shuffle=[True, False],
    batch_sampler=None,
    distributed=False,
    load_in_mem=False,
    pin_memory=True,
    drop_last=False,
    splits=['train', 'val'],
    normalize=True,
    rescale=True,
    **kwargs,
):
    if not isinstance(shuffle, list):
        shuffle = len(splits) * [shuffle]
    dataloaders = {
        split: get_heatvol_dataloader(
            name,
            data_root=root,
            split=split,
            size=size,
            resolution=resolution,
            dataset_type=dataset_type,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            shuffle=shuffle[i],
            load_in_mem=load_in_mem,
            pin_memory=pin_memory,
            drop_last=drop_last,
            distributed=distributed,
            normalize=normalize,
            rescale=rescale,
            **kwargs,
        )
        for i, split in enumerate(splits)
    }
    return dataloaders


def step_fn(input, target, model, criterion, optimizer=None, mode='train', **kwargs):
    # Compute output.
    output = model(input)
    loss = criterion(output, target)

    if mode == 'train':
        # Compute gradient and do SGD step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output, loss


def name_from_args(args):
    name = '_'.join(
        [
            args.model_name,
            args.basemodel_name,
            args.dataset.lower(),
            args.sampler_type,
            f'seg_count-{args.segment_count}'
            if args.sampler_type == 'TSNFrameSampler'
            else f'clip_length-{args.clip_length}_frame_step-{args.frame_step}',
            f'init-{"-".join([args.pretrained, args.init]) if args.pretrained else args.init}',
            f'optim-{args.optimizer}',
            f'lr-{args.lr}',
            f'sched-{args.scheduler}',
            f'bs-{args.batch_size}',
        ]
    )
    return name


def train_gandataset(
    train_loader, model, gan, criterion, optimizer, epoch, args, display=True
):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = images.size(0)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output, loss = step_fn(images, target, model, criterion, optimizer)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

        gan_images = gan(*gan.generate_input(batch_size))
        fake_target = torch.ones(batch_size).cuda(args.gpu, non_blocking=True).long()

        output, loss = step_fn(gan_images, fake_target, model, criterion, optimizer)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and display:
            progress.display(i)

    return top1.avg, losses.avg
