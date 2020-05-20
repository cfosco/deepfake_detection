import functools
import time
from operator import add
from typing import Union

import torch
import torch.nn as nn
import torchvision
import torchvision.models as torchvision_models
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import config as cfg
import data
import models as deepfake_models
from pretorched import models, optim, utils
from pretorched.data import samplers, transforms
from pretorched.metrics import accuracy
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
    model_name,
    basemodel_name='resnet18',
    pretrained='imagenet',
    init_name=None,
    num_classes=2,
) -> Union[
    deepfake_models.FrameDetector,
    deepfake_models.SeriesManipulatorDetector,
    deepfake_models.GradCamCaricatureModel,
]:
    if model_name == 'FrameDetector':
        basemodel = get_basemodel(
            basemodel_name,
            pretrained=pretrained,
            num_classes=num_classes,
            init_name=init_name,
        )
        return deepfake_models.FrameDetector(basemodel)
    elif model_name == 'SeriesManipulatorDetector':
        return deepfake_models.SeriesManipulatorDetector(
            manipulator_model=deepfake_models.MagNet(),
            detector_model=get_model(
                'FrameDetector', basemodel_name, pretrained=pretrained, init_name=init_name
            ),
        )
    elif model_name == 'CaricatureModel':
        return deepfake_models.GradCamCaricatureModel(
            face_model=deepfake_models.FaceModel(),
            fake_model=get_model(
                'FrameDetector', basemodel_name, pretrained, init_name=init_name
            ),
            mag_model=deepfake_models.MagNet(),
        )
    else:
        raise ValueError(f'Unreconized model type {model_name}')


def get_basemodel(
    model_name, num_classes=2, pretrained='imagenet', init_name=None, **kwargs
):
    if model_name in ['mxresnet18', 'mxresnet50']:
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
        'transform': data.get_transform(split=split, size=size),
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
    batch_size=64,
    num_workers=8,
    shuffle=True,
    load_in_mem=False,
    pin_memory=True,
    drop_last=False,
    distributed=False,
    segment_count=None,
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
        **kwargs,
    )
    loader_sampler = (
        DistributedSampler(dataset) if (distributed and split == 'train') else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=loader_sampler,
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
    distributed=False,
    load_in_mem=False,
    pin_memory=True,
    drop_last=False,
    splits=['train', 'val'],
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
            num_workers=num_workers,
            shuffle=shuffle[i],
            load_in_mem=load_in_mem,
            pin_memory=pin_memory,
            drop_last=drop_last,
            distributed=distributed,
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
            f'seg_count-{args.segment_count}',
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
