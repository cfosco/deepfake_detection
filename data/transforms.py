import torchvision

import torchvideo


def get_transform(name='DFDC', split='train', size=224, resolution=256,
                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                  normalize=True):
    norm = torchvideo.transforms.NormalizeVideo(mean=mean, std=std)
    cropping = {
        'train': torchvision.transforms.Compose([
            torchvideo.transforms.RandomResizedCropVideo(size),
            torchvideo.transforms.RandomHorizontalFlipVideo()]),
        'val': torchvision.transforms.Compose([
            torchvideo.transforms.ResizeVideo(resolution),
            torchvideo.transforms.CenterCropVideo(size),
        ])
    }.get(split, 'val')
    transform = torchvision.transforms.Compose([
        cropping,
        torchvideo.transforms.CollectFrames(),
        torchvideo.transforms.PILVideoToTensor(),
        norm if normalize else torchvideo.transforms.IdentityTransform(),
    ])
    return transform
