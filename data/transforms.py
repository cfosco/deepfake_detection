import torchvision

from pretorched.data import transforms


def get_transform(name='DFDC', split='train', size=224, resolution=256,
                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                  normalize=True, degrees=10):
    norm = transforms.NormalizeVideo(mean=mean, std=std)
    cropping = {
        'train': torchvision.transforms.Compose([
            transforms.RandomResizedCropVideo(size),
            transforms.RandomHorizontalFlipVideo(),
            transforms.RandomRotationVideo(degrees)]),
        'val': torchvision.transforms.Compose([
            transforms.ResizeVideo(resolution),
            transforms.CenterCropVideo(size),
        ])
    }.get(split, 'val')
    transform = torchvision.transforms.Compose([
        cropping,
        transforms.CollectFrames(),
        transforms.PILVideoToTensor(),
        norm if normalize else transforms.IdentityTransform(),
    ])
    return transform
