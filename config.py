import argparse
import os
import subprocess
from collections import defaultdict

import pretorched.models as models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

DATA_ROOT = os.getenv('DATA_ROOT', '~/Datasets')

ALL_DATASETS = [
    'DFDC',
    'FaceForensics',
    'CelebDF',
    'YouTubeDeepfakes',
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

num_classes_dict = {
    'ImageNet': 1000,
    'Places365': 365,
    'Hybrid1365': 1365,
    'Moments': 339,
    'kinetics-400': 400,
    'DFDC': 2,
    'FaceForensics': 2,
    'CelebDF': 2,
    'YouTubeDeepfakes': 2,
    'all': 2,
}

root_dirs = {
    'ImageNet': os.path.join(DATA_ROOT, 'ImageNet'),
    'Places365': os.path.join(DATA_ROOT, 'Places365'),
}

optimizer_defaults = {
    'SGD': {'momentum': 0.9, 'weight_decay': 1e-4},
}

scheduler_defaults = {'CosineAnnealingLR': {'T_max': 60}}


def has_libx264():
    out = subprocess.Popen(
        ['ffmpeg', '-loglevel', 'error', '-codecs'], stdout=subprocess.PIPE
    )
    out = subprocess.Popen(['grep', 'x264'], stdin=out.stdout, stdout=subprocess.PIPE)
    out = subprocess.run(['wc', '-l'], stdin=out.stdout, stdout=subprocess.PIPE)
    if out.returncode == 0:
        return '1' == out.stdout.decode().strip()


DEFAULT_VIDEO_CODEC = 'libx264' if has_libx264() else 'mpeg4'


def get_root_dirs(
    name, dataset_type='DeepfakeFrame', resolution=224, data_root=DATA_ROOT
):
    root_dirs = {
        'DFDC': {
            'DeepfakeFrame': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/frames'), {}
            ),
            'DeepfakeFaceFrame': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/face_frames'), {}
            ),
        },
        'ImageNet': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'ImageNet'), {}),
            'ImageFolder': defaultdict(lambda: os.path.join(data_root, 'ImageNet'), {}),
        },
        'Places365': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'Places365'), {}),
            'ImageFolder': defaultdict(
                lambda: os.path.join(data_root, 'Places365'), {}
            ),
        },
        'Hybrid1365': {
            'ImageHDF5': defaultdict(lambda: data_root, {}),
            'ImageFolder': defaultdict(lambda: data_root, {}),
        },
    }
    return root_dirs[name][dataset_type][resolution]


def get_metadata(
    name,
    split='train',
    dataset_type='DeepfakeFrame',
    record_set_type='DeepfakeSet',
    resolution=224,
    data_root=DATA_ROOT,
):
    root_dirs = {
        'DFDC': {
            'DeepfakeVideo': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/videos'), {}
            ),
            'DeepfakeZipVideo': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/videos'), {}
            ),
            'DeepfakeFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/facenet_videos'), {}
            ),
            'DeepfakeFaceHeatvolVideo': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/facenet_videos'), {},
            ),
            'DeepfakeZipFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/facenet_videos'), {}
            ),
            'DeepfakeFrame': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/frames'), {}
            ),
            'DeepfakeFaceFrame': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/facenet_frames'), {}
            ),
            'DeepfakeFaceCropFrame': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection/frames'), {}
            ),
        },
        'FaceForensics': {
            'DeepfakeVideo': defaultdict(
                lambda: os.path.join(data_root, 'FaceForensics'), {}
            ),
            'DeepfakeZipVideo': defaultdict(
                lambda: os.path.join(data_root, 'FaceForensics'), {}
            ),
            'DeepfakeFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'FaceForensics'), {}
            ),
            'DeepfakeZipFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'FaceForensics'), {}
            ),
            'DeepfakeFaceHeatvolVideo': defaultdict(
                lambda: os.path.join(data_root, 'FaceForensics'), {}
            ),
        },
        'CelebDF': {
            'DeepfakeVideo': defaultdict(
                lambda: os.path.join(data_root, 'CelebDF/videos'), {}
            ),
            'DeepfakeZipVideo': defaultdict(
                lambda: os.path.join(data_root, 'CelebDF/videos'), {}
            ),
            'DeepfakeFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'CelebDF/facenet_videos'), {}
            ),
            'DeepfakeZipFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'CelebDF/facenet_videos'), {}
            ),
            'DeepfakeFaceHeatvolVideo': defaultdict(
                lambda: os.path.join(data_root, 'CelebDF/facenet_videos'), {}
            ),
        },
        'YouTubeDeepfakes': {
            'DeepfakeVideo': defaultdict(
                lambda: os.path.join(data_root, 'YouTubeDeepfakes/videos'), {}
            ),
            'DeepfakeZipVideo': defaultdict(
                lambda: os.path.join(data_root, 'YouTubeDeepfakes/videos'), {}
            ),
            'DeepfakeFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'YouTubeDeepfakes/facenet_videos'), {}
            ),
            'DeepfakeZipFaceVideo': defaultdict(
                lambda: os.path.join(data_root, 'YouTubeDeepfakes/facenet_videos'), {}
            ),
            'DeepfakeFaceHeatvolVideo': defaultdict(
                lambda: os.path.join(data_root, 'YouTubeDeepfakes/facenet_videos'), {}
            ),
        },
        'ImageNet': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'ImageNet'), {}),
            'ImageFolder': defaultdict(lambda: os.path.join(data_root, 'ImageNet'), {}),
        },
        'Places365': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'Places365'), {}),
            'ImageFolder': defaultdict(
                lambda: os.path.join(data_root, 'Places365'), {}
            ),
        },
        'Hybrid1365': {
            'ImageHDF5': defaultdict(lambda: data_root, {}),
            'ImageFolder': defaultdict(lambda: data_root, {}),
        },
        'all': {
            'DeepfakeVideo': defaultdict(lambda: data_root, {}),
            'DeepfakeZipVideo': defaultdict(lambda: data_root, {}),
            'DeepfakeFaceVideo': defaultdict(lambda: data_root, {}),
            'DeepfakeZipFaceVideo': defaultdict(lambda: data_root, {}),
            'DeepfakeFaceHeatvolVideo': defaultdict(lambda: data_root, {},),
        },
    }
    #     print(name, dataset_type, resolution, root_dirs[name])
    root = root_dirs[name][dataset_type][resolution]
    fname = {'train': 'metadata.json', 'val': 'test_metadata.json'}.get(split, 'train')
    if name == 'DFDC' and split == 'val':
        pass
        # fname = 'aug_test_metadata.json'

    metafiles = {
        'DFDC': {
            'DeepfakeSet': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection', fname), {}
            ),
            'DeepfakeFaceSet': defaultdict(
                lambda: os.path.join(data_root, 'DeepfakeDetection', fname), {}
            ),
        },
        'FaceForensics': {
            'DeepfakeSet': defaultdict(
                lambda: os.path.join(data_root, 'FaceForensics', fname), {}
            ),
            'DeepfakeFaceSet': defaultdict(
                lambda: os.path.join(data_root, 'FaceForensics', fname), {}
            ),
        },
        'CelebDF': {
            'DeepfakeSet': defaultdict(
                lambda: os.path.join(data_root, 'CelebDF', fname), {}
            ),
            'DeepfakeFaceSet': defaultdict(
                lambda: os.path.join(data_root, 'CelebDF', fname), {}
            ),
        },
        'YouTubeDeepfakes': {
            'DeepfakeSet': defaultdict(
                lambda: os.path.join(data_root, 'YouTubeDeepfakes', fname), {}
            ),
            'DeepfakeFaceSet': defaultdict(
                lambda: os.path.join(data_root, 'YouTubeDeepfakes', fname), {}
            ),
        },
        'all': {
            'DeepfakeSet': defaultdict(lambda: data_root, {}),
            'DeepfakeFaceSet': defaultdict(lambda: data_root, {}),
            'DeepfakeFaceHeatvolVideo': defaultdict(lambda: data_root, {},),
        },
    }
    metafile = metafiles[name][record_set_type][resolution]
    blacklist_file = (
        os.path.join(
            data_root, {'DFDC': 'DeepfakeDetection'}.get(name, name), 'test_videos.json'
        )
        if (split == 'train')
        else None
    )
    # if name == 'DFDC' and split == 'train':
    # blacklist_file = os.path.join(data_root, 'DeepfakeDetection', 'aug_test_videos.json')
    return {'root': root, 'metafile': metafile, 'blacklist_file': blacklist_file}


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset', type=str, default='DFDC')
    parser.add_argument(
        '--data_root',
        metavar='DIR',
        default=None,
        help='path to data_root containing all datasets',
    )
    parser.add_argument(
        '-a',
        '--arch',
        metavar='ARCH',
        default='resnet3d50',
        choices=model_names,
        help='model architecture: '
        + ' | '.join(model_names)
        + ' (default: resnet3d50)',
    )
    parser.add_argument('--model_name', type=str, default='FrameDetector')
    parser.add_argument('--basemodel_name', type=str, default='resnet18')
    parser.add_argument('--segment_count', type=int, default=16)
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--frame_step', type=int, default=5)
    parser.add_argument('--dataset_type', type=str, default='DeepfakeFaceVideo')
    parser.add_argument('--record_set_type', type=str, default='DeepfakeFaceSet')
    parser.add_argument('--sampler_type', type=str, default='ClipSampler')
    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        dest='pretrained',
        help='use a pre-trained model',
    )
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--init', type=str, default='ortho')
    parser.add_argument('--test_split')

    parser.add_argument('--version', '-v', default=None)
    parser.add_argument(
        '-j',
        '--num_workers',
        default=12,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)',
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        metavar='N',
        help='number of total epochs to run',
    )
    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)',
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=64,
        type=int,
        metavar='N',
        help='mini-batch size (default: 64), this is the total '
        'batch size of all GPUs on the current node when '
        'using Data Parallel or Distributed Data Parallel',
    )
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.001,
        type=float,
        metavar='LR',
        help='initial learning rate',
        dest='lr',
    )
    parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M', help='momentum'
    )
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)',
        dest='weight_decay',
    )
    parser.add_argument(
        '-p',
        '--print-freq',
        default=10,
        type=int,
        metavar='N',
        help='print frequency (default: 10)',
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)',
    )
    parser.add_argument('--weights_dir', default='weights', type=str, metavar='PATH')
    parser.add_argument('--logs_dir', default='logs', type=str, metavar='PATH')
    parser.add_argument('--results_dir', default='results', type=str, metavar='PATH')
    parser.add_argument(
        '-e',
        '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set',
    )
    parser.add_argument(
        '--world-size',
        '-w',
        default=1,
        type=int,
        help='number of nodes for distributed training',
    )
    parser.add_argument(
        '--rank', '-r', default=0, type=int, help='node rank for distributed training'
    )
    parser.add_argument(
        '--dist-url',
        default='tcp://localhost:23456',
        type=str,
        help='url used to set up distributed training',
    )
    parser.add_argument(
        '--dist-backend', default='nccl', type=str, help='distributed backend'
    )
    parser.add_argument(
        '--seed', default=None, type=int, help='seed for initializing training. '
    )
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument(
        '--ddp',
        '-ddp',
        '--multiprocessing-distributed',
        action='store_true',
        dest='multiprocessing_distributed',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training',
    )
    return parser


def add_human_parser(parser):
    parser.add_argument('--ce_weight', type=float, default=1.5)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--cc_weight', type=float, default=-0.3)
    parser.add_argument('--init_weights', type=str, default=None)
    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = DATA_ROOT
    args.num_classes = num_classes_dict[args.dataset]
    return args


def parse_human_args():
    parser = get_parser()
    parser = add_human_parser(parser)
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = DATA_ROOT
    args.num_classes = num_classes_dict[args.dataset]
    return args
