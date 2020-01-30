import json
import os
import random
import shutil
import time
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

import pretorched
from pretorched import loggers
from pretorched.metrics import accuracy
from pretorched.runners.utils import AverageMeter, ProgressMeter

import core
import models
import config as cfg
# from torchvideo.internal.readers import _get_videofile_frame_count, _is_video_file, default_loader
from data import VideoFolder, video_collate_fn
from torchvideo.transforms import PILVideoToTensor
from data import faces
import cv2


# TEST_VIDEO_DIR = '/kaggle/input/deepfake-detection-challenge/test_videos/'
# SAMPLE_SUBMISSION_CSV = '/kaggle/input/deepfake-detection-challenge/sample_submission.csv'

TEST_VIDEO_DIR = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'test_videos')
SAMPLE_SUBMISSION_CSV = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'sample_submission.csv')
TARGET_FILE = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'test_targets.json')


def main(video_dir, margin=100, chunk_size=300):
    args = cfg.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = VideoFolder(video_dir, target_file=TARGET_FILE)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    fakenet = pretorched.resnet18(num_classes=2, pretrained=None)
    fakenet.load_state_dict({k.replace('module.model.', ''): v
                             for k, v in torch.load('models/data/'
                                                    'resnet18_dfdc_seg_count-24_init-imagenet-'
                                                    'ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs'
                                                    '-64_best.pth.tar')['state_dict'].items()})
    facenet = models.FaceModel(size=fakenet.input_size[-1],
                               device=device,
                               margin=margin,
                               min_face_size=50,
                               keep_all=True,
                               post_process=False,
                               select_largest=False,
                               chunk_size=chunk_size)

    fakenet.eval()
    detector = models.FrameModel(fakenet, normalize=True)
    model = models.DeepfakeDetector(facenet, detector)
    model.to(device)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda(device)
    sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub.label = 0.5
    sub = sub.set_index('filename', drop=False)

    preds, acc, loss = validate(dataloader, model, criterion, device=device)
    for filename, prob in preds.items():
        sub.loc[filename, 'label'] = prob

    # sub.to_csv('submission.csv', index=False)


def validate(val_loader, model, criterion, device='cuda', display=True, print_freq=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    preds = {}

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (filenames, images, target) in enumerate(val_loader):
            if device is not None:
                images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            probs = torch.softmax(output, 1)
            for fn, prob in zip(filenames, probs):
                preds[fn] = prob[1].item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and display:
                progress.display(i)

        if display:
            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return preds,top1.avg, losses.avg


if __name__ == '__main__':
    main(TEST_VIDEO_DIR)
