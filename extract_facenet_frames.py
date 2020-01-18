import json
import os
import random
import shutil
import sys
import time
import warnings
from multiprocessing.pool import Pool, ThreadPool

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

import config as cfg
import core
import models
import pretorched
from models import MTCNN, FaceModel
from pretorched import loggers
from pretorched.metrics import accuracy
from pretorched.runners.utils import AverageMeter, ProgressMeter
from pretorched.utils import chunk
from torchvideo.internal.readers import (_get_videofile_frame_count,
                                         _is_video_file, default_loader)
from torchvideo.transforms import PILVideoToTensor

STEP = 2
CHUNK_SIZE = 150
NUM_WORKERS = 2

try:
    PART = sys.argv[1]
except IndexError:
    PART = 'dfdc_train_part_0'
VIDEO_ROOT = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'videos')
FACE_ROOT = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'facenet_frames')
VIDEO_DIR = os.path.join(VIDEO_ROOT, PART)
FACE_DIR = os.path.join(FACE_ROOT, PART)


def read_frames(video, fps=30, step=1):
    # Open video file
    video_capture = cv2.VideoCapture(video)
    video_capture.set(cv2.CAP_PROP_FPS, fps)

    count = 0
    while video_capture.isOpened():
        # Grab a single frame of video
        ret = video_capture.grab()

        # Bail out when the video file ends
        if not ret:
            break
        if count % step == 0:
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            ret, frame = video_capture.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        count += 1


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, step=2, transform=None):
        self.root = root
        self.step = step
        self.videos_filenames = sorted([f for f in os.listdir(root) if f.endswith('.mp4')])

        if transform is None:
            transform = PILVideoToTensor(rescale=False)
        self.transform = transform

    def __getitem__(self, index):
        name = self.videos_filenames[index]
        video_filename = os.path.join(self.root, name)
        frames = read_frames(video_filename, step=self.step)
        frames = torch.stack(list(map(TF.to_tensor, frames))).transpose(0, 1)
        # frames = torch.tensor(read_frames(video_filename, step=self.step), dtype=torch.float32)
        # frames = default_loader(video_filename, list(range(0, 300, self.step)))
        # if self.transform is not None:
        # frames = self.transform(frames)
        return name, frames

    def __len__(self):
        return len(self.videos_filenames)


def save_image(args):
    image, filename = args
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        image.save(filename)
    except Exception:
        pass


def main():

    os.makedirs(FACE_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = VideoDataset(VIDEO_DIR, step=STEP)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, drop_last=False)

    size = 256
    margin = 50
    fdir_tmpl = 'face_{}'
    tmpl = '{:06d}.jpg'
    metadata_fname = 'faces_metadata.json'
    model = FaceModel(size=size,
                      device=device,
                      margin=margin,
                      min_face_size=50,
                      keep_all=True,
                      post_process=False,
                      select_largest=False)
    # facenet = models.FaceModel(select_largest=False)
    # fakenet = pretorched.resnet18(num_classes=2, pretrained=None)
    # fakenet.load_state_dict({k.replace('module.model.', ''): v
    #                          for k, v in torch.load('weights/half_split/'
    #                                                 'resnet18_dfdc_seg_count-24_init-imagenet-'
    #                                                 'ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs'
    #                                                 '-64_best.pth.tar')['state_dict'].items()})
    # fakenet.eval()
    # detector = models.FrameModel(fakenet, normalize=True)
    # model = models.DeepfakeDetector(facenet, detector)
    # model.to(device)

    cudnn.benchmark = True

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(dataset),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (filename, x) in enumerate(dataloader):
            print(f'Extracting faces from: {filename}')
            x = x.to(device)
            x.mul_(255.)
            bs, nc, d, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # [bs, d, nc, h, w]
            # x = x.reshape(-1, *x.shape[2:])  # [bs * d, nc, h, w]
            x = x.view(-1, *x.shape[2:])  # [bs * d, nc, h, w]

            save_dir = os.path.join(FACE_DIR, filename[0])
            os.makedirs(save_dir, exist_ok=True)
            save_paths = [os.path.join(save_dir, '{:06d}.jpg'.format(idx)) for idx in range(d)]
            faces_out = []
            for xx, ss in zip(chunk(x, CHUNK_SIZE), chunk(save_paths, CHUNK_SIZE)):
                # out = model.model(xx, smooth=True, save_path=ss)
                out = model.model(xx, smooth=True)
                out = torch.stack(out).cpu()
                faces_out.append(out)
            min_face = min([f.shape[1] for f in faces_out])
            faces_out = torch.cat([f[:, :min_face] for f in faces_out])
            # print('faces_out', faces_out.shape)
            face_images = {i: [Image.fromarray(ff.permute(1, 2, 0).numpy().astype(np.uint8)) for ff in f]
                           for i, f in enumerate(faces_out.permute(1, 0, 2, 3, 4))}

            metadata = {
                'filename': os.path.basename(filename[0]),
                'num_faces': len(face_images),
                'num_frames': [len(f) for f in face_images.values()],
                'dir_tmpl': fdir_tmpl,
                'im_tmpl': tmpl,
                'full_tmpl': os.path.join(fdir_tmpl, tmpl),
                'face_names': [fdir_tmpl.format(k) for k in face_images],
                'face_nums': list(face_images.keys()),
                'margin': margin,
                'size': size,
                'fps': 30,
            }

            with open(os.path.join(save_dir, metadata_fname), 'w') as f:
                json.dump(metadata, f)

            for face_num, faces in face_images.items():
                num_images = len(faces)
                out_filename = os.path.join(save_dir, fdir_tmpl.format(face_num), tmpl)
                with ThreadPool(num_images) as pool:
                    names = (out_filename.format(i) for i in range(1, num_images + 1))
                    list(tqdm(pool.imap(save_image, zip(faces, names)), total=num_images))
                    pool.close()
                    pool.join()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)


if __name__ == '__main__':
    main()

    # count = 0
    # for o in out:
    #     if o is None:
    #         try:
    #             o = out[count-1]
    #             out[] = o
    #         except IndexError:
    #             try:
    #                 o = out[i + 1]
    #                 out[i] = o
    #             except IndexError:
    #                 pass
    #     for j, f in enumerate(o):
    #         print(count, j,  o.shape)
    #     count += STEP
    # for i, o in enumerate(out):
    # if o is None:
    # try:
    # out[i] = out[i-1]
    # except IndexError:
    # pass
    # out = torch.stack(out)
    # out = out.view(bs, d, nc, *out.shape[-2:])
    # out = out.permute(0, 2, 1, 3, 4)  # [bs, nc, d, h, w]
    # print(f'out: {out.shape}')

    # for i, o in enumerate(out):
    #     if o is None:
    #         try:
    #             out[i] = out[i-1]
    #         except IndexError:
    #             pass
    # out = torch.stack(out)
    # out = out.view(bs, d, nc, *out.shape[-2:])
    # out = out.permute(0, 2, 1, 3, 4)  # [bs, nc, d, h, w]
    # print(f'out: {out.shape}')
    # compute output
    # output = model(images)
    # prob = torch.softmax(output, 1)[0, 1]
    # sub.loc[filename, 'label'] = prob.item()
    # print(f'filename: {filename}; prob: {prob:.3f}')
