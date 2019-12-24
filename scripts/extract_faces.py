#!/usr/bin/env python
import os
from multiprocessing.pool import ThreadPool
import subprocess

import torch

BATCH_SIZE = 300
DATA_ROOT = os.getenv('DATA_ROOT')
VIDEO_ROOT = os.path.join(DATA_ROOT, 'DeepfakeDetection', 'videos')
FACES_ROOT = os.path.join(DATA_ROOT, 'DeepfakeDetection', 'face_frames')

num_gpus = torch.cuda.device_count()

cmd = ' '.join(['CUDA_VISIBLE_DEVICES={} python -m fire data.faces videos_to_faces', VIDEO_ROOT, FACES_ROOT, '--batch_size', str(BATCH_SIZE)])


def run(device):
    c = cmd.format(device)
    print(c)
    # os.system(c
    subprocess.run(c, shell=True)


with ThreadPool(num_gpus) as pool:
    pool.map(run, range(num_gpus))
    pool.close()
    pool.join()
