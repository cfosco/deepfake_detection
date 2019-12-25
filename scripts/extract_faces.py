#!/usr/bin/env python
import os
from multiprocessing.pool import ThreadPool
import subprocess

import torch

PARTS = [16, 18, 19, 20, 21]
BATCH_SIZE = 32
DATA_ROOT = os.getenv('DATA_ROOT')
VIDEO_ROOT = os.path.join(DATA_ROOT, 'DeepfakeDetection', 'videos')
FACES_ROOT = os.path.join(DATA_ROOT, 'DeepfakeDetection', 'face_frames')

num_gpus = torch.cuda.device_count()

cmd = ' '.join(['CUDA_VISIBLE_DEVICES={} python -m fire data.faces videos_to_faces',
                VIDEO_ROOT, FACES_ROOT,
                '--batch_size', str(BATCH_SIZE),
                '--parts', str(PARTS).replace(' ', '')])
                #'--parts', ' '.join(list(map(str, PARTS)))])


def run(device):
    c = cmd.format(device)
    print(c)
    # os.system(c
    subprocess.run(c, shell=True)


with ThreadPool(num_gpus) as pool:
    pool.map(run, range(num_gpus))
    pool.close()
    pool.join()
