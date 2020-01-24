#!/usr/bin/env python
import contextlib
import json
import os
import sys
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import pretorched
from data import VideoFolder
from models import FaceModel, deepmmag
from pretorched.runners.utils import AverageMeter, ProgressMeter
from pretorched.utils import chunk

STEP = 1
CHUNK_SIZE = 300
NUM_WORKERS = 2
OVERWRITE = False
REMOVE_FRAMES = True

try:
    PART = sys.argv[1]
except IndexError:
    PART = 'dfdc_train_part_0'
VIDEO_ROOT = os.path.join(os.environ['SCRATCH_DATA_ROOT'], 'DeepfakeDetection', 'videos')
FACE_ROOT = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'facenet_smooth_frames')
VIDEO_DIR = os.path.join(VIDEO_ROOT, PART)
FACE_DIR = os.path.join(FACE_ROOT, PART)


def save_image(args):
    image, filename = args
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        image.save(filename, quality=95)
    except Exception:
        pass


def main():

    os.makedirs(FACE_DIR, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_motion_mag = deepmmag.get_motion_mag()
    dataset = VideoFolder(VIDEO_DIR, step=STEP)
    if not OVERWRITE:
        video_filenames = []
        for f in dataset.videos_filenames:
            frame_dir = os.path.join(FACE_DIR, f)
            if os.path.exists(frame_dir):
                if os.listdir(frame_dir):
                    print(f'Skipping {frame_dir}')
                    continue
            video_filenames.append(f)
        dataset.videos_filenames = video_filenames

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=False, drop_last=False)

    size = 360
    margin = 100
    fdir_tmpl = 'face_{}'
    tmpl = '{:06d}.jpg'
    metadata_fname = 'face_metadata.json'
    model = FaceModel(size=size,
                      device=device,
                      margin=margin,
                      min_face_size=50,
                      keep_all=True,
                      post_process=False,
                      select_largest=False,
                      chunk_size=150)
    cudnn.benchmark = True

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(dataset),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    dataloader = iter(dataloader)
    with torch.no_grad():
        end = time.time()
        for i in tqdm(range(len(dataloader)), total=len(dataloader)):
            with contextlib.suppress(RuntimeWarning):
                filename, x = next(dataloader)
                print(f'Extracting faces from: {filename}')
                bs, nc, d, h, w = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # [bs, d, nc, h, w]
                x = x.view(-1, *x.shape[2:])  # [bs * d, nc, h, w]

                save_dir = os.path.join(FACE_DIR, filename[0])
                os.makedirs(save_dir, exist_ok=True)
                save_paths = [os.path.join(save_dir, '{:06d}.jpg'.format(idx)) for idx in range(d)]
                faces_out = []
                torch.cuda.empty_cache()
                for xx, ss in zip(chunk(x, CHUNK_SIZE), chunk(save_paths, CHUNK_SIZE)):
                    xx = xx.to(device)
                    xx.mul_(255.)
                    out = model.model(xx, smooth=True)
                    if not out:
                        continue
                    out = torch.stack(out).cpu()
                    faces_out.append(out)
                min_face = min([f.shape[1] for f in faces_out])
                faces_out = torch.cat([f[:, :min_face] for f in faces_out])
                face_images = {i: [Image.fromarray(ff.permute(1, 2, 0).numpy().astype(np.uint8)) for ff in f]
                               for i, f in enumerate(faces_out.permute(1, 0, 2, 3, 4))}
                del x
                del xx
                torch.cuda.empty_cache()
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
                        list(pool.imap(save_image, zip(faces, names)))
                        pool.close()
                        pool.join()
                    video_path = os.path.join(save_dir, fdir_tmpl.format(face_num))
                    pretorched.data.utils.frames_to_video(f'{video_path}/*.jpg', video_path + '.mp4')
                    mm_out_dir = run_motion_mag(video=video_path, output=video_path + '_mm')
                    if REMOVE_FRAMES:
                        os.system(f'rm -rf {video_path}')
                        os.system(f'rm -rf {mm_out_dir}')
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)


if __name__ == '__main__':
    main()
