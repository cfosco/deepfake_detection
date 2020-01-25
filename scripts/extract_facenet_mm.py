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


try:
    PART = sys.argv[1]
except IndexError:
    PART = 'dfdc_train_part_0'
VIDEO_ROOT = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'videos')
FACE_ROOT = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection', 'facenet_smooth_frames')
VIDEO_DIR = os.path.join(VIDEO_ROOT, PART)
FACE_DIR = os.path.join(FACE_ROOT, PART)


def main(size=360, margin=100, fdir_tmpl='face_{}', tmpl='{:06d}.jpg', metadata_fname='face_metadata.json',
         step=1, chunk_size=100, num_workers=2, overwrite=False, remove_frames=True):

    os.makedirs(FACE_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = VideoFolder(VIDEO_DIR, step=step)
    if not overwrite:
        dataset.videos_filenames = filter_filenames(dataset.videos_filenames, FACE_DIR)

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=False, drop_last=False)

    model = FaceModel(size=size,
                      device=device,
                      margin=margin,
                      min_face_size=50,
                      keep_all=True,
                      post_process=False,
                      select_largest=False,
                      chunk_size=chunk_size)

    run_motion_mag = deepmmag.get_motion_mag()
    cudnn.benchmark = True

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(dataset), [batch_time], prefix='Facenet Extraction and MM: ')

    # switch to evaluate mode
    model.eval()
    dataloader = iter(dataloader)
    with torch.no_grad():
        end = time.time()
        for i in tqdm(range(len(dataloader)), total=len(dataloader)):
            with contextlib.suppress(RuntimeWarning):

                filenames, x = next(dataloader)
                print(f'Extracting faces from: {filenames}')

                faces_out = []
                torch.cuda.empty_cache()
                x = model.input_transform(x)
                out = model.model(x, smooth=True)
                if not out:
                    continue
                out = torch.stack(out).cpu()
                faces_out.append(out)

                min_face = min([f.shape[1] for f in faces_out])
                faces_out = torch.cat([f[:, :min_face] for f in faces_out])
                face_images = {i: [Image.fromarray(ff.permute(1, 2, 0).numpy().astype(np.uint8)) for ff in f]
                               for i, f in enumerate(faces_out.permute(1, 0, 2, 3, 4))}

                del x
                torch.cuda.empty_cache()

                for filename, face_images in zip(filenames, [face_images]):
                    save_dir = os.path.join(FACE_DIR, filename)
                    save_face_data(save_dir, face_images, run_motion_mag,
                                   size=size, margin=margin, fdir_tmpl=fdir_tmpl,
                                   tmpl=tmpl, metadata_fname=metadata_fname,
                                   remove_frames=remove_frames)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)


def save_face_data(save_dir, face_images, run_motion_mag, size=360, margin=100, fdir_tmpl='face_{}',
                   tmpl='{:06d}.jpg', metadata_fname='face_metadata.json', remove_frames=False):

    os.makedirs(save_dir, exist_ok=True)
    metadata = {
        'filename': os.path.basename(save_dir),
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
        names = (out_filename.format(i) for i in range(1, num_images + 1))
        save_images(faces, names, num_workers=num_images)

        video_path = os.path.join(save_dir, fdir_tmpl.format(face_num))
        pretorched.data.utils.frames_to_video(f'{video_path}/*.jpg', video_path + '.mp4')
        mm_out_dir = run_motion_mag(video=video_path, output=video_path + '_mm')
        if remove_frames:
            os.system(f'rm -rf {video_path}')
            os.system(f'rm -rf {mm_out_dir}')


def save_images(images, names, num_workers=None):
    num_workers = len(images) if num_workers is None else num_workers
    with ThreadPool(num_workers) as pool:
        list(pool.imap(save_image, zip(images, names)))
        pool.close()
        pool.join()


def save_image(args):
    image, filename = args
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        image.save(filename, quality=95)
    except Exception:
        pass


def filter_filenames(video_filenames, face_dir):
    filtered_filename = []
    for f in video_filenames:
        frame_dir = os.path.join(face_dir, f)
        if os.path.exists(frame_dir):
            if os.listdir(frame_dir):
                print(f'Skipping {frame_dir}')
                continue
        filtered_filename.append(f)
    return filtered_filename


if __name__ == '__main__':
    main()
