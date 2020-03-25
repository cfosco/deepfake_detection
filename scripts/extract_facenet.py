#!/usr/bin/env python
import argparse
import contextlib
import json
import os
import time
from multiprocessing import Process
from multiprocessing.pool import Pool, ThreadPool

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

import pretorched
from data import VideoFolder, VideoZipFile, video_collate_fn
from models import FaceModel, deepmmag
from pretorched.runners.utils import AverageMeter, ProgressMeter
from pretorched.utils import str2bool

DEEPFAKE_DATA_ROOT = os.path.join(os.environ['DATA_ROOT'], 'DeepfakeDetection')


def parse_args():
    parser = argparse.ArgumentParser(description='Face Extraction')
    parser.add_argument('--part', type=str, default='dfdc_train_part_0')
    parser.add_argument('--magnify_motion', default=False, type=str2bool)
    parser.add_argument('--overwrite', default=False, type=str2bool)
    parser.add_argument('--remove_frames', default=True, type=str2bool)
    parser.add_argument('--use_zip', default=True, type=str2bool)
    parser.add_argument('--video_rootdir', default='videos', type=str)
    parser.add_argument('--face_rootdir', default='facenet_videos', type=str)
    args = parser.parse_args()
    args.video_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.video_rootdir, args.part)
    args.face_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.face_rootdir, args.part)
    return args


def main(video_dir, face_dir, size=360, margin=100, fdir_tmpl='face_{}', tmpl='{:06d}.jpg',
         metadata_fname='face_metadata.json', step=1, batch_size=1, chunk_size=300, num_workers=2,
         overwrite=False, remove_frames=True, magnify_motion=False, use_zip=True, **kwargs):

    os.makedirs(face_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    if use_zip:
        video_dir = video_dir.rstrip('.zip') + '.zip'
        dataset = VideoZipFile(video_dir, step=step)
    else:
        dataset = VideoFolder(video_dir, step=step)

    print(f'Processing: {video_dir}')
    if not overwrite:
        dataset.videos_filenames = filter_filenames(dataset.videos_filenames, face_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=False, drop_last=False,
                            collate_fn=video_collate_fn)

    model = FaceModel(size=size,
                      device=device,
                      margin=margin,
                      min_face_size=50,
                      keep_all=True,
                      post_process=False,
                      select_largest=False,
                      chunk_size=chunk_size)

    run_motion_mag = deepmmag.get_motion_mag() if magnify_motion else None

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(dataset), [batch_time], prefix='Facenet Extraction and MM: ')

    # switch to evaluate mode
    model.eval()
    dataloader = iter(dataloader)
    with torch.no_grad():
        end = time.time()
        for i in tqdm(range(len(dataloader)), total=len(dataloader)):
            with contextlib.suppress(RuntimeWarning):

                filenames, x, _ = next(dataloader)
                print(f'Extracting faces from: {filenames}')

                face_images = model.get_faces(x)
                torch.cuda.empty_cache()

                for filename, face_images in zip(filenames, face_images):
                    save_dir = os.path.join(face_dir, filename)
                    save_face_data(save_dir, face_images, run_motion_mag,
                                   size=size, margin=margin, fdir_tmpl=fdir_tmpl,
                                   tmpl=tmpl, metadata_fname=metadata_fname,
                                   remove_frames=remove_frames)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)


def save_face_data(save_dir, face_images, run_motion_mag=None, size=360, margin=100, fdir_tmpl='face_{}',
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
        frames_to_video(video_path)
        # motion_magnification(video_path, remove_frames)
        if run_motion_mag is not None:
            mm_out_dir = run_motion_mag(video=video_path, output=video_path + '_mm')
        if remove_frames:
            os.system(f'rm -rf {video_path}')
            if run_motion_mag is not None:
                os.system(f'rm -rf {mm_out_dir}')


def frames_to_video(video_path):
    p = Process(target=pretorched.data.utils.frames_to_video,
                args=(f'{video_path}/*.jpg', video_path + '.mp4'),
                kwargs={'vcodec': 'mpeg4'})
    p.start()
    p.join()


def motion_magnification(video_path, remove_frames):
    p = Process(target=deepmmag.motion_magnify,
                kwargs={'video': video_path,
                        'output': video_path + '_mm',
                        'remove_frames': remove_frames})
    p.start()
    p.join()


def _motion_magnification(video_path, remove_frames):
    with Pool(2) as pool:
        res = pool.apply_async(deepmmag.motion_magnify,
                               kwds={'video': video_path,
                                     'output': video_path + '_mm',
                                     'remove_frames': remove_frames})
        mm_out_dir = res.get()
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
        frame_dir = os.path.join(face_dir, os.path.basename(f))
        if os.path.exists(frame_dir):
            if os.listdir(frame_dir):
                print(f'Skipping {frame_dir}')
                continue
        filtered_filename.append(f)
    return filtered_filename


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
