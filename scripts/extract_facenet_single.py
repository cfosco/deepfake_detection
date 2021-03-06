#!/usr/bin/env python
import argparse
import contextlib
import json
import os
import sys
import time
from multiprocessing import Process
from multiprocessing.pool import Pool, ThreadPool

import ffmpeg
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

import pretorched
from data import VideoFolder, VideoZipFile, video_collate_fn
from models import FaceModel
from pretorched.runners.utils import AverageMeter, ProgressMeter
from pretorched.utils import str2bool

try:
    sys.path.extend(['.', '..'])
    from config import DEFAULT_VIDEO_CODEC
except ModuleNotFoundError:
    raise ModuleNotFoundError


def parse_args():
    parser = argparse.ArgumentParser(description='Face Extraction')
    parser.add_argument('--dataset', type=str, default='DeepfakeDetection')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--part', type=str, default='dfdc_train_part_0')
    parser.add_argument('--magnify_motion', default=False, type=str2bool)
    parser.add_argument('--overwrite', default=False, type=str2bool)
    parser.add_argument('--remove_frames', default=True, type=str2bool)
    parser.add_argument('--use_zip', default=False, type=str2bool)
    parser.add_argument('--video_rootdir', default='videos', type=str)
    parser.add_argument('--face_rootdir', default='facenet_videos', type=str)
    parser.add_argument('--chunk_size', default=150, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    args = parser.parse_args()
    DEEPFAKE_DATA_ROOT = os.path.join(os.environ['DATA_ROOT'], args.dataset)
    if args.dataset == 'DeepfakeDetection':
        args.video_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.video_rootdir, args.part)
        args.face_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.face_rootdir, args.part)
    elif args.dataset == 'FaceForensics':
        args.video_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.part, args.video_rootdir)
        args.face_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.part, args.face_rootdir)
    else:
        args.video_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.video_rootdir, args.part)
        args.face_dir = os.path.join(DEEPFAKE_DATA_ROOT, args.face_rootdir, args.part)
    return args


def main(video_dir, face_dir, filename=None, size=360, margin=100, fdir_tmpl='face_{}', tmpl='{:06d}.jpg',
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

    if filename is not None:
        dataset.videos_filenames = [filename]

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

    if magnify_motion:
        from models import deepmmag
        run_motion_mag = deepmmag.get_motion_mag()
    else:
        run_motion_mag = None

    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Time', ':6.3f')
    # progress = ProgressMeter(len(dataset), [batch_time, data_time], prefix='Facenet Extraction and MM: ')

    # switch to evaluate mode
    model.eval()
    dataloader = iter(dataloader)
    with torch.no_grad():
        end = time.time()
        for i in tqdm(range(len(dataloader)), total=len(dataloader)):
            with contextlib.suppress(RuntimeWarning):

                filenames, x, _ = next(dataloader)
                # for f in filenames:
                # if os.path.exists(os.path.join(face_dir, os.path.basename(f))):
                # print(f'Skipping {f}')
                # continue
                face_images = model.get_faces(x, to_pil=magnify_motion)

                for filename, face_images in zip(filenames, face_images):
                    save_dir = os.path.join(face_dir, os.path.basename(filename))
                    save_face_data(save_dir, face_images, run_motion_mag,
                                   size=size, margin=margin, fdir_tmpl=fdir_tmpl,
                                   tmpl=tmpl, metadata_fname=metadata_fname,
                                   remove_frames=remove_frames)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                end = time.time()

                # progress.display(i)


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
        video_path = os.path.join(save_dir, fdir_tmpl.format(face_num))
        if run_motion_mag is None:
            array_to_video(faces, video_path + '.mp4')
        else:
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


def vidwrite(images, filename, framerate=30, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(filename, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
        .global_args('-loglevel', 'error')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
            .astype(np.uint8)
            .tobytes()
        )
    process.stdin.close()
    process.wait()


def array_to_video(images, filename):
    p = Process(target=vidwrite,
                args=(images, filename),
                kwargs={'vcodec': DEFAULT_VIDEO_CODEC})
    p.start()


def frames_to_video(video_path):
    p = Process(target=pretorched.data.utils.frames_to_video,
                args=(f'{video_path}/*.jpg', video_path + '.mp4'),
                kwargs={'vcodec': DEFAULT_VIDEO_CODEC})
    p.start()
    p.join()


def motion_magnification(video_path, remove_frames):
    from models import deepmmag
    p = Process(target=deepmmag.motion_magnify,
                kwargs={'video': video_path,
                        'output': video_path + '_mm',
                        'remove_frames': remove_frames})
    p.start()
    p.join()


def _motion_magnification(video_path, remove_frames):
    from models import deepmmag
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


def filter_filenames(video_filenames, face_root):
    filtered_filename = []
    for f in video_filenames:
        face_dir = os.path.join(face_root, os.path.basename(f))
        if os.path.exists(face_dir):
            fm = os.path.join(face_dir, 'face_metadata.json')
            try:
                with open(fm) as jf:
                    metadata = json.load(jf)
                for face_name in metadata['face_names']:
                    if not os.path.exists(os.path.join(face_dir, face_name + '.mp4')):
                        raise FileNotFoundError
            except FileNotFoundError:
                pass
            else:
                print(f'Skipping {face_dir}')
                continue
        filtered_filename.append(f)
    return filtered_filename


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
