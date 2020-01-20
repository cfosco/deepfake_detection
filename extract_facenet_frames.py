import json
import os
import sys
import time
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models import FaceModel
from pretorched.runners.utils import AverageMeter, ProgressMeter
from pretorched.utils import chunk

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
        self.transform = transform

    def __getitem__(self, index):
        name = self.videos_filenames[index]
        video_filename = os.path.join(self.root, name)
        frames = read_frames(video_filename, step=self.step)
        frames = torch.stack(list(map(TF.to_tensor, frames))).transpose(0, 1)
        if self.transform is not None:
            frames = self.transform(frames)
        return name, frames

    def __len__(self):
        return len(self.videos_filenames)


def save_image(args):
    image, filename = args
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        image.save(filename, quality=95)
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
        for i, (filename, x) in tqdm(enumerate(dataloader), total=len(dataloader)):
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
                    # list(tqdm(pool.imap(save_image, zip(faces, names)), total=num_images))
                    list(pool.imap(save_image, zip(faces, names)))
                    pool.close()
                    pool.join()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)


if __name__ == '__main__':
    main()
