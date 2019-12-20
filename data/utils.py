import functools
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import ffmpeg


def videos_to_frames(video_root, frame_root, num_workers=32):
    """videos_to_frames."""

    videos = []
    for r, d, f in os.walk(video_root):
        for file in f:
            if '.mp4' in file:
                videos.append(os.path.join(r, file))

    func = functools.partial(process_video, video_root=video_root, frame_root=frame_root)
    pool = Pool(num_workers)
    pool.map(func, videos)


def get_framedir_name(video, num_pdirs=1):
    name = '/'.join(video.rstrip('/').split('/')[-(num_pdirs + 1):])
    return name


def extract_frames(video, video_root='', frame_root='', tmpl='%06d.jpg', fps=30, num_pdirs=1):
    name = '/'.join(video.rstrip('/').split('/')[-(num_pdirs + 1):])
    out_filename = os.path.join(frame_root, name, tmpl)
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    (
        ffmpeg
        .input(video)
        .filter('fps', fps=fps, round='up')
        .output(out_filename)
        .run()
    )


def _extract_frames(in_filename, out_filename, tmpl='%06d.jpg', fps=30):
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    (
        ffmpeg
        .input(in_filename)
        .filter('fps', fps=fps, round='up')
        .output(out_filename)
        .run()
    )


def process_video(video, video_root='', frame_root='', tmpl='%06d.jpg', fps=30, num_pdirs=1):
    name = get_framedir_name(video, num_pdirs=num_pdirs)
    frame_dir = os.path.join(frame_root, name)
    if os.path.exists(frame_dir):
        if os.listdir(frame_dir):
            print(f'Skipping {frame_dir}')
            return
    out_filename = os.path.join(frame_dir, tmpl)
    _extract_frames(video, out_filename, tmpl=tmpl, fps=fps)


def generate_metadata(video_root, frame_root, filename='metadata.json'):
    metadata = {}
    for root, dirs, files in os.walk(video_root):
        for d in dirs:
            fname = os.path.join(root, d, filename)
            with open(fname, 'r') as f:
                data = json.load(f)
                for name, info in data.items():
                    frame_dir = os.path.join(frame_root, d, name)
                    num_frames = len(os.listdir(frame_dir))
                    data[name]['filename'] = name
                    data[name]['num_frames'] = num_frames
                    data[name]['path'] = os.path.join(d, name)
                metadata[d] = data
    with open(os.path.join(video_root, filename), 'w') as f:
        # json.dump(metadata, f)
        json.dump(metadata, f, indent=4)


def generate_test_metadata(test_list_file, video_root, frame_root, train_metadata_file='metadata.json'):
    with open(test_list_file) as f:
        test_videos = [x.strip() for x in f]

    with open(train_metadata_file) as f:
        train_metadata = json.load(f)
    missing = []
    test_metadata = defaultdict(dict)
    for test_video in test_videos:
        for part, data in train_metadata.items():
            try:
                rec = data[test_video]
            except KeyError:
                pass
            else:
                test_metadata[part][test_video] = rec
                break
        else:
            missing.append(test_video)

    print(len(missing))
    print(missing)





def verify_frames(video_dir, frame_root):
    """Verify frames."""
    videos = set(os.listdir(video_dir))
    frame_dirs = set(os.listdir(frame_root))
    missing = videos.difference(frame_dirs)
    for m in missing:
        print(missing)
