import contextlib
import functools
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import ffmpeg
import numpy as np
from tqdm import tqdm


def videos_to_frames(video_root, frame_root, num_workers=32):
    """videos_to_frames."""
    videos = []
    for r, d, f in os.walk(video_root):
        for file in f:
            if '.mp4' in file:
                videos.append(os.path.join(r, file))

    func = functools.partial(process_video, video_root=video_root, frame_root=frame_root)
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(func, videos), total=len(videos)))
        pool.close()
        pool.join()


def get_framedir_name(video, num_pdirs=1):
    name = '/'.join(video.rstrip('/').split('/')[-(num_pdirs + 1):])
    return name


def extract_frames(video, video_root='', frame_root='', tmpl='%06d.jpg', fps=30, qscale=2, num_pdirs=1):
    name = '/'.join(video.rstrip('/').split('/')[-(num_pdirs + 1):])
    out_filename = os.path.join(frame_root, name, tmpl)
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    (
        ffmpeg
        .input(video)
        .filter('fps', fps=fps)
        .output(out_filename, **{'qscale:v': qscale})
        .global_args('-loglevel', 'error')
        .run()
    )


def _extract_frames(in_filename, out_filename, tmpl='%06d.jpg', fps=30, qscale=8):
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    (
        ffmpeg
        .input(in_filename)
        .filter('fps', fps=fps)
        .output(out_filename, **{'qscale:v': qscale})
        .global_args('-loglevel', 'error')
        .run()
    )


def process_video(video, video_root='', frame_root='', tmpl='%06d.jpg', fps=30, num_pdirs=1):
    name = get_framedir_name(video, num_pdirs=num_pdirs)
    frame_dir = os.path.join(frame_root, name)
    if os.path.exists(frame_dir):
        if os.listdir(frame_dir):
            return
    out_filename = os.path.join(frame_dir, tmpl)
    _extract_frames(video, out_filename, tmpl=tmpl, fps=fps)


def generate_metadata(data_root, video_dir='videos', frames_dir='frames',
                      filename='metadata.json', missing_filename='missing_data.json',
                      faces_dirs=['face_frames', 'facenet_frames'],
                      face_metadata_fnames=['face_metadata.json', 'face_metadata.json'],
                      with_coords=False,
                      ):
    metadata = {}
    video_root = os.path.join(data_root, video_dir)
    frame_root = os.path.join(data_root, frames_dir)

    missing_frames = []
    missing_faces = defaultdict(list)

    for root, dirs, files in os.walk(video_root):
        for d in dirs:
            print(f'Processing: {d}')
            fname = os.path.join(root, d, filename)
            with open(fname, 'r') as f:
                data = json.load(f)
                for name, info in data.items():

                    # Frame metadata
                    frame_dir = os.path.join(frame_root, d, name)
                    data[name]['filename'] = name
                    data[name]['path'] = os.path.join(d, name)
                    try:
                        num_frames = len(os.listdir(frame_dir))
                    except Exception:
                        missing_frames.append((d, name))
                    else:
                        data[name]['num_frames'] = num_frames

                    # Face frame metadata
                    for face_dir, face_metadata_fname in zip(faces_dirs, face_metadata_fnames):
                        fd = os.path.join(data_root, face_dir, d, name)
                        try:
                            with open(os.path.join(fd, face_metadata_fname)) as f:
                                fdata = json.load(f)
                            if not with_coords:
                                with contextlib.suppress(KeyError):
                                    fdata.pop('face_coords')
                            data[name][face_dir] = fdata
                        except Exception:
                            missing_faces[face_dir].append((d, name))

                metadata[d] = data

    filename = 'coords_' + filename if with_coords else filename
    with open(os.path.join(data_root, filename), 'w') as f:
        json.dump(metadata, f)

    missing_data = {
        'frames': missing_frames,
        'faces': missing_faces,
    }
    with open(os.path.join(data_root, missing_filename), 'w') as f:
        json.dump(missing_data, f, indent=4)

    frame_counter = defaultdict(int)
    face_counter = {k: defaultdict(int) for k in faces_dirs}

    for x in missing_frames:
        frame_counter[x[0]] += 1

    for name, fd in missing_faces.items():
        for x in fd:
            face_counter[name][x[0]] += 1

    print(f'Missing Frames: ({len(missing_frames)})')
    for name, c in frame_counter.items():
        print(f'\t{name}: {c}')

    for name, fd in face_counter.items():
        print(f'Missing {name}: ({len(missing_faces[name])})')
        for n, c in fd.items():
            print(f'\t{name}:{n} Missing Face Frames: {c}')


def generate_test_metadata(data_root, test_list_file='test_videos.json', video_dir='videos',
                           frames_dir='frames', train_metadata_file='metadata.json',
                           test_metadata_file='test_metadata.json'):
    video_root = os.path.join(data_root, video_dir)
    frame_root = os.path.join(data_root, frames_dir)
    test_list_file = os.path.join(data_root, test_list_file)
    train_metadata_file = os.path.join(data_root, train_metadata_file)
    with open(test_list_file) as f:
        test_videos = json.load(f)

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

    with open(os.path.join(data_root, test_metadata_file), 'w') as f:
        json.dump(test_metadata, f, indent=4)

    print(f'Missing test videos ({len(missing)}): {missing}')


def verify_data(data_root, video_dir='videos', frame_dir='frames', face_dir='face_frames',
                missing_threshold=20):
    """Verify data."""
    video_root = os.path.join(data_root, video_dir)
    frame_root = os.path.join(data_root, frame_dir)
    face_root = os.path.join(data_root, face_dir)

    video_parts = set(d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d)))
    frame_parts = set(d for d in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, d)))
    face_parts = set(d for d in os.listdir(face_root) if os.path.isdir(os.path.join(face_root, d)))

    missing_frame_parts = video_parts.difference(frame_parts)
    missing_face_parts = video_parts.difference(face_parts)

    print(f'Missing frame parts: {missing_frame_parts}')
    print(f'Missing face parts: {missing_face_parts}')

    video_counts = defaultdict(int)
    for part in video_parts:
        if part in frame_parts.union(face_parts):
            video_counts[part] = len(os.listdir(os.path.join(video_root, part)))

    frame_counts = defaultdict(int)
    for part in frame_parts:
        frame_counts[part] = len(os.listdir(os.path.join(frame_root, part)))

    face_counts = defaultdict(int)
    for part in face_parts:
        face_counts[part] = len(os.listdir(os.path.join(face_root, part)))

    missing = defaultdict(list)
    for part in video_parts:
        nv = video_counts[part]
        if nv - frame_counts[part] > missing_threshold:
            missing['frame'].append(part)
        if nv - face_counts[part] > missing_threshold:
            missing['face'].append(part)
        print(
            f'Part: {part}\n'
            f'\t Videos: {video_counts[part]}\n'
            f'\t Frames: {frame_counts[part]}\n'
            f'\t Faces: {face_counts[part]}\n'
        )
    print(missing)


def save_image(args):
    image, filename = args
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)


def smooth_data(data, amount=1.0):
    if not amount > 0.0:
        return data
    data_len = len(data)
    ksize = int(amount * (data_len // 2))
    kernel = np.ones(ksize) / ksize
    return np.convolve(data, kernel, mode='same')
