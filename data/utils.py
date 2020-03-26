import contextlib
import functools
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import ffmpeg
import numpy as np
import torch
from tqdm import tqdm

from pretorched.data.utils import frames_to_video


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

    print(f'Finished Frames:')
    for name, c in frame_counter.items():
        if c == 0:
            print(f'\t{name}: {c}')

    for name, fd in face_counter.items():
        print(f'Missing {name}: ({len(missing_faces[name])})')
        for n, c in fd.items():
            print(f'\t{name}:{n} Missing Face Frames: {c}')
        print(f'Finished {name}:')
        for i in range(50):
            n = f'dfdc_train_part_{i}'
            if fd[n] == 0:
                print(f'\t{name}:{n}')


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


class FrameToVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except FileNotFoundError:
            record = self.dataset.record_set[index]
            video_dir = os.path.join(self.dataset.root, record.path)

            for fn in record.face_names:
                video_path = os.path.join(video_dir, fn)
                print(video_path)
                frames_to_video(f'{video_path}/*.jpg', video_path + '.mp4')
            return self[index]

    def __len__(self):
        return len(self.dataset)


def smooth(x, amount=0.2, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    data_len = len(x)
    window_len = max(1, int(amount * (data_len // 2)))

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # return y[(window_len // 2):-(window_len // 2)]
    return y[(window_len // 2 - 1):-(window_len // 2)][:data_len]


def generate_test_targets(data_root, video_dir='videos', filename='metadata.json', outfilename='test_targets.json'):
    label_mapping = {'FAKE': 1, 'REAL': 0}
    video_root = os.path.join(data_root, video_dir)

    for root, dirs, files in os.walk(video_root):
        for d in dirs:
            print(f'Processing: {d}')
            metadata = {}
            fname = os.path.join(root, d, filename)
            out_fname = os.path.join(root, d, outfilename)
            with open(fname, 'r') as f:
                data = json.load(f)
            for name, info in data.items():
                metadata[name] = label_mapping[info['label']]
            with open(out_fname, 'w') as f:
                json.dump(metadata, f)
