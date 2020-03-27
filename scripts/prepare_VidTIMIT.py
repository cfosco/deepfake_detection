import contextlib
import os

from pretorched.data.utils import frames_to_video

VidTIMIT_ROOT = os.path.join(os.environ.get('DATA_ROOT'), 'VidTIMIT')


def frames2video(root_dir=None):

    def process_subjectdir(sdir, ext='.mp4'):
        video_dir = os.path.join(sdir, 'video')
        for frame_dir in os.listdir(video_dir):
            frame_dirpath = os.path.join(video_dir, frame_dir)
            pattern = os.path.join(frame_dirpath, '%03d.jpg')
            for f in os.listdir(frame_dirpath):
                fname = os.path.join(frame_dirpath, f)
                with contextlib.suppress(FileExistsError):
                    os.symlink(fname, fname + '.jpg')

            output_video = os.path.join(video_dir, frame_dir + ext)
            frames_to_video(pattern, output_video, pattern_type='glob_sequence')

    root_dir = VidTIMIT_ROOT if root_dir is None else root_dir
    subject_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for sdir in subject_dirs:
        print(f'Processing: {sdir}')
        process_subjectdir(os.path.join(root_dir, sdir))


