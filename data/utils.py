import functools
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from PIL import Image

import cv2
import dlib
import ffmpeg
from tqdm import tqdm

import face_recognition


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

    with open(os.path.join(video_root, 'test_metadata.json'), 'w') as f:
        json.dump(test_metadata, f, indent=4)
    print(len(missing))
    print(missing)


def verify_frames(video_dir, frame_root):
    """Verify frames."""
    videos = set(os.listdir(video_dir))
    frame_dirs = set(os.listdir(frame_root))
    missing = videos.difference(frame_dirs)
    for m in missing:
        print(missing)


def extract_face_frame(image_file, v_margin=30, h_margin=15):
    frame = face_recognition.load_image_file(image_file)
    face_location = face_recognition.face_locations(frame)[0]
    top, right, bottom, left = face_location
    face_image = frame[top - v_margin:bottom + v_margin,
                       left - h_margin:right + h_margin]
    return face_image


def crop_face_location(frame, face_location, v_margin, h_margin, imsize):
    top, right, bottom, left = face_location
    mtop = max(top - v_margin, 0)
    mbottom = min(bottom + v_margin, frame.shape[0])
    mleft = max(left - h_margin, 0)
    mright = min(right + h_margin, frame.shape[1])
    face_image = frame[mtop:mbottom, mleft:mright]
    return Image.fromarray(face_image).resize((imsize, imsize))


def extract_faces(video, v_margin=100, h_margin=100, batch_size=32, fps=30, imsize=360):

    def process_batch(frames, imsize):
        face_batch = []
        batched_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

        for frameno, (frame, face_locations) in enumerate(zip(frames, batched_face_locations)):
            num_faces = len(face_locations)

            for face_location in face_locations:
                face_image = crop_face_location(frame, face_location, v_margin, h_margin, imsize)
                face_batch.append(face_image)
        return face_batch

    # Open video file
    video_capture = cv2.VideoCapture(video)
    video_capture.set(cv2.CAP_PROP_FPS, fps)
    faces = []
    frames = []
    frame_count = 0

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)

        if len(frames) == batch_size:
            # faces.extend(process_batch(frames, frame_count, batch_size))
            faces.extend(process_batch(frames, imsize))

            # Clear the frames array to start the next batch
            frames = []
    if frames:
        faces.extend(process_batch(frames, imsize))
    return faces


def save_image(args):
    image, filename = args
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)


def process_faces(video, video_root='', faces_root='', tmpl='{:06d}.jpg', fps=30, num_pdirs=1):
    name = get_framedir_name(video, num_pdirs=num_pdirs)
    frame_dir = os.path.join(faces_root, name)
    if os.path.exists(frame_dir):
        if os.listdir(frame_dir):
            print(f'Skipping {frame_dir}')
            return
    out_filename = os.path.join(frame_dir, tmpl)
    faces = extract_faces(video, v_margin=100, h_margin=100, batch_size=64, fps=fps, device_id=0, imsize=360)
    num_images = len(faces)
    with ThreadPool(num_images) as pool:
        names = (out_filename.format(i) for i in range(1, num_images + 1))
        list(tqdm(pool.imap(save_image, zip(faces, names)), total=num_images))
        pool.close()
        pool.join()


def videos_to_faces(video_root, faces_root):
    """videos_to_faces."""

    videos = []
    for r, d, f in os.walk(video_root):
        for file in f:
            if '.mp4' in file:
                videos.append(os.path.join(r, file))

    func = functools.partial(process_faces, video_root=video_root, faces_root=faces_root)
    for video in videos:
        func(video)


def extract_multi_faces(video, v_margin=100, h_margin=100, batch_size=64, fps=30, device_id=0, imsize=360):

    # dlib.cuda.set_device(device_id)

    def process_batch(frames, imsize):
        face_batch = []
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            for frame, face_location in zip(frames, face_locations):
                # Print the location of each face in this frame
                top, right, bottom, left = face_location
                mtop = max(top - v_margin, 0)
                mbottom = min(bottom + v_margin, frame.shape[0])
                mleft = max(left - h_margin, 0)
                mright = min(right + h_margin, frame.shape[1])
                # face_image = frame[top - v_margin:bottom + v_margin, left - h_margin:right + h_margin]
                face_image = frame[mtop:mbottom, mleft:mright]
                face_batch.append(Image.fromarray(face_image).resize((imsize, imsize)))
        return face_batch

    # Open video file
    video_capture = cv2.VideoCapture(video)
    video_capture.set(cv2.CAP_PROP_FPS, fps)
    faces = []
    frames = []
    frame_count = 0

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)

        if len(frames) == batch_size:
            # faces.extend(process_batch(frames, frame_count, batch_size))
            faces.extend(process_batch(frames, imsize))

            # Clear the frames array to start the next batch
            frames = []
    if frames:
        faces.extend(process_batch(frames, imsize))
    return faces
