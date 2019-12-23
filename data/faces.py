import functools
import json
import os
from collections import defaultdict
from contextlib import suppress
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import filelock

import face_recognition


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


def save_image(args):
    image, filename = args
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        image.save(filename)
    except Exception:
        pass


def process_faces(video, video_root='', faces_root='', fdir_tmpl='face_{}', tmpl='{:06d}.jpg',
                  fps=30, num_pdirs=1, batch_size=64, margin=100, imsize=360,
                  metadata_fname='face_metadata.json'):
    name = get_framedir_name(video, num_pdirs=num_pdirs)
    with suppress(filelock.FileLockException):
        with filelock.FileLock(f'.{os.path.basename(name)}', timeout=0.5):
            print(f'Processing: {name}')
            frame_dir = os.path.join(faces_root, name)
            if os.path.exists(frame_dir):
                if os.listdir(frame_dir):
                    print(f'Skipping {frame_dir}')
                    return
            face_images, face_coords = extract_faces(video, v_margin=margin, h_margin=margin,
                                                     batch_size=batch_size, fps=fps, imsize=imsize)
            metadata = {
                'filename': os.path.basename(name),
                'num_faces': len(face_coords),
                'num_frames': [len(f) for f in face_images.values()],
                'face_coords': face_coords,
                'dir_tmpl': fdir_tmpl,
                'im_tmpl': tmpl,
                'full_tmpl': os.path.join(fdir_tmpl, tmpl),
                'face_names': [fdir_tmpl.format(k) for k in face_coords],
                'face_nums': list(face_coords.keys()),
                'margin': margin,
                'size': imsize,
                'fps': fps,
            }

            os.makedirs(frame_dir, exist_ok=True)
            with open(os.path.join(frame_dir, metadata_fname), 'w') as f:
                json.dump(metadata, f)

            for face_num, faces in face_images.items():
                num_images = len(faces)
                out_filename = os.path.join(frame_dir, fdir_tmpl.format(face_num), tmpl)
                with ThreadPool(num_images) as pool:
                    names = (out_filename.format(i) for i in range(1, num_images + 1))
                    list(tqdm(pool.imap(save_image, zip(faces, names)), total=num_images))
                    pool.close()
                    pool.join()


def get_face_match(known_faces, face_encoding, tolerance=0.50):
    match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=tolerance)
    try:
        face_idx = match.index(True)
        return face_idx
    except ValueError:
        return get_face_match(known_faces, face_encoding, tolerance + 0.01)


def interp_face_locations(coords):
    coords = np.array(coords).ravel()
    n = len(coords)
    missing = coords.__eq__(None).ravel()
    if sum(missing) == 0:
        return coords
    inds = np.arange(n)[missing]
    x = np.arange(n)[~missing]
    y = coords[x]
    interps = []
    for dim in range(4):
        yc = np.array([e[dim] for e in y])
        out = np.interp(inds, x, yc).astype(int)
        interps.append(out)
    interps = np.stack(interps, axis=-1)
    for i, ind in enumerate(inds):
        coords[ind] = tuple([int(x) for x in interps[i]])
    return coords


def extract_faces(video, v_margin=100, h_margin=100, batch_size=32, fps=30, imsize=360):

    def process_multi_batch(frames, known_faces, face_coords, face_images, imsize):

        interp_inds = []
        batch_size = len(frames)
        batched_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

        for frameno, (frame, face_locations) in enumerate(zip(frames, batched_face_locations)):
            num_faces = len(face_locations)

            if num_faces < 1:
                face_coords[0].append(None)
                face_images[0].append(None)
                interp_inds.append(frameno)
            else:

                face_encodings = face_recognition.face_encodings(frame, face_locations)

                if not known_faces:
                    known_faces = face_encodings

                added = []
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):

                    # See if the face is a match for the known face(s)
                    if face_encoding is not None:
                        face_idx = get_face_match(known_faces, face_encoding)
                    else:
                        face_idx = i
                    if face_idx not in added:
                        face_image = crop_face_location(frame, face_location, v_margin, h_margin, imsize)

                        known_faces[face_idx] = face_encoding
                        face_images[face_idx].append(face_image)
                        face_coords[face_idx].append(face_location)
                        added.append(face_idx)

        if interp_inds:

            for face_num, fls in face_coords.items():
                if None not in fls:
                    break
                frame_count = len(fls)
                interped_fls = interp_face_locations(fls)
                face_coords[face_num] = interped_fls.tolist()

                for batch_idx in interp_inds:
                    idx = (frame_count - batch_size) + batch_idx
                    interped_coords = interped_fls[idx]
                    face_image = crop_face_location(frames[batch_idx], interped_coords, v_margin, h_margin, imsize)

                    face_images[face_num][idx] = face_image

        return known_faces, face_coords, face_images

    # Open video file
    video_capture = cv2.VideoCapture(video)
    video_capture.set(cv2.CAP_PROP_FPS, fps)

    frames = []
    frame_count = 0

    known_faces = []
    face_images = defaultdict(list)
    face_coords = defaultdict(list)

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
            known_faces, face_coords, face_images = process_multi_batch(frames, known_faces, face_coords, face_images, imsize)

            # Clear the frames array to start the next batch
            frames = []
    if frames:
        known_faces, face_coords, face_images = process_multi_batch(frames, known_faces, face_coords, face_images, imsize)
    return face_images, face_coords


def extract_face_frame(image_file, model='hog', v_margin=30, h_margin=15):
    frame = face_recognition.load_image_file(image_file)
    face_location = face_recognition.face_locations(frame, model=model)[0]
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


def get_framedir_name(video, num_pdirs=1):
    name = '/'.join(video.rstrip('/').split('/')[-(num_pdirs + 1):])
    return name
