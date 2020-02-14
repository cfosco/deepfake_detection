import functools
import json
import os
from collections import defaultdict
from contextlib import suppress
from multiprocessing.pool import ThreadPool, Pool

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import filelock

# import face_recognition


def videos_to_faces(video_root, faces_root, batch_size=32, parts=[]):
    """videos_to_faces."""

    videos = []
    if not parts:
        parts = list(range(50))

    for r, d, f in os.walk(video_root):
        if any([f'dfdc_train_part_{p}' in r for p in parts]):
            for file in f:
                if '.mp4' in file:
                    videos.append(os.path.join(r, file))

    func = functools.partial(process_faces, video_root=video_root, faces_root=faces_root,
                             batch_size=batch_size)
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
            if not face_images:
                return
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
    if sum(missing) == 0 or all(missing):
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


def extract_faces_with_encodings(video, v_margin=100, h_margin=100, batch_size=32, fps=30, imsize=360):

    frames = list(read_frames(video, fps=fps))

    known_faces = []
    interp_inds = []
    face_images = defaultdict(list)
    face_coords = defaultdict(list)
    batched_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0,
                                                                   batch_size=batch_size)

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
            interped_fls = interp_face_locations(fls)
            face_coords[face_num] = interped_fls.tolist()

            for idx in interp_inds:
                interped_coords = interped_fls[idx]
                face_image = crop_face_location(frames[idx], interped_coords, v_margin, h_margin, imsize)

                face_images[face_num][idx] = face_image

    return face_images, face_coords


def extract_faces_batched(video, v_margin=100, h_margin=100, batch_size=32, fps=30, imsize=360):

    def process_multi_batch(frames, known_faces, face_coords, face_images, imsize):

        interp_inds = []
        batch_size = len(frames)
        batched_face_locations = face_recognition.batch_face_locations(
            frames, number_of_times_to_upsample=0, batch_size=32)

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
            known_faces, face_coords, face_images = process_multi_batch(
                frames, known_faces, face_coords, face_images, imsize)

            # Clear the frames array to start the next batch
            frames = []
    if frames:
        known_faces, face_coords, face_images = process_multi_batch(
            frames, known_faces, face_coords, face_images, imsize)
    return face_images, face_coords


def enhance_image(image, brightness=2.0, color=1, contrast=1, sharpness=1):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if brightness > 0:
        enh_bri = ImageEnhance.Brightness(image)
        image = enh_bri.enhance(brightness)
    if color > 0:
        enh_col = ImageEnhance.Color(image)
        image = enh_col.enhance(color)
    if contrast > 0:
        enh_con = ImageEnhance.Contrast(image)
        image = enh_con.enhance(contrast)
    if sharpness > 0:
        enh_sha = ImageEnhance.Sharpness(image)
        image = enh_sha.enhance(sharpness)
    return np.array(image)


def enhance_frames(frames):
    return [enhance_image(f) for f in frames]


def extract_faces(video, v_margin=100, h_margin=100, batch_size=32, fps=30, imsize=360, step=1):

    def get_faces(frames, batch_size, attempt=0, retries=1):
        if attempt > retries:
            # raise ValueError(f'Could not find faces in video after {retries} retries!')
            print(f'Could not find faces in video after {retries} retries!')
            return (None, None)
        faces = face_recognition.batch_face_locations(frames,
                                                      number_of_times_to_upsample=0,
                                                      batch_size=batch_size)
        if not any(faces):
            print('Could not find any faces... trying again with "enhanced" frames')
            frames = enhance_frames(frames)
            return get_faces(frames, batch_size, attempt + 1, retries)
        return faces, frames

    frames = list(read_frames(video, fps=fps, step=step))

    known_coords = []
    interp_inds = []
    face_images = defaultdict(list)
    face_coords = defaultdict(list)

    batched_face_locations, frames = get_faces(frames, batch_size)
    if batched_face_locations is None:
        print(f'Could not find face frames for video: video')
        return (None, None)

    for frameno, (frame, face_locations) in enumerate(zip(frames, batched_face_locations)):
        num_faces = len(face_locations)

        if num_faces < 1:
            face_coords[0].append(None)
            face_images[0].append(None)
            interp_inds.append(frameno)
        else:
            if not known_coords:
                known_coords = face_locations

            added = []
            for i, face_location in enumerate(face_locations):

                diff = np.abs(np.array(face_location) - np.array(known_coords)).sum(1)
                face_idx = int(np.argmin(diff))

                if face_idx not in added:
                    face_image = crop_face_location(frame, face_location, v_margin, h_margin, imsize)

                    known_coords[face_idx] = face_location
                    face_images[face_idx].append(face_image)
                    face_coords[face_idx].append(face_location)
                    added.append(face_idx)

    if interp_inds:

        for face_num, fls in face_coords.items():
            if None not in fls:
                break
            interped_fls = interp_face_locations(fls)
            face_coords[face_num] = interped_fls.tolist()

            for idx in interp_inds:

                interped_coords = interped_fls[idx]
                face_image = crop_face_location(frames[idx], interped_coords, v_margin, h_margin, imsize)

                face_images[face_num][idx] = face_image

    return face_images, face_coords


def filter_real_metadata(metadata, req_face_data=True):
    real_metadata = defaultdict(dict)
    for part, part_data in metadata.items():
        for name, data in part_data.items():
            if data['label'] == 'REAL':
                if req_face_data:
                    if 'face_data' in data:
                        real_metadata[part][name] = data
                else:
                    real_metadata[part][name] = data
    return real_metadata


def get_encodings(frame, num_jitters=1, attempt=0, retries=1):
    if attempt > retries:
        # raise ValueError(f'Could not find faces in video after {retries} retries!')
        print(f'Could not find faces in video after {retries} retries!')
        return
    enc = face_recognition.face_encodings(frame, num_jitters=num_jitters)
    if not enc:
        print(f'Could not find any faces... trying again with {num_jitters} num_jitters')
        return get_encodings(frame, num_jitters + 1, attempt + 1, retries)
    return enc


def _old_match_actors(known_rec, metadata, face_num=0, data_root='.', num_frames=10, tolerance=0.6):
    face_data = known_rec['face_data']
    faces_dir = os.path.join(data_root, 'face_frames')
    known_path = os.path.join(faces_dir, known_rec['path'], face_data['face_names'][face_num])
    known_images = [face_recognition.load_image_file(os.path.join(known_path, '{:06d}.jpg'.format(i)))
                    for i in np.linspace(1, face_data['num_frames'][face_num], num_frames, dtype=int)]

    all_dists = []
    known_encodings = []
    for im in known_images:
        try:
            known_encodings.append(face_recognition.face_encodings(im, num_jitters=4)[0])
        except IndexError:
            continue
    # known_encodings = [face_recognition.face_encodings(im)[0] for im in known_images]

    matches = []
    not_matches = []
    for name, rec in metadata.items():
        face_data = rec['face_data']
        for face_num, face_name in enumerate(face_data['face_names']):
            frame_inds = np.linspace(1, face_data['num_frames'][face_num], num_frames, dtype=int)
            tmpl = os.path.join(faces_dir, rec['path'], face_data['full_tmpl'])
            unknown_images = [face_recognition.load_image_file(tmpl.format(face_num, ind))
                              for ind in frame_inds]
            unknown_encodings = []
            for unknown_im in unknown_images:
                try:
                    unknown_encodings.append(face_recognition.face_encodings(unknown_im, num_jitters=4)[0])
                except IndexError:
                    continue

            # results = [any(face_recognition.compare_faces(unknown_encodings, ke, tolerance=0.5))
                    #    for ke in known_encodings]

            dist = np.mean([np.mean(face_recognition.face_distance(known_encodings, uke))
                            for uke in unknown_encodings])
            print(len(known_encodings), len(unknown_encodings), dist)
            if dist <= tolerance:
                # if any(results):
                # matches.append(os.path.join(rec['path'], face_name))
                matches.append(unknown_images[0])
            else:
                # not_matches.append(os.path.join(rec['path'], face_name))
                not_matches.append(unknown_images[0])
    return matches, not_matches


def interp_nans(arr):
    arr = np.array(arr).ravel()
    n = len(arr)
    missing = np.isnan(arr)
    if sum(missing) == 0 or all(missing):
        return arr
    inds = np.arange(n)[missing]
    x = np.arange(n)[~missing]
    out = np.interp(inds, x, arr[x])
    arr[inds] = out
    return arr


def __match_actors(known_rec, metadata, face_num=0, data_root='.', num_frames=10, tolerance=0.6,
                   num_workers=24):
    face_data = known_rec['face_data']
    faces_root = os.path.join(data_root, 'face_frames')

    name = os.path.join(known_rec['filename'], face_data['face_names'][face_num])
    person = Person(name=name, faces_root=faces_root)
    person.add_record(known_rec)

    matches = []
    people = []
    for name, rec in metadata.items():
        face_data = rec['face_data']
        for face_num, face_name in enumerate(face_data['face_names']):
            pname = os.path.join(rec['filename'], face_name)
            p = Person(name=pname, faces_root=faces_root)
            p.add_record(rec, face_num)
            people.append(p)

    out = []
    dists = []
    psons = []
    with Pool(num_workers) as p:
        for dist, pson in p.imap_unordered(person.compare, people):
            print(f'Person: {pson.name} dist: {dist}')
            dists.append(dist)
            psons.append(pson)
        p.close()
        p.join()
    dists = interp_nans(dists)
    matches = dists <= tolerance
    out = list(zip(matches, psons, dists))
    return out


class Person:

    def __init__(self, name='', faces_root='face_frames', data=None):
        super().__init__()
        self.name = name
        self.faces_root = faces_root
        self.data = defaultdict(list) if data is None else data

        self.encodings = None

    def add_record(self, record, face_num=0):
        face_data = record['face_data']
        face_name = face_data['face_names'][face_num]
        num_frames = face_data['num_frames'][face_num]
        path = os.path.join(record['path'], face_name)
        self.data['records'].append(record)
        self.data['face_data'].append(face_data)
        self.data['paths'].append(path)
        self.data['face_num'].append(face_num)
        self.data['num_frames'].append(num_frames)

    def get_face_images(self, index=-1, num_frames=10):
        path = os.path.join(self.faces_root, self.data['paths'][index])
        frame_count = self.data['num_frames'][index]
        images = [face_recognition.load_image_file(os.path.join(path, '{:06d}.jpg'.format(i)))
                  for i in np.linspace(1, frame_count, num_frames, dtype=int)]
        return images

    def get_face_encodings(self, index=-1, num_frames=10):
        images = self.get_face_images(index, num_frames)
        encodings = []
        for im in images:
            try:
                encodings.append(face_recognition.face_encodings(im, num_jitters=1)[0])
            except IndexError:
                continue
        return encodings

    def compare(self, person, tolerance=0.6):
        if self.encodings is None:
            self.encodings = self.get_face_encodings(-1)

        dist = np.mean([
            np.mean(face_recognition.face_distance(self.encodings, uke))
            for uke in person.get_face_encodings(-1)])
        return (dist, person)

    @property
    def records(self):
        return self.data['records']

    @property
    def paths(self):
        return self.data['paths']

    @property
    def face_num(self):
        return self.data['face_num']

    def merge(self, people):
        for p in people:
            for rec, face_num in zip(p.records, p.face_num):
                self.add_record(rec, face_num)

    def __repr__(self):
        out = f'Person: {self.name} with {len(self.records)} records'
        return out

    all_dists = []
    known_encodings = []


class Matcher:

    def __init__(self, name='', data_root='', face_framedir='face_frames', num_workers=100):
        self.name = name
        self.data_root = data_root
        self.faces_root = os.path.join(data_root, face_framedir)
        self.people = []
        self.num_workers = num_workers

    def match_all(self, metadata, part='dfdc_train_part_0'):
        pqueue = self.get_people(metadata[part])
        while pqueue:
            p = pqueue.pop()
            matches, pqueue = self.match(p, pqueue, num_workers=self.num_workers)
            p.merge(matches)
            self.people.append(p)

        return self.people

    def get_people(self, metadata):
        people = []
        for name, rec in metadata.items():
            face_data = rec['face_data']
            for face_num, face_name in enumerate(face_data['face_names']):
                pname = os.path.join(rec['filename'], face_name)
                p = Person(name=pname, faces_root=self.faces_root)
                p.add_record(rec, face_num)
                people.append(p)
        return people

    def match(self, person, people, num_frames=10, tolerance=0.6, num_workers=12):
        dists = []
        psons = []
        with ThreadPool(num_workers) as p:
            for dist, pson in p.imap_unordered(person.compare, people):
                print(f'Person: {pson.name} dist: {dist}')
                dists.append(dist)
                psons.append(pson)
            p.close()
            p.join()
        dists = interp_nans(dists)
        match_inds = dists <= tolerance
        matches = []
        rest = []
        for ps, m in zip(psons, match_inds):
            if m:
                matches.append(ps)
            else:
                rest.append(ps)
        return matches, rest
        # m = [ps for ps, m in zip(psons, matches) ]

        # return matches, psons, dists
        # out = list(zip(matches, psons, dists))
        # return out

    def _match_actor(self, known_rec, metadata, face_num=0, data_root='.', num_frames=10, tolerance=0.6,
                     num_workers=24):
        face_data = known_rec['face_data']
        faces_root = os.path.join(data_root, 'face_frames')

        name = os.path.join(known_rec['filename'], face_data['face_names'][face_num])
        person = Person(name=name, faces_root=faces_root)
        person.add_record(known_rec)

        matches = []
        people = []
        for name, rec in metadata.items():
            face_data = rec['face_data']
            for face_num, face_name in enumerate(face_data['face_names']):
                pname = os.path.join(rec['filename'], face_name)
                p = Person(name=pname, faces_root=faces_root)
                p.add_record(rec, face_num)
                people.append(p)

        out = []
        dists = []
        psons = []
        with Pool(num_workers) as p:
            for dist, pson in p.imap_unordered(person.compare, people):
                print(f'Person: {pson.name} dist: {dist}')
                dists.append(dist)
                psons.append(pson)
            p.close()
            p.join()
        # dists = interp_nans(dists)
        matches = dists <= tolerance
        out = list(zip(matches, psons, dists))
        return out


def match_actors(data_root, metadata_file='metadata.json', people_metafile='people_metadata.json'):

    with open(os.path.join(data_root, metadata_file)) as f:
        metadata = json.load(f)

    real_metadata = filter_real_metadata(metadata)

    match_data = {}
    for part in real_metadata:
        if part not in ['dfdc_train_part_3']:
            continue
        print(f'Matching people in part: {part}')
        matcher = Matcher(name=part, data_root=data_root)

        matched = matcher.match_all(real_metadata, part=part)
        match_data[part] = {p.name: p.data for p in matched}
        [print(m) for m in matched]

        with open(os.path.join(data_root, f'{part}_{people_metafile}'), 'w') as f:
            json.dump(match_data, f)
