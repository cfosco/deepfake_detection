import cv2
import h5py
from joblib import Parallel, delayed
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter
import time

SIGMA=[6, 20, 20] # time, x, y
idx = 0


def load_video():
    cap = cv2.VideoCapture("../datafile_pilot/exp2_res/face_0.mp4")
    flag = True
    frames = []

    while flag:
        flag, fr = cap.read()
        if flag: frames.append(np.expand_dims(fr[:, :, ::-1], 0))
    return np.concatenate(frames, axis=0)


def heatmap_overlay(im, heatmap, colmap='hot'):
    cm_array = cm.get_cmap(colmap)
    im_array = np.asarray(im)
    heatmap_norm = (heatmap-np.min(heatmap))/float(np.max(heatmap)-np.min(heatmap))
    heatmap_hot = cm_array(heatmap_norm)
    res_final = im_array.copy()
    heatmap_rep = np.repeat(heatmap_norm[:, :, np.newaxis], 3, axis=2)
    res_final[...] = heatmap_hot[...,0:3]*255.0*heatmap_rep + im_array[...]*(1-heatmap_rep)
    return res_final


def generate_3Dheatmap(pos_arr, title_str=None, vis_hm=False):
    heatmap_3d = np.zeros((300, 360, 360), dtype=np.float)

    resc_factor = 360/540

    for r in pos_arr:
        f_idx = int(round(r[2] * 30))
        x = int(r[0]*resc_factor)
        if x < 0 or x >= 360:
            continue
        y = int(r[1]*resc_factor)
        if y < 0 or y >= 360:
            continue
        # heatmap_3d[f_idx][x, y] += 1
        heatmap_3d[f_idx][y, x] += 1

    st = time.time()
    heatmap_3d = gaussian_filter(heatmap_3d, SIGMA)
    print(time.time()-st)

    if vis_hm:
        if 'kmtqqzwtox' in title_str:
            frames = load_video()

    for i in range(300):
        f = heatmap_3d[i]
        if np.sum(f) == 0:
            continue
        f *= (1.0/np.sum(f))

        if vis_hm:
            if 'kmtqqzwtox' in title_str:
                if i >= len(frames):
                    break
                vis = heatmap_overlay(frames[i], heatmap_3d[i])
                plt.imshow(vis)
                plt.xticks([])
                plt.yticks([])
                plt.savefig('../datafile_pilot/exp2_res/frame_%d.jpeg'%i)
            else:
                plt.matshow(f)

            plt.title(title_str)
            plt.show(block=False)
            plt.pause(0.01)
            plt.close()

    if vis_hm:
        if 'kmtqqzwtox' in title_str:
            os.system("ffmpeg -r 30 -i ../datafile_pilot/exp2_res/frame_%01d.jpeg -vcodec mpeg4 -y ../datafile_pilot/exp2_res/movie.mp4")
        plt.close()
        plt.clf()

    return heatmap_3d


def main(parallel=True):
    res_dir = "../datafile_pilot/exp2_res/df_1level.json"
    with open(res_dir, 'r') as fp:
        data = json.load(fp)

    # print(len(data)) # 19, each a set from one turker
    # ['HITId', 'AssignmentId', 'WorkerId', 'AcceptTime', 'SubmitTime', 'TaskTime',
    # 'workerId', 'taskTime', 'feedback', 'results']
    # print(data[0].keys())

    taskTimes = []
    for d in data:
        taskTimes.append(float(d['taskTime'])/60.0)

    mt = np.mean(taskTimes)
    std_t = np.std(taskTimes)

    print(mt, std_t)

    n_catch = 0
    for t in taskTimes:
        if t <= mt - 1.0 * std_t:
            # print(t)
            # print("catch")
            n_catch += 1

    print("percentage of hits too short: ", n_catch/len(data))

    for d in data:
        fb = d['feedback']
        if fb is not None:
            print('feedback: ', fb)

    ressponses_dict = {}
    for d in data:
        print("data from worker: ", d['WorkerId'])
        res_dict = d['results']
        # print(type(res_dict)) # str
        res_dict = json.loads(res_dict)["outputs"]
        # print(len(res_dict)) # 10 videos in one level
        for k in res_dict.keys():
            url_k = list(res_dict[k].keys())  # video url
            base_name = os.path.basename(url_k[0][:-11])
            video_path = url_k[0][:url_k[0].find(base_name)-1]
            video_k = os.path.basename(video_path)
            video_k += '/'
            video_k += base_name

            pos_arr = np.asarray(list(res_dict[k].values())[0])  # list of array
            if video_k not in ressponses_dict.keys():
                ressponses_dict[video_k] = []

            ressponses_dict[video_k].append(pos_arr)

    print("-*-" * 30)

    video_keys = []
    heatmap_3D = []
    pos_data = []
    for k, v in ressponses_dict.items():
        # print('processing video ', k)
        pos_arr = np.asarray(v)
        pos_arr = np.vstack(pos_arr)
        # print(pos_arr.shape)
        pos_data.append(pos_arr)
        video_keys.append(k)
        if not parallel:
            hm = generate_3Dheatmap(pos_arr, title_str=k, vis_hm=True)
            heatmap_3D.append(hm)

    video_keys = np.asarray(video_keys)
    video_keys = video_keys.astype('|S35')

    if parallel:
        heatmap_3D = Parallel(n_jobs=16)(delayed(generate_3Dheatmap)(d) for d in pos_data)

    heatmap_3D = np.asarray(heatmap_3D)

    h5f = h5py.File("../datafile_pilot/exp2_res/hm.hdf5", 'w')
    h5f.create_dataset('video_keys', video_keys.shape, dtype=video_keys.dtype, data=video_keys)
    h5f.create_dataset('heatmaps', heatmap_3D.shape, dtype=heatmap_3D.dtype, data=heatmap_3D)
    h5f.close()


def test_hdf5():
    file_dir = "../datafile_pilot/exp2_res/hm.hdf5"
    h5f = h5py.File(file_dir, 'r')
    print(h5f.keys())
    video_keys = np.asarray(list(h5f['video_keys'])).astype('U')
    print(video_keys.shape)
    print(video_keys)
    hms = h5f['heatmaps']
    print(hms.shape)


if __name__ == '__main__':
    main(parallel=False)
    # test_hdf5()
