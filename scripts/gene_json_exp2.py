import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def getFakeVideosExp1():
    file_dir = "/Users/xi/Desktop/dfd/datafile_exp1/vids_for_exp1_1050pairs.json"
    with open(file_dir, 'r') as fp:
        data = json.load(fp)
    # print(len(data)) # 2
    # print(data.keys()) # real fake

    fake_videos = data['fake']
    # print(fake_videos)

    prefix = len("facenet_smooth_frames/")
    suffix = len("/face_0.mp4")

    video_keys = []
    folder_keys = []
    for s in fake_videos:
        k = s[prefix:-suffix]
        video_keys.append(k)
        p = k[:k.find('/')]
        folder_keys.append(p)
    return video_keys, set(folder_keys)


def getFakeVideos(video_keys, folder_keys):
    pref = "predictions_DeepfakeDetection_part_videos_" #    dfdc_train_part_27
    prediction_dir = "../datafile_exp1/resnet18_seg_count-24-dfdc"

    res = {}
    for p in folder_keys:
        pre_file = pref+p+".json"
        pre_file_dir = os.path.join(prediction_dir, pre_file)
        with open(pre_file_dir, 'r') as fp:
            data = json.load(fp)

        for d in data.keys():
            if d in video_keys:
                if data[d]['label'] == 1: # fake videos
                    res[d] = data[d]['prob']
    # print(res)
    # print(len(res))
    return res


def compileExp2SessionJsons(real_keys, practice_keys):
    n_level = 10
    trail_per_level = 10

    np.random.shuffle(real_keys)

    prefix = "http://visiongpu23.csail.mit.edu/scratch/datasets/DeepfakeDetection/facenet_smooth_frames/"
    for exp_id in range(5):
        output_json = {}
        output_json[0] = []
        for pi in range(trail_per_level):
            video_url = prefix + practice_keys[pi + trail_per_level * exp_id] + "/face_0.mp4"
            output_json[0].append(video_url)

        for i in np.arange(0, n_level):
            level_idx = int(i+1)
            output_json[level_idx] = []
            for ri in range(trail_per_level):
                idx = i * trail_per_level + ri + 100 * exp_id
                video_url = prefix + real_keys[idx] + "/face_0.mp4"
                output_json[level_idx].append(video_url)
        output_dir = "../datafile_pilot/exp2/vids_for_exp2_110pairs_id%d.json" % exp_id
        with open(output_dir, 'w') as outfile:
            json.dump(output_json, outfile)


def generateJsonFiles(video_dict):

    video_keys = np.asarray(list(video_dict.keys()))
    prob = np.asarray(list(video_dict.values()))
    sorted_idx = np.argsort(prob)
    # print(prob[sorted_idx])
    exp_idx = sorted_idx[:500]
    practice_idx = sorted_idx[500:550]

    compileExp2SessionJsons(video_keys[exp_idx], video_keys[practice_idx])


def quickCheckExp2Json():
    with open("../datafile_pilot/exp2/vids_for_exp2_110pairs_id4.json", 'r') as fp:
        data = json.load(fp)

    print(data.keys())
    print(len(data))
    print(data['0'])
    print(len(data['0']))


def main():
    video_keys, folder_keys = getFakeVideosExp1()
    fakevideo_dict = getFakeVideos(video_keys, folder_keys)
    generateJsonFiles(fakevideo_dict)


if __name__ == '__main__':
    main()
    quickCheckExp2Json()