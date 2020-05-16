#!/usr/bin/env bash

VIDEO_ROOT=${DATA_ROOT}/DeepfakeDetection/videos
FRAME_ROOT=${DATA_ROOT}/DeepfakeDetection/frames

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/../"

# python eval.py --part adversarial_test_videos
# python eval.py --part aug_test_videos/original
# python eval.py --part aug_test_videos/original_fps15
# python eval.py --part aug_test_videos/original_crf33
# python eval.py --part aug_test_videos/original_scale0.25
# python eval.py --part aug_test_videos/original_scale0.25_crf33
# python eval.py --part aug_test_videos/original_scale0.25_crf44
# python eval.py --part aug_test_videos/dfdc_train_part_33
# python eval.py --part aug_test_videos/dfdc_train_part_33_fps15
# python eval.py --part aug_test_videos/dfdc_train_part_33_crf33
# python eval.py --part aug_test_videos/dfdc_train_part_33_scale0.25

# python eval.py --dataset FaceForensics --part original_sequences/actors/c23 --default_target 0
python eval.py --dataset FaceForensics --part original_sequences/youtube/c23 --default_target 0
python eval.py --dataset FaceForensics --part manipulated_sequences/DeepFakeDetection/c23 --default_target 1 --whitelist_file test_videos.json
python eval.py --dataset FaceForensics --part manipulated_sequences/Deepfakes/c23 --default_target 1 --whitelist_file test_videos.json
python eval.py --dataset FaceForensics --part manipulated_sequences/Face2Face/c23 --default_target 1 --whitelist_file test_videos.json
python eval.py --dataset FaceForensics --part manipulated_sequences/FaceSwap/c23 --default_target 1 --whitelist_file test_videos.json
python eval.py --dataset FaceForensics --part manipulated_sequences/NeuralTextures/c23 --default_target 1 --whitelist_file test_videos.json

python eval.py --dataset CelebDF --part Celeb-synthesis --default_target 1
python eval.py --dataset CelebDF --part Celeb-real --default_target 0
python eval.py --dataset CelebDF --part YouTube-real --default_target 0
python eval.py --dataset CelebDF --part Celeb-synthesis --default_target 1