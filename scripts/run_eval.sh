#!/usr/bin/env bash

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/../"

DATASETS="DFDC FaceForensics CelebDF YouTubeDeepfakes"
for dataset in ${DATASETS}; do

    # CKPT=resnet18_dfdc_seg_count-24_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
    CKPT=FrameModel_resnet18_all_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-128_best.pth.tar
    python main.py \
        --dataset ${dataset} \
        --evaluate \
        -b 32 --segment_count 64 -a resnet18 --optimizer Ranger \
        --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
        --resume weights/${CKPT}
done
# --sampler_type ClipSampler \
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
# python eval.py --part test_videos
# python eval.py --dataset FaceForensics --part original_sequences/youtube/c23 --default_target 0
# python eval.py --dataset FaceForensics --part manipulated_sequences/DeepFakeDetection/c23 --default_target 1 --whitelist_file test_videos.json
# python eval.py --dataset FaceForensics --part manipulated_sequences/Deepfakes/c23 --default_target 1 --whitelist_file test_videos.json
# python eval.py --dataset FaceForensics --part manipulated_sequences/Face2Face/c23 --default_target 1 --whitelist_file test_videos.json
# python eval.py --dataset FaceForensics --part manipulated_sequences/FaceSwap/c23 --default_target 1 --whitelist_file test_videos.json
# python eval.py --dataset FaceForensics --part manipulated_sequences/NeuralTextures/c23 --default_target 1 --whitelist_file test_videos.json

# python eval.py --dataset CelebDF --part Celeb-synthesis --default_target 1 --whitelist_file
# python eval.py --dataset CelebDF --part Celeb-real --default_target 0
# python eval.py --dataset CelebDF --part YouTube-real --default_target 0
# python eval.py --dataset CelebDF --part Celeb-synthesis --default_target 1
