#!/usr/bin/env bash

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/../"

DATASETS="DFDC FaceForensics CelebDF YouTubeDeepfakes"
for dataset in ${DATASETS}; do

    if false; then
        CKPT=best_weights/FrameDetector_resnet18_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        python main.py \
            --model_name FrameDetector \
            --basemodel_name resnet18 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 24 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/FrameDetector_resnet18_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        python main.py \
            --model_name FrameDetector \
            --basemodel_name resnet18 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 24 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/VideoDetector_resneti3d18_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-128_best.pth.tar
        python main.py \
            --model_name VideoDetector \
            --basemodel_name resneti3d18 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 16 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/SeriesPretrainedSmallManipulatorDetector_resnet18_all_TSNFrameSampler_seg_count-8_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-60_best.pth.tar
        python main.py \
            --model_name SeriesPretrainedSmallManipulatorDetector \
            --basemodel_name resnet18 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 24 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/SeriesPretrainedFrozenMediumManipulatorDetector_resnet18_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-36_best.pth.tar
        python main.py \
            --model_name SeriesPretrainedFrozenMediumManipulatorDetector \
            --basemodel_name resnet18 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 24 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/FrameDetector_samxresnet18_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-36_best.pth.tar
        python main.py \
            --model_name FrameDetector  \
            --basemodel_name samxresnet18 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 24 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/FrameDetector_resnet34_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-48_best.pth.tar
        python main.py \
            --model_name FrameDetector  \
            --basemodel_name resnet34 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 64 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi
done
