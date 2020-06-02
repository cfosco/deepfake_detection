#!/usr/bin/env bash

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/../"

DATASETS="DFDC FaceForensics CelebDF YouTubeDeepfakes"
for dataset in ${DATASETS}; do
    if true; then
        CKPTS="
        weights/FrameDetector_mxresnet18_celebdf_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
        weights/FrameDetector_mxresnet18_faceforensics_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
        "
        for CKPT in ${CKPTS}; do
            echo ${CKPT}
            python main.py \
                --model_name FrameDetector \
                --basemodel_name mxresnet18 \
                --dataset ${dataset} --evaluate \
                -b 32 --segment_count 24 --optimizer Ranger \
                --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
                --resume ${CKPT}
        done
    fi

    if true; then
        CKPTS="
            weights/FrameDetector_mxresnet34_celebdf_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
            weights/FrameDetector_mxresnet34_faceforensics_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
        "
        for CKPT in ${CKPTS}; do
            echo ${CKPT}
            python main.py \
                --model_name FrameDetector \
                --basemodel_name mxresnet34 \
                --dataset ${dataset} --evaluate \
                -b 32 --segment_count 24 --optimizer Ranger \
                --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
                --resume ${CKPT}
        done
    fi

    if true; then
        CKPTS="
        weights/FrameDetector_resnet18_celebdf_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        weights/FrameDetector_resnet18_faceforensics_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        "
        for CKPT in ${CKPTS}; do
            echo ${CKPT}
            python main.py \
                --model_name FrameDetector \
                --basemodel_name resnet18 \
                --dataset ${dataset} --evaluate \
                -b 32 --segment_count 24 --optimizer Ranger \
                --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
                --resume ${CKPT}
        done

    fi

    if true; then
        CKPTS="
        weights/FrameDetector_resnet34_celebdf_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        weights/FrameDetector_resnet34_celebdf_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        weights/FrameDetector_resnet34_faceforensics_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        "
        for CKPT in ${CKPTS}; do
            echo ${CKPT}
            python main.py \
                --model_name FrameDetector \
                --basemodel_name resnet34 \
                --dataset ${dataset} --evaluate \
                -b 32 --segment_count 24 --optimizer Ranger \
                --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
                --resume ${CKPT}
        done
    fi

    if true; then
        CKPTS="
        weights/FrameDetector_resnet50_celebdf_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
        weights/FrameDetector_resnet50_faceforensics_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
        "
        for CKPT in ${CKPTS}; do
            echo ${CKPT}
            python main.py \
                --model_name FrameDetector \
                --basemodel_name resnet50 \
                --dataset ${dataset} --evaluate \
                -b 32 --segment_count 24 --optimizer Ranger \
                --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
                --resume ${CKPT}
        done
    fi

    if true; then
        CKPTS="
        weights/FrameDetector_samxresnet18_celebdf_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-24_best.pth.tar
        weights/FrameDetector_samxresnet18_faceforensics_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-24_best.pth.tar
        "
        for CKPT in ${CKPTS}; do
            echo ${CKPT}
            python main.py \
                --model_name FrameDetector \
                --basemodel_name samxresnet18 \
                --dataset ${dataset} --evaluate \
                -b 32 --segment_count 24 --optimizer Ranger \
                --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
                --resume ${CKPT}
        done
    fi

    if true; then
        CKPTS="
        weights/FrameDetector_samxresnet34_celebdf_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
        weights/FrameDetector_samxresnet34_faceforensics_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-32_best.pth.tar
        "
        for CKPT in ${CKPTS}; do
            echo ${CKPT}
            python main.py \
                --model_name FrameDetector \
                --basemodel_name samxresnet34 \
                --dataset ${dataset} --evaluate \
                -b 32 --segment_count 24 --optimizer Ranger \
                --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
                --resume ${CKPT}
        done
    fi

    if false; then
        CKPT=weights/FrameDetector_resnet18_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
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
            --model_name FrameDetector \
            --basemodel_name samxresnet18 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 24 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/FrameDetector_resnet34_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-48_best.pth.tar
        python main.py \
            --model_name FrameDetector \
            --basemodel_name resnet34 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 64 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=best_weights/FrameDetector_resnet34_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-48_best.pth.tar
        python main.py \
            --model_name FrameDetector \
            --basemodel_name resnet34 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 64 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=weights/FrameDetector_resnest50_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-56_best.pth.tar
        python main.py \
            --model_name FrameDetector \
            --basemodel_name resnest50 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 16 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

    if false; then
        CKPT=weights/FrameDetector_samxresnet34_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar
        python main.py \
            --model_name FrameDetector \
            --basemodel_name samxresnet34 \
            --dataset ${dataset} --evaluate \
            -b 32 --segment_count 64 --optimizer Ranger \
            --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
            --resume ${CKPT}
    fi

done
