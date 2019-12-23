#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py \
    --ddp -w 1 -r 0 -j 16 \
    -a resnet3d50 --pretrained moments \
    --dataset_type DeepfakeFaceFrame --record_set_type DeepfakeFaceSet