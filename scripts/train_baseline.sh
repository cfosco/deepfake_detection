#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py \
    --ddp -w 1 -r 0 -j 12 \
    -b 32 \
    --segment_count 32 \
    -a resnet3d50 --pretrained moments \
    --optimizer SGD \
    --dataset_type DeepfakeFaceFrame --record_set_type DeepfakeFaceSet