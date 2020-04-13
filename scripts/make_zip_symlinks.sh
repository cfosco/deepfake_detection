#!/usr/bin/env bash

VIDEO_ROOT=${DATA_ROOT}/DeepfakeDetection/videos
for i in {00..09}; do
    ln -s ${VIDEO_ROOT}/dfdc_train_part_${i}.zip ${VIDEO_ROOT}/dfdc_train_part_${i:1}.zip
done
