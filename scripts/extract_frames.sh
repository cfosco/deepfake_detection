#!/usr/bin/env bash

VIDEO_ROOT=/data/datasets/DeepfakeDetection/videos
FRAME_ROOT=/data/datasets/DeepfakeDetection/frames

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m fire data.utils videos_to_frames ${VIDEO_ROOT} ${FRAME_ROOT}