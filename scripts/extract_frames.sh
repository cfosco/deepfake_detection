#!/usr/bin/env bash

VIDEO_ROOT=${DATA_ROOT}/DeepfakeDetection/videos
FRAME_ROOT=${DATA_ROOT}/DeepfakeDetection/frames

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m fire data.utils videos_to_frames ${VIDEO_ROOT} ${FRAME_ROOT}
