#!/usr/bin/env bash

VIDEO_ROOT=${DATA_ROOT}/DeepfakeDetection/videos
FACES_ROOT=${DATA_ROOT}/DeepfakeDetection/face_frames

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m fire data.utils generate_metadata $DATA_ROOT/DeepfakeDetection
python -m fire data.utils generate_test_metadata $DATA_ROOT/DeepfakeDetection
