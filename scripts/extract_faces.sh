#!/usr/bin/env bash

VIDEO_ROOT=${DATA_ROOT}/DeepfakeDetection/videos
FACES_ROOT=${DATA_ROOT}/DeepfakeDetection/face_frames

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m fire data.faces videos_to_faces ${VIDEO_ROOT} ${FACES_ROOT}
