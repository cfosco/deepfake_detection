#!/usr/bin/env bash

ROOT=${DATA_ROOT}/DeepfakeDetection
source activate face_recognition_cpu

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m fire data.faces match_actors $ROOT