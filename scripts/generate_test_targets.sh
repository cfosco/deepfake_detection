#!/usr/bin/env bash

VIDEO_ROOT=${DATA_ROOT}/DeepfakeDetection/videos

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m fire data.utils generate_test_targets $DATA_ROOT/DeepfakeDetection
