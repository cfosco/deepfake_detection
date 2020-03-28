#!/usr/bin/env bash

ROOT=${DATA_ROOT}/YouTubeDeepfakes/

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m fire data.utils generate_YouTubeDeepfakes_metadata $ROOT
