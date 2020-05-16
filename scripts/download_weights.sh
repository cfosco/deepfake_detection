#!/usr/bin/env bash

ROOT=${DATA_ROOT}/FaceForensics/
USER=andonian
REMOTE_DIR='/nobackup/users/${USER}/forensics/deepfake_detection/weights'
REMOTE_SERVER='satori'
LOCAL_DIR='.'

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scp -r ${REMOTE_SERVER}:${REMOTE_DIR} ${LOCAL_DIR}
