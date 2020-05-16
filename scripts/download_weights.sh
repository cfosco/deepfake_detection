#!/usr/bin/env bash

USER=andonian
REMOTE_ROOT='/nobackup/users/${USER}/forensics/deepfake_detection/'
REMOTE_SERVER='satori'
LOCAL_DIR='.'

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scp -r ${REMOTE_SERVER}:${REMOTE_ROOT}/weights ${LOCAL_DIR}
scp -r ${REMOTE_SERVER}:${REMOTE_ROOT}/logs ${LOCAL_DIR}
