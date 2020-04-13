#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python main.py --ddp -b 32 --segment_count 32 --pretrained moments --optimizer SGD