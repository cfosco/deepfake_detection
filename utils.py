import contextlib
import os

import torch


def inspect_checkpoint(ckpt_file):
    ckpt = torch.load(ckpt_file, map_location='cpu')
    for key in ['state_dict', 'optimizer', 'history', 'args']:
        if key in ckpt:
            ckpt.pop(key)
    print(ckpt_file)
    for name, val in ckpt.items():
        print(f'\t {name}: {val}')


def inspect_all_checkpoints(checkpoint_dir):
    ckpt_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith('.tar') or f.endswith('.pth')
    ]
    for ckpt_file in ckpt_files:
        inspect_checkpoint(ckpt_file)


def clean_logs(log_dir):
    dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    for d in dirs:
        path = os.path.join(log_dir, d)
        clean_tensorboard_logdir(path)


def clean_tensorboard_logdir(logdir):
    dirs = [d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]
    for d in dirs:
        path = os.path.join(logdir, d)
        for f in os.listdir(path):
            filepath = os.path.join(path, f)
            size = os.path.getsize(filepath)
            if size < 20000:
                print(filepath, size)
                dirname = os.path.dirname(filepath)
                dst_dir = os.path.join('/home/andonian/trash/df_logs', dirname)
                os.makedirs(dst_dir, exist_ok=True)
                os.rename(filepath, os.path.join(dst_dir, os.path.basename(filepath)))
                with contextlib.suppress(OSError):
                    os.rmdir(dirname)
