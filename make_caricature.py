"""
Warning: this is an extremely ugly throwaway script...proceed with caution...
"""
import glob
import json
import os
import sys

import numpy as np
import torch

import core
import pretorched
import pretorched.visualizers as vutils
from data import VideoFolder, transforms
from pretorched.models import utils as mutils
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = os.environ.get('DATA_ROOT', '')
datadir = os.path.join(data_root, 'DeepfakeDetection', 'test_facenet_videos')
target_file = os.path.join(data_root, 'DeepfakeDetection', 'test_targets.json')
# outdir = 'notebooks/caricatures'

CHECKPOINTS = {
    'resnet18': '../best_weights/FrameDetector_resnet18_all_TSNFrameSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-64_best.pth.tar'
}


# def _make_caricatures(
#     video_dir,
#     out_dir,
#     amp=5,
#     mode='full',
#     size='Small',
#     batch_size=1,
#     basemodel_name='resnet18',
# ):

#     os.makedirs(outdir, exist_ok=True)
#     transform = transforms.get_transform(split='val', normalize=False, rescale=False)
#     dataset = VideoFolder(datadir, step=1, transform=transform)
#     dataset_orig = VideoFolder(datadir, step=1)

#     # model = core.get_model(f'SeriesPretrainedFrozen{size}ManipulatorAttnDetector')
#     model = core.get_model(f'SeriesPretrainedFrozen{size}ManipulatorDetector')

#     ckpt_file = CHECKPOINTS[basemodel_name]
#     ckpt = torch.load(ckpt_file)
#     state_dict = mutils.remove_prefix(ckpt['state_dict'])
#     model.detector_model.load_state_dict(state_dict, strict=False)

#     model = model.to(device)
#     model.eval()

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=False,
#         drop_last=False,
#     )
#     dataloader_orig = torch.utils.data.DataLoader(
#         dataset_orig,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=False,
#         drop_last=False,
#     )

#     for i, ((names, frames, target), (_, frames_orig, _)) in enumerate(
#         zip(dataloader, dataloader_orig)
#     ):
#         print(f'Processing [{i+1} / {len(dataloader)}]: {", ".join(names)}')
#         if i == 0:
#             continue

#         model = model.cpu()
#         model = model.to(device)
#         frames = frames.to(device)

#         attn_map = None
#         model.zero_grad()
#         if mode.lower() == 'gradcam':
#             normalize_attn = True
#             gcam_model = vutils.grad_cam.GradCAM(model.detector_model.model)
#             out, attn_map = gcam_forward(gcam_model, frames)
#             attn_map.detach_()
#             attn_map = attn_map.cpu()
#             del gcam_model

#         with torch.no_grad():
#             if mode == 'attn':
#                 normalize_attn = True
#                 out, attn_map = model.detector_model(frames)

#             del frames
#             frames_orig = frames_orig.to(device)
#             cari = model.manipulate(
#                 frames_orig,
#                 amp=torch.tensor(amp),
#                 attn_map=attn_map,
#                 normalize_attn=normalize_attn,
#             )
#             cari = cari.cpu()
#             del frames_orig
#             torch.cuda.empty_cache()

#         for n, (name, c) in enumerate(zip(names, cari)):
#             c = c.permute(1, 2, 3, 0)
#             outname = f'{name.replace(".mp4", "")}_cari_{size}_{mode}_amp{amp}' + '.mp4'
#             outfile = os.path.join(outdir, outname)
#             pretorched.data.utils.async_array_to_video(c, outfile)
#             if attn_map is not None:
#                 am = normalize(attn_map[n]).cpu()
#                 attn_outname = (
#                     f'{name.replace(".mp4", "")}_attn_{size}_{mode}_amp{amp}' + '.mp4'
#                 )
#                 attn_outfile = os.path.join(outdir, attn_outname)
#                 if mode not in ['gradcam']:
#                     pretorched.data.utils.async_array_to_video(
#                         255 * am.unsqueeze(-1).repeat(1, 1, 1, 3), attn_outfile,
#                     )
#                 heatmap_outname = (
#                     f'{name.replace(".mp4", "")}_heatmap_{size}_{mode}_amp{amp}'
#                     + '.mp4'
#                 )

#                 heatmap = [
#                     vutils.grad_cam.apply_heatmap(a, cc)
#                     for a, cc in zip(am.numpy(), c.numpy())
#                 ]
#                 heatmap_outfile = os.path.join(outdir, heatmap_outname)
#                 pretorched.data.utils.async_array_to_video(
#                     heatmap, heatmap_outfile,
#                 )


def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


def gcam_forward(
    gcam_model, frames, target_layer='layer4', fake_weighted=True, threshold=0.1
):
    # TODO: Finish
    outputs = []
    attn_maps = []
    for frame in frames:
        frame = frame.transpose(0, 1)
        gcam_model.model.zero_grad()
        probs, idx = gcam_model.forward(frame)
        print(probs.tolist())
        print(idx.tolist())
        gcam_model.backward(idx=idx[0])
        attn_map = gcam_model.generate(target_layer=target_layer)
        attn_map.detach_()
        gcam_model.model.zero_grad()
        del gcam_model.preds
        del gcam_model.probs
        del gcam_model.image

        print(idx.tolist()[0])
        outputs.append(idx.tolist()[0] == 1)
        attn_maps.append(attn_map)
    # TODO: smooth and threshold attn maps
    attn_maps = torch.stack(attn_maps)
    del probs
    del idx
    del gcam_model
    return outputs, attn_maps


def make_metadata(glob_pattern='cari_Small_gradcam_amp5'):
    with open(target_file) as f:
        in_data = json.load(f)

    outname = glob_pattern + '.json'
    if not glob_pattern.startswith('*'):
        glob_pattern = '*' + glob_pattern
    if not glob_pattern.endswith('*'):
        glob_pattern += '*'

    videos = glob.glob(os.path.join(outdir, glob_pattern))
    out_data = []
    for video in videos:
        filename = os.path.basename(video)
        name = filename.split('_')[0] + '.mp4'
        target = in_data[name]
        label = 'fake' if target == 1 else 'real'
        out_data.append({'filename': filename, 'label': label})

    print(len(out_data))
    with open(os.path.join('mturk', outname), 'w') as f:
        json.dump(out_data, f, indent=2)



def make_caricatures(
    datadir,
    outdir,
    amp=5,
    mode='full',
    size='Small',
    batch_size=1,
    basemodel_name='resnet18',
    is_single_video=False
):

    os.makedirs(outdir, exist_ok=True)
    transform = transforms.get_transform(split='val', normalize=False, rescale=False)
    dataset = VideoFolder(datadir, step=1, transform=transform)
    dataset_orig = VideoFolder(datadir, step=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = prepare_model(basemodel_name, size=size, device=device)

    if mode == 'gradcam':
        gcam_model = vutils.grad_cam.GradCAM(model.detector_model.model)


    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=False,
    #     drop_last=False,
    # )
    # dataloader_orig = torch.utils.data.DataLoader(
    #     dataset_orig,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=False,
    #     drop_last=False,
    # )

    for i, ((names, frames, target), (_, frames_orig, _)) in enumerate(
    zip(dataset, dataset_orig)):
        names = [names]
        frames = frames.unsqueeze(0)
        frames_orig = frames_orig.unsqueeze(0)
        print(f'Processing [{i} / {len(dataset)}]: {", ".join(names)}')

        process_gcam(model, gcam_model, frames, frames_orig, names, device, size, amp, mode, outdir)
        del frames
        del frames_orig
        del names
        del target
        torch.cuda.empty_cache()


def make_caricatures_single(input, outdir, amp=5, mode='full', size='Small', batch_size=1):
    '''Generates caicatures for a single video in mp4 format'''

    # TODO: Finish
    os.makedirs(outdir, exist_ok=True)
    transform = transforms.get_transform(split='val', normalize=False, rescale=False)

    tmp_frames = extract_frames(input)

    dataset = VideoFolder(tmp_frames, step=1, transform=transform)

def prepare_model(basemodel_name, size='Small', device=f'cuda'):

    print("Preparing Model...")
    model = core.get_model(f'SeriesPretrainedFrozen{size}ManipulatorDetector')
    ckpt_file = CHECKPOINTS[basemodel_name]

    print("Loading checkpoint")
    ckpt = torch.load(ckpt_file)
    state_dict = mutils.remove_prefix(ckpt['state_dict'])
    model.detector_model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model

def process_gcam(model, gcam_model, frames, frames_orig, names, device, size, amp, mode, outdir):

    frames = frames.to(device)
    model.zero_grad()

    normalize_attn = True
    frames = model.detector_model.norm(frames)
    input("Press Enter to continue...")
    do_mag, attn_map = gcam_forward(gcam_model, frames)
    input("Post gcam_forward")
    attn_map.detach_()
    attn_map = attn_map.cpu()
    del gcam_model

    with torch.no_grad():
        if mode == 'attn':
            normalize_attn = True
            out, attn_map = model.detector_model(frames)

        del frames
        torch.cuda.empty_cache()
        frames_orig = frames_orig.to(device)
        torch.cuda.empty_cache()
        a = torch.tensor(amp, requires_grad=False)

        # NOTE: Cannot use a batch size larger than 1!
        if len(do_mag) == 1:
            do_mag = do_mag[0]

        if do_mag:
            print('doing mag')
            cari = model.manipulate(
                frames_orig, amp=a, attn_map=attn_map, normalize_attn=normalize_attn,
            )
        else:
            print('skipping mag')
            cari = frames_orig
        del model
        cari = cari.cpu()
        del frames_orig
        torch.cuda.empty_cache()

    
    # Save caricatures and heatmaps
    for n, (name, c) in enumerate(zip(names, cari)):
        c = c.permute(1, 2, 3, 0)

        save_caricature_to_video(c, size, mode, amp, outdir, name)

        if attn_map is not None:
            save_heatmap(attn_map[n], c, size, mode, amp, outdir, name)

# def process(model, frames, frames_orig, names, device, size, amp, mode, outdir):

#     frames = frames.to(device)

#     do_mag = True
#     attn_map = None
#     model.zero_grad()
#     if mode.lower() == 'gradcam':
#         normalize_attn = True
#         gcam_model = vutils.grad_cam.GradCAM(model.detector_model.model)
#         frames = model.detector_model.norm(frames)
#         do_mag, attn_map = gcam_forward(gcam_model, frames)
#         attn_map.detach_()
#         attn_map = attn_map.cpu()
#         del gcam_model

#     with torch.no_grad():
#         if mode == 'attn':
#             normalize_attn = True
#             out, attn_map = model.detector_model(frames)

#         del frames
#         torch.cuda.empty_cache()
#         frames_orig = frames_orig.to(device)
#         torch.cuda.empty_cache()
#         a = torch.tensor(amp, requires_grad=False)

#         # NOTE: Cannot use a batch size larger than 1!
#         if len(do_mag) == 1:
#             do_mag = do_mag[0]

#         if do_mag:
#             print('doing mag')
#             cari = model.manipulate(
#                 frames_orig, amp=a, attn_map=attn_map, normalize_attn=normalize_attn,
#             )
#         else:
#             print('skipping mag')
#             cari = frames_orig
#         del model
#         cari = cari.cpu()
#         del frames_orig
#         torch.cuda.empty_cache()

    
#     # Save caricatures and heatmaps
#     for n, (name, c) in enumerate(zip(names, cari)):
#         c = c.permute(1, 2, 3, 0)
#         save_caricature_to_video(c, size, mode, amp, outdir, name)

#         if attn_map is not None:
#             save_heatmap(attn_map[n], c, size, mode, amp, outdir, name)
        


def save_caricature_to_video(c, size, mode, amp, outdir, name):
    
    outname = f'{name.replace(".mp4", "")}_cari_{size}_{mode}_amp{amp}' + '.mp4'
    outfile = os.path.join(outdir, outname)
    pretorched.data.utils.async_array_to_video(c, outfile)

def save_heatmap(am, c, size, mode, amp, outdir, name):

    am = normalize(am.cpu())

    attn_outname = (
        f'{name.replace(".mp4", "")}_attn_{size}_{mode}_amp{amp}' + '.mp4'
    )
    attn_outfile = os.path.join(outdir, attn_outname)
    if mode not in ['gradcam']:
        pretorched.data.utils.async_array_to_video(
            255 * am.unsqueeze(-1).repeat(1, 1, 1, 3), attn_outfile,
        )
        
    heatmap_outname = (
        f'{name.replace(".mp4", "")}_heatmap_{size}_{mode}_amp{amp}' + '.mp4'
    )

    print("am and c shapes:" + str(am.shape) + str(c.shape))

    heatmap = [
        vutils.grad_cam.apply_heatmap(a, cc)
        for a, cc in zip(am.numpy(), c.numpy())
    ]
    heatmap_outfile = os.path.join(outdir, heatmap_outname)
    pretorched.data.utils.async_array_to_video(
        heatmap, heatmap_outfile,
    )
    del heatmap
    del c
    del am

if __name__ == "__main__":

    # catch arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./deepfake_videos')
    parser.add_argument('--outdir', type=str, default='./caricatures')
    parser.add_argument('--basemodel', type=str, default='resnet18')
    parser.add_argument('--amp', type=int, default=10)
    parser.add_argument('--mode', type=str, default='gradcam')
    parser.add_argument('--idx', type=int, default=None)

    args = parser.parse_args()

    if args.input is None:
        raise ValueError('Must specify input directory or file')

    elif os.path.isdir(args.input):
        make_caricatures(args.input, args.outdir, amp=10, mode="gradcam")

    elif os.path.isfile(args.input):
        make_caricatures(args.input, args.outdir, amp=10, mode="gradcam", is_single_video=True)
