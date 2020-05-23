# deepfake_detection
Using Distortion to Detect Deepfakes

## Installation

First, you may optionally create a dedicated conda env for the project [TODO: (README) Add instructions]. Next,

```
git clone https://github.com/cfosco/deepfake_detection.git
TODO: (README) Finish installation instractions
```

<details>
  <summary> Dependencies (click to expand) </summary>

  ## Dependencies
  - PyTorch >= 1.4
  - torchvision
  - torchvideo
  - lintel
  - pretorched (`dev` branch)

  ```
git clone https://github.com/alexandonian/torchvideo.git && cd torchvideo && pip install -e . && cd ..
git clone https://github.com/alexandonian/lintel.git && cd lintel && pip install -e . && cd ..
git clone https://github.com/alexandonian/FileLock.git && cd FileLock && pip install -e . && cd ..
git clone https://github.com/alexandonian/pretorched-x.git && cd pretorched-x && pip install -e . && git checkout dev && cd ..
  ```

TODO: (README) Finish writing instructions.
</details>

### Setup environment for CSAIL vision cluster:
```
# Add Alex's python to path:
ANACONDA_HOME=/data/vision/oliva/scratch/andonian/anaconda3
PATH=${ANACONDA_HOME}/bin:$PATH

# Add env variable that specifies location to data

# NFS data root
export DATA_ROOT=/data/vision/oliva/scratch/datasets

# Local data root on select machines (e.g. visiongpu52)
# export DATA_ROOT=/mnt/data/datasets

# Run setup scripts:

bash scripts/setup_scripts.sh
bash notebooks/setup_notebooks.sh
```

## Dataset structure
The directory structure of $DATA_ROOT should be the following:
TODO: (README) Explain the structure in more detail
```
├── DeepfakeDetection
│   ├── metadata.json
│   ├── test_metadata.json
│   ├── test_videos.json
│   ├── videos
│   │   └── dfdc_train_part_0
│   │   └── ...
│   ├── facenet_videos
│   │   └── dfdc_train_part_0
│   │   └── ...
├── FaceForensics
│   ├── metadata.json
│   ├── test_metadata.json
│   ├── test_videos.json
│   ├── original_sequences
│   ├── manipulated_sequences
│   │   └── DeepFakeDetection
│   │   └── Deepfakes
│   │   └── Face2Face
│   │   └── FaceSwap
│   │   └── NeuralTextures
├── CelebDF
│   ├── metadata.json
│   ├── test_metadata.json
│   ├── test_videos.json
│   ├── videos
│   ├── facenet_videos
├── YouTubeDeepfakes
│   ├── metadata.json
│   ├── test_metadata.json
│   ├── test_videos.json
│   ├── videos
│   ├── facenet_videos
```


## Preprocessing

In order to accelerate training and testing, we first extract the faces from each video and store them in the corresponding `facenet_videos` dir. This only needs to be done once.

An example extraction can be run with the following command:

```
python scripts/extract_facenet.py --dataset FaceForensics --part manipulated_sequences/Face2Face/c23/ --magnify_motion False --chunk_size 100 --use_zip False --num_workers 4
```

## Training

The entrypoint into training is `main.py`. You'll need to specifiy several command line arguments, including the `model_name`, `dataset`. A full list of cmd line options are shown in `config.py`. Here is an example run:

```
python main.py \
    --model_name FrameDetector--basemodel_name resnet18 --dataset DFDC \
    --batch-size 128 --segment_count 16 --optimizer Ranger --pretrained imagenet
    --num_workers 12 --dataset_type DeepfakeFaceVideo
```
Checkpoints are stored in `weights_dir` (default: weights) and logs in `logs_dir` (default: logs)

## Evaluation

There are currently two ways to evaluate the performance of a model.

1. Run a single evaluation from `main.py` with the `--evaluate` flag, specifying a saved checkpoint to load using the `resume` cmd line arg. For example,

```
CKPT=weights/FrameModel_resnet18_all_ClipSampler_seg_count-16_init-imagenet-ortho_optim-Ranger_lr-0.001_sched-CosineAnnealingLR_bs-128_best.pth.tar

python main.py \
    --evaluate \ --dataset ${dataset} \
    -b 32 --segment_count 64 -a resnet18 --optimizer Ranger \
    --pretrained imagenet -j 4 --dataset_type DeepfakeFaceVideo \
    --resume ${CKPT}
```

2. Run `eval.py` - this currently includes the entire pipeline, including an initial face extraction step, which is skipped in (1) due to the preprocessing step from above.

An example:
```
python eval.py \
    --dataset FaceForensics \
    --part manipulated_sequences/DeepFakeDetection/c23 \
    --default_target 1 \
    --whitelist_file test_videos.json
```
TODO: eval.py script needs to be cleaned

Results of the evaluation are stored in `results_dir` (default: results).

## Description of available models

1. `Detector` - a base class representing a deepfake detector with the following subclasses:
    - `FrameDetector`:  Baseline 2D CNN detector (e.g. resnet18 with average pooling over the per-frame logits)
    - `VideoDetector`:  Baseline 3D CNN detector (e.g. resnet3d18)
2. Detectors with self-attention: An improvment over the base detectors via the addition of self-attention blocks:
    - Specify `--model_name FrameDetector` and `--basemodel_name mxresnet18` for a FrameDetector with self attention.
3. `SeriesManipulatorDetector`:
    - Unsupervised manipulation is applied to the video before passing through the detector.
    - Pretrained MagNetDetector Hybrid can be achieved by first loading MagNet and Detector weights before training.
4. `SharedAttnCaricatureDetector`: Learned self-attention maps are applied to caricature module
5. `GradCamCaricatureModel` - simply for post-training caricature generation for human perception.
