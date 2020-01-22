import functools
import os

import setproctitle
import tensorflow as tf
from configobj import ConfigObj
from tqdm import tqdm
from validate import Validator

from .deep_motion_mag.magnet import MagNet3Frames


def _run_motion_mag(args):
    configspec = ConfigObj(args.config_spec, raise_errors=True)
    config = ConfigObj(args.config_file,
                       configspec=configspec,
                       raise_errors=True,
                       file_error=True)
    # Validate to get all the default values.
    config.validate(Validator())
    if not os.path.exists(config['exp_dir']):
        # checkpoint directory.
        os.makedirs(os.path.join(config['exp_dir'], 'checkpoint'))
        # Tensorboard logs directory.
        os.makedirs(os.path.join(config['exp_dir'], 'logs'))
        # default output directory for this experiment.
        os.makedirs(os.path.join(config['exp_dir'], 'sample'))
    network_type = config['architecture']['network_arch']
    exp_name = config['exp_name']
    setproctitle.setproctitle('{}_{}_{}'.format(args.phase, network_type, exp_name))
    tfconfig = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        model = MagNet3Frames(sess, exp_name, config['architecture'])
        checkpoint = config['training']['checkpoint_dir']
        if args.phase == 'train':
            train_config = config['training']
            if not os.path.exists(train_config['checkpoint_dir']):
                os.makedirs(train_config['checkpoint_dir'])
            model.train(train_config)
        elif args.phase == 'run':
            model.run(checkpoint,
                      args.vid_dir,
                      args.frame_ext,
                      args.out_dir,
                      args.amplification_factor,
                      args.velocity_mag)
        elif args.phase == 'run_temporal':
            model.run_temporal(checkpoint,
                               args.vid_dir,
                               args.frame_ext,
                               args.out_dir,
                               args.amplification_factor,
                               args.fl,
                               args.fh,
                               args.fs,
                               args.n_filter_tap,
                               args.filter_type)
        else:
            raise ValueError('Invalid phase argument. '
                             'Expected ["train", "run", "run_temporal"], '
                             'got ' + args.phase)


class Parser:
    def add_argument(self, name, dest='var', default=None, help='', required=True, type='', action=''):
        exec("self." + dest + " = default")


def parse_args():
    parser = Parser()

    parser.add_argument('--phase', dest='phase', default=phase,
                        help='train, test, run, interactive')
    parser.add_argument('--config_file', dest='config_file', required=True, default=config_file,
                        help='path to config file')
    parser.add_argument('--config_spec', dest='config_spec', default=config_spec,
                        help='path to config spec file')

    # for inference
    parser.add_argument('--vid_dir', dest='vid_dir', default=vid_dir,
                        help='Video folder to run the network on.')
    parser.add_argument('--frame_ext', dest='frame_ext', default=frame_ext,
                        help='Video frame file extension.')
    parser.add_argument('--out_dir', dest='out_dir', default=out_dir,
                        help='Output folder of the video run.')
    parser.add_argument('--amplification_factor', dest='amplification_factor',
                        type=float, default=amplification_factor,
                        help='Magnification factor for inference.')
    parser.add_argument('--velocity_mag', dest='velocity_mag', action='store_true', default=velocity_mag,
                        help='Whether to do velocity magnification.')

    # For temporal operation.
    parser.add_argument('--fl', dest='fl', type=float, default=fl,
                        help='Low cutoff Frequency.')
    parser.add_argument('--fh', dest='fh', type=float, default=fh,
                        help='High cutoff Frequency.')
    parser.add_argument('--fs', dest='fs', type=float, default=fs,
                        help='Sampling rate.')
    parser.add_argument('--n_filter_tap', dest='n_filter_tap', type=int, default=n_filter_tap,
                        help='Number of filter tap required.')
    parser.add_argument('--filter_type', dest='filter_type', type=str, default=filter_type,
                        help='Type of filter to use, must be Butter or FIR.')

    return parser


def get_model(config_spec, config_file):
    configspec = ConfigObj(config_spec, raise_errors=True)
    config = ConfigObj(config_file,
                       configspec=configspec,
                       raise_errors=True,
                       file_error=True)

    # Validate to get all the default values.
    config.validate(Validator())
    exp_name = '%(dataset)s_%(variant)s'
    setproctitle.setproctitle(exp_name)
    tfconfig = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    return MagNet3Frames(sess, exp_name, config['architecture'])


def run_motion_mag_folder(data_dir, output_dir, config_spec, config_file, checkpoint):
    model = get_model(config_spec, config_file)
    for vid in tqdm(os.listdir(data_dir)):
        print('running motion mag for vid:', vid)
        input_video = os.path.join(data_dir, vid)
        output_video = os.path.join(output_dir, vid)
        run_motion_mag_one_video(model, checkpoint, 'run_temporal', input_video, output_video)


def run_motion_mag_face_folder(data_dir, output_dir, config_spec, config_file, checkpoint,
                               suffix='_mm'):
    model = get_model(config_spec, config_file)
    for vid in tqdm(os.listdir(data_dir)):
        print('running motion mag for vid:', vid)
        video_dir = os.path.join(data_dir, vid)
        for d in os.listdir(video_dir):
            in_face_dir = os.path.join(video_dir, d)
            if os.path.isdir(in_face_dir):
                out_face_dir = os.path.join(video_dir, d + suffix)
                run_motion_mag_one_video(model, checkpoint, 'run_temporal', in_face_dir, out_face_dir)


def run_motion_mag_one_video(model, checkpoint, phase, video, output,
                             amplification_factor=4, velocity_mag=True,
                             fl=0.5, fh=1, fs=30, n_filter_tap=2,
                             filter_type='differenceOfIIR',
                             frame_ext='jpg'):
    if phase == 'run':
        model.run(checkpoint,
                  video,
                  frame_ext,
                  output,
                  amplification_factor,
                  velocity_mag)

    elif phase == 'run_temporal':
        model.run_temporal(checkpoint,
                           video,
                           frame_ext,
                           output,
                           amplification_factor,
                           fl, fh, fs,
                           n_filter_tap,
                           filter_type)
        #    fixed_size=256)


# Define parameters
dir_path = os.path.dirname(os.path.realpath(__file__))
exp_name = 'o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3'
config_file = os.path.join(dir_path, 'deep_motion_mag', 'configs', exp_name + '.conf')
config_spec = os.path.join(dir_path, 'deep_motion_mag', 'configs', 'configspec.conf')
checkpoint = os.path.join(dir_path, 'deep_motion_mag', 'data', 'training', exp_name, 'checkpoint')

DATA_ROOT = os.environ['DATA_ROOT']
PART = 'dfdc_train_part_2'
vid_dir = os.path.join(DATA_ROOT, 'DeepfakeDetection', 'facenet_smooth_frames', PART)
out_dir = os.path.join(DATA_ROOT, 'DeepfakeDetection', 'test_videos_mm')
os.makedirs(out_dir, exist_ok=True)


def get_motion_mag(device='/device:GPU:1'):
    with tf.device(device):
        run_motion_mag = functools.partial(run_motion_mag_one_video,
                                           model=get_model(config_spec, config_file),
                                           checkpoint=checkpoint,
                                           phase='run_temporal'
                                           )
        return run_motion_mag
# run_motion_mag_folder(vid_dir, out_dir, config_spec, config_file, checkpoint)
# run_motion_mag_face_folder(vid_dir, out_dir, config_spec, config_file, checkpoint)

# run_motion_mag_all_folders(base_dirs, config_spec, config_file)


# def old_run_motion_mag_all_folders(base_dirs, config_spec, config_file):
#     configspec = ConfigObj(config_spec, raise_errors=True)
#     config = ConfigObj(config_file,
#                        configspec=configspec,
#                        raise_errors=True,
#                        file_error=True)

#     # Validate to get all the default values.
#     config.validate(Validator())

#     network_type = 'ynet_3frames'
#     exp_name = '%(dataset)s_%(variant)s'
#     setproctitle.setproctitle(exp_name)
#     tfconfig = tf.ConfigProto(allow_soft_placement=True,
#                               log_device_placement=False)
#     tfconfig.gpu_options.allow_growth = True

#     with tf.Session(config=tfconfig) as sess:
#         model = MagNet3Frames(sess, exp_name, config['architecture'])
#         checkpoint = '../deep_motion_mag/data/training/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3/checkpoint'
#         for bd in base_dirs:
#             print("Magnifying vids in", bd)
#             for vid in tqdm(os.listdir(bd)):
#                 print('running motion mag for vid:', vid)
#                 input_folder = os.path.join(bd, vid)
#                 output_folder = os.path.join(bd + '_mm', vid)
# #                 os.makedirs(output_folder, exist_ok=True)
#                 run_motion_mag_one_video(model, checkpoint, 'run_temporal', input_folder, output_folder)
