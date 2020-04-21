import ffmpeg
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def vidwrite(images, filename, framerate=30, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(filename, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
        .global_args('-loglevel', 'error')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
            .astype(np.uint8)
            .tobytes()
        )
    process.stdin.close()
    process.wait()
