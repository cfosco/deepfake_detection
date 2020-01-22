import time
import os
import torch.utils.data as data
import pretorched.data
import torch
import sys
sys.path.append('../')

try:
    import data
    import core
except Exception:
    pass


class FrameToVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except FileNotFoundError:
            record = self.dataset.record_set[index]
            video_dir = os.path.join(self.dataset.root, record.path)

            for fn in record.face_names:
                video_path = os.path.join(video_dir, fn)
                print(video_path)
                pretorched.data.utils.frames_to_video(f'{video_path}/*.jpg', video_path + '.mp4')
            return self[index]

    def __len__(self):
        return len(self.dataset)


dataloader = core.get_dataloader('DFDC', data_root='/data/datasets', resolution=256, size=224, segment_count=16,
                                 dataset_type='DeepfakeFaceVideo', record_set_type='DeepfakeFaceSet',
                                 batch_size=1, pin_memory=False)
dataset = FrameToVideoDataset(dataloader.dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=12, shuffle=False)


end = time.time()
for i, d in enumerate(dataloader):
    if i % 100 == 0:
        print(i, d[0].shape, time.time() - end)
        end = time.time()
