import torch


class FrameModel(torch.nn.Module):

    frame_dim = 2

    def __init__(self, model, consensus_func=torch.mean):
        super().__init__()
        self.model = model
        self.consensus_func = consensus_func

    def forward(self, x):
        x = x.permute(2, 0, 1, 3, 4)
        return self.consensus_func(torch.stack([self.model(f) for f in x]), dim=0)

    @property
    def input_size(self):
        return self.model.input_size
