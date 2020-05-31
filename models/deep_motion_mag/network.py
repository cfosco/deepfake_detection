"""
need to check
1) conv initializer (std)
2) bias
"""

import torch
import torch.nn as nn
import numpy as np


def _make_layer(block, in_planes, out_planes, num_layers, kernel_size=3, stride=1):
    layers = []
    for i in range(num_layers):
        layers.append(block(in_planes, out_planes, kernel_size, stride))
    return nn.Sequential(*layers)


def gaussian(shape_x, shape_y, mu_x=0.0, mu_y=0.0, sig_x=1.0, sig_y=1.0):

    x = np.linspace(-1.0, 1.0, shape_x)
    y = np.linspace(-1.0, 1.0, shape_y)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-0.5 * (((X - mu_x) / sig_x) ** 2 + ((Y - mu_y) / sig_y) ** 2))

    return g


class ResBlock(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        p = (kernel_size - 1) // 2
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(
            in_planes, output_planes, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.pad2 = nn.ReflectionPad2d(p)
        self.conv2 = nn.Conv2d(
            in_planes, output_planes, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv1(self.pad1(x)))
        y = self.conv2(self.pad2(y))
        return y + x


class ConvBlock(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=7, stride=1):
        super(ConvBlock, self).__init__()
        p = 3
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(
            in_planes, output_planes, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv1(self.pad1(x)))


class ConvBlockAfter(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=3, stride=1):
        super(ConvBlockAfter, self).__init__()
        p = 1
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(
            in_planes, output_planes, kernel_size=kernel_size, stride=stride, bias=False
        )

    def forward(self, x):
        return self.conv1(self.pad1(x))


class Encoder(nn.Module):
    def __init__(self, num_resblk):
        super().__init__()
        # common representation
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, bias=False)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False)
        self.resblks = _make_layer(ResBlock, 32, 32, num_resblk)
        self.relu = nn.ReLU(inplace=True)

        # texture representation
        self.pad1_text = nn.ReflectionPad2d(1)
        self.conv1_text = nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False)
        self.resblks_text = _make_layer(ResBlock, 32, 32, 2)

        # shape representation
        self.pad1_shape = nn.ReflectionPad2d(1)
        self.conv1_shape = nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False)
        self.resblks_shape = _make_layer(ResBlock, 32, 32, 2)

    def forward(self, x):
        c = self.relu(self.conv1(self.pad1(x)))
        c = self.relu(self.conv2(self.pad2(c)))
        c = self.resblks(c)

        v = self.relu(self.conv1_text(self.pad1_text(c)))
        v = self.resblks_text(v)

        m = self.relu(self.conv1_shape(self.pad1_shape(c)))
        m = self.resblks_shape(m)

        return v, m  # v: texture, m: shape


class Manipulator(nn.Module):
    def __init__(self, num_resblk):
        super().__init__()
        self.convblks = _make_layer(ConvBlock, 32, 32, 1, kernel_size=7, stride=1)
        self.convblks_after = _make_layer(
            ConvBlockAfter, 32, 32, 1, kernel_size=3, stride=1
        )
        self.resblks = _make_layer(
            ResBlock, 32, 32, num_resblk, kernel_size=3, stride=1
        )

        # testing embedding manipulation
        g = gaussian(shape_x=180, shape_y=180, mu_x=0.0, mu_y=0.4, sig_x=0.7, sig_y=0.3)
        # import matplotlib.pyplot as plt

        self.attn_map = None
        self.gaussian_tensor = torch.from_numpy(g).float().repeat(1, 32, 1, 1)
        # print("self.gaussian_tensor.shape", self.gaussian_tensor.shape)

    def forward(self, x_a, x_b, amp, attn_map=None):
        diff = x_b - x_a
        diff = self.convblks(diff)

        if attn_map is None:
            attn_map = self.attn_map
        if attn_map is not None:
            scaled_attn_map = self._format_attn_map(attn_map, diff.size())
            diff = diff * scaled_attn_map.to(diff.device)

        diff = (amp - 1.0) * diff
        diff = self.convblks_after(diff)
        diff = self.resblks(diff)

        return x_b + diff

    def _format_attn_map(self, attn_map, size):
        n, c, h, w = size
        if attn_map.ndim == 3:
            attn_map = attn_map.unsqueeze(1)  # [num_frames, 1, h, w]
        elif attn_map.ndim == 2:
            attn_map = attn_map.unsqueeze(0).unsqueeze(0)
        scaled_attn_map = nn.functional.interpolate(attn_map, size=(h, w), mode='area')
        scaled_attn_map = scaled_attn_map.expand(*size)
        return scaled_attn_map

    def manip(self, x_a, amp, attn_map=None):
        diff = self.convblks(x_a)

        if attn_map is None:
            attn_map = self.attn_map
        if attn_map is not None:
            scaled_attn_map = self._format_attn_map(attn_map, diff.size())
            diff = diff * scaled_attn_map.to(diff.device)

        diff = (amp - 1.0) * diff
        diff = self.convblks_after(diff)
        diff = self.resblks(diff)

        return x_a + diff


class Decoder(nn.Module):
    def __init__(self, num_resblk):
        super().__init__()
        # texture
        self.upsample_text = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad_text = nn.ReflectionPad2d(1)
        self.conv1_text = nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # common blocks
        self.resblks = _make_layer(ResBlock, 64, 64, num_resblk)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False)
        self.pad2 = nn.ReflectionPad2d(3)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=7, stride=1, bias=False)

    def forward(self, v, m):  # v: texture, m: shape
        v = self.relu(self.conv1_text(self.pad_text(self.upsample_text(v))))

        c = torch.cat([v, m], 1)
        c = self.resblks(c)
        c = self.upsample(c)
        c = self.relu(self.conv1(self.pad1(c)))
        c = self.conv2(self.pad2(c))

        return c


class MagNet(nn.Module):
    def __init__(self, num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=9, amp=1.0):
        super().__init__()
        self.amp = amp
        self.encoder = Encoder(num_resblk=num_resblk_enc)
        self.manipulator = Manipulator(num_resblk=num_resblk_man)
        self.decoder = Decoder(num_resblk=num_resblk_dec)

        # initialize conv weights(xavier)
        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))

    def pre_process(self, x):
        return x / 127.5 - 1.0

    def post_process(self, o):
        o = o - o.min()
        o = o / o.max()
        o = o * 255
        return o

    def forward(self, x_a, x_b, x_c, amp):  # v: texture, m: shape
        v_a, m_a = self.encoder(x_a)
        v_b, m_b = self.encoder(x_b)
        v_c, m_c = self.encoder(x_c)

        m_enc = self.manipulator(m_a, m_b, amp)

        y_hat = self.decoder(v_b, m_enc)

        return y_hat, (v_a, m_a), (v_b, m_b), (v_c, m_c)

    def manipulate(self, x, amp=None, attn_map=None):
        if amp is None:
            amp = self.amp
        v, m = self.encoder(x)
        m_enc = self.manipulator.manip(m, amp, attn_map)
        y_hat = self.decoder(v, m_enc)
        return y_hat

    def manipulate_video(
        self,
        x,
        amp=None,
        attn_maps=None,
        mode='temporal',
        fh=1,
        fl=0.5,
        fs=30,
        pre_process=True,
        post_process=True,
    ):
        if pre_process:
            x = self.pre_process(x)

        if mode == 'temporal':
            # temporal mode (difference of IIR)
            # copy filter coefficients and follow codes from https://github.com/12dmodel/deep_motion_mag
            mag_frames = []
            filter_b = [fh - fl, fl - fh]
            filter_a = [-1.0 * (2.0 - fh - fl), (1.0 - fl) * (1.0 - fh)]

            if amp is None:
                amp = self.amp

            if attn_maps is None:
                attn_maps = [None] * x.size(2)

            x_state = []
            y_state = []

            mag_frames = [x[:, :, 0]]
            for i in range(1, x.size(2)):
                xb = x[:, :, i]  # [bs, 3, h, w] (i-th frame)

                vb, mb = self.encoder(xb)
                # TODO: Determine if "detach" needs to be removed
                x_state.insert(0, mb.detach())

                while len(x_state) < len(filter_b):
                    # TODO: Determine if "detach" needs to be removed
                    x_state.insert(0, mb.detach())
                if len(x_state) > len(filter_b):
                    x_state = x_state[: len(filter_b)]
                y = torch.zeros_like(mb)
                for i in range(len(x_state)):
                    y += x_state[i] * filter_b[i]
                for i in range(len(y_state)):
                    y -= y_state[i] * filter_a[i]

                # TODO: Determine if "detach" needs to be removed
                y_state.insert(0, y.detach())
                if len(y_state) > len(filter_a):
                    y_state = y_state[: len(filter_a)]

                mb_m = self.manipulator(0.0, y, amp, attn_map=attn_maps[i])
                mb_m += mb - y

                y_hat = self.decoder(vb, mb_m)
                mag_frames.append(y_hat)

            mag_frames = torch.stack(mag_frames, 2)
        if post_process:
            mag_frames = self.post_process(mag_frames)
        return mag_frames


if __name__ == '__main__':
    model = MagNet()
    print(model)
