

import torch


class MagNet3Frames(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MagNet3Frames, self).__init__()


    def _encoder(self, image):
        enc = res_encoder(image,
                            layer_dims=self.encoder_dims,
                            num_resblk=self.num_enc_resblk)

        texture_enc = enc
        shape_enc = enc

        if self.use_texture_conv:
            stride = 2 if self.texture_downsample else 1
            texture_conv = torch.nn.Conv2d(texture_enc.shape[1], self.texture_dims, kernel_size=3, stride=stride, activation='relu')

        else:
            assert self.texture_dims == self.encoder_dims, \
                "Texture dim ({}) must match encoder dim ({}) " \
                "if texture_conv is not used.".format(self.texture_dims,
                                                      self.encoder_dims)
            assert not self.texture_downsample, \
                "Must use texture_conv if texture_downsample."

        if self.use_shape_conv:
            shape_conv = torch.nn.Conv2d(shape_enc, self.shape_dims, 3, 1, activation='relu')

        else:
            assert self.shape_dims == self.encoder_dims, \
                "Shape dim ({}) must match encoder dim ({}) " \
                "if shape_conv is not used.".format(self.shape_dims,
                                                    self.encoder_dims)

        for i in range(self.num_texture_resblk):
            texture_enc = residual_block(texture_enc, self.texture_dims, 3, 1)

        for i in range(self.num_shape_resblk):
            shape_enc = residual_block(shape_enc, self.shape_dims, 3, 1)

        return texture_enc, shape_enc


    def _decoder(self, texture_enc, shape_enc):
        if self.texture_downsample:
            texture_enc = torch.nn.functional.interpolate(texture_enc, texture_enc.shape[1:3]*2)
            texture_enc = torch.nn.pad(texture_enc)
            texture_enc = torch.nn.Conv2d(texture_enc, self.texture_dims, 3, 1, padding='VALID', activation='relu')

        enc = torch.cat((texture_enc, shape_enc), dim=3)

        return res_decoder(enc,
                           layer_dims=self.decoder_dims,
                           out_channels=self.n_channels,
                           num_resblk=self.num_dec_resblk)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
