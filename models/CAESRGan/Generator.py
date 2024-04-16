import torch 
import torch.nn as nn 
import torch.nn.functional as F

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from .CrossAttnEncoder import CrossAttnEncoder

# def make_layer(block, n_layers):
#     layers = []
#     for _ in range(n_layers):
#         layers.append(block())
#     return nn.Sequential(*layers)



class Generator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(Generator, self).__init__()
        # RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        # self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        # self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        # self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        backward_strides = [2, 2, 2]
        strides = [1] * (nb - len(backward_strides)) + backward_strides
        self.cross_attn_encoder = CrossAttnEncoder(
            in_nc, nf, nb, gc, strides=strides, pivot_layer=nb - len(backward_strides) - 1
        )

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.cross_attn_encoder(x)

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


# import torch 
# import torch.nn as nn 
# import torch.nn.functional as F

# from .CrossAttnEncoder import CrossAttnEncoder
# from .UpsamplePixShuffle import UpsamplePixShuffle
# from .UpsampleInterpolate import UpsampleInterpolate

# class Generator(nn.Module):
#     def __init__(self,in_channel = 3, out_channel = 3, latent_dim=64, noRRDBBlock = 23):
#         super().__init__()   
#         # encoder 

#         # backward strides: 1, 1, 1, .. 2, 2, 2
#         backward_strides = [2, 2, 2]

#         # set strides 
#         strides = [1] * (noRRDBBlock - len(backward_strides)) + backward_strides

#         # define cross attention encoder block 
#         self.cross_attn_encoder = CrossAttnEncoder(
#             in_channel, 
#             noRRDBBlock, 
#             latent_dim, 
#             strides=strides,
#             pivot_layer=noRRDBBlock - len(backward_strides) - 1,
#         )

#         self.upsample = nn.Sequential(
#             # UpsamplePixShuffle(2, latent_dim, latent_dim),
#             # UpsamplePixShuffle(2, latent_dim, latent_dim),
#             UpsampleInterpolate(2, latent_dim, latent_dim),
#             UpsampleInterpolate(2, latent_dim, latent_dim)
#         )

#         self.out_conv = nn.Sequential(
#             nn.Conv2d(latent_dim, 3, 3, 1, 1)
#         )
    
#     def forward(self, x):
#         # encode features 
#         ca_x = self.cross_attn_encoder(x)
       
#         # upsample features
#         x = self.upsample(ca_x)

#         out = self.out_conv(x)

#         return out
        