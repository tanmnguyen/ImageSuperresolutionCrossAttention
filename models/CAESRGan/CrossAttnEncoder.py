import sys 
sys.path.append("../")

import configs 
import torch
import torch.nn as nn 
from .RRDB import RRDB
from .CrossAttention import CrossAttention

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class CrossAttnEncoder(nn.Module):
    def __init__(self, in_nc, nf, nb, gc, strides, pivot_layer):
        super().__init__()
        self.pivot_layer = pivot_layer 
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = [RRDB(nf=nf, gc=gc, stride=strides[i]).to(configs.device) for i in range(nb)]

        cumulative_strides = strides 
        for i in range(1, len(strides)):
            cumulative_strides[i] *= cumulative_strides[i-1]
        dims = [int((nf ** 2) / (cumulative_strides[i] ** 2)) for i in range(len(cumulative_strides))]
        self.cross_attns = [CrossAttention(dims[0], dims[i+1], nf).to(configs.device) for i in range(len(dims) - 1)]

        self.attn_conv = nn.Conv2d(nf * (len(strides) - self.pivot_layer - 1), nf, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.conv_first(x)

        rrdb_out, prev = [], fea 
        for rrdb in self.RRDB_trunk:
            prev = rrdb(prev)
            rrdb_out.append(prev)
            

        # cross attention between features map 0 with 1, 2, 3...
        cross_attn_output = [] 
        for i in range(self.pivot_layer, len(self.cross_attns)):
            cross_attn_output.append(self.cross_attns[i](rrdb_out[self.pivot_layer], rrdb_out[i+1]))

        # stack the cross attention output along the channel dimension 
        cross_attn_output = torch.cat(cross_attn_output, dim=1)

        out = self.attn_conv(cross_attn_output)
        out = self.leaky_relu(out)
        out = self.trunk_conv(out)
        fea = fea + out

        return fea


# from .ResidualInResidualDenseBlock import ResidualInResidualDenseBlock

# class CrossAttnEncoder(nn.Module):
#     def __init__(self, in_channel, noRRDBBlock, latent_dim, strides, pivot_layer):
#         super().__init__()
#         self.pivot_layer = pivot_layer
#         self.conv1 = nn.Conv2d(in_channel, latent_dim, 3, 1, 1)

#         self.RRDB_layers = [ResidualInResidualDenseBlock(stride=strides[i]).to(configs.device) for i in range(noRRDBBlock)]

#         # cross attention layers 
#         cumulative_strides = strides 
#         for i in range(1, len(strides)):
#             cumulative_strides[i] *= cumulative_strides[i-1]

#         dims = [int((latent_dim ** 2) / (cumulative_strides[i] ** 2)) for i in range(len(cumulative_strides))]
#         self.cross_attns = [CrossAttention(dims[0], dims[i+1], latent_dim).to(configs.device) for i in range(len(dims) - 1)]

#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(latent_dim * (len(strides) - self.pivot_layer - 1), latent_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)
#         )

#     def forward(self, x):
#         first_conv = self.conv1(x)

#         # encode features 
#         RRDB_layer_output, prev = [], first_conv 
#         for RRDB_layer in self.RRDB_layers:
#             prev = RRDB_layer(prev)
#             RRDB_layer_output.append(prev)

#         # cross attention between features map 0 with 1, 2, 3...
#         cross_attn_output = [] 
#         for i in range(self.pivot_layer, len(self.cross_attns)):
#             cross_attn_output.append(self.cross_attns[i](RRDB_layer_output[self.pivot_layer], RRDB_layer_output[i+1]))

#         # stack the cross attention output along the channel dimension 
#         cross_attn_output = torch.cat(cross_attn_output, dim=1)

#         RRDB_conv2 = self.conv_layers(cross_attn_output)
#         RRDB_full_block = torch.add(RRDB_conv2, first_conv)

#         return RRDB_full_block