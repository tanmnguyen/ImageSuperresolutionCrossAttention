import sys 
sys.path.append("../")

import configs 
import torch
import torch.nn as nn 

from .CrossAttention import CrossAttention
from .ResidualInResidualDenseBlock import ResidualInResidualDenseBlock

class CrossAttnEncoder(nn.Module):
    def __init__(self, in_channel, noRRDBBlock, latent_dim, strides):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, latent_dim, 3, 1, 1)

        self.RRDB_layers = [ResidualInResidualDenseBlock(stride=strides[i]).to(configs.device) for i in range(noRRDBBlock)]

        # cross attention layers 
        cumulative_strides = strides 
        for i in range(1, len(strides)):
            cumulative_strides[i] *= cumulative_strides[i-1]

        dims = [int((latent_dim ** 2) / (cumulative_strides[i] ** 2)) for i in range(len(cumulative_strides))]
        self.cross_attns = [CrossAttention(dims[0], dims[i+1], latent_dim).to(configs.device) for i in range(len(dims) - 1)]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(latent_dim * (len(strides) - 1), latent_dim, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)
        )

    def forward(self, x):
        first_conv = self.conv1(x)

        # encode features 
        RRDB_layer_output, prev = [], first_conv 
        for RRDB_layer in self.RRDB_layers:
            prev = RRDB_layer(prev)
            RRDB_layer_output.append(prev)

        # cross attention between features map 0 with 1, 2, 3...
        cross_attn_output = [] 
        for i in range(len(self.cross_attns)):
            cross_attn_output.append(self.cross_attns[i](RRDB_layer_output[0], RRDB_layer_output[i+1]))

        # stack the cross attention output along the channel dimension 
        cross_attn_output = torch.cat(cross_attn_output, dim=1)

        RRDB_conv2 = self.conv_layers(cross_attn_output)
        RRDB_full_block = torch.add(RRDB_conv2, first_conv)

        return RRDB_full_block