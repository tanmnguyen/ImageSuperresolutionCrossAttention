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
        self.conv1 = nn.Conv2d(in_channel, 64, 3, 1, 1)

        self.RRDB_layers = [ResidualInResidualDenseBlock(stride=strides[i]).to(configs.device) for i in range(noRRDBBlock)]

        # cross attention layers 
        cumulative_strides = strides 
        for i in range(1, len(strides)):
            cumulative_strides[i] *= cumulative_strides[i-1]

        dims = [int((64 ** 2) / (cumulative_strides[i] ** 2)) for i in range(len(cumulative_strides))]
        self.cross_attns = [CrossAttention(dims[i], dims[i+1], 64).to(configs.device) for i in range(len(dims) - 1)]

        # self.RRDB_block = nn.Sequential(*RRDB_layer)
        self.RRDB_conv2 = nn.Conv2d(64, latent_dim, 3, 1, 1)

    def forward(self, x):
        first_conv = self.conv1(x)

        # encode features 
        RRDB_layer_output, prev = [], first_conv 
        for RRDB_layer in self.RRDB_layers:
            prev = RRDB_layer(prev)
            RRDB_layer_output.append(prev)

        # cross attention in reverse 
        for i in range(len(self.cross_attns) - 1):
            RRDB_layer_output[-2-i] = self.cross_attns[-1-i](RRDB_layer_output[-2-i], RRDB_layer_output[-1-i])
            
        CA_RRDB_block = RRDB_layer_output[0]
        RRDB_conv2 = self.RRDB_conv2(CA_RRDB_block)
        RRDB_full_block = torch.add(RRDB_conv2, first_conv)

        return RRDB_full_block