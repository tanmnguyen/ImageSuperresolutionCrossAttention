import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .CrossAttnEncoder import CrossAttnEncoder
from .UpsamplePixShuffle import UpsamplePixShuffle
from .UpsampleInterpolate import UpsampleInterpolate
class Generator(nn.Module):
    def __init__(self,in_channel = 3, out_channel = 3, latent_dim=64, noRRDBBlock = 23):
        super().__init__()   
        # encoder 

        # backward strides: 1, 1, 1, .. 2, 2, 2
        backward_strides = [2, 2, 2]

        # set strides 
        strides = [1] * (noRRDBBlock - len(backward_strides)) + backward_strides

        # define cross attention encoder block 
        self.cross_attn_encoder = CrossAttnEncoder(
            in_channel, 
            noRRDBBlock, 
            latent_dim, 
            strides=strides,
            pivot_layer=noRRDBBlock - len(backward_strides) - 1,
        )

        self.upsample = nn.Sequential(
            # UpsamplePixShuffle(2, latent_dim, latent_dim),
            # UpsamplePixShuffle(2, latent_dim, latent_dim),
            UpsampleInterpolate(2, latent_dim, latent_dim),
            UpsampleInterpolate(2, latent_dim, latent_dim)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(latent_dim, 3, 3, 1, 1)
        )
    
    def forward(self, x):
        # encode features 
        ca_x = self.cross_attn_encoder(x)
       
        # upsample features
        x = self.upsample(ca_x)

        out = self.out_conv(x)

        return out
        