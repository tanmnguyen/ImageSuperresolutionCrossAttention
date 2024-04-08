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
        # set strides to 1 except for the last 2 layers 
        strides = [1] * (noRRDBBlock - 2) + [2, 2]
        self.cross_attn_encoder = CrossAttnEncoder(
            in_channel, 
            noRRDBBlock, 
            latent_dim, 
            strides=strides,
            pivot_layer=noRRDBBlock - 3,
        )

        # upsampler 
        # self.upsample1 = UpsamplePixShuffle(2, latent_dim, latent_dim)
        self.upsample1 = UpsampleInterpolate(2, latent_dim, latent_dim)
        # self.upsample2 = UpsampleInterpolate(2, latent_dim, latent_dim)
        self.upsample2 = UpsamplePixShuffle(2, latent_dim, latent_dim)

        
        self.out_conv = nn.Conv2d(latent_dim, out_channel, 3, 1, 1)
    
    def forward(self, x):
        # encode features 
        RRDB_full_block = self.cross_attn_encoder(x)

        # upsample features
        upconv_block1 = self.upsample1(RRDB_full_block)
        upconv_block2 = self.upsample2(upconv_block1)
        out = self.out_conv(upconv_block2)

        return out
        