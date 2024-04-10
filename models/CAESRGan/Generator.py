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

        # upsampler 
        self.upsample_pxshuffl1 = UpsamplePixShuffle(2, latent_dim, latent_dim)
        self.upsample_interpol1 = UpsampleInterpolate(2, latent_dim, latent_dim)

        self.upsample_pxshuffl2 = UpsamplePixShuffle(2, latent_dim, latent_dim)
        self.upsample_interpol2 = UpsampleInterpolate(2, latent_dim, latent_dim)

        # self.upsample1 = UpsampleInterpolate(2, latent_dim, latent_dim)
        # self.upsample2 = UpsampleInterpolate(2, latent_dim, latent_dim)
        # self.upsample2 = UpsamplePixShuffle(2, latent_dim, latent_dim)

        self.out_conv = nn.Conv2d(latent_dim, out_channel, 3, 1, 1)
    
    def forward(self, x):
        # encode features 
        RRDB_full_block = self.cross_attn_encoder(x)

        # upsample features
        # upconv_block1 = self.upsample1(RRDB_full_block)
        # upconv_block2 = self.upsample2(upconv_block1)

        up1 = self.upsample_pxshuffl1(RRDB_full_block)
        up2 = self.upsample_interpol1(up1)
        x = up1 + up2 

        up3 = self.upsample_pxshuffl2(x)
        up4 = self.upsample_interpol2(up3)
        x = up3 + up4

        out = self.out_conv(x)

        return out
        