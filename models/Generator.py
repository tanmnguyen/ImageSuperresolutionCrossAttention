import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .ResidualInResidualDenseBlock import ResidualInResidualDenseBlock

class Generator(nn.Module):
    def __init__(self,in_channel = 3, out_channel = 3, noRRDBBlock = 23):
        super().__init__()   
        self.conv1 = nn.Conv2d(in_channel, 64, 3, 1, 1)

        RRDB_layer = []
        for _ in range(noRRDBBlock):
            RRDB_layer.append(ResidualInResidualDenseBlock())
        self.RRDB_block =  nn.Sequential(*RRDB_layer)

        self.RRDB_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv = nn.Conv2d(64, 64, 3, 1, 1)

        self.out_conv = nn.Conv2d(64, out_channel, 3, 1, 1)
    
    def forward(self, x):
        first_conv = self.conv1(x)
        RRDB_full_block = torch.add(self.RRDB_conv2(self.RRDB_block(first_conv)),first_conv)
        upconv_block1 = self.upconv(F.interpolate(RRDB_full_block, scale_factor = 2))
        upconv_block2 = self.upconv(F.interpolate(upconv_block1, scale_factor = 2))
        out = self.out_conv(upconv_block2)
        
        return out
        