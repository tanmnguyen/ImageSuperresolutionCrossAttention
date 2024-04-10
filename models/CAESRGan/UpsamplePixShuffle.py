import torch.nn as nn
from icnr import ICNR

class UpsamplePixShuffle(nn.Module):
    def __init__(self, upscale_factor, in_channels, out_channels):
        super().__init__()   
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        weight = ICNR(self.conv1.weight, initializer=nn.init.kaiming_normal_,
              upscale_factor=upscale_factor)
        self.conv1.weight.data.copy_(weight)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        return x
        