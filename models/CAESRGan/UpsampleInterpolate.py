import torch.nn as nn
import torch.nn.functional as F

class UpsampleInterpolate(nn.Module):
    def __init__(self, upscale_factor, in_channels, out_channels):
        super().__init__()   
        self.upscale_factor = upscale_factor
        self.upconv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.upscale_factor)
        x = self.upconv(x)

        return x
        