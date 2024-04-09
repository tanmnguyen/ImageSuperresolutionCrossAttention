import torch.nn as nn

class UpsamplePixShuffle(nn.Module):
    def __init__(self, upscale_factor, in_channels, out_channels):
        super().__init__()   
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        return x
        