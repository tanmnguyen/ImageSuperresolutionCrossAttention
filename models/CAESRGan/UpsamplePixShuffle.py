import torch 
import torch.nn as nn

def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

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
        