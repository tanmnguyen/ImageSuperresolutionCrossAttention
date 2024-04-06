import torch 
import torch.nn as nn

from .ResidualDenseBlock import ResidualDenseBlock

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 32, beta = 0.2, stride = 2):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(in_channel, out_channel)
        self.RDB2 = ResidualDenseBlock(in_channel, out_channel)
        self.RDB3 = ResidualDenseBlock(in_channel, out_channel)
        self.b = beta
        self.maxpool = nn.MaxPool2d(stride)
    
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        x = x + self.b * out
        x = self.maxpool(x)
        return x
        