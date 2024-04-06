import torch 
import torch.nn as nn

from .ResidualDenseBlock import ResidualDenseBlock

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 32, beta = 0.2):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channel, out_channel)
        self.b = beta
    
    def forward(self, x):
        out = self.RDB(x)
        out = self.RDB(out)
        out = self.RDB(out)
        
        return x + self.b * out
        