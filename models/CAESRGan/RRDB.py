import torch.nn as nn 
from .ResidualDenseBlock_5C import ResidualDenseBlock_5C

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, stride=1):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        self.maxpool = nn.MaxPool2d(stride)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = out * 0.2 + x
        out = self.maxpool(out)
        return out