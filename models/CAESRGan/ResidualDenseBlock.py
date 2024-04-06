import torch 
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self,in_channel = 64,inc_channel = 32, beta = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, inc_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channel + inc_channel, inc_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel + 2 * inc_channel, inc_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channel + 3 * inc_channel, inc_channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channel + 4 * inc_channel,  in_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU()
        self.b = beta
        
    def forward(self, x):
        block1 = self.lrelu(self.conv1(x))
        block2 = self.lrelu(self.conv2(torch.cat((block1, x), dim = 1)))
        block3 = self.lrelu(self.conv3(torch.cat((block2, block1, x), dim = 1)))
        block4 = self.lrelu(self.conv4(torch.cat((block3, block2, block1, x), dim = 1)))
        out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim = 1))
        
        return x + self.b * out
            