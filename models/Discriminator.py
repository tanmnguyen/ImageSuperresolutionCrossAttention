import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, ResNet50_Weights

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.resnet(x)
    
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3,64,3,padding=1,bias=False)
#         self.conv2 = nn.Conv2d(64,64,3,stride=2,padding=1,bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64,128,3,padding=1,bias=False)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128,128,3,stride=2,padding=1,bias=False)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.conv5 = nn.Conv2d(128,256,3,padding=1,bias=False)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256,256,3,stride=2,padding=1,bias=False)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.conv7 = nn.Conv2d(256,512,3,padding=1,bias=False)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
#         self.bn8 = nn.BatchNorm2d(512)
#         self.fc1 = nn.Linear(512*16*16,1024)
#         self.fc2 = nn.Linear(1024,1)
#         self.drop = nn.Dropout(0.3)
        
#     def forward(self,x):
#         block1 = F.leaky_relu(self.conv1(x))
#         block2 = F.leaky_relu(self.bn2(self.conv2(block1)))
#         block3 = F.leaky_relu(self.bn3(self.conv3(block2)))
#         block4 = F.leaky_relu(self.bn4(self.conv4(block3)))
#         block5 = F.leaky_relu(self.bn5(self.conv5(block4)))
#         block6 = F.leaky_relu(self.bn6(self.conv6(block5)))
#         block7 = F.leaky_relu(self.bn7(self.conv7(block6)))
#         block8 = F.leaky_relu(self.bn8(self.conv8(block7)))
#         block8 = block8.view(-1,block8.size(1)*block8.size(2)*block8.size(3))
#         block9 = F.leaky_relu(self.fc1(block8))
# #         block9 = block9.view(-1,block9.size(1)*block9.size(2)*block9.size(3))
#         block10 = torch.sigmoid(self.drop(self.fc2(block9)))
#         return block9