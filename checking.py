import torch 
from models.Original.RRDBNet_arch import RRDBNet


model = RRDBNet(3, 3, 64, 23, gc=32)
lr_img = torch.rand(1, 3, 64, 64)
hr_img = model(lr_img)

print(hr_img.shape)