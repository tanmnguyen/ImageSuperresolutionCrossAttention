import sys 
sys.path.append("../")

import configs 
from torchvision import models
from torchvision.models.vgg import VGG19_Weights

vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).to(configs.device)