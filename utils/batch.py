import sys 
sys.path.append('../')

import torch 
import torchvision.transforms as transforms


torch.manual_seed(17)

hr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

hr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image):
    
    return image

def collate_fn(batch):
    # process low resolution images
    lr_images = [hr_transform(item[0]) for item in batch]
    # process high resolution images
    hr_images = [hr_transform(item[1]) for item in batch]

    # stack 
    lr_images = torch.stack(lr_images)
    hr_images = torch.stack(hr_images)

    return lr_images, hr_images