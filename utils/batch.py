import sys 
sys.path.append('../')

import h5py
import torch
import numpy as np 
import torchvision.transforms as transforms

torch.manual_seed(17)
class BatchHandler():
    def __init__(self, lr_hdf5_file: str, hr_hdf5_file: str):
        super().__init__()
        self.lr_hdf5_file = lr_hdf5_file
        self.hr_hdf5_file = hr_hdf5_file

        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def collate_fn(self, batch):
        with h5py.File(self.lr_hdf5_file, 'r') as lr_file, h5py.File(self.hr_hdf5_file, 'r') as hr_file:
            lr_imgs = [self.lr_transform(np.array(lr_file[key])) for key in batch]
            hr_imgs = [self.hr_transform(np.array(hr_file[key])) for key in batch]

            lr_imgs = torch.stack(lr_imgs)
            hr_imgs = torch.stack(hr_imgs)

            return lr_imgs, hr_imgs
    