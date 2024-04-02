import ast 
import h5py 
import numpy as np
from torch.utils.data import Dataset

class ImageSuperResDataset(Dataset):
    def __init__(self, lr_hdf5_file: str, hr_hdf5_file):
        self.lr_hdf5_file = lr_hdf5_file
        self.hr_hdf5_file = hr_hdf5_file
        # open files
        with h5py.File(self.lr_hdf5_file, 'r') as lr_hdf5_file, h5py.File(self.hr_hdf5_file, 'r') as hr_hdf5_file:
            self.keys = set(lr_hdf5_file.keys()) & set(hr_hdf5_file.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = list(self.keys)[idx]

        with h5py.File(self.lr_hdf5_file, 'r') as lr_hdf5_file, h5py.File(self.hr_hdf5_file, 'r') as hr_hdf5_file:
            lr_img = np.array(lr_hdf5_file[key])
            hr_img = np.array(hr_hdf5_file[key])
            return lr_img, hr_img
        
        # lr_img = np.array(self.lr_hdf5_file[key])
        # hr_img = np.array(self.hr_hdf5_file[key])

        return lr_img, hr_img