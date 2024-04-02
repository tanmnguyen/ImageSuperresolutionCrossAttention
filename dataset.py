import ast 
import h5py 
import numpy as np
from torch.utils.data import Dataset

class ImageSuperResDataset(Dataset):
    def __init__(self, lr_hdf5_file: str, hr_hdf5_file):
        # open files
        self.lr_hdf5_file = h5py.File(lr_hdf5_file, 'r')
        self.hr_hdf5_file = h5py.File(hr_hdf5_file, 'r')

        # intersection of keys
        self.keys = set(self.lr_hdf5_file.keys()) & set(self.hr_hdf5_file.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = list(self.keys)[idx]
        lr_img = np.array(self.lr_hdf5_file[key])
        hr_img = np.array(self.hr_hdf5_file[key])

        return lr_img, hr_img