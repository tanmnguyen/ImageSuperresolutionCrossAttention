import ast 
import h5py 
import numpy as np
from torch.utils.data import Dataset

class ImageSuperResDataset(Dataset):
    def __init__(self, lr_hdf5_file: str, hr_hdf5_file: str):
        # open files
        with h5py.File(lr_hdf5_file, 'r') as lr_f, h5py.File(hr_hdf5_file, 'r') as hr_f:
            self.keys = set(lr_f.keys()) & set(hr_f.keys())

    def __len__(self):
        return 100
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = list(self.keys)[idx]
        return key 