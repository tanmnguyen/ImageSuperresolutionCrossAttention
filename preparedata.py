# This file prepare the dataset from raw data image
import os 
import ast 
import cv2 
import h5py
import json
import argparse 
import numpy as np
from tqdm import tqdm

def get_cropped(image, resolution: int, stride: int):
    ys = list(range(0, image.shape[0] - resolution, stride))
    xs = list(range(0, image.shape[1] - resolution, stride))
    if ys[-1] + resolution < image.shape[0]:
        ys.append(image.shape[0] - resolution)
    if xs[-1] + resolution < image.shape[1]:
        xs.append(image.shape[1] - resolution)

    cropped_images = []
    for y in ys:
        for x in xs:
            cropped_images.append(image[y:y + resolution, x:x + resolution])

    return cropped_images

def build_dataset(data_dir, hr: int, stride: int, downfactor: int, lr_hdf5_file: str, hr_hdf5_file: str):
    # read high res image paths 
    img_dir = os.path.join(data_dir)
    # high res image files 
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # sort files 
    img_files.sort()

    with h5py.File(f"{lr_hdf5_file}", 'w') as lr_f, h5py.File(hr_hdf5_file, 'w') as hr_f:    
        for fname in tqdm(img_files, desc=f"Building dataset for {data_dir}"):
            img = cv2.imread(os.path.join(img_dir, fname))
            cropped_images = get_cropped(img, hr, stride)
            for i, hr_img in enumerate(cropped_images):
                lr_img = cv2.resize(hr_img, (hr // downfactor, hr // downfactor), interpolation=cv2.INTER_CUBIC)
                lr_f.create_dataset(f"{fname}_{i}", data=lr_img)
                hr_f.create_dataset(f"{fname}_{i}", data=hr_img)



def main(args):
    hr = args.resolution
    stride = args.stride 

    # prepare data for train set 
    build_dataset(
        data_dir=os.path.join(args.data, "train"), 
        hr=hr,
        stride=stride,
        downfactor=args.downfactor,
        lr_hdf5_file=os.path.join(args.data, "lr_train.hdf5"),
        hr_hdf5_file=os.path.join(args.data, "hr_train.hdf5")
    )

    # prepare data for valid set
    build_dataset(
        data_dir=os.path.join(args.data, "valid"), 
        hr=hr,
        stride=stride,
        downfactor=args.downfactor,
        lr_hdf5_file=os.path.join(args.data, "lr_valid.hdf5"),
        hr_hdf5_file=os.path.join(args.data, "hr_valid.hdf5")
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help=   "Path to data file. This should contains two folders: train and valid.\
                                Each folder should contains ground truth images")
    
    parser.add_argument('-resolution',
                        '--resolution',
                        type=int,
                        default=256,
                        required=False,
                        help="Cropped resolution for the cropped images. Default is 256.")
    
    parser.add_argument('-downfactor',
                        '--downfactor',
                        type=int,
                        default=4,
                        required=False,
                        help="Down factor for the low resolution images. Default is 4.")
    
    parser.add_argument('-stride',
                        '--stride',
                        type=int,
                        default=256,
                        required=False,
                        help="Stride factor for the cropped images. Default is 128.")
    
    args = parser.parse_args()
    main(args)
