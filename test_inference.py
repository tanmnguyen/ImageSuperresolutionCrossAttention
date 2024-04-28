import os 
import cv2 
import h5py 
import torch
import argparse

import matplotlib.pyplot as plt
from utils.networks import load_gen
from utils.batch import BatchHandler


output_dir = 'inference_output'
os.makedirs(output_dir, exist_ok=True)

def inference(batch_handler, model, img):
    # convert to tensor 
    img = batch_handler.lr_transform(img)
    img = img.unsqueeze(0)
    # inference 
    output = model(img)
    output = batch_handler.detransform(output)
    return output

def main(args):
    # load batch handler 
    batch_handler = BatchHandler(None, None)
    # loader generator model 
    model = load_gen(args.weight)

    lr_hdf5 = os.path.join(args.valid, 'lr_train.hdf5')
    hr_hdf5 = os.path.join(args.valid, 'hr_train.hdf5')

    with h5py.File(lr_hdf5, 'r') as lr_f, h5py.File(hr_hdf5, 'r') as hr_f:
        for key in lr_f.keys():
            lr = lr_f[key][()]
            hr = hr_f[key][()]
            # pred hr image 
            gen_hr = inference(batch_handler, model, lr)
            # resize low res image into high res dim
            lr = cv2.resize(lr, (gen_hr.shape[1], gen_hr.shape[0]), interpolation=cv2.INTER_CUBIC)

            # convert color 
            # save image using cv2 
            cv2.imwrite(os.path.join(output_dir, f'{key}_lr.png'), lr)
            cv2.imwrite(os.path.join(output_dir, f'{key}_hr.png'), hr)
            cv2.imwrite(os.path.join(output_dir, f'{key}_gen_hr.png'), gen_hr)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-valid',
                        '--valid',
                        required=True,
                        help='Path to valid directory')
    
    parser.add_argument('-weight',
                        '--weight',
                        required=True,
                        help='Path to weight file')
    
    args = parser.parse_args()
    main(args)