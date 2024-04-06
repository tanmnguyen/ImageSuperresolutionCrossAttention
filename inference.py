import cv2 
import h5py 
import torch
import argparse

from utils.batch import BatchHandler
from models.Generator import Generator


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
    model = Generator(noRRDBBlock=7)
    model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))

    # read image 
    with h5py.File(args.hdf5, 'r') as f:
        for key in f.keys():
            img = f[key][()]
            hr  = inference(batch_handler, model, img)

            cv2.imshow("low res", img)
            cv2.imshow("high res", hr)
            cv2.waitKey(0)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-hdf5',
                        '--hdf5',
                        required=True,
                        help='Path to hdf5 file')
    
    parser.add_argument('-weight',
                        '--weight',
                        required=True,
                        help='Path to weight file')
    
    args = parser.parse_args()
    main(args)