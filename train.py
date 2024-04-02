import os
import configs 
import argparse 

from utils.batch import collate_fn
from torch.utils.data import DataLoader
from dataset import ImageSuperResDataset

def main(args):
    train_dataset = ImageSuperResDataset(
        lr_hdf5_file=os.path.join(args.data, "lr_train.hdf5"),
        hr_hdf5_file=os.path.join(args.data, "hr_train.hdf5")
    )

    valid_dataset = ImageSuperResDataset(
        lr_hdf5_file=os.path.join(args.data, "lr_valid.hdf5"),
        hr_hdf5_file=os.path.join(args.data, "hr_valid.hdf5")
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=configs.batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=configs.batch_size, 
        collate_fn=collate_fn, 
        shuffle=False,
    )

    for lr_img, hr_img in train_dataloader:
        print(lr_img.shape, hr_img.shape)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # parser.add_argument('-config',
    #                     '--config',
    #                     required=True,
    #                     help="path to config file")
    
    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="Path to data file")
    
    args = parser.parse_args()
    main(args)