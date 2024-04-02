import os
import configs 
import argparse 
import torch.optim as optim

from utils.batch import collate_fn
from torch.utils.data import DataLoader
from dataset import ImageSuperResDataset

from models.VGG import vgg
from models.Generator import Generator
from utils.metrics.Losses import Losses
from models.Discriminator import Discriminator

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

    gen = Generator().to(configs.device)
    disc = Discriminator().to(configs.device)

    gen_optimizer = optim.Adam(gen.parameters(),lr=0.0002)
    disc_optimizer = optim.Adam(disc.parameters(),lr=0.0002)

    for lr_img, hr_img in train_dataloader:
        disc_loss, gen_loss = Losses().calculateLoss(
            disc_optimizer=disc_optimizer, 
            gen_optimizer=gen_optimizer, 
            vgg=vgg, 
            discriminator=disc, 
            generator=gen, 
            LR_image=lr_img, 
            HR_image=hr_img
        )

        print(disc_loss, gen_loss)
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