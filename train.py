import os
import time 
import torch 
import configs 
import argparse 
import torch.optim as optim

from utils.batch import BatchHandler
from torch.utils.data import DataLoader
from dataset import ImageSuperResDataset
from utils.steps import train_net, valid_net
from utils.general import get_time, count_params
from utils.io import logging, plot_learning_curve

from models.VGG import vgg
from models.Generator import Generator
from models.Discriminator import Discriminator

# format time to print month day year hour minute second
result_dir = os.path.join(configs.result_dir, f'{get_time()}')
os.makedirs(result_dir, exist_ok=True)

# define log file 
log_file = os.path.join(result_dir, 'log.txt')

def main(args):
    train_dataset = ImageSuperResDataset(
        lr_hdf5_file=os.path.join(args.data, 'lr_train.hdf5'),
        hr_hdf5_file=os.path.join(args.data, 'hr_train.hdf5')
    )

    valid_dataset = ImageSuperResDataset(
        lr_hdf5_file=os.path.join(args.data, 'lr_valid.hdf5'),
        hr_hdf5_file=os.path.join(args.data, 'hr_valid.hdf5')
    )

    train_batch_handler = BatchHandler(
        lr_hdf5_file=os.path.join(args.data, 'lr_train.hdf5'), 
        hr_hdf5_file=os.path.join(args.data, 'hr_train.hdf5')
    )

    valid_batch_handler = BatchHandler(
        lr_hdf5_file=os.path.join(args.data, 'lr_valid.hdf5'), 
        hr_hdf5_file=os.path.join(args.data, 'hr_valid.hdf5')
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=configs.batch_size, 
        collate_fn=train_batch_handler.collate_fn, 
        shuffle=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=configs.batch_size, 
        collate_fn=valid_batch_handler.collate_fn, 
        shuffle=False,
    )

    gen = Generator(noRRDBBlock=7).to(configs.device)
    disc = Discriminator().to(configs.device)

    logging(f'Generator:\n {gen}', log_file)
    logging(f'Discriminator:\n {disc}', log_file)
    logging(f'Device: {configs.device}', log_file)
    logging(f'Generator Parameters: {count_params(gen)}', log_file)
    logging(f'Discriminator Parameters: {count_params(disc)}', log_file)

    gen_optimizer = optim.Adam(gen.parameters(),lr=0.0002)
    disc_optimizer = optim.Adam(disc.parameters(),lr=0.0002)

    train_history, valid_history, opt_gen_loss = [], [], float('inf')
    for epoch in range(configs.epochs):
        train_history.append(train_net(train_dataloader, gen_optimizer, disc_optimizer, vgg, disc, gen))
        train_disc_loss, train_gen_loss = train_history[-1]['disc_loss'], train_history[-1]['gen_loss']
        logging(
            f'[Train] Epoch: {epoch + 1}/{configs.epochs} | Disc Loss: {train_disc_loss} | Gen Loss: {train_gen_loss}',
            log_file
        )

        valid_history.append(valid_net(valid_dataloader, vgg, disc, gen))
        valid_disc_loss, valid_gen_loss = valid_history[-1]['disc_loss'], valid_history[-1]['gen_loss']
        logging(
            f'[Valid] Epoch: {epoch + 1}/{configs.epochs} | Disc Loss: {valid_disc_loss} | Gen Loss: {valid_gen_loss}\n',
            log_file
        )

        if valid_history[-1]['gen_loss'] < opt_gen_loss:
            opt_gen_loss = valid_history[-1]['gen_loss']
            torch.save(gen.state_dict(), os.path.join(result_dir, f'best_gen_ep{epoch}.pth'))
            torch.save(disc.state_dict(), os.path.join(result_dir, f'best_disc_ep{epoch}.pth'))

    # plot learning curve 
    plot_learning_curve(train_history, valid_history, result_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # parser.add_argument('-config',
    #                     '--config',
    #                     required=True,
    #                     help='path to config file')
    
    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help='Path to data file')
    
    args = parser.parse_args()
    main(args)