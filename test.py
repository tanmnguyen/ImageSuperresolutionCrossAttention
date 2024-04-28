import os
import torch 
import configs 
import argparse 

from tqdm import tqdm 
from utils.general import get_time
from utils.networks import load_gen_from_ckpt
from utils.batch import BatchHandler
from torch.utils.data import DataLoader
from dataset import ImageSuperResDataset
from utils.metrics.quality import PSNR_fn

# format time to print month day year hour minute second
result_dir = os.path.join(configs.result_dir, f'{get_time()}')
os.makedirs(result_dir, exist_ok=True)

# define log file 
log_file = os.path.join(result_dir, 'log.txt')

def main(args):
    valid_dataset = ImageSuperResDataset(
        lr_hdf5_file=os.path.join(args.data, 'lr_valid.hdf5'),
        hr_hdf5_file=os.path.join(args.data, 'hr_valid.hdf5')
    )

    valid_batch_handler = BatchHandler(
        lr_hdf5_file=os.path.join(args.data, 'lr_valid.hdf5'), 
        hr_hdf5_file=os.path.join(args.data, 'hr_valid.hdf5')
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=configs.batch_size, 
        collate_fn=valid_batch_handler.collate_fn, 
        shuffle=False,
    )

    # list all folder starts with epoch_
    epoch_folders = [folder for folder in os.listdir(args.checkpoint) if folder.startswith('epoch_')]
    epoch_folders.sort(key=lambda x: int(x.split('_')[-1]))
    for epoch_folder in epoch_folders:
        gen_model = load_gen_from_ckpt(args.checkpoint, epoch_folder)
        print("Validating", epoch_folder)

        avg_pnsr = 0.0
        for i, (lr_img, hr_img) in enumerate(tqdm(valid_dataloader)):
            lr_img = lr_img.to(configs.device)
            hr_img = hr_img.to(configs.device)

            with torch.no_grad():
                fake_data = gen_model(lr_img)
                # compute PSNR
                psnr = PSNR_fn(fake_data, hr_img)
                # add to avg_pnsr
                avg_pnsr += psnr.item()

        avg_pnsr /= len(valid_dataloader)
        print(f'Epoch: {epoch_folder} | PSNR: {avg_pnsr}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help='Path to data file')
    
    parser.add_argument('-checkpoint',
                        '--checkpoint',
                        required=True,
                        help='Path to checkpoint directory')
    
    args = parser.parse_args()
    main(args)