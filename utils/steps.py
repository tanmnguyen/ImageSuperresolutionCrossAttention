import numpy as np

import torch
from tqdm import tqdm
from utils.io import logging
from utils.metrics.Losses import Losses

def train_net(train_dataloader, gen_optimizer, disc_optimizer, vgg, disc, gen, log_file):
    disc.train()
    gen.train()

    epoch_disc, epoch_gen = 0.0, 0.0
    for i, (lr_img, hr_img) in enumerate(tqdm(train_dataloader)):
        disc_loss, gen_loss = Losses().calculateLossWithGrad(
            disc_optimizer=disc_optimizer, 
            gen_optimizer=gen_optimizer, 
            vgg=vgg, 
            discriminator=disc, 
            generator=gen, 
            LR_image=lr_img, 
            HR_image=hr_img
        )
        epoch_disc += disc_loss.detach().cpu()
        epoch_gen += gen_loss.detach().cpu()

        if i % 100 == 0:
            logging(f"Discriminator Loss: {epoch_disc / (i + 1)} | Generator Loss: {epoch_gen / (i + 1)}", log_file)

    return {
        "disc_loss": epoch_disc / len(train_dataloader),
        "gen_loss": epoch_gen / len(train_dataloader)
    }

def valid_net(valid_dataloader, vgg, disc, gen):
    disc.eval()
    gen.eval()

    epoch_disc, epoch_gen = 0.0, 0.0
    for lr_img, hr_img in tqdm(valid_dataloader):
        with torch.no_grad():
            disc_loss, gen_loss = Losses().calculateLoss(
                vgg=vgg, 
                discriminator=disc, 
                generator=gen, 
                LR_image=lr_img, 
                HR_image=hr_img
            )
            epoch_disc += disc_loss.detach().cpu()
            epoch_gen += gen_loss.detach().cpu()

    return {
        "disc_loss": epoch_disc / len(valid_dataloader),
        "gen_loss": epoch_gen / len(valid_dataloader)
    }