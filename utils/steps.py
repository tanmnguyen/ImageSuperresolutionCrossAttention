import numpy as np

from tqdm import tqdm
from utils.metrics.Losses import Losses

def train_net(train_dataloader, gen_optimizer, disc_optimizer, vgg, disc, gen):
    epoch_disc_losses, epoch_gen_losses = [], []
    for lr_img, hr_img in tqdm(train_dataloader):
        disc_loss, gen_loss = Losses().calculateLossWithGrad(
            disc_optimizer=disc_optimizer, 
            gen_optimizer=gen_optimizer, 
            vgg=vgg, 
            discriminator=disc, 
            generator=gen, 
            LR_image=lr_img, 
            HR_image=hr_img
        )
        epoch_disc_losses.append(disc_loss.detach())
        epoch_gen_losses.append(gen_loss.detach())

    return {
        "disc_loss": np.mean(epoch_disc_losses),
        "gen_loss": np.mean(epoch_gen_losses)
    }

def valid_net(valid_dataloader, vgg, disc, gen):
    epoch_disc_losses, epoch_gen_losses = [], []
    for lr_img, hr_img in tqdm(valid_dataloader):
        disc_loss, gen_loss = Losses().calculateLoss(
            vgg=vgg, 
            discriminator=disc, 
            generator=gen, 
            LR_image=lr_img, 
            HR_image=hr_img
        )
        epoch_disc_losses.append(disc_loss.detach())
        epoch_gen_losses.append(gen_loss.detach())

    return {
        "disc_loss": np.mean(epoch_disc_losses),
        "gen_loss": np.mean(epoch_gen_losses)
    }