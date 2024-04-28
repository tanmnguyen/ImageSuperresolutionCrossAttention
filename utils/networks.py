import sys 
sys.path.append("../")

import os 
import torch
import configs 

def load_gen_from_ckpt(checkpoint, epoch_folder):
    epoch_dir = os.path.join(checkpoint, epoch_folder)
    files = os.listdir(epoch_dir)
    for file in files:
        if file.startswith('gen'):
            try:
                # load state dict 
                from models.ESRGan.Generator import Generator
                gen = Generator(3, 3, 64, 23, gc=32).to(configs.device)
                gen.load_state_dict(torch.load(os.path.join(epoch_dir, file), map_location=configs.device))
            except:
                from models.CAESRGan.Generator import Generator
                gen = Generator(3, 3, 64, 5, gc=32).to(configs.device)
                gen.load_state_dict(torch.load(os.path.join(epoch_dir, file), map_location=configs.device))
            # return generator
            return gen 
        
def load_gen(weight_path):
    try:
        from models.ESRGan.Generator import Generator
        gen = Generator(3, 3, 64, 23, gc=32).to(configs.device)
        gen.load_state_dict(torch.load(weight_path, map_location=configs.device))
    except:
        from models.CAESRGan.Generator import Generator
        gen = Generator(3, 3, 64, 5, gc=32).to(configs.device)
        gen.load_state_dict(torch.load(weight_path, map_location=configs.device))
    return gen 