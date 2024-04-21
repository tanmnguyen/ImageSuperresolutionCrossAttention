import torch 

# training configuration file 
batch_size = 4
hr_width = 256
hr_height = 256
epochs = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'
result_dir = "results"