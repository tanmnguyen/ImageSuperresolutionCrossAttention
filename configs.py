import torch 

# training configuration file 
batch_size = 8
hr_width = 256
hr_height = 256
epochs = 30
device = 'cuda' if torch.cuda.is_available() else torch.device('mps') 
result_dir = "results"