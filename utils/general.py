import time 
def get_time():
    return time.strftime("%m-%d-%Y-%H-%M-%S")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)