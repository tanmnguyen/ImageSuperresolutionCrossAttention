import torch 
from models.Generator import Generator

print("create dummy input")
inp = torch.randn(8, 3, 64, 64)

print("create Generator model")
gen = Generator(noRRDBBlock=2)

print("forward pass")
out = gen(inp)

print("output shape:", out.shape)

# to cuda 
gen = gen.to('cuda')
inp = inp.to('cuda')

print("forward pass")
out = gen(inp)

print("output shape:", out.shape)