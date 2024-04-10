import torch 

def PSNR_fn(img1, img2, max_pixel = 1.0):
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return torch.mean(psnr)

# img1 = torch.rand(5, 3, 256, 256)
# img2 = torch.rand(5, 3, 256, 256)

# print(psnr(img1, img2))