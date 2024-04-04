# ImageSuperresolutionCrossAttention
Image Superresolution with Cross Attention 

Note that the images are read and trained on cv2 BGR format 
## Install 
```bash 
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
## Prepare dataset 
```bash
python preparedata.py --data dataset/DIV2k
```
This command line creates `train.hdf5` and `valid.hdf5` file 
