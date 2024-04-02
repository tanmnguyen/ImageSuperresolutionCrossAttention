# ImageSuperresolutionCrossAttention
Image Superresolution with Cross Attention 

Note that the images are read and trained on cv2 BGR format 
## Prepare dataset 
```bash
python preparedata.py --data dataset/DIV2k
```
This command line creates `train.hdf5` and `valid.hdf5` file 
