#!/home/rafatmatting/anaconda3/envs/ml/bin/python3
import numpy
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# lrs = [1e-1, 1e-3, 1e-4, 1e-5]

for lr in numpy.linspace(.1,1,1):
    for net in ["MODNet"]:#"UNet", "DFM", "GFM", "MODNet"]:
        os.system(f"python train.py --model-type={net} --dataset-name=AMD --gpu-indices=0,1,2,3 --num-workers=16 --batch-size=16 --epochs=200 --version=resize --learning-rate={lr}")
# os.system(f"python train.py --model-type=DFM --dataset-name=AMD_cropped --gpu-indices=0,1,2,3 --num-workers=12 --batch-size=16 --epochs=60 --learning-rate=.01")