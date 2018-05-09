########################################################################################################
# Generate a standard stack of deformed images. So messy.
########################################################################################################

import os, sys, h5py
import math, random, time, collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

import numpy as np, cv2  # don't move this

global_checkpoint = time.time()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default="3", required=False)
parser.add_argument('-dataroot', type=str, default="m2w1024n100.h5", required=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from utils.providers import StackDset
dset = StackDset('data/stacks/' + args.dataroot)   # train and test same, for now
img_w = dset.W * 5/4   # width, accounting for padding
img_q = dset.H   # quantity aka height
print('')

from utils.helpers import save_fig, save_field, save_gif
from utils.transformations import chop, pad, rot90, translate_rotate, crack, tone_patch


### main loop

itr = 0;
while itr < 100:

    print(itr)

    stack = dset.get_stack(itr % dset.H)   # (H, W, W), np
    stack = torch.Tensor(stack).float() / 255

    new_stack = torch.zeros((50,1280,1280))

    for si in range(img_q):
        source = stack[si].unsqueeze(0)

        if si % 2 == 0:
            disp, angle = 16, 4
            source = chop(source, pix=disp)
            source = tone_patch(source)
            source = translate_rotate(source, disp=disp, angle=angle)
            source = crack(source)
            source = chop(source, pix=disp)
        source = pad(source)

        new_stack[si] = source.data * 255


    new_stack = new_stack.cpu().numpy()

    with h5py.File("standard_all_aug_dset.h5", "a") as f:
        f.create_dataset(str(itr), data = new_stack)
    
    if itr == 50:  
        save_gif(new_stack, "standard_all_aug_sample.gif") 


    itr += 1

    
