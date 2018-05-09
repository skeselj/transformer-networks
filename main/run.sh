#!/bin/bash

# These are things I need
# - saved feature maps
# - z-axis of aligned stack
# - unet head-to-head

# I am running the UNet with div by 10

# e.g.
# python train_pairs.py pairs/rigid/temp


#python train.py find_most_problematic-get_train_curves/5e3 -sc=5e3 -gpu=3 & 

#python infer.py test_run -gpu=3

#python train.py unet/1e1 -sc=1e1 -gpu=1 & 
python train.py unet/div100/1e2 -sc=1e2 -gpu=2 & 
#python train.py unet/1e3 -sc=1e3 -gpu=3 &

#python train.py unet/1e4 -sc=1e4 -gpu=0 & 
#python train.py unet/1e5 -sc=1e5 -gpu=1 & 


