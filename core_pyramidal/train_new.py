########################################################################################################
# Train a pyramid network (on basic vertically cracked dataset)
# I copied this from Eric's directory and then removed the parts that didn't make sense
########################################################################################################

# the usual gang
import os, sys, time, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy, scipy.ndimage as img
from scipy.misc import imsave
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import numpy as np, cv2  # don't move this
import pdb  # python debugger

PINKY = 'gs://neuroglancer/pinky40_v11/image'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='dump', required=False)
parser.add_argument('--burn', help='only load one sample', action='store_true')
parser.add_argument('--epochs', type=int, default=10, required=False)
parser.add_argument('--batch_size', type=int, default=1, required=False)  # currently breaks if != 1
parser.add_argument('--n_levels', help='in the pyramid', type=int, default=1, required=False)
parser.add_argument('--width', help='of an image', type=int, default=256, required=False)
parser.add_argument('--lr', help='you know what it is', type=float, default=2e-5, required=False)
args = parser.parse_args()

log_period = 1 if args.burn else 100
log_path = "logs/" + args.name + "/"
if not os.path.isdir(log_path): os.makedirs(log_path)

# data
from providers import VerticalCrackDset
dset  = VerticalCrackDset()
loader = torch.utils.data.DataLoader(dataset=dset, batch_size=args.batch_size)

# model
from pyramid import PyramidTransformer
model = PyramidTransformer(size=args.n_levels, dim=args.width).cuda()
for p in model.parameters(): p.requires_grad = True
model.train()

# optim
from losses import basicLoss
criterion = basicLoss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

# loop
from analysis_helpers import save_image, save_field
itr = 0
print '\n========= TRAIN LOOP ========='
for e in range(args.epochs):
    print '------ Epoch', e, '-------'
    for frame, target in loader:
        scheduler.step()

        # turn the crank
        stack = torch.cat([frame.unsqueeze(1), target.unsqueeze(1)], dim=1)
        stack = torch.autograd.Variable(stack).cuda().float()
        frame = torch.autograd.Variable(frame).cuda().float()
        target = torch.autograd.Variable(target).cuda().float()
        pred, field, residuals = model.forward(stack)
        
        loss = criterion(target, pred, field)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # show and tell
        if itr % log_period == 0:
            print '... iteration', itr, '...'
            save_image(frame.data.cpu().numpy().squeeze(), log_path + 'frame' + str(itr))
            save_image(target.data.cpu().numpy().squeeze(), log_path + 'target' + str(itr))
            save_image(pred.data.cpu().numpy().squeeze(), log_path + 'pred' + str(itr))
            save_field(field.data.cpu().numpy().squeeze(), log_path + 'field' + str(itr))
        itr += 1

        if args.burn: break
    if args.burn: break
