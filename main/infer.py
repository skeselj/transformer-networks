########################################################################################################
# Given a model, infers with it. This file is messy.
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
parser.add_argument('name', type=str)
parser.add_argument('--justtry', help='check to see if things work', action='store_true')
parser.add_argument('-nitr', type=int, default=1, required=False)
parser.add_argument('-sc', help='smoothness coef', type=float, default=3e+5, required=False)  
parser.add_argument('-gpu', type=str, default="2", required=False)
parser.add_argument('-dataroot', type=str, default="m2w1024n100.h5", required=False)
parser.add_argument('-modelfile', type=str, required=False,
                    default="logs/stacks/final_figures/all_aug/wild--disp16angle4--best/1e3/_model.pt")
args = parser.parse_args()

log_path = "logs/stacks/inference/" + args.name + "/"
if not os.path.isdir(log_path): os.makedirs(log_path)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sc = args.sc

from utils.providers import StackDset
dset = StackDset('data/stacks/' + args.dataroot)   # train and test same, for now
img_w = dset.W * 5/4   # width, accounting for padding
img_q = dset.H   # quantity aka height
print('')

from models.shared_pyramid import Transformer
model = Transformer(num_lvl=5, base_dim=img_w).cuda()
model.load_state_dict(torch.load(args.modelfile))

print('-- Setup took %ds' % int(time.time()-global_checkpoint))


# plotting utilities

from utils.helpers import save_fig, save_field

def draw_plot(loss_lists, names, itr):
    """ Save a plots of all losses """
    lines = []
    for i in range(len(loss_lists)):
        line, = plt.plot(loss_lists[i], label=names[i])
        lines.append(line)
    plt.legend(handles=lines)
    plt.xlabel('Iteration'); plt.ylabel('Loss')
    name = names[0] if len(names)==1 else ''
    plt.savefig(log_path + '_plot_' + str(itr) + "_" + name + '.png')
    plt.plot(); plt.clf()

def draw_main(source, target, pred, field, itr):
    """ Save the figures that define a sample: the source, target, pred, and field"""
    save_fig(source[0], log_path + str(itr) + '_source')
    save_fig(target[0], log_path + str(itr) + '_target')
    save_fig(pred[0], log_path + str(itr) + '_pred')
    save_field(field[0], log_path + str(itr) + '_field')
    save_fig(torch.abs(pred[0]-target[0]), log_path + str(itr) + '_mydiff')
    save_fig(torch.abs(source[0]-target[0]), log_path + str(itr) + '_ogdiff')

def draw_residuals(residuals, itr):
    """ Save the residuals """
    for i, res in enumerate(residuals):
        save_field(res[0], log_path + str(itr) + '_residual_' + str(i), isResidual=True)

def smooth_losses(losses, p=100):
    losses = np.array(losses)
    smoothed_losses = [losses[p*i:p*(i+1)] for i in range(len(losses)//p)]
    return smoothed_losses

# inference utilities

from utils.losses import mse_mutual, smo_l2d1
from utils.transformations import chop, pad, rot90, translate_rotate, crack, tone_patch
optimizer = torch.optim.Adam(model.parameters(), lr=1-4, betas=[0.9,0.999])

mse_losses, smo_losses = [], []
worst_loss = 0

def train_step(source, target):
    """ Run a sample pair through the net, forward only. """
    min_lvl = 0
    pred, field, residuals = model.forward(source, target, min_lvl)  # (B,W,W), (B,W,W,2), [(B,W,W,2)]
    
    mse_losses.append(mse_mutual(target, pred).data[0])
    smo_losses.append((args.sc * smo_l2d1(field)).data[0])
    loss = mse_mutual(target, pred) + sc * smo_l2d1(field)
    
    optimizer.zero_grad()

    return loss.data[0], pred, field, residuals

def train_itr(stack):
    """ Run all the pairs in a stack (both forward and backward ones. Apply transformations here.
    Return the average loss and the (pred, field, residuals) for the last pair that was run. """
    avg_loss = 0
    for si in range(2*img_q):
        reverse = (si >= img_q)
        si = si % (img_q-1)   # did this to make sure 100 in each stack
        source, target = stack[si].clone().unsqueeze(0), stack[si+1].clone().unsqueeze(0)
        if reverse: source, target = target, source

        disp, angle = 16, 4
        source, target = chop(source, pix=disp), chop(target, pix=disp)
        source = tone_patch(source)
        source = translate_rotate(source, disp=disp, angle=angle)
        source = crack(source); source, target = rot90(source, target)
        source, target = chop(source, pix=disp), chop(target, pix=disp)
        source, target = pad(source), pad(target)

        source, target = Variable(source.data, volatile=True), Variable(target.data, volatile=True)

        loss, pred, field, residuals = train_step(source.unsqueeze(0), target.unsqueeze(0))
        avg_loss += loss / (2*img_q)

        del source, target

    return avg_loss, source, target, pred, field, residuals


### main loop

model.eval()
itr = 0;
while itr < args.nitr:
    iter_checkpoint = time.time()

    stack = dset.get_stack(itr % dset.H)   # (H, W, W), np
    stack = torch.Tensor(stack).float() / 2
    avg_loss, source, target, pred, field, residuals = train_itr(stack)
    print('.... stack %d: avg_loss = %.5e, time = %.2fs' % (itr, avg_loss, time.time()-iter_checkpoint))

    draw_checkpoint = time.time()
    #draw_main(source, target, pred, field, itr)
    #draw_plot([mse_losses, smo_losses], ['mse', 'smo'], itr)
    #draw_plot([mse_losses], ['mse'], itr); draw_plot([smo_losses], ['smo'], itr)
    #if (itr+1) % 10 == 0: draw_residuals(residuals, itr)
    #print('.. drawing figures and plot, time = %.2fs' % (time.time()-draw_checkpoint))

    itr += 1

print('-- Entire program took %dm' % (int(time.time()-global_checkpoint)/60))

