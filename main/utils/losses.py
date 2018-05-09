# Define different loss functions

import math

import torch
from torch.autograd import Variable


def mse(im1, im2):   # b, w, w
    """ average MSE between all pixel pairs, includin 0 """
    w = im1.size()[-1]
    squerrs = (im1-im2)**2
    return torch.sum(squerrs) / (w*w)

def mse_mutual(im1, im2):
    """ average MSE between all pixel pairs that don't include 0 """
    mask = ( (im1 != 0) & (im2 != 0) ).float()
    squerrs = (im1-im2)**2
    return (mask*squerrs).sum() / mask.sum()

def mse_burn(im1, im2, k=100):
    """ average MSE where the difference between a zero-pixel and a non-zero pixel is:
        (1) scaled by k  (2) not squared (only abs) """
    errs = (im1-im2).abs()
    squerrs = errs**2
    n0_map = torch.eq(im1, 0) + torch.eq(im2, 0)
    valids = torch.eq(n0_map, 0).float()
    deviants = torch.eq(n0_map, 1).float()
    return ( (valids*squerrs).sum() + k*(deviants*errs).sum() ) / valids.sum()


def smo_mesh(f):
    """ energy in system of springs that start with identity grid """
    w = f.size()[2]
    nx = (f[:,1:,:,:]-f[:,:w-1,:,:]).norm(p=2, dim=3)
    ny = (f[:,:,1:,:]-f[:,:,:w-1,:]).norm(p=2, dim=3)
    ex = (nx - 1)**2
    ey = (ny - 1)**2
    return (ex.sum() + ey.sum()) / (w*w)
    

def smo_l1d1(f):
    """ smoothness error that encorporates l1 norm between all direct neighbors """
    w = f.size()[2]
    dx1 = torch.sum( (f[:,1:,:,:] - f[:,:w-1,:,:]).abs() )
    dy1 = torch.sum( (f[:,:,1:,:] - f[:,:,:w-1,:]).abs() )
    return (dx1+dy1) / (w*w)

def smo_l2d1(f):
    """ smoothness error that encorporates l1 norm between all direct neighbors """
    w = f.size()[2]
    dx1 = torch.sum( (f[:,1:,:,:] - f[:,:w-1,:,:])**2 )
    dy1 = torch.sum( (f[:,:,1:,:] - f[:,:,:w-1,:])**2 )
    return (dx1+dy1) / (w*w)

def smo_l2d2(f):   # b, w, w, 2
    """ smoothness error that encorporates all neighbors <= 2 units away and scales them by 1/dist"""
    w = f.size()[2]
    dx1 = torch.sum( (f[:,1:,:,:] - f[:,:w-1,:,:])**2 )
    dx2 = torch.sum( (f[:,2:,:,:] - f[:,:w-2,:,:])**2 )
    dy1 = torch.sum( (f[:,:,1:,:] - f[:,:,:w-1,:])**2 )
    dy2 = torch.sum( (f[:,:,2:,:] - f[:,:,:w-2,:])**2 )
    dxy1 = torch.sum( (f[:,1:,1:,:] - f[:,:w-1,:w-1,:])**2 )
    s = 2*(dx1+dy1) + (dx2+dy2) + math.sqrt(2)*dxy1
    return s / (w*w)


def l2_norm(f):
    _, w, h, t = tuple(f.size())
    torch.sum(f * f) / (w * h * t)
