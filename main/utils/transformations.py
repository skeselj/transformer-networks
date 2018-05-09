# Houses functions that alter data

import numpy as np
import math, random

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def pad(img, m=0.125):
    h, w, _ = img.size()

    marg = int(m*w)
    new_w = w + 2*marg

    out = Variable(torch.zeros(h, new_w, new_w)).cuda()
    out[:, marg:marg+w, marg:marg+w] = img

    return out


def chop(img, pix=8):
    h, w, _ = img.size()

    side = random.randint(0,3)
    if pix == 0: return img
    if side == 0:
        img[:, :pix, :] = 0
    if side == 1:
        img[:, :, :pix] = 0
    if side == 2:
        img[:, w-pix:, :] = 0
    if side == 3:
        img[:, :, w-pix:] = 0

    return img


def rot90(im1, im2):
    if type(im1) == Variable: im1 = im1.data
    if type(im2) == Variable: im2 = im2.data

    k = random.randint(0,3)

    im1 = im1.squeeze().cpu().numpy()
    im1 = np.rot90(im1, k=k).copy()
    im1 = torch.Tensor(im1).unsqueeze(0)
    
    im2 = im2.squeeze().cpu().numpy()
    im2 = np.rot90(im2, k=k).copy()
    im2 = torch.Tensor(im2).unsqueeze(0)
    
    return im1, im2


def translate_rotate(img, disp=16, angle=2):
    """ displacement in pixels, angle in degrees"""
    h, w, _ = tuple(img.size())

    [dx, dy] = [random.uniform(-disp, disp)/w for i in range(2)]
    angle = random.uniform(-angle,angle) * math.pi / 180
    
    thetas = np.zeros((h, 2, 3))

    thetas[:,0,2] = dx
    thetas[:,1,2] = dy

    thetas[:,0,0] += np.cos(angle)
    thetas[:,1,0] -= np.sin(angle)
    thetas[:,0,1] += np.sin(angle)
    thetas[:,1,1] += np.cos(angle)
    
    thetas = Variable(torch.Tensor(thetas), requires_grad=False)   # H, 2, 3
    grid = F.affine_grid(thetas, torch.Size((h, 2, w, w)))
    res = F.grid_sample(img.unsqueeze(1), grid).squeeze()
    del grid, thetas

    return res.unsqueeze(0)


def tone_patch(img):
    h, w, _ = img.size()

    x, y = random.randint(0,w-1), random.randint(0,w-1),
    pw = random.randint(1, max(1, w-max(x,y)-1))

    if random.choice([0,1]) == 0:
        img[:, x:x+pw, y:y+pw] = img[:, x:x+pw, y:y+pw]  + (1-img[:, x:x+pw, y:y+pw])/2
    else:
        img[:, x:x+pw, y:y+pw] = img[:, x:x+pw, y:y+pw] / 2

    return img


def crack(img):
    if type(img) == Variable: img = img.data

    h, w, _ = img.size()
    if h != 1: raise ValueError("Height of image to be cracked must be 1")
    chunk = img[0].cpu().numpy()

    # this will be the image we output            
    im = np.random.uniform(low=0.0, high=1.0/255, size=chunk.shape)

    # decide on parameters of movement
    movements = [[0,+1],[0,-1],[+1,0]]
    odds = [1,2,3]  # pro of each movement
    random.shuffle(odds)
    odds[2] += 1   # give down a boost

    # find start & end (bounds) of crack in each row
    bounds = [[float('inf'),-float('inf')] for i in xrange(w)]
    margin = 0.125
    pi, pj = 0, random.randint(2*int(w*margin), 6*int(w*margin))
    while pi>=0 and pi<w and pj>=0 and pj<w:
        if pj < bounds[pi][0]: bounds[pi][0] = pj
        if pj > bounds[pi][1]: bounds[pi][1] = pj
        choice = random.randint(1,sum(odds))
        if choice <= odds[0]:
            pi, pj = pi+movements[0][0], pj+movements[0][1]
        elif choice <= odds[0]+odds[1]:
            pi, pj = pi+movements[1][0], pj+movements[1][1]
        else:
            pi, pj = pi+movements[2][0], pj+movements[2][1]
    i_bounds = 0
    while i_bounds < w and bounds[i_bounds] != [float('inf'), -float('inf')]:
        i_bounds += 1
    bounds = bounds[:i_bounds-1]

    # decide on distance and angle on each side of crack
    dist_r = np.random.uniform(4.0, 12.0)
    angle_r = np.random.uniform(-math.pi/4, math.pi/4)
    di_r, dj_r = int(dist_r*math.sin(angle_r)), int(dist_r*math.cos(angle_r))
    dist_l = np.random.uniform(4.0, 12.0)
    angle_l = np.random.uniform(math.pi*3/4, math.pi*5/4)
    di_l, dj_l = int(dist_l*math.sin(angle_l)), int(dist_l*math.cos(angle_l))

    # fill in the parts of the result image around the crack            
    for pi in range(len(bounds)):
        bi_l, bi_r = bounds[pi][0], bounds[pi][1]
        for pj in reversed(range(0,bi_l)):  # left 
            newi, newj = pi+di_l, pj+dj_l
            if newi<0 or newi>=w or newj<0 or newj>=w: break
            im[newi,newj] = chunk[pi,pj]
        for pj in range(bi_l+1,w):  # right 
            newi, newj = pi+di_r, pj+dj_r
            if newi<0 or newi>=w or newj<0 or newj>=w: break
            im[newi,newj] = chunk[pi,pj]

    # fill in what's below, if anything
    if bounds[-1][0] <= w/2:  # left
        di_b = di_r
        sj_lo, sj_hi = 0, w+dj_l
        tj_lo, tj_hi = -dj_l, w
    elif bounds[-1][1] > w/2:  # right    
        di_b = di_l
        sj_lo, sj_hi = dj_r, w
        tj_lo, tj_hi = 0, w-dj_r
    for pi in range(len(bounds),w):  # fill in 
        newi = pi+di_b
        if newi>=w: break
        im[newi, tj_lo:tj_hi] = chunk[pi, sj_lo:sj_hi]

    return torch.Tensor(im).unsqueeze(0)


