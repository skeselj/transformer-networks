########################################################################################################
# Define the a SpyNet like model
########################################################################################################

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample  # in (N x C x IH x IW), out (N x OH x OW x 2)


def get_identity_grid(dim):
    gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
    I = np.stack(np.meshgrid(gx, gy))   # (2, dim, dim)
    I = np.expand_dims(I, 0)   # (1, 2, dim, dim)
    I = Variable(torch.Tensor(I)).cuda()
    I = I.permute(0,2,3,1)   # (1, dim, dim, 2)
    return I


# single level
class Level(nn.Module):
    def __init__(self, k=5, act=nn.ReLU()):
        super(Level, self).__init__()
        p = (k-1)/2

        self.convS1 = nn.Conv2d( 1, 32, k, padding=p) 
        self.convS2 = nn.Conv2d(32, 64, k, padding=p)

        self.seqS = nn.Sequential(self.convS1, act,    # S - separate
                                  self.convS2, act)

        self.convJ1 = nn.Conv2d(128, 64, k, padding=p)
        self.convJ2 = nn.Conv2d( 64, 32, k, padding=p)
        self.convJ3 = nn.Conv2d( 32 , 2, k, padding=p)
        self.convJ3.weight.data /= 10; self.convJ3.bias.data /= 10

        self.seqJ = nn.Sequential(self.convJ1, act,    # J - joined
                                  self.convJ2, act,
                                  self.convJ3)

    def forward(self, source_maps, target_maps):  # each: (B, F, W, W)
        source_maps = self.seqS(source_maps)
        target_maps = self.seqS(target_maps)
        out = self.seqJ(torch.cat([source_maps, target_maps], 1))   # (B, 2, W, W)
        return out.permute(0,2,3,1)   # (B, W, W, 2)


# pyramid 
class Pyramid(nn.Module):
    def __init__(self, num_lvl, min_lvl, base_dim):
        super(Pyramid, self).__init__()
        print('-- Building PyramidNet with %d levels' % num_lvl)
        self.num_lvl = num_lvl
        self.min_lvl = min_lvl

        self.f_up = lambda x: nn.Upsample(scale_factor=x, mode='bilinear')
        self.up = self.f_up(2)
        self.down = nn.AvgPool2d(2, 2)

        self.Levels = nn.ModuleList([Level() for level in xrange(num_lvl-min_lvl)])
        self.sup_I = get_identity_grid(base_dim // 2**num_lvl) 

    def forward(self, source, target, lvl):   # stack: B x 2 x _ x _
        # one step above the top, we just get the identity
        if lvl == self.num_lvl:
            I = self.sup_I.repeat(source.size()[0], 1, 1, 1)  # B x 2 x _ x _
            return I, [ I ]

        # otherwise, take what's above and do work if you're included in the pyramid
        field_so_far, residuals_so_far = self.forward(self.down(source), self.down(target), lvl+1)
        field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)   # B x _ x _ x 2
        updated_source = grid_sample(source, field_so_far)
        if lvl >= self.min_lvl:
            residual = self.Levels[lvl-self.min_lvl].forward(updated_source, target) # B x W x W x 2
        else:
            residual = Variable(torch.zeros(field_so_far.size())).cuda()

        residuals_so_far.insert(0, residual)
        return residual + field_so_far, residuals_so_far


# wrapper
class PyramidTransformer(nn.Module):
    def __init__(self, num_lvl, min_lvl, base_dim):
        super(PyramidTransformer, self).__init__()
        self.pyramid = Pyramid(num_lvl, min_lvl, base_dim)

    def forward(self, source, target):
        field, residuals = self.pyramid.forward(source, target, lvl=0)
        pred = grid_sample(source, field).squeeze(1)
        
        present_mask = (pred.clone()).gt(0).float().unsqueeze(3).repeat(1,1,1,2) 
        absent_mask = 1 - present_mask  
        identity = get_identity_grid(pred.size()[-1])
        field = field * present_mask + identity * absent_mask

        return pred, field, residuals
    
