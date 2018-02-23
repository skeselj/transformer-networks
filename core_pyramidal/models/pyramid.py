########################################################################################################
# Define the a SpyNet like model
########################################################################################################

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample  # in (N x C x IH x IW), out (N x OH x OW x 2)


# function G makes up a single level
#  input: current version of frame, target
#  ouput: residual, probs
class G(nn.Module):
    def __init__(self, dim=256, k=7, f=nn.ReLU()):
        super(G, self).__init__()
        self.dim = dim
        print 'building G for dim:', dim, 'with kernel size ', k
        p = (k-1)/2
        self.conv1 = nn.Conv2d(2, 32, k, padding=p)
        self.conv2 = nn.Conv2d(32, 64, k, padding=p)
        self.conv3 = nn.Conv2d(64, 32, k, padding=p)
        self.conv4 = nn.Conv2d(32, 16, k, padding=p)
        self.conv5 = nn.Conv2d(16, 2, k, padding=p)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2, f,
                                 self.conv3, f,
                                 self.conv4, f,
                                 self.conv5)

    def forward(self, x):  # x dims are probs (B, 2, W, W)
        return torch.mul(self.seq(x).permute(0,2,3,1), 0.1)


# main pyramid class, defines forward function at each level
class Pyramid(nn.Module):
    def get_identity_grid(self, dim):
        gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
        I = np.stack(np.meshgrid(gx, gy))
        I = np.expand_dims(I, 0)
        I = torch.autograd.Variable(torch.Tensor(I), requires_grad=False).cuda()
        I = I.permute(0,2,3,1)
        return I

    def __init__(self, size, dim):
        super(Pyramid, self).__init__()
        rdim = dim / (2 ** (size))
        print '\n------- Constructing PyramidNet with size', size, \
              '(' + str(size-1) + ' downsamples) [' + str(rdim) + '] -------'
        self.size = size
        self.mlist = nn.ModuleList([G(dim / (2 ** level)) for level in xrange(size)])
        self.f_up = lambda x: nn.Upsample(scale_factor=x, mode='bilinear')
        self.up = self.f_up(2)
        self.down = nn.AvgPool2d(2, 2)
        self.I = self.get_identity_grid(rdim)
        self.Zero = self.I - self.I

    def forward(self, stack, idx=0):
        if idx < self.size:
            field_so_far, residuals_so_far = self.forward(self.down(stack), idx + 1) # (B, dim, dim, 2)
            field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)
        else:
            return self.I, [ self.I ]
        resampled_source = grid_sample(stack[:,0:1], field_so_far)  # for some reason
        new_stack = torch.cat((resampled_source, stack[:,1:2]),1)
        residual = self.mlist[idx](new_stack)
        residuals_so_far.insert(0, residual)
        return residual + field_so_far, residuals_so_far

    
from analysis_helpers import save_image, save_field   # temp
# actual top level class
class PyramidTransformer(nn.Module):
    def __init__(self, size=4, dim=256):
        super(PyramidTransformer, self).__init__()
        self.pyramid = Pyramid(size, dim)

    def select_module(self, idx):
        for g in self.pyramid.mlist:
            g.requires_grad = False
        self.pyramid.mlist[idx].requires_grad = True

    def select_all(self):
        for g in self.pyramid.mlist:
            g.requires_grad = True
        
    def forward(self, x, idx=0):
        field, residuals = self.pyramid(x, idx)
        pred = grid_sample(x[:,0:1,:,:], field)        
        return pred, field, residuals
