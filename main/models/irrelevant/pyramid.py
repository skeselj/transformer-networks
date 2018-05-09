########################################################################################################
# Define the a SpyNet like model
########################################################################################################

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample  # in (N x C x IH x IW), out (N x OH x OW x 2)


# single level
class G(nn.Module):
    def __init__(self, k=5, act=nn.ReLU()):
        super(G, self).__init__()

        p = (k-1)/2
        self.conv1 = nn.Conv2d( 2, 10, k, padding=p)
        self.conv2 = nn.Conv2d(10, 10, k, padding=p)
        self.conv3 = nn.Conv2d(10, 10, k, padding=p)
        self.conv4 = nn.Conv2d(10, 10, k, padding=p)
        self.conv5 = nn.Conv2d(10,  2, k, padding=p)

        self.seq = nn.Sequential(self.conv1, act,
                                 self.conv2, act,
                                 self.conv3, act,
                                 self.conv4, act,
                                 self.conv5)

        #for m in self.modules():
        #    print(m)
        #    if isinstance(m, nn.Conv2d):
        #        m.weight.data.fill_(0)
        #        m.bias.data.fill_(0)
        #        m.weight = nn.Parameter(torch.div(m.weight.data, 10))
        #        m.bias = nn.Parameter(torch.div(m.bias.data, 10))

    def forward(self, x):  # in (B, 2, W, W)
        return self.seq(x).permute(0,2,3,1)   # out (B, W, W, 2)


# pyramid 
class Pyramid(nn.Module):
    def get_identity_grid(self, dim):
        gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
        I = np.stack(np.meshgrid(gx, gy))   # (2, dim, dim)
        I = np.expand_dims(I, 0)   # (1, 2, dim, dim)
        I = Variable(torch.Tensor(I), requires_grad=False).cuda()
        I = I.permute(0,2,3,1)   # (1, dim, dim, 2)
        return I

    def __init__(self, nlevels):
        super(Pyramid, self).__init__()
        print('--- Building PyramidNet with %d levels' % nlevels)
        self.nlevels = nlevels
        self.mlist = nn.ModuleList([G() for level in xrange(nlevels)])

        self.f_up = lambda x: nn.Upsample(scale_factor=x, mode='bilinear')
        self.up = self.f_up(2)
        self.down = nn.AvgPool2d(2, 2)

        self.I_initialized = False

    def forward(self, stack, idx):   # stack: B x 2 x _ x _
        if not self.I_initialized:    # I do this here so we don't have to specify dim
            _, _, w, _ = stack.size()
            self.I = self.get_identity_grid(w / 2**self.nlevels)
            self.I_initialized = True

        # one step above the top, we just get the identity
        if idx == self.nlevels:
            I = self.I.repeat(stack.size()[0], 1, 1, 1)  # B x 2 x _ x _
            return I, [ I ]

        # in our pyramid, take what's above and do work
        frame, target = stack[:,0:1,:,:], stack[:,1:2,:,:]
        field_so_far, residuals_so_far = self.forward(self.down(stack), idx+1) # B, _, _, 2
        field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)   # B x _ x _ x 2

        updated_frame = grid_sample(frame, field_so_far)        
        new_stack = torch.cat((updated_frame, target), 1)
        residual = self.mlist[idx].forward(new_stack) # B x W x W x 2            
        residuals_so_far.insert(0, residual)

        return residual + field_so_far, residuals_so_far


# wrapper    
class PyramidTransformer(nn.Module):
    def __init__(self, nlevels):
        super(PyramidTransformer, self).__init__()
        self.pyramid = Pyramid(nlevels)

    def forward(self, x):
        field, residuals = self.pyramid.forward(x, idx=0)
        pred = grid_sample(x[:,0:1,:,:], field).squeeze()   # sample frame with field
        return pred, field, residuals


    def select_module(self, idx):
        for i, g in enumerate(self.pyramid.mlist):
            if i != idx:
                g.requires_grad = False
    def select_subpyramid(self, idx):
        for i, g in enumerate(self.pyramid.mlist):
            if i >= idx:
                g.requires_grad = True
            
    
