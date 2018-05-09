########################################################################################################
# Define the a SpyNet like model
########################################################################################################

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample  # in (N x C x IH x IW), out (N x OH x OW x 2)



class DownConv(nn.Module):
    def __init__(self, k=5, f=nn.ReLU()):
        super(DownConv, self).__init__()
        p = (k-1) // 2

# some custom thing eric did
class DConv(nn.Module):
    def __init__(self, infm, outfm, k, padding, dilation=1, groups=1, f=nn.ReLU(inplace=True)):
        assert infm == outfm

        super(DConv, self).__init__()
        self.f = f
        self.conv = nn.Conv2d(infm, outfm, k, padding=padding, groups=groups, dilation=dilation)
        weights = torch.zeros((outfm, infm, k, k)).normal_(0, 0.01)
        for i in range(infm):
            weights[i,i,k//2,k//2] = 1
        self.conv.weight = nn.Parameter(weights)
        self.conv.bias.data /= 10

    def forward(self, x):
        return self.f(self.conv(x))

# dilation level
class DG(nn.Module):
    def __init__(self, k=5, f=nn.ReLU(), t=1):   # t = ntargets
        super(DG, self).__init__()
        print('building DG wtih %dx%d kernels and %d targets' % (k, k, t))
        p = (k-1) // 2; d = (k+1) // 2
        self.f = f
        fm = 32 * (t+1)
        self.conv1 = nn.Conv2d(t+1, fm, k, padding=p, groups=t+1)
        self.conv2 = nn.Conv2d(fm, fm, k, padding=p)
        self.conv3 = DConv(fm, fm, k, padding=p*d, dilation=d)
        self.conv4 = DConv(fm, fm, k, padding=p*d*2, dilation=d*2)
        self.conv5 = DConv(fm, fm, k, padding=p*d*4, dilation=d*4)
        self.conv6 = DConv(fm, fm, k, padding=p*d*8, dilation=d*8)
        self.conv7 = nn.Conv2d(fm, 16, 3, padding=1)
        self.conv8 = nn.Conv2d(16, 2, 3, padding=1)
        self.conv8.weight.data /= 10
        self.conv8.bias.data /= 10

    def forward(self, x):
        out = self.f(self.conv1(x))
        out = self.f(self.conv2(out))
        out = self.f(self.conv3(out))
        out = self.f(self.conv4(out))
        out = self.f(self.conv5(out))
        out = self.f(self.conv6(out))
        out = self.f(self.conv7(out))
        out = self.f(self.conv8(out))
        return out.permute(0,2,3,1)


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

        #self.mlist = nn.ModuleList([G() for level in xrange(nlevels)])
        self.mlist = nn.ModuleList([DG() for level in xrange(nlevels)])

        self.f_up = lambda x: nn.Upsample(scale_factor=x, mode='bilinear')
        self.up = self.f_up(2)
        self.down = nn.AvgPool2d(2, 2)

        self.I_initialized = False

    def forward(self, stack, idx, lastlevel):   # stack: B x 2 x _ x _
        if not self.I_initialized:    # I do this here so we don't have to specify dim
            _, _, w, _ = stack.size()
            self.I = self.get_identity_grid(w / 2**self.nlevels)
            self.I_initialized = True

        # top level: return identity
        if idx == self.nlevels:
            I = self.I.repeat(stack.size()[0], 1, 1, 1)  # B x 2 x _ x _
            return I, [ I ]
        # non-top level: run levels above
        frame, target = stack[:,0:1,:,:], stack[:,1:2,:,:]
        field_so_far, residuals_so_far = self.forward(self.down(stack), idx+1, lastlevel) # B x _ x _ x 2
        field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)   # B x _ x _ x 2

        # included level: do work
        if idx >= lastlevel:
            updated_frame = grid_sample(frame, field_so_far)        
            new_stack = torch.cat((updated_frame, target), 1)
            residual = self.mlist[idx](new_stack) # B x W x W x 2
        # excluded level: pass it on
        else:
            residual = Variable(torch.zeros(field_so_far.size()), requires_grad=False).cuda().detach()

        residuals_so_far.insert(0, residual)
        return residual + field_so_far, residuals_so_far


# wrapper    
class PyramidTransformer(nn.Module):
    def __init__(self, nlevels=5):
        super(PyramidTransformer, self).__init__()
        self.pyramid = Pyramid(nlevels)

    def forward(self, x, lastlevel=2):
        field, residuals = self.pyramid.forward(x, idx=0, lastlevel=lastlevel)
        pred = grid_sample(x[:,0:1,:,:], field).squeeze()   # sample frame with field
        return pred, field, residuals

    def select_module(self, idx):
        for g in self.pyramid.mlist:
            g.requires_grad = False
        self.pyramid.mlist[idx].requires_grad = True

    def select_all(self):
        for g in self.pyramid.mlist:
            g.requires_grad = True
        
    
