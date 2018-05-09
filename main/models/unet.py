########################################################################################################
# Define the a UNet like model
########################################################################################################

import numpy as np

import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample


class DownSlab(nn.Module):
    def __init__(self, n, k=5, act=nn.ReLU()):
        super(DownSlab, self).__init__()
        p = (k-1)/2
        conv1 = nn.Conv2d(n, 2*n, k, padding=p)
        conv2 = nn.Conv2d(2*n, 2*n, k, padding=p)
        down = nn.MaxPool2d(2, 2)
        self.seq = nn.Sequential(conv1, act, conv2, act, down)
    def forward(self, stack):
        return self.seq(stack)

class UpSlab(nn.Module):
    def __init__(self, n, k=5, act=nn.ReLU()):
        super(UpSlab, self).__init__()
        p = (k-1)/2
        conv1 = nn.Conv2d(n, n//2, k, padding=p)
        conv2 = nn.Conv2d(n//2, n//2, k, padding=p)
        up = nn.ConvTranspose2d(n//2, n//2, k, stride=2, padding=p-1)
        self.seq = nn.Sequential(conv1, act, conv2, act, up)
    def forward(self, stack):
        return self.seq(stack)[:,:,:-1,:-1]

class CustomSlab(nn.Module):
    def __init__(self, n1, n2, n3, k=5, act=nn.ReLU()):
        super(CustomSlab, self).__init__()
        p = (k-1)/2
        self.conv1 = nn.Conv2d(n1, n2, k, padding=p)
        self.conv2 = nn.Conv2d(n2, n3, k, padding=p)
        self.seq = nn.Sequential(self.conv1, act, self.conv2, act)
    def forward(self, stack):
        return self.seq(stack)

# 5 downsamples, ~3M params, UNet style
class UNet(nn.Module):
    def __init__(self, n=40, k=5, act=nn.ReLU()):
        super(UNet, self).__init__()

        self.inslab = CustomSlab(2, n, n)
        self.d1 = DownSlab(n)
        self.d2 = DownSlab(2*n)
        self.d3 = DownSlab(4*n)
        self.d4 = DownSlab(8*n)
        self.d5 = DownSlab(16*n)
        self.u5 = UpSlab(32*n)
        self.u4 = UpSlab(16*n)
        self.u3 = UpSlab(8*n)
        self.u2 = UpSlab(4*n)
        self.u1 = UpSlab(2*n)
        self.outslab = CustomSlab(n, n, 2)

    def forward(self, stack):
        a0 = self.inslab(stack)
        a1 = self.d1(a0)
        a2 = self.d2(a1)
        a3 = self.d3(a2)
        a4 = self.d4(a3)
        bottom = self.d5(a4)
        b4 = self.u5(bottom)
        b4 += a4
        b3 = self.u4(b4)
        b3 += a3
        b2 = self.u3(b3)
        b2 += a2
        b1 = self.u2(b2)
        b1 += a1
        b0 = self.u1(b1)
        b0 += a0
        out = self.outslab(b0)
        return out.div(10)


def get_identity_grid(dim):
    gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
    I = np.stack(np.meshgrid(gx, gy))   # (2, dim, dim)
    I = np.expand_dims(I, 0)   # (1, 2, dim, dim)
    I = Variable(torch.Tensor(I)).cuda()
    I = I.permute(0,2,3,1)   # (1, dim, dim, 2)
    return I

# wrapper
class Transformer(nn.Module):
    def __init__(self, n=40, k=5):
        super(Transformer, self).__init__()
        self.unet = UNet(n=40, k=5)

    def forward(self, source, target, min_lvl):   # legacies, famine
        stack = torch.cat([source, target], 1)

        residual = self.unet.forward(stack).permute(0,2,3,1)  # (B, W, W, 2)
        identity = get_identity_grid(stack.size()[-1])

        field = residual + identity
        pred = grid_sample(source, field).squeeze(1)
        
        present_mask = (pred.clone()).gt(0).float().unsqueeze(3).repeat(1,1,1,2) 
        absent_mask = 1 - present_mask
        field = field * present_mask + identity * absent_mask

        return pred, field, None
    
