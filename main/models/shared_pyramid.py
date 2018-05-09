########################################################################################################
# Define the a SpyNet like model that leverages shared weights:
#   - as it downsamples
#   - at each level
########################################################################################################

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample  # in (N x C x IH x IW), out (N x OH x OW x 2)

#import sys, os    # for saving feature maps
#sys.path.append('..')
#from utils.helpers import save_fig
#import time


def get_identity_grid(dim):
    gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
    I = np.stack(np.meshgrid(gx, gy))   # (2, dim, dim)
    I = np.expand_dims(I, 0)   # (1, 2, dim, dim)
    I = Variable(torch.Tensor(I)).cuda()
    I = I.permute(0,2,3,1)   # (1, dim, dim, 2)
    return I


# generates feature maps
class DownConv(nn.Module):
    def __init__(self, n_in, down, k=5, act=nn.ReLU()):
        super(DownConv, self).__init__()
        p = (k-1)/2
        self.n_in = n_in
        self.isDown = down

        self.conv1 = nn.Conv2d(n_in, 3, k, padding=p)
        self.conv2 = nn.Conv2d(3, 3, k, padding=p)
        self.down = nn.AvgPool2d(2, 2)
        
        self.seq_down = nn.Sequential(self.conv1, self.conv2, self.down, act)
        self.seq_stay = nn.Sequential(self.conv1, self.conv2, act)

    def forward(self, stack):
        image = stack[:,0,:,:].unsqueeze(1)
        if self.isDown:
            image = self.down(image)
            feature_maps = self.seq_down(stack)
        else:
            feature_maps = self.seq_stay(stack)
        return torch.cat([image, feature_maps], 1)


# single level
class Level(nn.Module):
    def __init__(self, k=5, act=nn.ReLU()):
        super(Level, self).__init__()
        p = (k-1)/2

        self.convS1 = nn.Conv2d( 4, 32, k, padding=p) 
        self.convS2 = nn.Conv2d(32, 64, k, padding=p)

        self.seqS = nn.Sequential(self.convS1, act,    # S - separate
                                  self.convS2, act)

        self.convJ1 = nn.Conv2d(128, 64, k, padding=p)
        self.convJ2 = nn.Conv2d( 64, 32, k, padding=p)
        self.convJ3 = nn.Conv2d( 32 , 2, k, padding=p)
        
        self.seqJ = nn.Sequential(self.convJ1, act,    # J - joined
                                  self.convJ2, act,
                                  self.convJ3)

        self.convJ3.weight.data /= 100; self.convJ3.bias.data /= 100

    def forward(self, source_maps, target_maps):  # each: (B, F, W, W)
        source_maps = self.seqS(source_maps)
        target_maps = self.seqS(target_maps)
        out = self.seqJ(torch.cat([source_maps, target_maps], 1))   # (B, 2, W, W)
        return out.permute(0,2,3,1)   # (B, W, W, 2)


# pyramid 
class Pyramid(nn.Module):
    def __init__(self, num_lvl, base_dim):
        super(Pyramid, self).__init__()
        print('-- Building PyramidNet with %d levels' % num_lvl)
        self.num_lvl = num_lvl

        self.f_up = lambda x: nn.Upsample(scale_factor=x, mode='bilinear')
        self.up = self.f_up(2)

        self.Levels = nn.ModuleList([Level() for level in xrange(num_lvl)])
        
        Downs = [DownConv(n_in=1, down=False)]
        for i in range(num_lvl-1): Downs += [DownConv(n_in=4, down=True)]
        self.Downs = nn.ModuleList(Downs)

        self.sup_I = get_identity_grid(base_dim // 2**(num_lvl))

    def forward(self, source, target, lvl, min_lvl):   # stack: B x 2 x _ x _
        # at the start, compute all the feature maps we'll need
        if lvl == 0:
            self.source_fmaps, self.target_fmaps = [], []
            source_fmap, target_fmap = source, target
            for i in range(self.num_lvl):
                source_fmap, target_fmap = self.Downs[i](source_fmap), self.Downs[i](target_fmap)
                self.source_fmaps += [source_fmap]; self.target_fmaps += [target_fmap]

            #print('leggo')   # saving features
            #stamp = str(int(time.time())*100//100)
            #for level in range(5):
            #    source_fmaps_np = self.source_fmaps[level].data.squeeze().cpu().numpy()
            #    target_fmaps_np = self.target_fmaps[level].data.squeeze().cpu().numpy()
            #    for num in range(4):
            #        this_source = source_fmaps_np[num]
            #        this_source /= this_source.max()
            #        save_fig(this_source, ""
            #                 + stamp + "_source_fmap_" + str(level) + "_" + str(num))
            #        this_target = target_fmaps_np[num]
            #        this_target /= this_target.max()
            #        save_fig(this_target, ""
            #                 + stamp + "_target_fmap_" + str(level) + "_" + str(num))

        # one step above the top, we just get the identity
        if lvl == self.num_lvl:
            I = self.sup_I.repeat(1, 1, 1, 1)  # B x 2 x _ x _
            return I, []

        # below that, take the field from above
        source_fmaps, target_fmaps = self.source_fmaps[lvl], self.target_fmaps[lvl]
        field_so_far, residuals_so_far = self.forward(source_fmaps, target_fmaps, lvl+1, min_lvl)
        field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)   # B x _ x _ x 2
        field_so_far = field_so_far.repeat(4,1,1,1)
        updated_source_fmaps = grid_sample(source_fmaps.permute(1,0,2,3), field_so_far).permute(1,0,2,3)
        field_so_far = field_so_far[0].unsqueeze(0)

        # if you in the game, put in work
        if lvl >= min_lvl:
            residual = self.Levels[lvl].forward(updated_source_fmaps, target_fmaps)
        else:   # your time will come
            residual = Variable(torch.zeros(field_so_far.size())).cuda()

        residuals_so_far.insert(0, residual)
        return residual + field_so_far, residuals_so_far


# wrapper
class Transformer(nn.Module):
    def __init__(self, num_lvl, base_dim):
        super(Transformer, self).__init__()
        self.pyramid = Pyramid(num_lvl, base_dim)

    def forward(self, source, target, min_lvl):
        field, residuals = self.pyramid.forward(source, target, lvl=0, min_lvl=min_lvl)
        pred = grid_sample(source, field).squeeze(1)
        
        present_mask = (pred.clone()).gt(0).float().unsqueeze(3).repeat(1,1,1,2) 
        absent_mask = 1 - present_mask  
        identity = get_identity_grid(pred.size()[-1])
        field = field * present_mask + identity * absent_mask

        return pred, field, residuals
    
