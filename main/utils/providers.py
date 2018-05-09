# Define providers, which get data for us

import os, h5py
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


### Stacks

class StackDset(Dataset):
    """ Uses a stack to give you pairs, and can also give you the whole stack
    """
    def __init__(self, datafile):
        self.datafile = datafile
        hf = h5py.File(self.datafile, 'r')
        self.n_stacks = len(hf.keys())
        H, Wx, Wy = hf['0'].shape
        assert Wx == Wy
        hf.close()
        
        self.H = H
        self.W = Wx
        self.size_stack = 2 * (H-1)
        self.length = self.size_stack * self.n_stacks
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.datafile, 'a') as f:
            stack = np.array(f[str(idx // self.size_stack)])
            idx = idx % self.size_stack
        isReversed = idx // (self.size_stack//2)
        idx = idx % (self.size_stack//2)
        frame, target = stack[idx,:,:], stack[idx+1,:,:]
        if isReversed: frame, target = target, frame
        return frame, target

    def get_stack(self, idx):
        with h5py.File(self.datafile, 'a') as f:
            stack = np.array(f[str(idx)])
        return stack


### Pairs
    
class LumpH5Dset(Dataset):
    """ sample a datatset that is stored in a single h5py file  """
    
    def __init__(self, datafile):
        self.datafile = datafile
        hf = h5py.File(self.datafile, 'r')

        len_f = hf['frame'].shape[0]
        len_t = hf['target'].shape[0]
        if len_f != len_t:
            raise ValueError('length mismatch %d != %d' % (len_f, len_t))

        height_f, width_f = hf['frame'].shape[1], hf['frame'].shape[2]
        height_t, width_t = hf['target'].shape[1], hf['frame'].shape[2]
        if height_f != height_t:
            raise ValueError('height mismatch %d != %d' % (height_f, height_t))
        if width_f != width_t:
            raise ValueError('width mismatch %d != %d' % (width_f, width_t))
        if width_f != height_f:
            raise ValueError('h-w mismatch %d != %d' % (width_f, height_f))
        
        hf.close()

        self.length = len_f
        self.dim = height_f
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hf = h5py.File(self.datafile, 'r')
        frame = hf['frame'][idx,:,:]
        target = hf['target'][idx,:,:]
        hf.close()
        return frame, target

    
class HybridDset(Dataset):
    """ Build multiple lump datsets """

    def __init__(self, datafiles):
        self.datafiles = datafiles
        self.dsets = [LumpH5Dset(datafile) for datafile in datafiles]
        self.lens = [len(dset) for dset in self.dsets]
        self.clens = reduce(lambda c,x: c + [c[-1] + x], self.lens, [0])  # includes 0
        for i, dset in enumerate(self.dsets):
            if i != len(self.dsets)-1:
                assert dset.dim == self.dsets[i+1].dim
        self.dim = self.dsets[0].dim
        
        
    def __len__(self):
        return self.clens[-1]
    
    def __getitem__(self, idx):
        for i, clen in enumerate(self.clens):
            if idx < clen:
                return self.dsets[i-1][idx-self.clens[i-1]]


"""

### Artificial
import torchsample
class SmoothBrightDset(Dataset):
    def __init__(self, dim):
        factor = 255.0 / dim / 2
        main_img = torch.zeros((1,dim,dim))
        for i in range(dim):
            for j in range(dim):
                main_img[0,i,j] = factor * (i + j)
        img1 = main_img
        img2 = torchsample.transforms.Rotate(90)(main_img)
        img3 = torchsample.transforms.Rotate(180)(main_img)
        img4 = torchsample.transforms.Rotate(270)(main_img)
        self.imgs = [img1, img2, img3, img4]

    def get(self):
        return self.imgs[random.randint(0,3)]

"""
