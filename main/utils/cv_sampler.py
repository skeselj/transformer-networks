import torch  # get a warning if I don't do this lol
import cloudvolume as cv
import numpy as np
import httplib2shim
from torch.autograd import Variable

class Sampler(object):
    def __init__(self, source, mip=1, dim=512, is_test=False):
        httplib2shim.patch()
        if source == 'basil': source = 'gs://neuroglancer/basil_v0/raw_image' 
        elif source == 'pinky40_unaligned': source = 'gs://neuroglancer/pinky40_v11/image' 
        self.vol = cv.CloudVolume(source, mip=mip)
        self.dim = dim
        self.vol_info = self.vol.info['scales'][0]
        self.vol_size = self.vol_info['size']
        self.vol_offsets = self.vol_info['voxel_offset']
        self.adj_dim = self.dim * 2 ** self.vol.mip
        self.is_test = is_test

    def chunk_at_global_coords(self, xyz, xyz_):
        factor = 2 ** self.vol.mip
        x, x_ = xyz[0]/factor, xyz_[0]/factor
        y, y_ = xyz[1]/factor, xyz_[1]/factor
        z, z_ = xyz[2], xyz_[2]
        if x < self.vol_offsets[0]/factor or \
           x_ >= (self.vol_offsets[0]+self.vol_size[0])/factor or \
           y < self.vol_offsets[1]/factor or \
           y_ >= (self.vol_offsets[1]+self.vol_size[1])/factor or \
           z < self.vol_offsets[2] or \
           z_ >= self.vol_offsets[2]+self.vol_size[2]:
            raise ValueError("%d:%d %d:%d %d:%d not inbounds" % (x,x_,y,y_,z,z_))
        else:
            return np.squeeze(self.vol[x:x_, y:y_, z:z_])  # can raise exception
            
    def random_sample(self, train=True, offsets=None, size=None, split=True):
        if offsets is None:
            offsets = self.vol_offsets
        if size is None:
            size = self.vol_size
        x = np.random.randint(offsets[0], offsets[0] + size[0] - self.adj_dim)
        y = np.random.randint(offsets[1], offsets[1] + size[1] - self.adj_dim)
        if train:  # train on z = 0 or 1 mod 10$
            z = np.random.randint(offsets[2], offsets[2] + size[2] - 1)
            while z % 10 == 0 or z % 10 == 1:
                z = np.random.randint(offsets[2], offsets[2] + size[2] - 1)
        else:  # test on all layers
            z = (np.random.randint(offsets[2], offsets[2] + size[2] - 1) / 10) * 10
            
        this_chunk = self.chunk_at_global_coords(
                                      (x,y,z),
                                      (x+self.adj_dim, y+self.adj_dim, z+1))  # watch the 1
        #print 'drawing random sample at', x, y, z, 'train is set to', train
        return this_chunk

    def get_chunk(self):
        chunk = None
        while chunk is None:
            chunk = self.random_sample(train = not self.is_test)
            counts = dict(zip(*np.unique(chunk, return_counts=True)))  # can be replaced with where
            if 0 in counts:
                print '  tossing a chunk with a 0'
                chunk = None
        return chunk
