import torch  # get a warning if I don't do this lol
import cloudvolume as cv
import numpy as np
import httplib2shim
from torch.autograd import Variable

class Sampler(object):
    def __init__(self, source='gs://neuroglancer/pinky40_v11/image', mip=1, dim=512):
        httplib2shim.patch()
        self.vol = cv.CloudVolume(source, mip=mip)
        self.dim = dim
        self.vol_info = self.vol.info['scales'][0]
        self.vol_size = self.vol_info['size']
        self.vol_offsets = self.vol_info['voxel_offset']
        self.adj_dim = self.dim * 2 ** self.vol.mip

    def chunk_at_global_coords(self, xyz, xyz_):
        factor = 2 ** self.vol.mip
        x, x_ = xyz[0]/factor, xyz_[0]/factor
        y, y_ = xyz[1]/factor, xyz_[1]/factor
        z, z_ = xyz[2], xyz_[2]
        squeezed = None
        while squeezed is None:
            try:
                squeezed = np.squeeze(self.vol[x:x_, y:y_, z:z_])
            except Exception as e:
                print e
        #print squeezed.shape
        return squeezed

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
