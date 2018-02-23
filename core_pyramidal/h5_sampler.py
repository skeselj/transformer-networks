import numpy as np
import torch
from torchvision.transforms import *
from torch.autograd import Variable
import h5py
import scipy
    
class Sampler(object):
    def get_file(self, source, num, cuda=True, dim=256):
        h5f = h5py.File(source, 'r')
        data = np.reshape(h5f['main'][:num], (num,-1,dim,dim))
        data = torch.FloatTensor(data)
        if cuda:
            data = data.cuda()
        h5f.close()
        return data

    def __init__(self, data_train='../data/prealigned_train_mip2.h5', data_test='../data/prealigned_test_mip2.h5', train_count=5000, test_count=500, cuda=True, dim=1024):
        self.data_train = self.get_file(data_train, train_count, cuda=cuda, dim=dim) / 255
        self.data_test = self.get_file(data_test, test_count, cuda=cuda, dim=dim) / 255
        self.count_train, self.count_test = train_count, test_count
        self.epoch()

    def epoch(self):
        self.queue = np.random.permutation(self.count_train)
        self.idx = 0

    def swap(self, chunk):
        x0 = chunk[:,0:1].clone()
        x1 = chunk[:,1:2].clone()
        if chunk.size()[1] == 3:
            x2 = chunk[:,2:3].clone()
            return torch.cat((x1,x0,x2), 1)
        else:
            return torch.cat((x1,x0), 1)
        
    def random_sample(self, train=True):
        if not train:
            idx = np.random.randint(self.count_test)
            chunk = self.data_test[idx:idx+1]
            chunk = self.swap(chunk)
#            y = chunk[:,0:1,:,:]
#            dd = 5
#            s2 = y.size()[-1] / 2
#            y[:,:,:s2-dd,:] = y[:,:,dd:s2,:]
#            y[:,:,s2+dd:,:] = y[:,:,s2:-dd,:]
#            y[:,:,s2-dd:s2+dd,:] = 0
#            chunk[:,0,:,:] = y
            return Variable(chunk)
        else:
            if self.idx >= self.count_train:
                self.epoch()
            chunk = self.data_train[self.queue[self.idx]:self.queue[self.idx]+1]
            self.idx += 1
            chunk = self.swap(chunk)
            return Variable(chunk)
