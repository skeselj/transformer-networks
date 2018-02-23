# Define providers, which get data for us

import os, h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class VerticalCrackDset(Dataset):
    def __init__(self):
        self.datadir = "data/vcrack/"
    def __len__(self):
        return 20*250  # hard-coded, for now
    def __getitem__(self, idx):
        f = idx / 20
        i = idx % 20
        a = h5py.File(self.datadir + str(f) + ".h5", 'r')['main']
        return a[i,1,:,:], a[i,0,:,:]
        
