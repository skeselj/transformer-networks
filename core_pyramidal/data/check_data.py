import os
import h5py
import sys

datadir = "vcrack/"

N = 250  # num files
q = 20  # num samples in each file

for i in range(N):
    try:
        f = h5py.File(datadir + str(i) + ".h5")
        if len(f['main']) != q:
            print i, 'too short', len(f['main'])
    except:
        print i, 'missing'
