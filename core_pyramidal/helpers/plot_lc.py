import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

import sys
import os

from numpy import genfromtxt

if len(sys.argv) == 1:
    print 'Pass the path to a test logfile to graph its learning curve.'
    exit()

path = sys.argv[1]
names = None
if len(sys.argv) > 3:
    names = sys.argv[3:]

data = genfromtxt(path, delimiter=',')
print 'count:', data.shape[0]
n = 100
N = data.shape[0] / n
if data.shape[0] % N != 0:
    data = data[:-(data.shape[0] % N),:]
time = np.mean(data[:,0].reshape(-1,N), axis=1)
loss = np.mean(data[:,1].reshape(-1,N), axis=1)
test = np.mean(data[:,-1].reshape(-1,N), axis=1)

print loss.shape, np.mean(loss[-10:]), np.mean(test[-10:])

fig = plt.figure()
color=iter(cm.rainbow(np.linspace(0,1,5)))
#plt.plot(time, loss, c=next(color))

plt.semilogy(np.arange(0,loss.shape[0]*N,N), loss, linewidth=0.3)
plt.semilogy(np.arange(0,loss.shape[0]*N,N), test, linewidth=0.3)
fig.savefig(os.path.dirname(path) + '/loss_' + os.path.basename(path) + '.png')
