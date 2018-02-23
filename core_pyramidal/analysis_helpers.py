########################################################################################################
# Some functions to help me take a peek into what these nets are up to
########################################################################################################

import os, h5py
import numpy as np
import matplotlib, matplotlib.pyplot as plt

def save_image(image, name):
    plt.imsave(name + '.png', 1 - image, cmap='Greys')

def save_field(field, name):
    X, Y = np.meshgrid(np.arange(-1, 1, 2.0/field.shape[-2]), np.arange(-1, 1, 2.0/field.shape[-2]))
    U, V = np.squeeze(np.vsplit(np.swapaxes(field,0,-1),2))
    colors = np.arctan2(U,V)   # true angle
    plt.title('Field')
    plt.gca().invert_yaxis()
    Q = plt.quiver(X, Y, U, V, colors, scale=6, width=0.002, angles='uv', pivot='tail')
    qk = plt.quiverkey(Q, 10.0, 10.0, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
    plt.savefig(name + '.png')
    plt.clf()
