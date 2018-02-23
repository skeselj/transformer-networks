#####################################################################################
# analysis_helpers.py
#
# Functions for looking at images and vector fields,
# functions for managing all of my recorded runs.
#####################################################################################

import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


img_w = 128

# Visualizing curves and samples

def display_img(a, b, c):
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(a, cmap='Greys')
    plt.subplot(1,3,2)
    plt.title("Transformed")
    plt.imshow(b, cmap='Greys')
    plt.subplot(1,3,3)
    plt.title("Predicted")
    plt.imshow(c, cmap='Greys')
    plt.show()

def display_v(V_pred, name):
    X, Y = np.meshgrid(np.arange(-1, 1, 2.0/V_pred.shape[-2]), np.arange(-1, 1, 2.0/V_pred.shape[-2]))
    U, V = np.squeeze(np.vsplit(np.swapaxes(V_pred,0,-1),2))
    colors = np.arctan2(U,V)   # true angle
    plt.title('V_pred')
    plt.gca().invert_yaxis()
    Q = plt.quiver(X, Y, U, V, colors, scale=6, width=0.002, angles='uv', pivot='tail')
    qk = plt.quiverkey(Q, 10.0, 10.0, 2, r'$2 \frac{m}{s}$', labelpos='E', \
                       coordinates='figure')

    plt.savefig(name + '.png')
    plt.clf()

def vis_grid(sample_file):
    f = h5py.File(sample_file, 'r')

    for i in range(9):
        img_orig = f["img_orig"][i,:,:]
        img_tran = f["img_tran"][i,:,:]
        img_pred = f["img_pred"][i,:,:]
        v_pred   = f["v_pred"][i,:,:,:]
        
        plt.subplot(3,3,i+1)
        
        X, Y = np.meshgrid(np.arange(-1, 1, 2.0/img_w), np.arange(-1, 1, 2.0/img_w))
        U, V = np.squeeze(np.vsplit(np.swapaxes(v_pred,0,-1),2))
        colors = np.arctan2(U,V)   # true angle
        plt.gca().invert_yaxis()
        Q = plt.quiver(X, Y, U, V, colors, scale=6, width=0.002, angles='uv', pivot='tail')
        qk = plt.quiverkey(Q, 10.0, 10.0, 2, r'$2 \frac{m}{s}$', labelpos='E', \
                           coordinates='figure')

    plt.show()
    f.close()
    
def vis_sample(sample_file, i=0):
    f = h5py.File(sample_file, 'r')
    img_orig = f["img_orig"][i,:,:]
    img_tran = f["img_tran"][i,:,:]
    img_pred = f["img_pred"][i,:,:]
    v_pred   = f["v_pred"][i,:,:,:]
    display_img(img_orig, img_tran, img_pred)
    display_v(v_pred)
    f.close()

def vis_curve(curve_file, showTest=False):
    f = open(curve_file, 'r')

    train_iters = []; train_losses = []
    test_iters = []; test_losses = []
    
    isTrain = True
    for l in f.readlines():
        arr = l.strip().split()
        if arr[0] == 'Test...': isTrain = False
        if len(arr) != 7: continue
        if isTrain:
            train_iters.append(float(arr[3][:-1]))
            train_losses.append(float(arr[6].split('=')[-1]))
        else:
            test_iters.append(float(arr[3][:-1]))
            test_losses.append(float(arr[6].split('=')[-1]))
            
    plt.plot(train_iters, train_losses, c='b')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training-- end: ' + str(train_losses[-1]))
    plt.grid(True)
    plt.show()
            
    if showTest:
        plt.plot(test_iters, test_losses, c='b')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Testing-- end: ' + str(test_losses[-1]))
        plt.grid(True)
        plt.show()

    f.close()

