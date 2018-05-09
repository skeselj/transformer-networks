#######################################################################################################
# Some functions to help me take a peek into what these nets are up to
#######################################################################################################

import os, h5py, math
import cv2

import numpy as np
import torch

import matplotlib, matplotlib.pyplot as plt

def save_fig(img, name):
    if type(img) == torch.autograd.Variable:
        img = img.data
    if type(img) == torch.Tensor:
        img = img.cpu().numpy().squeeze()

    plt.set_cmap('Greys_r')
    plt.imshow(img)
    plt.savefig(name)
    plt.title(name)
    plt.clf()
    
def save_h5(stack, filename):
    h5f = h5py.File(filename, "w")
    h5f.create_dataset(u'main', data=stack)
    h5f.close()

from moviepy.editor import ImageSequenceClip
def save_gif(stack, filename):
    fps = 8; scale = 1.0
    assert len(stack.shape) == 3
    stack = stack[..., np.newaxis] * np.ones(3)   # RGB
    clip = ImageSequenceClip(list(stack), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps, verbose=False)

def save_field(field, name, isResidual=False):
    if type(field) != np.ndarray:
        field = field.data.cpu().numpy().squeeze()

    plt.set_cmap('hsv')
    fig = plt.figure(figsize=(10,10))
    
    w = field.shape[0]
    gx, gy = np.linspace(-1, 1, w), np.linspace(-1, 1, w)
    X, Y = np.meshgrid(gx, gy)
    U, V = np.squeeze(np.vsplit(np.moveaxis(field,[0,1,2],[1,2,0]),2))  # watch this line
    if not isResidual: U, V = U - X, V - Y
    colors = np.arctan2(U, V) 

    ax = plt.subplot(1,1,1)
    ax.invert_yaxis()
    Q = ax.quiver(X, Y, U, V, colors, 
                  scale = 1, width = 0.003, 
                  angles='uv', pivot='tail')
    ax.set_title('field')
    fig.savefig(name)
    plt.close(fig) 



    
# remove margin from each side of an set images
def rem_margin(a, m=0.125):
    if type(a) == np.ndarray:
        s = a.shape
    elif type(a) == torch.Tensor or type(a) == torch.autograd.variable.Variable:
        s = a.size()
    else:
        raise TypeError("Don't know what to do with type %s" % (type(a)))
    
    if len(s) == 2:
        h, w = s
        p = int(math.ceil(w*m))
        return a[p:w-p,p:w-p]
    elif len(s) == 3:
        b, h, w = s
        p = int(math.ceil(w*m))
        return a[:,p:w-p,p:w-p]
    elif len(s) == 4:
        b, h, w, t = s
        p = int(math.ceil(w*m))
        return a[:,p:w-p,p:w-p,:]
    else:
        raise ValueError("a.size() = %d" % a.size())

# add margin from each side of set of images    
def add_margin(a, m=0.125):
    if type(a) == np.ndarray:
        s = a.shape
    elif type(a) == torch.Tensor or type(a) == torch.autograd.variable.Variable:
        s = a.size()
    else:
        raise TypeError("Don't know what to do with type %s" % (type(a)))
    
    if len(s) == 2:
        h, w = s
        p = int(w*m)
        nw = int(w*(1+2*m))
        res = np.zeros((nw,nw))
        res[p:p+w, p:p+w] = a
    elif len(s) == 3:
        b, h, w = s
        p = int(w*m)
        nw = int(w*(1+2*m))
        res = np.zeros((b,nw,nw))
        res[:, p:p+w, p:p+w] = a
    elif len(s) == 4:
        b, h, w, t = s
        p = int(math.ceil(w*m))
        nw = int(w*(1+2*m))
        res = np.zeros((b,nw,nw,t))
        res[:, p:p+w, p:p+w, :] = a
    else:
        raise ValueError("a.size() = %d" % a.size())
    return res


# this is an old, messed up function that deserves to be deleted

def save_iter(name, frame, target, pred, field, residuals, margin=0.0):
    # turn into numpy
    frame = frame[0].data.cpu().numpy().squeeze()
    target = target[0].data.cpu().numpy().squeeze()
    pred = pred[0].data.cpu().numpy().squeeze()
    field = field[0].data.cpu().numpy().squeeze()
    residuals = [res[0].data.cpu().numpy().squeeze() for res in residuals]
    
    # save data
    #h5f = h5py.File(name + "_data.h5", "w")
    #h5f.create_dataset(u'frame', data=frame)
    #h5f.create_dataset(u'target', data=target)
    #h5f.create_dataset(u'pred', data=pred)
    #h5f.create_dataset(u'field', data=field)
    #h5f.close()
    
    # save images
    plt.set_cmap('Greys_r')
    fig = plt.figure(1, figsize=(10, 10))

    ax = plt.subplot(2,2,1)
    ax.imshow(frame)
    ax.set_xlabel('frame')
    ax.xaxis.set_label_position('top')

    ay = plt.subplot(2,2,2)
    ay.imshow(target)
    ay.set_xlabel('target')
    ay.xaxis.set_label_position('top')
    
    bx = plt.subplot(2,2,3)
    bx.imshow(pred)
    bx.set_xlabel('pred')

    by = plt.subplot(2,2,4)
    by.imshow(np.abs(target-pred))
    by.set_xlabel('| pred - target |')

    fig.savefig(name + "_images")
    plt.close(fig) 
    
    # save field
    plt.set_cmap('rainbow')
    fig = plt.figure(figsize=(10,10))
    
    w = field.shape[0]
    gx, gy = np.linspace(-1, 1, w), np.linspace(-1, 1, w)
    X, Y = np.meshgrid(gx, gy)
    U, V = np.squeeze(np.vsplit(np.moveaxis(field,[0,1,2],[1,2,0]),2))  # watch this line
    #m_fact = 1.0 / (1-2*margin)
    #U, V = U - X*(1-2*margin), V - Y*(1-2*margin)
    U, V = U - X, V - Y
    colors = np.arctan2(U, V) 

    ax = plt.subplot(1,1,1)
    ax.invert_yaxis()
    Q = ax.quiver(X, Y, U, V, colors, 
                  scale = 2, width = 0.0001, 
                  angles='uv', pivot='tail')
    ax.set_title('field')
    fig.savefig(name + '_field')
    plt.close(fig) 

    # save residuals
    nr = len(residuals)-1
    fig = plt.figure(figsize=(2*10,10*nr/2))
    axs = [fig.add_subplot((nr+1)/2, 2, i+1) for i in range(nr)]

    for i, res in enumerate(residuals[:-1]):
        w = res.shape[0]
        gx, gy = np.linspace(-1, 1, w), np.linspace(-1, 1, w)
        X, Y = np.meshgrid(gx, gy)
        U, V = np.squeeze(np.vsplit(np.moveaxis(res,[0,1,2],[1,2,0]),2))  # watch this line
        colors = np.arctan2(U, V)
        axs[i].set_title("Residual %d (%d x %d)" % (i,w,w))
        axs[i].invert_yaxis()
        Q = axs[i].quiver(X, Y, U, V, colors, 
                          scale = 2.0 / (1.25)**i, width = 0.0001 * (1.25)**i, 
                          angles='uv', pivot='tail')

    fig.savefig(name + '_residuals')
    plt.close(fig)
    plt.clf()
