# This will train a pyramid net"work
# I copied this from Eric's directory and then removed the parts that didn't make sense

import os, sys, time
import argparse
import pdb  # python debugger

import torch, torch.nn.functional as F, torch.nn as nn, torch.autograd import Variable
import numpy as np, cv2
import collections
import matplotlib, matplotlib.pyplot as plt
matplotlib.use('Agg')
import scipy, scipy.ndimage as img, scipy.misc.imsave as imsave
#from scipy.misc import imsave

from analysis_helpers import display_v
from pyramid import PyramidTransformer
from helpers import gif
from h5_sampler import Sampler


PINKY = 'gs://neuroglancer/pinky40_v11/image'
ZFISH = 'gs://neuroglancer/zfish_v1/image'
PIRIFORM = 'gs://neuroglancer/piriform_v0/image'


if __name__ == '__main__':    # need to get rid of this
    
    # loss and helpers
    
    def smoothness_penalty(fields, order=1):
        factor = lambda f: f.size()[2] / 1024.0
        dx =     lambda f: (f[:,1:,:,:] - f[:,:-1,:,:]) * factor(f)
        dy =     lambda f: (f[:,:,1:,:] - f[:,:,:-1,:]) * factor(f)
        for idx in range(order):
            # given k-th derivatives, compute (k+1)-th
            fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
        # sum along last axis (x/y channel)
        square_errors = map(lambda f: torch.sum(f ** 2, -1), fields) 
        return sum(map(torch.mean, square_errors))

    def save_chunk(chunk, name):
        plt.imsave(name + '.png', 1 - chunk, cmap='Greys')

    def center(var, dims, d):
        if not isinstance(d, collections.Sequence):
            d = [d for i in range(len(dims))]
        for idx, dim in enumerate(dims):
            if d[idx] == 0:
                continue
            var = var.narrow(dim, d[idx]/2, var.size()[dim] - d[idx])
        return var

    
    # arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--test', help='only load one sample', action='store_true')
    parser.add_argument('--lambda1', type=float, default=0.0)
    parser.add_argument('--lambda2', type=float, default=10000.0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--trunc', type=int, default=0)
    parser.add_argument('--lr', help='starting learning rate', type=float, default=0.00002)
    parser.add_argument('--it', help='number of training iterations', type=int, default=1000000)
    parser.add_argument('--info_period', help='iterations between outputs', type=int, default=200)
    parser.add_argument('--batch_size', help='size of batch', type=int, default=1)
    args = parser.parse_args()

    name = args.name
    trunclayer = args.trunc
    skiplayers = args.skip
    size = args.size
    dim = args.dim
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    log_path = 'out/' + name + '/'
    log_file = log_path + name + '.log'
    it = args.it
    lr = args.lr
    info_period = args.info_period
    batch_size = args.batch_size
    print args

    if not os.path.isdir(log_path):
        os.makedirs(log_path)

        
    # model and training
        
    model = PyramidTransformer(size=size, dim=dim, skip=skiplayers).cuda()
    for p in model.parameters():
        p.requires_grad = True
    model.train(True)

    c = 600   # count
    sampler = Sampler(train_count=c, test_count=10 if args.taest else 500)
    downsample = lambda x: nn.AvgPool2d(2**x,2**x)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    start_time = time.time()
    mse = torch.nn.MSELoss()
    history = []
    updates = 0
    
    print '=========== BEGIN TRAIN LOOP ============'

    def random_sample(train=True, csample=64, coutput=64):
        csample /= (2 ** trunclayer)
        coutput /= (2 ** trunclayer)
        X = downsample(trunclayer)(sampler.random_sample(train=train))
        #gif(log_path + name + 'test' + str(t), np.squeeze(X.data.cpu().numpy()) * 255, fps=1)
        y = X[:,-1:,:,:]
        X = center(X, (-1,-2), csample)
        y = center(y, (-1,-2), csample)
        pred, field, residuals = model(X[:,:2], trunclayer)
        fieldc = center(field, (-2,-3), coutput)
        irfield = field - model.pyramid.get_identity_grid(field.size()[-2])
        irfieldc = center(irfield, (-2,-3), coutput)
        predc = center(pred, (-1,-2), coutput)
        yc = center(y, (-1,-2), coutput)
        Xc = center(X, (-1,-2), coutput)
        err = mse(predc, yc)
        return Xc[:,0:1,:,:], Xc[:,1:2,:,:], predc, irfieldc, err, residuals

#    model.select_module(trunclayer)

    for t in range(it):
        if t % (info_period * (200 if not args.test else 1)) == 0 and t > 0:
            if trunclayer > 0:
                trunclayer -= 1
#                model.select_module(trunclayer)
            else:
                pass
#                model.select_all()

        if not args.inference_only:
            #if trunclayer == 0:
            #    scheduler.step()

            # Get inputs/ground truth
            a, b, pred, field, err_train, residuals = random_sample(train=True, csample=0)
            penalty1 = smoothness_penalty(residuals[:-1], 1)
            penalty2 = smoothness_penalty(residuals[:-1], 2)
            ((err_train + lambda1 * penalty1 + lambda2 * penalty2) / batch_size).backward()

            # Update model
            if t % batch_size == 0:
                a, b, pred, field, err_test, residuals = random_sample(train=False)
                print 'TRUNC:', trunclayer, err_train.data[0], err_test.data[0], penalty1.data[0], penalty2.data[0]

                # Save some info
                history.append((time.time() - start_time, err_train.data[0], err_test.data[0]))

                optimizer.step()
                model.zero_grad()
                updates += 1

        if t % info_period == 0:
            if not args.inference_only:
                torch.save(model.state_dict(), 'pt/' + name + '.pt')

            if not args.inference_only:
                print 'Writing status to: ', log_file
                with open(log_file, 'a') as log:
                    for tr in history:
                        log.write(str(tr[0]) + ', ' + str(tr[1]) + ', ' + str(tr[2]) + '\n')

            a, b, pred, field, err, residuals = random_sample(train=False)
            a = np.squeeze(a.data.cpu().numpy())
            b = np.squeeze(b.data.cpu().numpy())
            pred = np.squeeze(pred.data.cpu().numpy())
            gif(log_path + name + 'src' + str(t//info_period), np.stack((a, pred)) * 255)
            gif(log_path + name + 'trgt' + str(t//info_period), np.stack((pred, b)) * 255)
            display_v(field.data.cpu().numpy(), log_path + name + '_field' + str(t//info_period))
            for idx, rfield in enumerate(residuals[:-1]):
                step = len(residuals) - 1
                #display_v(rfield.data.cpu().numpy()[:,::2 ** (step - idx),::2 ** (step - idx),:] * 2 ** 3, log_path + name + '_rfield' + str(idx) + '_' + str(t//info_period))
                display_v(rfield.data.cpu().numpy() * 2 ** 3, log_path + name + '_rfield' + str(idx) + '_' + str(t//info_period))
            history = []

    if not args.inference_only:
        torch.save(model.state_dict(), 'pt/' + name + '.pt')
