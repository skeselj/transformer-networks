# Define different loss functions

import torch

def basicLoss(frame, target, field, order=1):
    frame = frame.unsqueeze(1)
    if frame.size() != target.size():
        raise ValueError(frame.size(), '!=', target.size())
    fields = [field]  # for now, do this, m8!

    #mse_term = torch.nn.MSELoss()(frame, target)
    mse_term = 0

    factor = lambda f: f.size()[2] / 1024.0   # f.size()[2] is width
    dx =     lambda f: (f[:,1:,:,:] - f[:,:-1,:,:]) * factor(f)
    dy =     lambda f: (f[:,:,1:,:] - f[:,:,:-1,:]) * factor(f)
    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])  # given kth derivs, compute (k+1)th
    square_errors = map(lambda f: torch.sum(f ** 2, -1), fields)  # sum along last axis (x/y channel)
    smoothness_term = sum(map(torch.mean, square_errors))

    return mse_term + smoothness_term
