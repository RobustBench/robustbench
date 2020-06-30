import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def clean_accuracy(model, x, y, bs=100):
    acc = 0.
    n_batches = math.ceil(x.shape[0] / bs)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * bs:(counter + 1) * bs].cuda()
            y_curr = y[counter * bs:(counter + 1) * bs].cuda()
    
            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
        
    return acc / x.shape[0]