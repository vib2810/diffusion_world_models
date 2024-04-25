# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import pickle
import numpy as np
import torch

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # set torch to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Adapted from https://diffusion-policy.cs.columbia.edu/ 
# normalize data
def get_data_stats(data):
    # data shape is (N, ep_len, dim)
    data = data.reshape(-1,data.shape[-1])
    # data shape is (N*ep_len, dim)
    stats = {
        'min': torch.min(data, axis=0).values,
        'max': torch.max(data, axis=0).values,
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # TODO: add checks to not get nans
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data