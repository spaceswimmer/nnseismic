import os.path
import numpy as np
import torch
from torch.utils import data

def min_max_norm(x):
    """Normalize the input by its min and max"""
    x = x - np.min(x)
    x = x / np.max(x)
    return x

def mea_std_norm(x):
    """Normalize the input by its mean and standard deviation"""
    x = (x - np.mean(x)) / np.std(x)
    return x

def trace_mea_std_norm(x0):
    x = (x0 - np.mean(x0, axis = -3, keepdims=True)) / np.std(x0, axis = -3, keepdims=True)
    return x

def ordinal_embedding(x):
    reserve_bit = 2
    x = np.round(min_max_norm(x)*(10**reserve_bit))
    return x

class Dataset(data.Dataset):
    def __init__(self, root_dir, list_IDs, transform=None, only_load_input=False):
        'Initialization'
        self.list_IDs  = list_IDs
        self.root_dir  = root_dir
        self.transform = transform
        self.only_load_input = only_load_input
        self.seis_dir  = os.path.join(root_dir,'seis')
        self.reserve_bit = 0
        if not self.only_load_input:
            self.rgt_dir = os.path.join(root_dir,'rgt')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        seis_path = os.path.join(self.seis_dir, ID)
        if not self.only_load_input:
            rgt_path  = os.path.join(self.rgt_dir,  ID)
       
        X = np.fromfile(seis_path, dtype=np.float32)
        X = self.transform(X)
        X = mea_std_norm(X)
        X = torch.from_numpy(X)

        if not self.only_load_input:
            Y = np.fromfile(rgt_path,  dtype=np.float32)
            Y = self.transform(Y)
            Y = mea_std_norm(Y)
            Y = torch.from_numpy(Y)
            return X, Y
        else:
            return X
