import numpy as np

def HorizontalFlip(data, axis = 1):
    return np.flip(data, axis=axis)

def VerticallFlip(data, rev=False):
    if rev:
        return -np.flip(data, axis=3)
    return np.flip(data, axis=3)