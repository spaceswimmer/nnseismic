import numpy as np
import torch

class Reshape(object):
    """Reshape the data to a given size."""
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, data):

        data = np.reshape(data, self.output_size)

        return data

class ToTensor(object):
    """Convert ndarrys in sample to Tensors"""
    def __call__(self, data):
        data = data.transpose((3, 2, 1, 0))
        return data

def VerticalFlip(dat):
    return torch.flip(dat, dims=[2])

def VerticalFlip_reverse(dat):
    return -torch.flip(dat, dims=[2])

def HorizontalFlip1(dat):
    return torch.flip(dat, dims=[3])

def HorizontalFlip2(dat):
    return torch.flip(dat, dims=[4])
