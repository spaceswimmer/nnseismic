import torch
import numpy as np
import torch.nn as nn
from .ssim3d import MultiScaleSSIMLoss3d as ms_ssim3d

class ssim3DLoss(nn.Module):
    """Calculate loss function of RGT"""
    def __init__(self):
        super(ssim3DLoss, self).__init__()
        self.ssim = ms_ssim3d()
        self.name = "SSIM"
    def forward(self, output, target):
        loss = (1 - self.ssim(output, target))
        return loss
    def getLossName(self):
        return self.name

    """Calculate loss function of RGT"""
    def __init__(self):
        super(mse3DLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.name = "MSE"
    def forward(self, output, target):
        loss = self.mse(output, target)
        return loss
    def getLossName(self):
        return self.name
  
